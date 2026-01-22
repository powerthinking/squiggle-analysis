from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from squiggle_core.schemas.probe_summar import (
    ProbeSummary,
    probe_summary_to_probe_events_candidate_rows,
    probe_summary_to_probe_summaries_row,
)
from squiggle_core.schemas.probe_tables import (
    probe_events_candidates_schema,
    probe_summaries_schema,
)


def _cast_list(xs: Any, f) -> List[Any]:
    if xs is None:
        return []
    if not isinstance(xs, list):
        raise TypeError(f"Expected list, got {type(xs)}")
    return [f(x) for x in xs]


def _f32(x: Any) -> float:
    if x is None:
        return float("nan")
    return float(x)


def _i16(x: Any) -> int:
    return int(x)


def _i32(x: Any) -> int:
    return int(x)


def _i64(x: Any) -> int:
    return int(x)


def _normalize_probe_summaries_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)

    for k in [
        "seed",
        "steps",
    ]:
        if k in out and out[k] is not None:
            out[k] = _i32(out[k])

    if "capture_steps_used" in out and out["capture_steps_used"] is not None:
        out["capture_steps_used"] = _cast_list(out["capture_steps_used"], _i64)

    if "layers_covered" in out and out["layers_covered"] is not None:
        out["layers_covered"] = _cast_list(out["layers_covered"], _i16)

    if "affected_layers" in out and out["affected_layers"] is not None:
        out["affected_layers"] = _cast_list(out["affected_layers"], _i16)

    list_f32_cols = [
        "A_eff_rank_pre",
        "A_eff_rank_post",
        "A_eff_rank_delta",
        "A_sv_entropy_pre",
        "A_sv_entropy_post",
        "A_sv_entropy_delta",
        "A_sparsity_pre",
        "A_sparsity_post",
        "A_sparsity_delta",
        "A_principal_angle_post_vs_pre",
        "B_drift_velocity_by_layer",
        "B_drift_accel_by_layer",
        "B_volatility_by_layer",
        "B_alignment_velocity_by_layer",
        "signature_vector",
    ]
    for c in list_f32_cols:
        if c in out and out[c] is not None:
            out[c] = _cast_list(out[c], _f32)

    f32_cols = [
        "B_drift_velocity_global",
        "B_drift_accel_global",
        "B_volatility_global",
        "B_alignment_velocity_global",
        "signature_norm",
        "magnitude",
        "coherence",
        "novelty",
        "DIS",
    ]
    for c in f32_cols:
        if c in out and out[c] is not None:
            out[c] = _f32(out[c])

    return out


def _normalize_probe_events_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if "seed" in out and out["seed"] is not None:
        out["seed"] = _i32(out["seed"])
    if "timewarp_tolerance_steps" in out and out["timewarp_tolerance_steps"] is not None:
        out["timewarp_tolerance_steps"] = _i32(out["timewarp_tolerance_steps"])
    if "t_step" in out and out["t_step"] is not None:
        out["t_step"] = _i64(out["t_step"])
    if "layers" in out and out["layers"] is not None:
        out["layers"] = _cast_list(out["layers"], _i16)
    if "strength" in out and out["strength"] is not None:
        out["strength"] = _f32(out["strength"])
    if "supporting_key" in out and out["supporting_key"] is None:
        out["supporting_key"] = []
    if "supporting_val" in out and out["supporting_val"] is not None:
        out["supporting_val"] = _cast_list(out["supporting_val"], _f32)
    return out


def _iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    files = sorted(root.glob("*.json"))
    for p in files:
        if p.name.startswith("_"):
            continue
        yield p


def _load_probe_summary_json(path: Path, *, strict: bool = True) -> Optional[ProbeSummary]:
    try:
        txt = path.read_text(encoding="utf-8")
        return ProbeSummary.model_validate_json(txt)
    except Exception:
        if strict:
            raise
        return None


def _build_tables_from_json_dir(summaries_dir: Path, *, strict: bool = True) -> Tuple[pa.Table, pa.Table]:
    probe_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    for path in _iter_json_files(summaries_dir):
        ps = _load_probe_summary_json(path, strict=strict)
        if ps is None:
            continue

        probe_rows.append(_normalize_probe_summaries_row(probe_summary_to_probe_summaries_row(ps)))
        for r in probe_summary_to_probe_events_candidate_rows(ps):
            event_rows.append(_normalize_probe_events_row(r))

    probe_table = pa.Table.from_pylist(probe_rows, schema=probe_summaries_schema)
    event_table = pa.Table.from_pylist(event_rows, schema=probe_events_candidates_schema)
    return probe_table, event_table


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _unique_list(table: pa.Table, col: str, *, limit: int = 50000) -> List[Any]:
    if col not in table.column_names:
        return []
    arr = table[col]
    if isinstance(arr, pa.ChunkedArray) and arr.num_chunks > 1:
        arr = arr.combine_chunks()
    # Drop nulls
    arr = pc.drop_null(arr)
    if len(arr) == 0:
        return []
    u = pc.unique(arr)
    if isinstance(u, pa.ChunkedArray) and u.num_chunks > 1:
        u = u.combine_chunks()
    if len(u) > limit:
        # don’t explode manifests
        return u.slice(0, limit).to_pylist() + [f"...(+{len(u)-limit} more)"]
    return u.to_pylist()


def _minmax_timestamp(table: pa.Table, col: str) -> Dict[str, Optional[str]]:
    if col not in table.column_names:
        return {"min": None, "max": None}
    arr = table[col]
    if arr.num_chunks > 1:
        arr = arr.combine_chunks()
    arr = pc.drop_null(arr)
    if len(arr) == 0:
        return {"min": None, "max": None}
    mn = pc.min(arr).as_py()
    mx = pc.max(arr).as_py()
    # serialize to ISO
    def iso(x):
        if x is None:
            return None
        if isinstance(x, datetime):
            return x.astimezone(timezone.utc).isoformat()
        return str(x)
    return {"min": iso(mn), "max": iso(mx)}


def _null_counts(table: pa.Table, cols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in cols:
        if c not in table.column_names:
            continue
        arr = table[c]
        if isinstance(arr, pa.ChunkedArray) and arr.num_chunks > 1:
            arr = arr.combine_chunks()
        s = pc.sum(pc.is_null(arr)).as_py()
        out[c] = int(0 if s is None else s)
    return out


def _build_manifest(
    out_dir: Path,
    probe_table: pa.Table,
    event_table: pa.Table,
    probe_path: Path,
    events_path: Path,
    *,
    compute_hashes: bool = True,
) -> Dict[str, Any]:
    created_at = _utc_now().isoformat()

    files: Dict[str, Any] = {}
    for name, pth, tbl in [
        ("probe_summaries", probe_path, probe_table),
        ("probe_events_candidates", events_path, event_table),
    ]:
        stat = pth.stat()
        files[name] = {
            "path": pth.name,
            "bytes": stat.st_size,
            "rows": tbl.num_rows,
            "sha256": _sha256_file(pth) if compute_hashes else None,
        }

    # Keep unique lists bounded
    probe_unique = {
        "run_id": _unique_list(probe_table, "run_id"),
        "analysis_id": _unique_list(probe_table, "analysis_id"),
        "family_id": _unique_list(probe_table, "family_id"),
        "probe_name": _unique_list(probe_table, "probe_name"),
        "schema_version": _unique_list(probe_table, "schema_version"),
        "signature_version": _unique_list(probe_table, "signature_version"),
        "score_version": _unique_list(probe_table, "score_version"),
    }

    events_unique = {
        "detector_version": _unique_list(event_table, "detector_version"),
        "event_type": _unique_list(event_table, "event_type"),
    }

    manifest: Dict[str, Any] = {
        "manifest_version": "experiment_parquet_manifest@1.0",
        "created_at_utc": created_at,
        "directory": out_dir.as_posix(),
        "files": files,

        "probe_summaries": {
            "unique": probe_unique,
            "created_at_utc_minmax": _minmax_timestamp(probe_table, "created_at_utc"),
            "null_counts": _null_counts(
                probe_table,
                cols=[
                    "run_id", "analysis_id", "family_id", "probe_name",
                    "probe_config_hash", "layers_covered", "signature_vector",
                    "magnitude", "coherence", "novelty", "DIS",
                ],
            ),
        },

        "probe_events_candidates": {
            "rows": int(event_table.num_rows),
            "unique": events_unique,
            "created_at_utc_minmax": _minmax_timestamp(event_table, "created_at_utc"),
            "null_counts": _null_counts(
                event_table,
                cols=[
                    "run_id", "analysis_id", "family_id", "probe_name",
                    "event_id", "event_type", "t_step", "layers", "strength",
                ],
            ),
        },
    }

    return manifest


def write_experiment_parquet(
    summaries_dir: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
    strict: bool = True,
    write_manifest: bool = True,
    compute_hashes: bool = True,
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Writes one Parquet file per experiment:
      - probe_summaries.parquet
      - probe_events_candidates.parquet
    and optionally writes:
      - _manifest.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_path = out_dir / "probe_summaries.parquet"
    events_path = out_dir / "probe_events_candidates.parquet"
    manifest_path = out_dir / "_manifest.json"

    if not overwrite:
        for p in (probe_path, events_path, manifest_path):
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    probe_table, event_table = _build_tables_from_json_dir(summaries_dir, strict=strict)

    pq.write_table(
        probe_table,
        probe_path.as_posix(),
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    pq.write_table(
        event_table,
        events_path.as_posix(),
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    mp: Optional[Path] = None
    if write_manifest:
        manifest = _build_manifest(
            out_dir=out_dir,
            probe_table=probe_table,
            event_table=event_table,
            probe_path=probe_path,
            events_path=events_path,
            compute_hashes=compute_hashes,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        mp = manifest_path

    return probe_path, events_path, mp


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate ProbeSummary JSONs and write one Parquet set per experiment."
    )
    ap.add_argument("--summaries-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--non-strict", action="store_true")
    ap.add_argument("--no-manifest", action="store_true", help="Do not write _manifest.json")
    ap.add_argument("--no-hash", action="store_true", help="Do not compute sha256 for parquet files")
    args = ap.parse_args()

    summaries_dir = Path(args.summaries_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    probe_path, events_path, manifest_path = write_experiment_parquet(
        summaries_dir=summaries_dir,
        out_dir=out_dir,
        overwrite=args.overwrite,
        strict=not args.non_strict,
        write_manifest=not args.no_manifest,
        compute_hashes=not args.no_hash,
    )

    print("Wrote:")
    print("  -", probe_path)
    print("  -", events_path)
    if manifest_path:
        print("  -", manifest_path)


if __name__ == "__main__":
    main()
