"""Attribution module for trace window extraction and trigger scoring.

This module supports Experiment 2 (Trigger Mining) and Experiment 5 (Impact Curriculum)
by providing functions to:
1. Extract item_ids from sample traces in windows before events
2. Score items by their frequency of appearance before events across seeds
3. Rank trigger candidates for corpus construction
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass
class TraceEntry:
    """A single entry from sample_trace.jsonl (schema v2).

    Schema v2 fields:
    - step: optimizer step number
    - micro: microbatch index within step (0 to accumulation_steps-1)
    - pos: position within microbatch (0 to batch_size-1)
    - phase: curriculum phase name
    - phase_idx: curriculum phase index
    - family_id: problem family
    - item_id: unique item identifier
    """

    step: int
    item_id: str
    micro: int = 0
    pos: int = 0
    phase: str | None = None
    phase_idx: int = -1
    family_id: str | None = None


@dataclass
class EventWindow:
    """Window of training steps associated with an event."""

    event_id: str
    """Unique identifier for this event."""

    start_step: int
    """Start of the window (inclusive)."""

    end_step: int
    """End of the window (inclusive, typically event peak)."""

    metric: str
    """Event metric (e.g., 'effective_rank')."""

    layer: int
    """Layer where event occurred."""

    run_id: str | None = None
    """Run this window is from."""


@dataclass
class TriggerCandidate:
    """A candidate trigger item with scoring information."""

    item_id: str
    """The item identifier."""

    event_count: int
    """Number of events this item preceded."""

    seed_count: int
    """Number of seeds where this item preceded events."""

    total_occurrences: int
    """Total times this item appeared in event windows."""

    families: set[str] = field(default_factory=set)
    """Families this item belongs to."""

    events: list[str] = field(default_factory=list)
    """Event IDs this item preceded."""


@dataclass
class TraceMeta:
    """Metadata from sample_trace.meta.json (schema v2)."""

    schema_version: int
    run_id: str
    seed: int
    split: str
    sampler_mode: str
    trace_file: str


def load_trace_meta(trace_path: Path) -> TraceMeta | None:
    """Load trace metadata from companion .meta.json file.

    Args:
        trace_path: Path to sample_trace.jsonl

    Returns:
        TraceMeta if meta file exists, None otherwise
    """
    meta_path = trace_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        data = json.load(f)

    return TraceMeta(
        schema_version=data.get("schema_version", 1),
        run_id=data.get("run_id", ""),
        seed=data.get("seed", 0),
        split=data.get("split", "train"),
        sampler_mode=data.get("sampler_mode", "unknown"),
        trace_file=data.get("trace_file", "sample_trace.jsonl"),
    )


def load_trace(trace_path: Path) -> list[TraceEntry]:
    """Load sample trace from JSONL file.

    Args:
        trace_path: Path to sample_trace.jsonl

    Returns:
        List of TraceEntry objects ordered by (step, micro, pos)

    Expected JSONL format (schema v2, one record per line):
        {"step":0,"micro":0,"pos":0,"phase":"warmup","phase_idx":0,"family_id":"modular_arithmetic","item_id":"omr:train:123"}

    Also supports legacy schema v1:
        {"step":0,"phase":"warmup","family_id":"modular_arithmetic","item_id":"omr:train:123"}
    """
    entries = []

    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            entries.append(
                TraceEntry(
                    step=data["step"],
                    item_id=data["item_id"],
                    micro=data.get("micro", 0),
                    pos=data.get("pos", data.get("batch_idx", 0)),
                    phase=data.get("phase"),
                    phase_idx=data.get("phase_idx", -1),
                    # Support both "family_id" (v2) and "family" (legacy)
                    family_id=data.get("family_id") or data.get("family"),
                )
            )

    # Sort by (step, micro, pos) for deterministic ordering
    entries.sort(key=lambda e: (e.step, e.micro, e.pos))
    return entries


def load_trace_as_df(trace_path: Path) -> pd.DataFrame:
    """Load sample trace as DataFrame.

    Args:
        trace_path: Path to sample_trace.jsonl

    Returns:
        DataFrame with columns: step, micro, pos, phase, phase_idx, family_id, item_id
        (schema v2) or step, phase, family_id, item_id (legacy v1)
    """
    records = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    # Sort by (step, micro, pos) if available
    sort_cols = ["step"]
    if "micro" in df.columns:
        sort_cols.append("micro")
    if "pos" in df.columns:
        sort_cols.append("pos")
    return df.sort_values(sort_cols).reset_index(drop=True)


def extract_event_windows(
    events_df: pd.DataFrame,
    window_steps: int = 20,
    run_id: str | None = None,
) -> list[EventWindow]:
    """Extract windows before each event from events dataframe.

    Args:
        events_df: DataFrame with columns: step (or peak_step), metric, layer
        window_steps: Number of steps before event to include in window
        run_id: Optional run_id to attach to windows

    Returns:
        List of EventWindow objects
    """
    windows = []

    # Determine step column name
    step_col = "peak_step" if "peak_step" in events_df.columns else "step"

    for idx, row in events_df.iterrows():
        peak_step = int(row[step_col])
        start_step = max(0, peak_step - window_steps)

        event_id = f"{run_id or 'run'}_{row['metric']}_L{row['layer']}_s{peak_step}"

        windows.append(
            EventWindow(
                event_id=event_id,
                start_step=start_step,
                end_step=peak_step,
                metric=row["metric"],
                layer=int(row["layer"]),
                run_id=run_id,
            )
        )

    return windows


def extract_window_items(
    trace: list[TraceEntry],
    window: EventWindow,
) -> list[str]:
    """Extract item_ids from trace entries within a window.

    Args:
        trace: List of TraceEntry objects (sorted by step)
        window: EventWindow defining the step range

    Returns:
        List of item_ids that appeared in the window
    """
    return [
        entry.item_id
        for entry in trace
        if window.start_step <= entry.step <= window.end_step
    ]


def extract_all_window_items(
    trace_path: Path,
    windows: list[EventWindow],
) -> dict[str, list[str]]:
    """Extract item_ids for all windows from a trace file.

    Args:
        trace_path: Path to sample_trace.jsonl
        windows: List of EventWindow objects

    Returns:
        Dict mapping event_id to list of item_ids in that window
    """
    trace = load_trace(trace_path)

    result = {}
    for window in windows:
        items = extract_window_items(trace, window)
        result[window.event_id] = items

    return result


def score_trigger_candidates(
    window_items: dict[str, list[str]],
    min_events: int = 1,
) -> pd.DataFrame:
    """Score items by frequency of appearance before events.

    Args:
        window_items: Dict mapping event_id to list of item_ids
        min_events: Minimum number of events an item must precede

    Returns:
        DataFrame with columns: item_id, event_count, total_occurrences, score
        Sorted by score descending
    """
    # Count occurrences per item
    item_events: dict[str, set[str]] = {}
    item_counts: dict[str, int] = {}

    for event_id, items in window_items.items():
        item_counter = Counter(items)
        for item_id, count in item_counter.items():
            if item_id not in item_events:
                item_events[item_id] = set()
                item_counts[item_id] = 0
            item_events[item_id].add(event_id)
            item_counts[item_id] += count

    # Build results
    records = []
    for item_id in item_events:
        event_count = len(item_events[item_id])
        if event_count < min_events:
            continue

        total_occurrences = item_counts[item_id]
        # Score: event_count weighted by frequency
        score = event_count * (1 + 0.1 * total_occurrences)

        records.append(
            {
                "item_id": item_id,
                "event_count": event_count,
                "total_occurrences": total_occurrences,
                "score": score,
            }
        )

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


def score_triggers_across_seeds(
    seed_window_items: dict[int, dict[str, list[str]]],
    min_seeds: int = 1,
    min_events: int = 1,
) -> pd.DataFrame:
    """Score items by appearance before events across multiple seeds.

    This identifies items that consistently precede events regardless of seed,
    which are the strongest trigger candidates.

    Args:
        seed_window_items: Dict mapping seed to (event_id -> item_ids)
        min_seeds: Minimum number of seeds where item must appear
        min_events: Minimum total events item must precede

    Returns:
        DataFrame with columns: item_id, seed_count, event_count, score
        Sorted by score descending
    """
    # Aggregate across seeds
    item_seeds: dict[str, set[int]] = {}
    item_events: dict[str, set[str]] = {}
    item_counts: dict[str, int] = {}

    for seed, window_items in seed_window_items.items():
        for event_id, items in window_items.items():
            item_counter = Counter(items)
            for item_id, count in item_counter.items():
                if item_id not in item_seeds:
                    item_seeds[item_id] = set()
                    item_events[item_id] = set()
                    item_counts[item_id] = 0

                item_seeds[item_id].add(seed)
                item_events[item_id].add(f"{seed}:{event_id}")
                item_counts[item_id] += count

    # Build results
    records = []
    for item_id in item_seeds:
        seed_count = len(item_seeds[item_id])
        event_count = len(item_events[item_id])

        if seed_count < min_seeds or event_count < min_events:
            continue

        # Score: seed coverage × event coverage × frequency bonus
        score = seed_count * event_count * (1 + 0.05 * item_counts[item_id])

        records.append(
            {
                "item_id": item_id,
                "seed_count": seed_count,
                "event_count": event_count,
                "total_occurrences": item_counts[item_id],
                "score": score,
            }
        )

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df


def compute_trigger_set(
    trigger_scores: pd.DataFrame,
    top_k: int = 50,
    min_score: float | None = None,
) -> list[str]:
    """Extract top-K trigger items from scored candidates.

    Args:
        trigger_scores: DataFrame from score_trigger_candidates()
        top_k: Maximum number of triggers to return
        min_score: Optional minimum score threshold

    Returns:
        List of item_ids in score order
    """
    df = trigger_scores.copy()

    if min_score is not None:
        df = df[df["score"] >= min_score]

    return df.head(top_k)["item_id"].tolist()


def find_consensus_triggers(
    test_dir: Path,
    window_steps: int = 20,
    min_seeds: int = 2,
    top_k: int = 50,
) -> pd.DataFrame:
    """Find trigger candidates that appear across seeds in a test.

    Convenience function that:
    1. Loads consensus events from test
    2. Extracts windows for each seed's events
    3. Scores items across seeds
    4. Returns top triggers

    Args:
        test_dir: Path to test directory containing runs
        window_steps: Steps before event to include
        min_seeds: Minimum seeds where item must appear
        top_k: Maximum triggers to return

    Returns:
        DataFrame of trigger candidates with scores
    """
    # This is a placeholder - actual implementation would:
    # 1. Read test.yaml to get run_ids and seeds
    # 2. For each run, load events and trace
    # 3. Extract windows and items
    # 4. Aggregate and score

    raise NotImplementedError(
        "find_consensus_triggers requires test directory structure. "
        "Use extract_event_windows() and score_triggers_across_seeds() directly."
    )
