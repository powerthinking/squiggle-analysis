import torch


def test_analysis_writes_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUIGGLE_DATA_ROOT", str(tmp_path))

    from squiggle_core import paths
    from squiggle_analysis.run import run_analysis

    run_id = "golden_analysis_smoke"

    step_dir = paths.captures_dir(run_id) / "step_000000"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "sample_meta.json").write_text('{"source": "test"}')

    x = torch.randn(2, 3, 8)
    torch.save(x, step_dir / "resid_layer_00.pt")

    run_analysis(run_id=run_id, force=True)

    assert paths.geometry_state_path(run_id).exists()
    assert paths.events_path(run_id).exists()
    assert paths.report_md_path(run_id).exists()
