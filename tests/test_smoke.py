def test_import():
    import squiggle_analysis

    assert hasattr(squiggle_analysis, "run_analysis")
