"""Per-section LLM analysis for notebook use.

This module provides lightweight analysis functions designed to be called
once per visualization/data section in a Jupyter notebook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from squiggle_core.llm import LLMClient, LLMRequest, build_squiggle_context


@dataclass
class SectionAnalysis:
    """Result from a single section analysis."""

    interpretation: str  # 2-3 sentence interpretation
    key_observations: list[str]  # Bullet points of notable findings
    questions: list[str]  # Follow-up questions raised
    raw_response: str  # Full response for debugging
    error: str | None = None  # Error message if analysis failed


# Shared context (cached after first call)
_CACHED_SQUIGGLE_CONTEXT: str | None = None


def _get_squiggle_context() -> str:
    """Get cached squiggle context for prompts."""
    global _CACHED_SQUIGGLE_CONTEXT
    if _CACHED_SQUIGGLE_CONTEXT is None:
        _CACHED_SQUIGGLE_CONTEXT = build_squiggle_context()
    return _CACHED_SQUIGGLE_CONTEXT


def _make_section_request(
    section_type: str,
    section_prompt: str,
    data_context: str,
    backend: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> SectionAnalysis:
    """Make a focused LLM request for a specific section.

    Args:
        section_type: Type of section (e.g., "raster_plot", "iou_curve")
        section_prompt: Specific instructions for this section type
        data_context: The data/numbers to interpret
        backend: LLM backend
        model: Model to use (default: gpt-4o-mini for speed/cost)
        temperature: Sampling temperature

    Returns:
        SectionAnalysis with interpretation and observations
    """
    client = LLMClient(backend=backend)  # type: ignore[arg-type]

    system_prompt = f"""You are an expert ML researcher analyzing Squiggle training run comparisons.

{_get_squiggle_context()}

Your task: Provide a brief, focused analysis of ONE specific visualization or data section.
Be concrete and cite specific numbers. Do not be generic."""

    user_prompt = f"""SECTION TYPE: {section_type}

{section_prompt}

DATA/CONTEXT:
{data_context}

Respond in JSON format:
{{
    "interpretation": "2-3 sentences explaining what this data shows",
    "key_observations": ["observation 1", "observation 2", "observation 3"],
    "questions": ["follow-up question 1", "follow-up question 2"]
}}"""

    llm_request = LLMRequest(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=800,
        response_format="json",
    )

    try:
        response = client.complete(llm_request)

        # Parse response
        if isinstance(response.content, dict):
            parsed = response.content
        else:
            parsed = json.loads(response.content)

        return SectionAnalysis(
            interpretation=parsed.get("interpretation", ""),
            key_observations=parsed.get("key_observations", []),
            questions=parsed.get("questions", []),
            raw_response=response.raw_response,
        )

    except Exception as e:
        return SectionAnalysis(
            interpretation="",
            key_observations=[],
            questions=[],
            raw_response="",
            error=str(e),
        )


def analyze_raster_plot(
    run_a_info: dict,
    run_b_info: dict,
    **kwargs,
) -> SectionAnalysis:
    """Analyze event raster plots comparing two runs.

    Args:
        run_a_info: Dict with keys: run_id, seed, n_events, layer_range, step_range
        run_b_info: Same structure for run B
        **kwargs: Passed to _make_section_request (backend, model, temperature)
    """
    section_prompt = """Analyze these event raster plots showing detected change points across layers and steps.

Consider:
- Are events clustered in specific layers or distributed?
- Are there temporal patterns (early burst, late clustering)?
- Do the two runs show similar spatial patterns?
- Any obvious asymmetries between runs?"""

    data_context = f"""Run A: {run_a_info.get('run_id', 'unknown')[:30]} (seed={run_a_info.get('seed', '?')})
  - Total events: {run_a_info.get('n_events', '?')}
  - Layer range with events: {run_a_info.get('layer_range', '?')}
  - Step range: {run_a_info.get('step_range', '?')}

Run B: {run_b_info.get('run_id', 'unknown')[:30]} (seed={run_b_info.get('seed', '?')})
  - Total events: {run_b_info.get('n_events', '?')}
  - Layer range with events: {run_b_info.get('layer_range', '?')}
  - Step range: {run_b_info.get('step_range', '?')}"""

    return _make_section_request("raster_plot", section_prompt, data_context, **kwargs)


def analyze_common_events(
    n_common: int,
    step_tolerance: int,
    common_events_summary: dict,
    run_a_total: int,
    run_b_total: int,
    **kwargs,
) -> SectionAnalysis:
    """Analyze common events overlay.

    Args:
        n_common: Number of common events found
        step_tolerance: Step tolerance used for matching
        common_events_summary: Dict with layer/metric distribution
        run_a_total: Total events in run A
        run_b_total: Total events in run B
    """
    section_prompt = """Analyze the common events found between two runs.

Consider:
- What fraction of events are shared?
- Is this overlap surprisingly high or low?
- Which layers/metrics contribute most to common events?
- What does this tell us about reproducibility?"""

    jaccard = n_common / (run_a_total + run_b_total - n_common) if (run_a_total + run_b_total - n_common) > 0 else 0

    data_context = f"""Common events: {n_common} (strict matching with ±{step_tolerance} step tolerance)
Run A total: {run_a_total}
Run B total: {run_b_total}
Implied Jaccard: {jaccard:.1%}

Distribution:
{json.dumps(common_events_summary, indent=2)}"""

    return _make_section_request("common_events", section_prompt, data_context, **kwargs)


def analyze_iou_invariance(
    overlap_result: dict,
    curve_data: list[dict],
    **kwargs,
) -> SectionAnalysis:
    """Analyze IoU-based invariance metrics.

    Args:
        overlap_result: Result from compute_window_overlap_invariance
        curve_data: List of dicts from compute_invariance_curve
    """
    section_prompt = """Analyze the IoU-based event matching results.

Consider:
- How does Jaccard change with IoU threshold?
- What does the IoU distribution tell us about event alignment quality?
- Is there a natural cutoff where matching collapses?
- How should we interpret the greedy efficiency?"""

    curve_summary = ""
    if curve_data:
        curve_summary = "\nIoU Curve:\n"
        for row in curve_data[:10]:
            curve_summary += f"  τ={row.get('iou_threshold', '?'):.2f}: Jaccard={row.get('jaccard', 0):.1%}, matched={row.get('n_matched', 0)}\n"

    dist = overlap_result.get("iou_distribution", {})
    data_context = f"""Window Overlap Invariance (at τ={overlap_result.get('iou_threshold', '?')}):
- Jaccard: {overlap_result.get('jaccard_overlap', 0):.1%}
- Matched pairs: {overlap_result.get('n_matched_pairs', 0)}
- Mean IoU of matches: {overlap_result.get('mean_iou', 0):.3f}
- Greedy efficiency: {overlap_result.get('greedy_efficiency', 0):.1%}

IoU Distribution:
- Max: {dist.get('max', '?')}
- P99: {dist.get('p99', '?')}
- P95: {dist.get('p95', '?')}
- Median: {dist.get('median', '?')}
{curve_summary}"""

    return _make_section_request("iou_invariance", section_prompt, data_context, **kwargs)


def analyze_iou_distribution(
    iou_values: list[float],
    thresholds: list[float] = [0.2, 0.3, 0.4],
    **kwargs,
) -> SectionAnalysis:
    """Analyze IoU distribution histogram.

    Args:
        iou_values: List of all IoU values from matching
        thresholds: Key thresholds to highlight
    """
    import numpy as np

    section_prompt = """Analyze the IoU distribution histogram.

Consider:
- What is the shape of the distribution (unimodal, bimodal, skewed)?
- What fraction of matches exceed each threshold?
- Is there evidence of distinct matching quality tiers?
- What does the max IoU tell us about alignment limits?"""

    if iou_values:
        arr = np.array(iou_values)
        threshold_stats = {f"τ≥{t}": f"{(arr >= t).sum()}/{len(arr)} ({(arr >= t).mean():.1%})" for t in thresholds}

        data_context = f"""IoU Distribution (n={len(iou_values)}):
- Mean: {np.mean(arr):.3f}
- Median: {np.median(arr):.3f}
- Std: {np.std(arr):.3f}
- Max: {np.max(arr):.3f}
- Min: {np.min(arr):.3f}

Threshold crossings:
{json.dumps(threshold_stats, indent=2)}"""
    else:
        data_context = "No IoU values available (no matching pairs found)."

    return _make_section_request("iou_distribution", section_prompt, data_context, **kwargs)


def analyze_trajectory_comparison(
    correlation_matrix: Any,  # pandas DataFrame
    metric: str,
    layers: list[int],
    **kwargs,
) -> SectionAnalysis:
    """Analyze trajectory correlation comparison.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        metric: The metric being compared (e.g., "effective_rank")
        layers: Layers included in analysis
    """
    section_prompt = """Analyze the trajectory correlation between runs.

Consider:
- How correlated are the learning trajectories?
- Do certain layers show higher/lower correlation?
- Does high correlation + low event overlap indicate selection sensitivity?
- What does this tell us about signal vs event invariance?"""

    # Extract correlation values
    try:
        if hasattr(correlation_matrix, "values"):
            corr_str = correlation_matrix.to_string()
            mean_corr = correlation_matrix.values[0, 1] if correlation_matrix.shape[0] >= 2 else "N/A"
        else:
            corr_str = str(correlation_matrix)
            mean_corr = "N/A"
    except Exception:
        corr_str = str(correlation_matrix)
        mean_corr = "N/A"

    data_context = f"""Metric: {metric}
Layers: {layers}

Correlation Matrix:
{corr_str}

Mean cross-run correlation: {mean_corr}"""

    return _make_section_request("trajectory", section_prompt, data_context, **kwargs)


def analyze_composite_events(
    composite_invariance: dict,
    run_a_composites: int,
    run_b_composites: int,
    **kwargs,
) -> SectionAnalysis:
    """Analyze composite event invariance.

    Args:
        composite_invariance: Result from compute_composite_invariance
        run_a_composites: Number of composites in run A
        run_b_composites: Number of composites in run B
    """
    section_prompt = """Analyze the composite event comparison.

Consider:
- How does composite invariance compare to single-metric invariance?
- Is there asymmetry in composite coverage between runs?
- If strength correlation is available, what does it show?
- Are composites more or less stable than individual events?"""

    data_context = f"""Composite Events:
- Run A: {run_a_composites} composites
- Run B: {run_b_composites} composites
- Matched: {composite_invariance.get('n_matched', 0)}
- Jaccard: {composite_invariance.get('jaccard', 0):.1%}
- Coverage of A: {composite_invariance.get('coverage_a', 0):.1%}
- Coverage of B: {composite_invariance.get('coverage_b', 0):.1%}
- Strength correlation: {composite_invariance.get('strength_correlation', 'N/A')}"""

    return _make_section_request("composite_events", section_prompt, data_context, **kwargs)


def analyze_phase_distribution(
    phase_a: dict,
    phase_b: dict,
    **kwargs,
) -> SectionAnalysis:
    """Analyze event phase distribution comparison.

    Args:
        phase_a: Phase analysis result for run A
        phase_b: Phase analysis result for run B
    """
    section_prompt = """Analyze the event phase distribution comparison.

Consider:
- Do both runs have similar phase distributions?
- Are events concentrated in shaping, transition, or locking phases?
- Does any phase show stronger invariance than others?
- What does asymmetry in phases suggest about learning dynamics?"""

    data_context = f"""Phase Distribution:

Run A phases: {json.dumps(phase_a.get('phase_counts', {}), indent=2)}
Run B phases: {json.dumps(phase_b.get('phase_counts', {}), indent=2)}"""

    return _make_section_request("phase_distribution", section_prompt, data_context, **kwargs)


def analyze_retention(
    retention_a: dict,
    retention_b: dict,
    **kwargs,
) -> SectionAnalysis:
    """Analyze retention metrics comparison.

    Args:
        retention_a: Retention info dict for run A
        retention_b: Retention info dict for run B
    """
    section_prompt = """Analyze the retention metrics comparison.

Consider:
- Are retention rates similar between runs?
- What does the candidate density tell us about signal intensity?
- Does density difference explain event count differences?
- Are pre-warmup vs post-warmup retention rates consistent?"""

    data_context = f"""Retention Metrics:

Run A:
- Candidates: {retention_a.get('n_candidates', '?')}
- Selected: {retention_a.get('n_selected', '?')}
- Retention rate: {retention_a.get('retention_rate', '?')}
- Pre-warmup retention: {retention_a.get('pre_retention_rate', '?')}
- Post-warmup retention: {retention_a.get('post_retention_rate', '?')}
- Candidates per series (post): {retention_a.get('mean_candidates_post_per_series', '?')}

Run B:
- Candidates: {retention_b.get('n_candidates', '?')}
- Selected: {retention_b.get('n_selected', '?')}
- Retention rate: {retention_b.get('retention_rate', '?')}
- Pre-warmup retention: {retention_b.get('pre_retention_rate', '?')}
- Post-warmup retention: {retention_b.get('post_retention_rate', '?')}
- Candidates per series (post): {retention_b.get('mean_candidates_post_per_series', '?')}"""

    return _make_section_request("retention", section_prompt, data_context, **kwargs)


def analyze_summary(
    trajectory_correlation: float,
    iou_jaccard: float,
    composite_jaccard: float,
    iou_threshold: float,
    n_common_strict: int,
    iou_max: float | None = None,
    **kwargs,
) -> SectionAnalysis:
    """Analyze overall summary and provide final interpretation.

    Args:
        trajectory_correlation: Mean trajectory correlation
        iou_jaccard: Jaccard from IoU matching
        composite_jaccard: Jaccard for composites
        iou_threshold: IoU threshold used
        n_common_strict: Common events from strict matching
        iou_max: Maximum IoU observed (if available)
    """
    section_prompt = """Provide a final summary interpretation.

Consider:
- Signal invariance (trajectory) vs Event invariance (Jaccard) discrepancy
- What is the overall reliability of event detection across seeds?
- What are the main limitations of this comparison?
- What should be done before trusting these events for downstream analysis?"""

    data_context = f"""Summary Metrics:

Signal Invariance:
- Trajectory correlation: {trajectory_correlation:.3f}

Event Invariance:
- Common events (strict): {n_common_strict}
- IoU Jaccard (τ={iou_threshold}): {iou_jaccard:.1%}
- Composite Jaccard: {composite_jaccard:.1%}
- Max IoU observed: {iou_max if iou_max else 'N/A'}

Interpretation Guide:
- High trajectory + low Jaccard = selection sensitivity (events detected but at different peaks)
- Low trajectory + low Jaccard = genuine signal difference
- High trajectory + high Jaccard = strong reproducibility"""

    return _make_section_request("summary", section_prompt, data_context, **kwargs)


def analyze_overall(
    section_summaries: list[dict],
    n_runs: int,
    run_seeds: list[str],
    **kwargs,
) -> SectionAnalysis:
    """Provide an overall qualitative synthesis of all section analyses.

    This function takes summaries from all previous sections and synthesizes
    them into a comprehensive overall assessment.

    Args:
        section_summaries: List of dicts, each with keys:
            - section_name: Name of the section
            - interpretation: Key finding from that section
            - observations: List of key observations
        n_runs: Number of runs being compared
        run_seeds: List of seed values for each run
        **kwargs: Passed to _make_section_request (backend, model, temperature)

    Returns:
        SectionAnalysis with overall synthesis
    """
    section_prompt = """Provide a comprehensive overall synthesis of all the analyses above.

Your task is to:
1. Identify the most important findings across all sections
2. Explain what these findings mean for the reliability and reproducibility of the detected events
3. Highlight any contradictions or tensions between different metrics
4. Provide actionable recommendations for the researcher
5. Assess overall confidence in using these events for downstream analysis

Be specific and cite the data from the sections. Do not be generic."""

    # Build context from all sections
    sections_text = ""
    for summary in section_summaries:
        sections_text += f"\n### {summary.get('section_name', 'Unknown')}\n"
        sections_text += f"**Interpretation:** {summary.get('interpretation', 'N/A')}\n"
        if summary.get("observations"):
            sections_text += "**Key Observations:**\n"
            for obs in summary["observations"]:
                sections_text += f"- {obs}\n"

    data_context = f"""COMPARISON OVERVIEW:
- Number of runs: {n_runs}
- Seeds: {', '.join(str(s) for s in run_seeds)}

SECTION SUMMARIES:
{sections_text}

SYNTHESIS REQUIREMENTS:
- Weigh signal invariance (trajectory correlation) vs event invariance (Jaccard)
- Consider whether low event overlap is due to selection sensitivity or genuine signal differences
- Identify which events/findings are most trustworthy across seeds
- Recommend next steps for validation or further analysis"""

    return _make_section_request(
        "overall_synthesis",
        section_prompt,
        data_context,
        **kwargs,
    )


def display_analysis(analysis: SectionAnalysis, section_name: str = "") -> None:
    """Display a section analysis result in a notebook-friendly format.

    Args:
        analysis: SectionAnalysis result
        section_name: Optional section name for header
    """
    from IPython.display import Markdown, display

    if analysis.error:
        display(Markdown(f"**⚠️ Analysis Error:** {analysis.error}"))
        return

    md_parts = []

    if section_name:
        md_parts.append(f"### 🤖 LLM Analysis: {section_name}\n")

    if analysis.interpretation:
        md_parts.append(f"**Interpretation:** {analysis.interpretation}\n")

    if analysis.key_observations:
        md_parts.append("**Key Observations:**")
        for obs in analysis.key_observations:
            md_parts.append(f"- {obs}")
        md_parts.append("")

    if analysis.questions:
        md_parts.append("**Questions Raised:**")
        for q in analysis.questions:
            md_parts.append(f"- {q}")

    display(Markdown("\n".join(md_parts)))
