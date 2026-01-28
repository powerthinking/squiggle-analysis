"""Prompt templates for LLM report analysis."""

import json

from squiggle_analysis.llm_analysis.schemas import ANALYSIS_SCHEMA

SYSTEM_PROMPT = """You are SquiggleReportReviewer, an expert research engineer reviewing Squiggle analysis reports and run comparisons.

{squiggle_context}

Rules:
- Be skeptical. Prefer explanations that are testable.
- If you suspect missing information (e.g., analysis flags not shown), explicitly mark that as a limitation and propose how to verify.
- Never invent run settings, metrics, file contents, or experiment details not present in the input.
- Focus on actionable guidance: what to change, what to measure, and why.
- Do not give generic advice. Tie every major point to evidence in the report (quote small snippets or refer to specific reported values).
- Output must be valid JSON only, matching the provided schema. No markdown, no extra text."""

DEVELOPER_PROMPT = """TASK
Analyze the provided Squiggle report(s) and/or compare report, and produce an expert qualitative review.

ANALYSIS MODE
Check run_context.analysis_mode to determine what type of analysis this is:
- "single_run": Analyzing ONE run. Seed invariance CANNOT be assessed (requires multiple runs).
  Focus on: event distribution, retention metrics, phase coverage, potential anomalies.
  Do NOT mention lack of seed invariance data as a limitation - it's expected for single runs.
- "comparison": Analyzing MULTIPLE runs with different seeds. Seed invariance CAN be assessed.
  Focus on: common events, Jaccard similarity, trajectory correlation, retention differences.

DETECTION CONFIG
The run_context includes detection_config (or detection_configs for comparison) with the exact
parameters used for event detection. Reference these values when interpreting results:
- warmup_fraction / warmup_end_step: Events before this are pre-warmup
- max_pre_warmup: Budget for pre-warmup events (0 means all early events blocked)
- peak_suppression_radius: Step distance for non-maximum suppression
- max_events_per_series: Total event budget per (layer, metric) series
- adaptive_k: Threshold sensitivity (higher = fewer events)

PRIORITIES (in order)
1) Note the detection parameters and how they affect interpretation.
2) Interpret the results: what is stable vs unstable across phases, metrics, and layers.
3) Identify "load-bearing" evidence and cite it using exact numeric values from the input.
4) Propose next actions that are minimal-effort and high-information (diagnostic experiments).
5) Suggest improvements to the reporting/compare tooling (e.g., additional metrics, plots).

OUTPUT FORMAT
Return ONLY JSON with exactly this structure:

{schema}

STYLE
- Be concise but specific.
- Use the report's vocabulary (warmup gating, retention, suppression, top-k, composites, wavefronts).
- Reference the detection_config values when they're relevant to interpretation.
- Do not restate the whole report; interpret it."""

USER_PROMPT_TEMPLATE = """INPUTS

run_context:
{run_context_json}

primary_report_md:
{primary_report}

compare_report_md:
{compare_report}

artifacts_manifest:
{artifacts}

user_question:
{user_question}"""


def format_system_prompt(squiggle_context: str) -> str:
    """Format system prompt with squiggle context."""
    return SYSTEM_PROMPT.format(squiggle_context=squiggle_context)


def format_developer_prompt() -> str:
    """Format developer prompt with schema."""
    return DEVELOPER_PROMPT.format(schema=json.dumps(ANALYSIS_SCHEMA, indent=2))


def format_user_prompt(
    run_context: dict,
    primary_report: str | None,
    compare_report: str | None,
    artifacts: list[str] | None,
    user_question: str | None,
) -> str:
    """Format user prompt with input data.

    Args:
        run_context: Dict with run metadata (ids, seeds, configs)
        primary_report: Single-run report.md content
        compare_report: Comparison report content
        artifacts: List of generated artifact paths
        user_question: Optional specific question to ask

    Returns:
        Formatted user prompt string
    """
    return USER_PROMPT_TEMPLATE.format(
        run_context_json=json.dumps(run_context, indent=2),
        primary_report=primary_report or "N/A",
        compare_report=compare_report or "N/A",
        artifacts="\n".join(artifacts) if artifacts else "N/A",
        user_question=user_question
        or "Provide an expert qualitative analysis and recommended next actions.",
    )
