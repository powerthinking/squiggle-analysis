"""Main analyzer for LLM-based report interpretation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from squiggle_core.llm import LLMClient, LLMRequest, build_squiggle_context

from squiggle_analysis.llm_analysis.prompts import (
    format_developer_prompt,
    format_system_prompt,
    format_user_prompt,
)
from squiggle_analysis.llm_analysis.schemas import validate_analysis_response


@dataclass
class AnalysisRequest:
    """Request for LLM analysis of a report."""

    run_context: dict[str, Any]  # run_ids, seeds, config fingerprints
    primary_report: str | None = None  # Single-run report.md content
    compare_report: str | None = None  # Comparison report content
    artifacts: list[str] = field(default_factory=list)  # Paths to generated plots/parquets
    user_question: str | None = None  # Optional specific question


@dataclass
class AnalysisResult:
    """Result from LLM analysis."""

    analysis: dict[str, Any]  # Parsed JSON analysis
    provenance: dict[str, Any]  # Model info, prompt hash, timestamp
    validation_errors: list[str]  # Schema validation errors (if any)
    raw_response: str  # Raw LLM response for debugging


def analyze_report(
    request: AnalysisRequest,
    backend: str = "openai",
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 4000,
    validate: bool = True,
) -> AnalysisResult:
    """Send report to LLM for qualitative analysis.

    Args:
        request: AnalysisRequest with run context and report content
        backend: LLM backend ("openai" or "anthropic")
        model: Model to use (default: gpt-4o)
        temperature: Sampling temperature (default: 0.2 for consistency)
        max_tokens: Max response tokens
        validate: Whether to validate response against schema

    Returns:
        AnalysisResult with analysis, provenance, and validation info
    """
    client = LLMClient(backend=backend)  # type: ignore[arg-type]

    # Build prompts
    squiggle_context = build_squiggle_context()
    system_prompt = format_system_prompt(squiggle_context)
    developer_prompt = format_developer_prompt()
    user_prompt = format_user_prompt(
        run_context=request.run_context,
        primary_report=request.primary_report,
        compare_report=request.compare_report,
        artifacts=request.artifacts,
        user_question=request.user_question,
    )

    # Create LLM request
    llm_request = LLMRequest(
        system_prompt=system_prompt,
        developer_prompt=developer_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format="json",
    )

    # Send request
    response = client.complete(llm_request)

    # Parse response
    if isinstance(response.content, dict):
        analysis = response.content
    else:
        # Try to parse as JSON
        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            analysis = {"error": "Failed to parse response as JSON", "raw": response.content}

    # Validate if requested
    validation_errors: list[str] = []
    if validate and isinstance(analysis, dict) and "error" not in analysis:
        validation_errors = validate_analysis_response(analysis)

    return AnalysisResult(
        analysis=analysis,
        provenance={
            "model_id": response.model_id,
            "prompt_hash": response.prompt_hash,
            "timestamp": response.timestamp,
            "usage": response.usage,
            "backend": backend,
        },
        validation_errors=validation_errors,
        raw_response=response.raw_response,
    )


def write_analysis_result(result: AnalysisResult, output_path: Path) -> None:
    """Write analysis result to JSON file.

    Args:
        result: AnalysisResult from analyze_report
        output_path: Path to write JSON file
    """
    output = {
        "analysis": result.analysis,
        "provenance": result.provenance,
    }

    if result.validation_errors:
        output["validation_errors"] = result.validation_errors

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
