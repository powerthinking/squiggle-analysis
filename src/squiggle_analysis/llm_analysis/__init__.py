"""LLM-based qualitative analysis of Squiggle reports."""

from squiggle_analysis.llm_analysis.analyzer import AnalysisRequest, analyze_report
from squiggle_analysis.llm_analysis.schemas import ANALYSIS_SCHEMA, validate_analysis_response
from squiggle_analysis.llm_analysis.section_analyzer import (
    SectionAnalysis,
    analyze_raster_plot,
    analyze_common_events,
    analyze_iou_invariance,
    analyze_iou_distribution,
    analyze_trajectory_comparison,
    analyze_composite_events,
    analyze_phase_distribution,
    analyze_retention,
    analyze_summary,
    analyze_overall,
    display_analysis,
)

__all__ = [
    # Full report analysis
    "AnalysisRequest",
    "analyze_report",
    "ANALYSIS_SCHEMA",
    "validate_analysis_response",
    # Per-section analysis (for notebooks)
    "SectionAnalysis",
    "analyze_raster_plot",
    "analyze_common_events",
    "analyze_iou_invariance",
    "analyze_iou_distribution",
    "analyze_trajectory_comparison",
    "analyze_composite_events",
    "analyze_phase_distribution",
    "analyze_retention",
    "analyze_summary",
    "analyze_overall",
    "display_analysis",
]
