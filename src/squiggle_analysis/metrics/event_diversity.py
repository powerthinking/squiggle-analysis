"""Event diversity metrics for distinguishing rich vs uniform event patterns.

These metrics help identify runs with genuine learning dynamics vs runs where
events are dominated by schedule artifacts (e.g., all events at the same step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class EventDiversityMetrics:
    """Metrics describing the diversity and richness of detected events."""

    # Score variance across layers (higher = more layer-specific variation)
    score_variance_layers: float

    # Temporal dispersion: std of change point steps (higher = more spread out)
    temporal_dispersion: float

    # Metric participation entropy: Shannon entropy of metric distribution
    # Higher = more metrics participating equally
    metric_participation_entropy: float

    # Layer delay distribution: variation in event timing across layers
    layer_delay_std: float
    layer_delay_max: float

    # Number of distinct event steps
    n_distinct_steps: int

    # Number of layers with events
    n_layers_with_events: int

    # Total number of events
    n_events: int

    # Combined diversity score (0-1, higher = more diverse)
    diversity_score: float


def _shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy of a count distribution."""
    p = counts / counts.sum()
    p = p[p > 0]  # Remove zeros to avoid log(0)
    return float(-np.sum(p * np.log(p)))


def compute_event_diversity(events_df: pd.DataFrame) -> EventDiversityMetrics:
    """
    Compute event diversity metrics from an events_candidates DataFrame.

    Args:
        events_df: DataFrame with columns: layer, metric, step, score

    Returns:
        EventDiversityMetrics with computed values

    High diversity indicates:
    - Events spread across different steps (not all at same step)
    - Multiple metrics participating in events
    - Scores varying across layers
    - Different layers having events at different times

    Low diversity indicates:
    - Events concentrated at single step (schedule artifact)
    - Single metric dominating
    - Uniform scores across all layers
    """
    if events_df.empty:
        return EventDiversityMetrics(
            score_variance_layers=0.0,
            temporal_dispersion=0.0,
            metric_participation_entropy=0.0,
            layer_delay_std=0.0,
            layer_delay_max=0.0,
            n_distinct_steps=0,
            n_layers_with_events=0,
            n_events=0,
            diversity_score=0.0,
        )

    # Filter to single-metric events (not composite)
    single_metric_df = events_df[events_df["metric"] != "__composite__"]

    n_events = len(events_df)
    n_distinct_steps = events_df["step"].nunique()
    n_layers_with_events = events_df["layer"].nunique()

    # 1. Score variance across layers
    if "score" in events_df.columns:
        layer_mean_scores = events_df.groupby("layer")["score"].mean()
        score_variance_layers = float(layer_mean_scores.var()) if len(layer_mean_scores) > 1 else 0.0
    else:
        score_variance_layers = 0.0

    # 2. Temporal dispersion (std of step values)
    temporal_dispersion = float(events_df["step"].std()) if len(events_df) > 1 else 0.0

    # 3. Metric participation entropy
    if not single_metric_df.empty:
        metric_counts = single_metric_df["metric"].value_counts().values
        metric_participation_entropy = _shannon_entropy(metric_counts)
    else:
        metric_participation_entropy = 0.0

    # 4. Layer delay distribution (variation in event timing per layer)
    if n_layers_with_events > 1:
        layer_median_steps = events_df.groupby("layer")["step"].median()
        layer_delay_std = float(layer_median_steps.std())
        layer_delay_max = float(layer_median_steps.max() - layer_median_steps.min())
    else:
        layer_delay_std = 0.0
        layer_delay_max = 0.0

    # 5. Compute combined diversity score (0-1)
    # Normalize each component and combine
    components = []

    # Temporal diversity: proportion of distinct steps relative to events
    if n_events > 0:
        temporal_diversity = min(1.0, n_distinct_steps / n_events)
        components.append(temporal_diversity)

    # Metric diversity: entropy relative to max possible
    if not single_metric_df.empty:
        n_metrics = single_metric_df["metric"].nunique()
        max_entropy = np.log(n_metrics) if n_metrics > 1 else 1.0
        metric_diversity = metric_participation_entropy / max_entropy if max_entropy > 0 else 0.0
        components.append(metric_diversity)

    # Layer spread: proportion of layers with events (assuming 24 layers)
    layer_diversity = min(1.0, n_layers_with_events / 24.0)
    components.append(layer_diversity)

    # Score variance (normalized by typical variance)
    if score_variance_layers > 0:
        # Use sigmoid-like scaling
        score_diversity = min(1.0, score_variance_layers / 0.01)
        components.append(score_diversity)

    diversity_score = float(np.mean(components)) if components else 0.0

    return EventDiversityMetrics(
        score_variance_layers=score_variance_layers,
        temporal_dispersion=temporal_dispersion,
        metric_participation_entropy=metric_participation_entropy,
        layer_delay_std=layer_delay_std,
        layer_delay_max=layer_delay_max,
        n_distinct_steps=n_distinct_steps,
        n_layers_with_events=n_layers_with_events,
        n_events=n_events,
        diversity_score=diversity_score,
    )
