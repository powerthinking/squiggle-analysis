"""Trajectory extraction and plotting for geometry metrics.

Provides tools to extract metric trajectories (e.g., effective_rank vs step)
for specific layers and generate comparison plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def extract_metric_trajectories(
    geometry_df: pd.DataFrame,
    layers: List[int] = [0, 12, 23],
    metric: str = "effective_rank",
) -> pd.DataFrame:
    """
    Extract metric trajectories for specified layers.

    Args:
        geometry_df: DataFrame with columns: step, layer, metric, value
        layers: List of layer indices to extract
        metric: Metric name to extract (e.g., "effective_rank", "sv_entropy")

    Returns:
        DataFrame with columns: step, layer, value
        Sorted by step, suitable for plotting
    """
    filtered = geometry_df[
        (geometry_df["layer"].isin(layers)) &
        (geometry_df["metric"] == metric)
    ][["step", "layer", "value"]].copy()

    return filtered.sort_values(["layer", "step"]).reset_index(drop=True)


def pivot_trajectories(trajectories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot trajectories to wide form for easy plotting.

    Args:
        trajectories_df: Output from extract_metric_trajectories

    Returns:
        DataFrame with step as index, layer_X columns for values
    """
    return trajectories_df.pivot(
        index="step",
        columns="layer",
        values="value"
    ).rename(columns=lambda x: f"layer_{x}")


def plot_trajectories(
    trajectories_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Metric Trajectory",
    ylabel: str = "Value",
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot metric trajectories for multiple layers.

    Args:
        trajectories_df: Output from extract_metric_trajectories
        output_path: If provided, save figure to this path
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    layers = sorted(trajectories_df["layer"].unique())
    colors = plt.cm.viridis([i / len(layers) for i in range(len(layers))])

    for i, layer in enumerate(layers):
        layer_data = trajectories_df[trajectories_df["layer"] == layer]
        ax.plot(
            layer_data["step"],
            layer_data["value"],
            label=f"Layer {layer}",
            color=colors[i],
            linewidth=2,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.close(fig)


def compare_run_trajectories(
    run_geometry_dfs: dict[str, pd.DataFrame],
    layers: List[int] = [0, 12, 23],
    metric: str = "effective_rank",
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Extract and compare trajectories across multiple runs.

    Args:
        run_geometry_dfs: Dict mapping run_id to geometry DataFrame
        layers: Layers to extract
        metric: Metric to compare
        output_dir: If provided, save comparison plots

    Returns:
        Dict mapping run_id to trajectory DataFrame
    """
    trajectories = {}

    for run_id, geom_df in run_geometry_dfs.items():
        traj = extract_metric_trajectories(geom_df, layers=layers, metric=metric)
        traj["run_id"] = run_id
        trajectories[run_id] = traj

    if output_dir:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plots")
            return trajectories

        output_dir = Path(output_dir)

        for layer in layers:
            fig, ax = plt.subplots(figsize=(10, 6))

            for run_id, traj in trajectories.items():
                layer_data = traj[traj["layer"] == layer]
                ax.plot(layer_data["step"], layer_data["value"], label=run_id, linewidth=2)

            ax.set_xlabel("Step")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Step (Layer {layer})")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = output_dir / f"{metric}_layer_{layer}_comparison.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {plot_path}")

    return trajectories
