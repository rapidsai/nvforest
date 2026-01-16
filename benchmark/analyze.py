#!/usr/bin/env python
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""
Analyze and visualize benchmark results.

Usage:
    python -m cuforest.benchmark.analyze results.csv
    python -m cuforest.benchmark.analyze results.csv --output speedup_plots.png
"""

import os
from typing import Optional

import click
import pandas as pd


def plot_speedup_heatmaps(
    df: pd.DataFrame,
    title: str = "cuforest Speedups",
    output_filename: str = "speedup_heatmaps.png",
    speedup_column: str = "speedup",
) -> None:
    """
    Generate speedup heatmaps from benchmark results.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results DataFrame with columns: max_depth, num_trees,
        num_features, batch_size, and the speedup column.
    title : str
        Title for the plot.
    output_filename : str
        Output file path for the plot.
    speedup_column : str
        Name of the column containing speedup values.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install them with: pip install matplotlib seaborn"
        )

    # Get unique values for each parameter
    max_depth_values = sorted(df["max_depth"].unique())
    num_trees_values = sorted(df["num_trees"].unique())

    # Create grid of subplots
    fig, axes = plt.subplots(
        nrows=len(max_depth_values),
        ncols=len(num_trees_values),
        figsize=(len(num_trees_values) * 4, len(max_depth_values) * 4),
        constrained_layout=True,
    )

    # Handle single row/column case
    if len(max_depth_values) == 1:
        axes = axes.reshape(1, -1)
    if len(num_trees_values) == 1:
        axes = axes.reshape(-1, 1)

    min_speedup = df[speedup_column].min()
    max_speedup = df[speedup_column].max()

    for i, max_depth in enumerate(max_depth_values):
        for j, num_trees in enumerate(num_trees_values):
            ax = axes[i, j]

            # Filter data for the current max_depth and num_trees
            subset = df[(df["max_depth"] == max_depth) & (df["num_trees"] == num_trees)]

            if subset.empty:
                ax.set_visible(False)
                continue

            # Pivot data for heatmap
            heatmap_data = subset.pivot_table(
                index="num_features", columns="batch_size", values=speedup_column
            )
            heatmap_data.columns = pd.Index(
                [f"{x:.0e}" for x in heatmap_data.columns], name="Batch Size"
            )

            # Plot heatmap
            sns.heatmap(
                heatmap_data,
                ax=ax,
                vmin=min_speedup,
                vmax=max_speedup,
                center=1.0,
                cmap="RdBu",
                annot=True,
                fmt=".1f",
                cbar=False,
            )

            if i == 0:
                ax.set_title(f"Tree count: {num_trees}")
            if i == len(max_depth_values) - 1:
                ax.set_xlabel("Batch Size")
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(f"Maximum Depth: {max_depth}\nFeature Count")
            else:
                ax.set_ylabel("")

    fig.suptitle(title, fontsize=24)
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved plot to {output_filename}")


def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics from benchmark results.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary statistics grouped by framework, model_type, and device.
    """
    summary = (
        df.groupby(["framework", "model_type", "device"])
        .agg(
            {
                "speedup": ["mean", "median", "min", "max", "std"],
                "native_time": ["mean", "sum"],
                "cuforest_time": ["mean", "sum"],
            }
        )
        .round(3)
    )

    return summary


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    # Overall stats
    print(f"\nTotal benchmark runs: {len(df)}")
    print(f"Frameworks tested: {', '.join(df['framework'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    print(f"Devices: {', '.join(df['device'].unique())}")

    # Speedup stats
    print("\n" + "-" * 40)
    print("SPEEDUP STATISTICS (native / cuforest)")
    print("-" * 40)

    for framework in df["framework"].unique():
        print(f"\n{framework.upper()}:")
        fw_df = df[df["framework"] == framework]

        for device in fw_df["device"].unique():
            dev_df = fw_df[fw_df["device"] == device]
            valid_speedups = dev_df["speedup"].dropna()

            if len(valid_speedups) > 0:
                print(f"  {device}:")
                print(f"    Mean speedup:   {valid_speedups.mean():.2f}x")
                print(f"    Median speedup: {valid_speedups.median():.2f}x")
                print(f"    Min speedup:    {valid_speedups.min():.2f}x")
                print(f"    Max speedup:    {valid_speedups.max():.2f}x")

    # Best configurations
    print("\n" + "-" * 40)
    print("TOP 5 CONFIGURATIONS BY SPEEDUP")
    print("-" * 40)

    top_5 = df.nlargest(5, "speedup")[
        ["framework", "model_type", "device", "num_trees", "max_depth", "batch_size", "speedup"]
    ]
    print(top_5.to_string(index=False))

    print("\n" + "=" * 60)


@click.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file for speedup heatmap plot.",
)
@click.option(
    "--framework",
    "-f",
    default=None,
    help="Filter results to specific framework.",
)
@click.option(
    "--device",
    "-d",
    default=None,
    type=click.Choice(["cpu", "gpu"]),
    help="Filter results to specific device.",
)
@click.option(
    "--plot-only",
    is_flag=True,
    help="Only generate plot, skip summary output.",
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Only print summary, skip plot generation.",
)
def analyze(
    results_file: str,
    output: Optional[str],
    framework: Optional[str],
    device: Optional[str],
    plot_only: bool,
    summary_only: bool,
):
    """Analyze benchmark results from RESULTS_FILE."""
    df = pd.read_csv(results_file)

    # Apply filters
    if framework:
        df = df[df["framework"] == framework]
    if device:
        df = df[df["device"] == device]

    if df.empty:
        raise click.ClickException("No data matches the specified filters.")

    # Print summary
    if not plot_only:
        print_summary(df)

    # Generate plot
    if not summary_only:
        if output is None:
            base_name = os.path.splitext(results_file)[0]
            output = f"{base_name}_speedup.png"

        title = "cuforest Speedup vs Native Inference"
        if framework:
            title += f" ({framework})"
        if device:
            title += f" [{device.upper()}]"

        try:
            plot_speedup_heatmaps(df, title=title, output_filename=output)
        except ImportError as e:
            if not plot_only:
                print(f"\nNote: {e}")
            else:
                raise click.ClickException(str(e))


if __name__ == "__main__":
    analyze()
