"""Plot rubric distribution by level (example vs cluster) and axis.

Usage:
    python -m simple-evals.plot_rubric_distribution \
        --results-dir outputs/healthbench \
        --output-dir outputs/healthbench
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _load_rubric_items(results_dir: str) -> list[dict]:
    """Load rubric items from the first *_allresults.json found."""
    base = Path(results_dir)
    files = sorted(base.glob("*_allresults.json"))
    if not files:
        raise FileNotFoundError(f"No *_allresults.json files found in {results_dir}")

    data = json.loads(files[0].read_text())
    examples = data.get("metadata", {}).get("example_level_metadata", [])

    rows = []
    for ex in examples:
        for ri in ex.get("rubric_items", []):
            pts = float(ri["points"])
            level = "unknown"
            axes = []
            for tag in ri.get("tags", []):
                if tag.startswith("level:"):
                    level = tag.replace("level:", "")
                elif tag.startswith("axis:"):
                    axes.append(tag.replace("axis:", ""))
            for axis in axes:
                rows.append({
                    "level": level,
                    "axis": axis,
                    "polarity": "positive" if pts > 0 else "negative",
                    "points": pts,
                    "abs_points": abs(pts),
                })
    return rows


def print_summary_table(df: pd.DataFrame):
    """Print summary table grouped by level and axis."""
    grouped = df.groupby(["level", "axis"]).agg(
        count=("points", "size"),
        pos=("polarity", lambda x: (x == "positive").sum()),
        neg=("polarity", lambda x: (x == "negative").sum()),
        avg_weight=("points", "mean"),
        sum_abs=("abs_points", "sum"),
    ).reset_index()

    # Compute share within each level
    level_totals = grouped.groupby("level")["sum_abs"].transform("sum")
    grouped["share"] = grouped["sum_abs"] / level_totals

    grouped = grouped.sort_values(["level", "sum_abs"], ascending=[True, False])
    grouped["avg_weight"] = grouped["avg_weight"].round(2)
    grouped["sum_abs"] = grouped["sum_abs"].astype(int)
    grouped["share"] = grouped["share"].apply(lambda x: f"{x:.1%}")

    print("\n=== Rubric distribution by level and axis ===")
    print(grouped.to_markdown(index=False))


def plot_stacked_bars(df: pd.DataFrame, output_dir: Path):
    """Stacked bar chart: X = level, stacked by axis."""
    pivot = df.groupby(["level", "axis"]).size().unstack(fill_value=0)

    # Sort axes by total count descending
    axis_order = pivot.sum().sort_values(ascending=False).index.tolist()
    pivot = pivot[axis_order]

    # Sort levels: example first, cluster second
    level_order = [l for l in ["example", "cluster"] if l in pivot.index]
    pivot = pivot.loc[level_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("Set2", n_colors=len(axis_order))

    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    bar_width = 0.5

    for i, axis in enumerate(axis_order):
        values = pivot[axis].values
        bars = ax.bar(x, values, bar_width, bottom=bottom, label=axis, color=colors[i])
        # Add count labels inside bars if large enough
        for j, (val, bot) in enumerate(zip(values, bottom)):
            if val > pivot.values.sum() * 0.03:  # only label if > 3% of total
                ax.text(x[j], bot + val / 2, f"{val:,}", ha="center", va="center", fontsize=9)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("example", "instance-specific").replace("cluster", "consensus") for l in pivot.index])
    ax.set_ylabel("Number of rubric items")
    ax.set_title("Rubric Distribution by Level and Axis")
    ax.legend(title="Axis", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    save_path = output_dir / "rubric_distribution_by_level.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot rubric distribution by level and axis.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing *_allresults.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots (default: results-dir)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rubric_items(args.results_dir)
    df = pd.DataFrame(rows)

    print_summary_table(df)
    plot_stacked_bars(df, output_dir)


if __name__ == "__main__":
    main()
