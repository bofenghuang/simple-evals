import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Constants from aggregate_subset_scores.py and healthbench_eval.py
INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
INPUT_PATH_CONSENSUS = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"
)


def _get_subset_prompt_ids(subset_name: str) -> set[str]:
    """
    Fetch prompt_ids for a given HealthBench subset from the public JSONL or pediatric HF datasets.
    """
    if subset_name in ("hard", "consensus"):
        input_path = INPUT_PATH_HARD if subset_name == "hard" else INPUT_PATH_CONSENSUS
        ids: set[str] = set()
        with urllib.request.urlopen(input_path) as f:
            for raw_line in f:
                try:
                    line = raw_line.decode("utf-8")
                except Exception:
                    line = raw_line
                try:
                    obj = json.loads(line)
                    pid = obj.get("prompt_id")
                    if isinstance(pid, str):
                        ids.add(pid)
                except Exception:
                    continue
        return ids

    # For pediatric subsets, try importing datasets if available
    try:
        from datasets import load_dataset

        PEDIATRIC_DATASETS = {
            "pediatric": ("bofenghuang/healthbench-pediatric", "pediatric"),
            "pediatric_hard": ("bofenghuang/healthbench-hard-pediatric", "pediatric"),
            "pediatric_consensus": ("bofenghuang/healthbench-consensus-pediatric", "pediatric"),
        }
        if subset_name in PEDIATRIC_DATASETS:
            dataset_name, config_name = PEDIATRIC_DATASETS[subset_name]
            ds = load_dataset(dataset_name, config_name, split="test")
            return set(ds["prompt_id"])
    except ImportError:
        print(f"Warning: datasets library not found. Cannot load prompt_ids for {subset_name}.")

    return set()


def _compute_clipped_stats(values: list, stat: str):
    if not values:
        return 0 if stat != "n_samples" else 0
    if stat == "mean":
        return np.clip(np.mean(values), 0, 1)
    elif stat == "n_samples":
        return len(values)
    elif stat == "bootstrap_std":
        if len(values) < 2:
            return 0
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [np.clip(np.mean(s), 0, 1) for s in bootstrap_samples]
        return np.std(bootstrap_means)
    return 0


def parse_model_name(filepath):
    """
    Extracts a clean model name from the filename.
    Matches logic in aggregate_subset_scores.py
    """
    fname = Path(filepath).stem
    # Remove '_allresults' if present
    name = re.sub(r"_allresults$", "", fname)
    # Remove 'healthbench_' prefix
    name = re.sub(r"^healthbench_", "", name)
    # Remove date suffix _YYYYMMDD_HHMMSS
    name = re.sub(r"_\d{8}_\d{6}$", "", name)
    # Remove potential '_medium', '_high', '_low', '_minimal' suffix if present
    # name = re.sub(r"_(medium|high|low|minimal)$", "", name)
    return name


def main():
    parser = argparse.ArgumentParser(description="Plot HealthBench scores by axis.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory containing JSON result files and where the plot will be saved.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["hard", "consensus", "pediatric", "pediatric_hard", "pediatric_consensus"],
        default=None,
        help="Subset to filter and plot; if provided, uses *_allresults.json files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist.")
        return

    subset_ids = None
    if args.subset:
        print(f"Filtering for subset: {args.subset}")
        subset_ids = _get_subset_prompt_ids(args.subset)
        # Search for *_allresults.json files
        json_files = list(output_dir.rglob("*_allresults.json"))
    else:
        # Search recursively for summary JSON files
        json_files = list(output_dir.rglob("*.json"))
        # Filter out allresults.json
        json_files = [f for f in json_files if not f.name.endswith("_allresults.json")]

    print(f"Found {len(json_files)} result files in {output_dir}.")

    all_data = []

    for f in sorted(json_files):
        model_name = parse_model_name(f)

        # tmp: filter models
        if model_name not in [
            "gpt-5_medium",
            "gemini-3-flash-preview_medium",
            "claude-opus-4-5-20251101_medium",
            "claude-sonnet-4-5-20250929",
        ]:
            # print(f"Skipping {model_name}")
            continue

        try:
            with open(f, "r") as jfile:
                data = json.load(jfile)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

        if args.subset:
            # Re-aggregate scores for the subset
            metadata = data.get("metadata", {})
            example_level_metadata = metadata.get("example_level_metadata", [])

            # Filter examples
            filtered_examples = [ex for ex in example_level_metadata if ex.get("prompt_id") in subset_ids]

            if not filtered_examples:
                print(f"  Warning: No examples found for {model_name} in subset {args.subset}")
                continue

            # Aggregate scores by axis
            axis_values = defaultdict(list)
            overall_scores = []

            for ex in filtered_examples:
                if "score" in ex:
                    overall_scores.append(ex["score"])

                # Group criteria by axis tags
                for rubric_item in ex.get("rubric_items", []):
                    # We only care about criteria_met
                    val = 1.0 if rubric_item.get("criteria_met") else 0.0
                    for tag in rubric_item.get("tags", []):
                        if tag.startswith("axis:"):
                            axis_values[tag].append(val)

            # Compute stats for axes
            for axis_tag, values in axis_values.items():
                axis_name = axis_tag.replace("axis:", "")
                all_data.append(
                    {
                        "model": model_name,
                        "axis": axis_name,
                        "score": _compute_clipped_stats(values, "mean"),
                        "bootstrap_std": _compute_clipped_stats(values, "bootstrap_std"),
                    }
                )

            # Compute stats for overall
            if overall_scores:
                all_data.append(
                    {
                        "model": model_name,
                        "axis": "OVERALL",
                        "score": _compute_clipped_stats(overall_scores, "mean"),
                        "bootstrap_std": _compute_clipped_stats(overall_scores, "bootstrap_std"),
                    }
                )

            print(f"  Processed {model_name}: {len(axis_values) + (1 if overall_scores else 0)} axes found (subset filtered).")

        else:
            # Extract axis scores from summary JSON
            axes_found = 0
            for key, value in data.items():
                if key.startswith("axis:") and not (key.endswith(":bootstrap_std") or key.endswith(":n_samples")):
                    axis_name = key.replace("axis:", "")
                    std_key = f"{key}:bootstrap_std"
                    bootstrap_std = data.get(std_key, 0)

                    all_data.append({"model": model_name, "axis": axis_name, "score": value, "bootstrap_std": bootstrap_std})
                    axes_found += 1

            if "overall_score" in data:
                all_data.append(
                    {
                        "model": model_name,
                        "axis": "OVERALL",
                        "score": data["overall_score"],
                        "bootstrap_std": data.get("overall_score:bootstrap_std", 0),
                    }
                )
                axes_found += 1

            print(f"  Processed {model_name}: {axes_found} axes found.")

    if not all_data:
        print("No data found to plot.")
        return

    df = pd.DataFrame(all_data)

    # Plotting
    sns.set_theme(style="whitegrid")

    # Sort models and axes for consistent display
    # Put OVERALL at the beginning
    axes_list = sorted([a for a in df["axis"].unique() if a != "OVERALL"])
    if "OVERALL" in df["axis"].unique():
        axes_list = ["OVERALL"] + axes_list

    models_list = sorted(df["model"].unique())

    # We'll use matplotlib directly for more control over error bars
    fig, ax = plt.subplots(figsize=(max(12, len(axes_list) * 2), 8))

    x_coords = range(len(axes_list))
    # Adjust bar width based on number of models
    width = 0.8 / max(1, len(models_list))

    palette = sns.color_palette("husl", len(models_list))
    model_colors = dict(zip(models_list, palette))

    for i, model in enumerate(models_list):
        model_df = df[df["model"] == model]
        # Align with axes list
        model_df = model_df.set_index("axis").reindex(axes_list).reset_index()

        offsets = [x + (i - (len(models_list) - 1) / 2) * width for x in x_coords]

        # Filter out NaN scores if any axis is missing for a model
        mask = ~model_df["score"].isna()
        if not mask.any():
            continue

        x_final = [offsets[j] for j in range(len(offsets)) if mask[j]]
        y_final = model_df["score"][mask]
        err_final = model_df["bootstrap_std"][mask]

        ax.bar(x_final, y_final, width=width, label=model, color=model_colors[model], alpha=0.8)

        ax.errorbar(x_final, y_final, yerr=err_final, fmt="none", ecolor="black", capsize=3, elinewidth=1)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(axes_list, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("HealthBench: Score by Axis for Different Models")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 1.1)

    # Add horizontal grid lines every 0.1
    ax.set_yticks([i / 10 for i in range(11)])

    plt.tight_layout()

    fname = f"scores_by_axis_{args.subset}.png" if args.subset else "scores_by_axis.png"
    save_path = output_dir / fname
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
