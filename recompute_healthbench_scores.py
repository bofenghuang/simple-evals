import argparse
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_norm_rng = np.random.default_rng(seed=42)

from .utils.stats import bootstrap_ci, compute_agreement, compute_separability, compute_spearman

# Public dataset endpoints (duplicated to avoid import-time package issues)
INPUT_PATH = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
INPUT_PATH_CONSENSUS = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"

# Global trackers for raw rubric point extremes (before normalization)
GLOBAL_MIN_POINT_RAW: float | None = None
GLOBAL_MAX_POINT_RAW: float | None = None

def _update_global_point_extremes_raw(values: list[float]) -> None:
    global GLOBAL_MIN_POINT_RAW, GLOBAL_MAX_POINT_RAW
    for v in values:
        if GLOBAL_MIN_POINT_RAW is None or v < GLOBAL_MIN_POINT_RAW:
            GLOBAL_MIN_POINT_RAW = float(v)
        if GLOBAL_MAX_POINT_RAW is None or v > GLOBAL_MAX_POINT_RAW:
            GLOBAL_MAX_POINT_RAW = float(v)


def _get_subset_prompt_ids(subset_name: str) -> set[str]:
    """
    Fetch prompt_ids for a given HealthBench subset from the public JSONL.
    Allowed subset_name values: 'hard' or 'consensus'.
    """
    input_path = INPUT_PATH_HARD if subset_name == "hard" else INPUT_PATH_CONSENSUS
    ids: set[str] = set()
    with urllib.request.urlopen(input_path) as f:
        for raw_line in f:
            try:
                line = raw_line.decode("utf-8")
            except Exception:
                # best-effort fallback without decode (shouldn't happen)
                line = raw_line
            try:
                obj = json.loads(line)
                pid = obj.get("prompt_id")
                if isinstance(pid, str):
                    ids.add(pid)
            except Exception:
                continue
    return ids


def _get_prompt_theme_mapping() -> dict[str, list[str]]:
    """Fetch prompt_id → [theme tags] mapping from the main HealthBench JSONL.

    Reads example_tags from each example and extracts tags starting with 'theme:'.
    Returns {prompt_id: [theme1, theme2, ...]}.
    """
    mapping: dict[str, list[str]] = {}
    with urllib.request.urlopen(INPUT_PATH) as f:
        for raw_line in f:
            try:
                line = raw_line.decode("utf-8")
            except Exception:
                line = raw_line
            try:
                obj = json.loads(line)
                pid = obj.get("prompt_id")
                if not isinstance(pid, str):
                    continue
                example_tags = obj.get("example_tags", [])
                themes = [t.replace("theme:", "") for t in example_tags if t.startswith("theme:")]
                if themes:
                    mapping[pid] = themes
            except Exception:
                continue
    return mapping


def _normalize_points_list(values: list[float], mode: str) -> list[float]:
    if mode == "none":
        return [float(v) for v in values]
    if mode == "ternary":
        mapped = []
        for v in values:
            if v > 0:
                mapped.append(1.0)
            elif v < 0:
                mapped.append(-1.0)
            else:
                mapped.append(0.0)
        return mapped
    if mode == "random":
        mapped = []
        for v in values:
            if v > 0:
                mapped.append(float(_norm_rng.integers(1, 11)))  # [1, 10]
            elif v < 0:
                mapped.append(float(_norm_rng.integers(-10, 0)))  # [-10, -1]
            else:
                mapped.append(0.0)
        return mapped
    if mode == "seven":
        abs_max = max((abs(v) for v in values), default=0.0)
        if abs_max == 0:
            return [0.0 for _ in values]
        scale = 3.0 / abs_max
        scaled = [max(-3.0, min(3.0, float(v) * scale)) for v in values]
        return [float(max(-3, min(3, int(round(x))))) for x in scaled]
    # Fallback: no normalization
    return [float(v) for v in values]


def filter_rubric_by_polarity(
    rubric_items: list[dict[str, Any]],
    polarity: str = "all",
) -> list[dict[str, Any]]:
    """Filter rubric items by point polarity.

    - all: no filtering
    - positive: keep only items with points > 0
    - negative: keep only items with points < 0
    """
    if polarity == "all":
        return rubric_items
    if polarity == "positive":
        return [item for item in rubric_items if float(item["points"]) > 0]
    if polarity == "negative":
        return [item for item in rubric_items if float(item["points"]) < 0]
    return rubric_items


def _sample_rubric_items(
    rubric_items: list[dict[str, Any]],
    sample_frac: float = 1.0,
) -> list[dict[str, Any]]:
    """Randomly subsample rubric items per example."""
    if sample_frac >= 1.0 or len(rubric_items) == 0:
        return rubric_items
    k = max(1, int(np.ceil(len(rubric_items) * sample_frac)))
    indices = _norm_rng.choice(len(rubric_items), size=k, replace=False)
    return [rubric_items[i] for i in sorted(indices)]


def prepare_rubric_items(
    rubric_items: list[dict[str, Any]],
    sample_frac: float = 1.0,
    polarity: str = "all",
) -> list[dict[str, Any]]:
    """Apply sampling then polarity filtering. Returns the effective rubric items."""
    sampled = _sample_rubric_items(rubric_items, sample_frac)
    return filter_rubric_by_polarity(sampled, polarity)


def calculate_score_from_rubric_items(
    effective_items: list[dict[str, Any]],
    norm: str = "none",
    polarity: str = "all",
) -> float | None:
    """
    Compute score from pre-filtered rubric items.
      - For all/positive: denominator = sum of positive points, numerator = sum where criteria_met
      - For negative: score = 1 - (n_negative_met / n_negative_total) (avoidance rate)
      - Return None if no applicable rubric items
    """
    if not effective_items:
        return None

    raw_points = [float(item["points"]) for item in effective_items]
    _update_global_point_extremes_raw(raw_points)
    normalized_points = _normalize_points_list(raw_points, norm)

    if polarity == "negative":
        n_total = len(effective_items)
        n_met = sum(1 for item in effective_items if bool(item.get("criteria_met", False)))
        return 1.0 - (n_met / n_total)

    # all / positive: standard scoring
    total_possible_points = sum(p for p in normalized_points if p > 0)
    if total_possible_points == 0:
        return None

    achieved_points = 0.0
    for item, p in zip(effective_items, normalized_points):
        if bool(item.get("criteria_met", False)) is True:
            achieved_points += p
    return achieved_points / total_possible_points


def recompute_for_file(
    path: Path,
    out_dir: Path | None = None,
    norm: str = "none",
    polarity: str = "all",
    sample_frac: float = 1.0,
    prompt_id_filter: set[str] | None = None,
) -> dict[str, Any]:
    """
    Load a *_allresults.json file, recompute per-sample overall scores, compare to originals,
    and emit a compact report.
    """
    with path.open("r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    example_level_metadata: list[dict[str, Any]] = metadata.get("example_level_metadata", [])

    # Optional prefilter on subset by prompt_id
    if prompt_id_filter is not None:
        example_level_metadata = [
            s for s in example_level_metadata if s.get("prompt_id") in prompt_id_filter
        ]

    results: list[dict[str, Any]] = []
    mismatches = 0
    positive_denominator_zero = 0

    for sample in example_level_metadata:
        rubric_items = sample.get("rubric_items", [])
        effective_items = prepare_rubric_items(rubric_items, sample_frac=sample_frac, polarity=polarity)
        recomputed = calculate_score_from_rubric_items(effective_items, norm=norm, polarity=polarity)
        original = sample.get("score", None)

        if recomputed is None:
            positive_denominator_zero += 1

        # Compute delta only if both are present
        delta = None
        if recomputed is not None and original is not None:
            delta = float(recomputed) - float(original)
            if abs(delta) > 1e-9:
                mismatches += 1

        num_effective = len(effective_items)
        num_positive_items = sum(1 for it in effective_items if float(it["points"]) > 0)

        results.append(
            {
                "prompt_id": sample.get("prompt_id"),
                "completion_id": sample.get("completion_id"),
                "recomputed_overall_score": recomputed,
                "original_overall_score": original,
                "delta": delta,
                "num_rubric_items": len(rubric_items),
                "num_effective_items": num_effective,
                "num_positive_items": num_positive_items,
                "num_negative_items": num_effective - num_positive_items,
                "latency_seconds": sample.get("latency_seconds"),
            }
        )

    raw_counts = [r["num_rubric_items"] for r in results]
    effective_counts = [r["num_effective_items"] for r in results]
    positive_counts = [r["num_positive_items"] for r in results]
    negative_counts = [r["num_negative_items"] for r in results]

    summary = {
        "file": str(path),
        "num_samples": len(example_level_metadata),
        "num_mismatches_gt_1e-9": mismatches,
        "num_no_positive_points": positive_denominator_zero,
        "total_rubric_items": sum(raw_counts),
        "total_effective_items": sum(effective_counts),
        "avg_rubrics_per_example": float(np.mean(raw_counts)) if raw_counts else 0,
        "avg_effective_per_example": float(np.mean(effective_counts)) if effective_counts else 0,
        "min_effective_per_example": min(effective_counts) if effective_counts else 0,
        "max_effective_per_example": max(effective_counts) if effective_counts else 0,
        "avg_positive_per_example": float(np.mean(positive_counts)) if positive_counts else 0,
        "avg_negative_per_example": float(np.mean(negative_counts)) if negative_counts else 0,
    }

    report = {
        "summary": summary,
        "samples": results,
    }

    # Compute aggregated recomputed stats per file
    recomputed_values = [r["recomputed_overall_score"] for r in results if r["recomputed_overall_score"] is not None]
    original_values = [r["original_overall_score"] for r in results if r["original_overall_score"] is not None]
    n_samples = len(recomputed_values)

    if n_samples > 0:
        score_mean = float(np.clip(np.mean(recomputed_values), 0, 1))
        ci = bootstrap_ci(recomputed_values)
    else:
        score_mean = None
        ci = {"lower_bound": None, "upper_bound": None}

    if original_values:
        original_mean = float(np.clip(np.mean(original_values), 0, 1))
        original_ci = bootstrap_ci(original_values)
    else:
        original_mean = None
        original_ci = {"lower_bound": None, "upper_bound": None}

    latency_values = [r["latency_seconds"] for r in results if r.get("latency_seconds") is not None]
    avg_latency_seconds = float(np.mean(latency_values)) if latency_values else None

    summary.update(
        {
            "score": score_mean,
            "score:n_samples": n_samples,
            "ci_lower": ci["lower_bound"],
            "ci_upper": ci["upper_bound"],
            "avg_latency_seconds": avg_latency_seconds,
            "original_score": original_mean,
            "original_ci_lower": original_ci["lower_bound"],
            "original_ci_upper": original_ci["upper_bound"],
        }
    )

    out_dir = out_dir or path.parent
    out_path = out_dir / f"{path.stem}_recomputed_scores.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    return summary


def _parse_model_name(file_path: Path, input_dir_name: str) -> str:
    pretty_name = file_path.stem.replace("_allresults", "")
    pretty_name = re.sub(rf"^{re.escape(input_dir_name)}_", "", pretty_name)
    pretty_name = re.sub(r"_(\d+)_(\d+)$", "", pretty_name)
    return pretty_name


def compute_axis_model_stats(
    all_files: list[Path],
    input_dir_name: str,
    norm: str = "none",
    polarity: str = "all",
    sample_frac: float = 1.0,
    prompt_id_filter: set[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-axis per-model stats.

    Per-axis score = mean of binary criteria_met across rubric items tagged with that axis
    (filtered by sampling and polarity if specified).
    "overall" pseudo-axis uses the recomputed weighted overall score.

    Returns: {axis_name: {model_name: {score, ci_lower, ci_upper}}}
    """
    # axis_name -> model_name -> list of per-rubric-item binary values
    axis_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # model_name -> list of per-example overall scores (for "overall")
    full_values: dict[str, list[float]] = defaultdict(list)

    for file_path in all_files:
        model_name = _parse_model_name(file_path, input_dir_name)
        with file_path.open("r") as f:
            data = json.load(f)

        examples = data.get("metadata", {}).get("example_level_metadata", [])
        if prompt_id_filter is not None:
            examples = [ex for ex in examples if ex.get("prompt_id") in prompt_id_filter]

        for ex in examples:
            rubric_items = ex.get("rubric_items", [])
            # Prepare once: sample + filter
            effective_items = prepare_rubric_items(rubric_items, sample_frac=sample_frac, polarity=polarity)

            # Per-axis: binary criteria_met
            for ri in effective_items:
                val = 1.0 if ri.get("criteria_met") else 0.0
                for tag in ri.get("tags", []):
                    if tag.startswith("axis:"):
                        axis_name = tag.replace("axis:", "")
                        axis_values[axis_name][model_name].append(val)

            # Overall: recomputed weighted score
            score = calculate_score_from_rubric_items(effective_items, norm=norm, polarity=polarity)
            if score is not None:
                full_values[model_name].append(score)

    # Compute bootstrap CI for each axis + full
    result: dict[str, dict[str, dict[str, float]]] = {}

    for axis_name, model_vals in sorted(axis_values.items()):
        result[axis_name] = {}
        for model_name, vals in model_vals.items():
            ci = bootstrap_ci(vals)
            result[axis_name][model_name] = {
                "score": ci["mean"],
                "ci_lower": ci["lower_bound"],
                "ci_upper": ci["upper_bound"],
            }

    result["overall"] = {}
    for model_name, vals in full_values.items():
        ci = bootstrap_ci(vals)
        result["overall"][model_name] = {
            "score": ci["mean"],
            "ci_lower": ci["lower_bound"],
            "ci_upper": ci["upper_bound"],
        }

    return result


def compute_theme_model_stats(
    all_files: list[Path],
    input_dir_name: str,
    theme_map: dict[str, list[str]],
    norm: str = "none",
    polarity: str = "all",
    sample_frac: float = 1.0,
    prompt_id_filter: set[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-theme per-model stats.

    Groups examples by their theme tags, computes overall score per example,
    then aggregates per theme per model with bootstrap CI.
    "overall" pseudo-theme includes all examples.

    Returns: {theme_name: {model_name: {score, ci_lower, ci_upper}}}
    """
    # theme_name -> model_name -> list of per-example scores
    theme_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # model_name -> list of all per-example scores (for "overall")
    full_values: dict[str, list[float]] = defaultdict(list)

    for file_path in all_files:
        model_name = _parse_model_name(file_path, input_dir_name)
        with file_path.open("r") as f:
            data = json.load(f)

        examples = data.get("metadata", {}).get("example_level_metadata", [])
        if prompt_id_filter is not None:
            examples = [ex for ex in examples if ex.get("prompt_id") in prompt_id_filter]

        for ex in examples:
            rubric_items = ex.get("rubric_items", [])
            effective_items = prepare_rubric_items(rubric_items, sample_frac=sample_frac, polarity=polarity)
            score = calculate_score_from_rubric_items(effective_items, norm=norm, polarity=polarity)
            if score is None:
                continue

            full_values[model_name].append(score)

            prompt_id = ex.get("prompt_id", "")
            themes = theme_map.get(prompt_id, [])
            for theme in themes:
                theme_values[theme][model_name].append(score)

    # Compute bootstrap CI
    result: dict[str, dict[str, dict[str, float]]] = {}

    for theme_name, model_vals in sorted(theme_values.items()):
        result[theme_name] = {}
        for model_name, vals in model_vals.items():
            ci = bootstrap_ci(vals)
            result[theme_name][model_name] = {
                "score": ci["mean"],
                "ci_lower": ci["lower_bound"],
                "ci_upper": ci["upper_bound"],
            }

    result["overall"] = {}
    for model_name, vals in full_values.items():
        ci = bootstrap_ci(vals)
        result["overall"][model_name] = {
            "score": ci["mean"],
            "ci_lower": ci["lower_bound"],
            "ci_upper": ci["upper_bound"],
        }

    return result


def print_group_analysis(
    group_model_stats: dict[str, dict[str, dict[str, float]]],
    group_label: str = "axis",
    out_dir: Path | None = None,
    norm: str = "none",
):
    """Print per-group metrics table, pairwise agreement matrix, and save heatmap.

    group_label: "axis" or "theme" (used in titles/filenames).
    """
    groups = sorted(k for k in group_model_stats if k != "overall")
    all_groups = ["overall"] + groups

    # --- Per-group metrics table (vs overall) ---
    print(f"\n=== Per-{group_label} metrics vs overall (norm={norm}) ===")
    rows = []
    for group in all_groups:
        stats = group_model_stats.get(group, {})
        sep = compute_separability(stats)
        if group == "overall":
            rows.append({
                group_label: group,
                "separability": f"{sep['separability']:.1%}",
                "agreement_vs_overall": "-",
                "spearman_vs_overall": "-",
            })
        else:
            agr = compute_agreement(stats, group_model_stats["overall"])
            spr = compute_spearman(stats, group_model_stats["overall"])
            rows.append({
                group_label: group,
                "separability": f"{sep['separability']:.1%}",
                "agreement_vs_overall": f"{agr['agreement']:.2f}",
                "spearman_vs_overall": f"{spr['correlation']:.2f}",
            })
    print(pd.DataFrame(rows).to_markdown(index=False))

    # --- Pairwise agreement matrix ---
    print(f"\n=== Pairwise {group_label} agreement matrix ===")
    n = len(all_groups)
    agreement_matrix = np.ones((n, n))
    for i, g_a in enumerate(all_groups):
        for j, g_b in enumerate(all_groups):
            if i == j:
                continue
            agr = compute_agreement(group_model_stats[g_a], group_model_stats[g_b])
            agreement_matrix[i, j] = agr["agreement"]

    df_matrix = pd.DataFrame(agreement_matrix, index=all_groups, columns=all_groups)
    print(df_matrix.round(2).to_markdown())

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Pairwise Agreement Between {group_label.title()}s (norm={norm})")
    plt.tight_layout()

    save_dir = out_dir or Path(".")
    save_path = save_dir / f"{group_label}_agreement_heatmap_{norm}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nHeatmap saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute per-sample HealthBench scores from *_allresults.json using rubric_items."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="One or more files or directories. Directories will be scanned for '*_allresults.json'.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["none", "ternary", "seven", "random"],
        default="none",
        help="Normalization for rubric points: none (raw), ternary (-1,0,1), seven (-3..3), random (random int preserving sign).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["hard", "consensus"],
        default=None,
        help="If set, prefilter to the given HealthBench subset by prompt_id.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory for recomputed reports. Defaults to the source file directory.",
    )
    parser.add_argument(
        "--polarity",
        type=str,
        choices=["all", "positive", "negative"],
        default="all",
        help="Filter rubric items by point polarity: all (default), positive (points > 0), negative (points < 0).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of rubric items to sample per example (0, 1]. Default 1.0 (no sampling).",
    )
    parser.add_argument(
        "--axis-analysis",
        action="store_true",
        help="Compute per-axis metrics (separability, agreement, spearman) and save agreement heatmap.",
    )
    parser.add_argument(
        "--theme-analysis",
        action="store_true",
        help="Compute per-theme metrics (separability, agreement, spearman) and save agreement heatmap.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    prompt_id_filter = _get_subset_prompt_ids(args.subset) if args.subset else None
    all_files: list[Path] = []

    for inp in args.inputs:
        p = Path(inp)
        if p.is_file() and p.name.endswith("_allresults.json"):
            all_files.append(p)
        elif p.is_dir():
            all_files.extend(sorted(p.glob("**/*_allresults.json")))
        else:
            # Skip non-matching single files silently
            continue

    if not all_files:
        raise SystemExit("No *_allresults.json files found in inputs.")

    summaries = []
    for file_path in all_files:
        summaries.append(
            recompute_for_file(
                file_path,
                out_dir,
                norm=args.norm,
                polarity=args.polarity,
                sample_frac=args.sample_frac,
                prompt_id_filter=prompt_id_filter,
            )
        )

    # Build per-model stats keyed by pretty name
    recomputed_model_stats: dict[str, dict] = {}
    original_model_stats: dict[str, dict] = {}
    rows = []
    for stats in summaries:
        fname = stats.get("file", "")
        pretty_name = Path(fname).stem.replace("_allresults", "")
        pretty_name = re.sub(rf"^{re.escape(args.inputs[0].split('/')[-1])}_", "", pretty_name)
        pretty_name = re.sub(r"_(\d+)_(\d+)$", "", pretty_name)

        recomputed_model_stats[pretty_name] = {
            "score": stats.get("score"),
            "ci_lower": stats.get("ci_lower"),
            "ci_upper": stats.get("ci_upper"),
        }
        original_model_stats[pretty_name] = {
            "score": stats.get("original_score"),
            "ci_lower": stats.get("original_ci_lower"),
            "ci_upper": stats.get("original_ci_upper"),
        }
        rows.append(
            {
                "model": pretty_name,
                "score": stats.get("score"),
                "n_samples": stats.get("score:n_samples"),
                "ci_lower": stats.get("ci_lower"),
                "ci_upper": stats.get("ci_upper"),
                "latency_seconds": stats.get("avg_latency_seconds"),
            }
        )

    if rows:
        label_parts = [f"norm={args.norm}", f"polarity={args.polarity}"]
        if args.sample_frac < 1.0:
            label_parts.append(f"sample_frac={args.sample_frac}")
        print(f"\n=== Recomputed scores ({', '.join(label_parts)}) ===")
        df = pd.DataFrame(rows).sort_values(by="score", ascending=False)
        df["score"] = df.apply(
            lambda r: f"{r['score']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
            if r["score"] is not None and r["ci_lower"] is not None
            else "NA",
            axis=1,
        )
        df = df.drop(columns=["ci_lower", "ci_upper"])
        if "latency_seconds" in df.columns:
            df["latency_seconds"] = df["latency_seconds"].round(2)
        df = df[["model", "score", "latency_seconds", "n_samples"]]
        print(df.to_markdown(index=False))

        # Separability for recomputed benchmark
        sep = compute_separability(recomputed_model_stats)
        print(
            f"\nSeparability: {sep['separability']:.1%} "
            f"({sep['n_separable']}/{sep['n_pairs']} model pairs confidently separated)"
        )

        # Agreement and Spearman vs original (meaningful when any recompute param differs from defaults)
        if args.norm != "none" or args.polarity != "all" or args.sample_frac < 1.0:
            agreement = compute_agreement(recomputed_model_stats, original_model_stats)
            spearman = compute_spearman(recomputed_model_stats, original_model_stats)

            print(f"\n=== Comparison vs original (norm=none) ===")
            print(
                f"Agreement: {agreement['agreement']:.2f} "
                f"({agreement['n_ref_separable']} reference-separable pairs)"
            )
            print(
                f"Spearman:  {spearman['correlation']:.2f} "
                f"(p={spearman['p_value']:.4e})"
            )

            # Also show original separability for reference
            sep_orig = compute_separability(original_model_stats)
            print(
                f"Original separability: {sep_orig['separability']:.1%} "
                f"({sep_orig['n_separable']}/{sep_orig['n_pairs']} pairs)"
            )

    # Per-axis analysis
    if args.axis_analysis:
        input_dir_name = args.inputs[0].split("/")[-1]
        axis_stats = compute_axis_model_stats(
            all_files, input_dir_name, norm=args.norm, polarity=args.polarity,
            sample_frac=args.sample_frac, prompt_id_filter=prompt_id_filter,
        )
        print_group_analysis(axis_stats, group_label="axis", out_dir=out_dir, norm=args.norm)

    # Per-theme analysis
    if args.theme_analysis:
        input_dir_name = args.inputs[0].split("/")[-1]
        theme_map = _get_prompt_theme_mapping()
        theme_stats = compute_theme_model_stats(
            all_files, input_dir_name, theme_map, norm=args.norm, polarity=args.polarity,
            sample_frac=args.sample_frac, prompt_id_filter=prompt_id_filter,
        )
        print_group_analysis(theme_stats, group_label="theme", out_dir=out_dir, norm=args.norm)

    # Rubric summary stats
    if summaries:
        total_examples = sum(s["num_samples"] for s in summaries)
        avg_raw = np.mean([s["avg_rubrics_per_example"] for s in summaries])
        avg_eff = np.mean([s["avg_effective_per_example"] for s in summaries])
        avg_pos = np.mean([s["avg_positive_per_example"] for s in summaries])
        avg_neg = np.mean([s["avg_negative_per_example"] for s in summaries])
        min_r = min(s["min_effective_per_example"] for s in summaries)
        max_r = max(s["max_effective_per_example"] for s in summaries)
        pt_min = GLOBAL_MIN_POINT_RAW if GLOBAL_MIN_POINT_RAW is not None else "NA"
        pt_max = GLOBAL_MAX_POINT_RAW if GLOBAL_MAX_POINT_RAW is not None else "NA"

        print(f"\n=== Rubric stats ===")
        print(f"Models: {len(summaries)}, Avg examples/model: {total_examples / len(summaries):.0f}")
        if args.sample_frac < 1.0 or args.polarity != "all":
            print(f"Raw rubrics/example: {avg_raw:.1f}")
        print(f"Effective rubrics/example: {avg_eff:.1f} avg, {min_r}-{max_r} range")
        print(f"Positive: {avg_pos:.1f}/example ({avg_pos/avg_eff:.0%}), Negative: {avg_neg:.1f}/example ({avg_neg/avg_eff:.0%})")
        print(f"Point range: [{pt_min}, {pt_max}]")


if __name__ == "__main__":
    main()


