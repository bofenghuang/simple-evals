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

from .utils.stats import bootstrap_ci, compute_agreement, compute_separability, compute_spearman

# Public dataset endpoints (duplicated to avoid import-time package issues)
INPUT_PATH_HARD = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
INPUT_PATH_CONSENSUS = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"

# Global trackers for extreme normalized rubric points across all processed files
GLOBAL_MIN_POINT: float | None = None
GLOBAL_MAX_POINT: float | None = None
GLOBAL_MIN_POINT_RAW: float | None = None
GLOBAL_MAX_POINT_RAW: float | None = None

# Global frequency counters for rubric points (raw and normalized)
GLOBAL_POINT_COUNTS: dict[float, int] = {}
GLOBAL_POINT_COUNTS_RAW: dict[float, int] = {}

def _update_global_point_extremes(values: list[float]) -> None:
    global GLOBAL_MIN_POINT, GLOBAL_MAX_POINT
    for v in values:
        if GLOBAL_MIN_POINT is None or v < GLOBAL_MIN_POINT:
            GLOBAL_MIN_POINT = float(v)
        if GLOBAL_MAX_POINT is None or v > GLOBAL_MAX_POINT:
            GLOBAL_MAX_POINT = float(v)

def _update_global_point_extremes_raw(values: list[float]) -> None:
    global GLOBAL_MIN_POINT_RAW, GLOBAL_MAX_POINT_RAW
    for v in values:
        if GLOBAL_MIN_POINT_RAW is None or v < GLOBAL_MIN_POINT_RAW:
            GLOBAL_MIN_POINT_RAW = float(v)
        if GLOBAL_MAX_POINT_RAW is None or v > GLOBAL_MAX_POINT_RAW:
            GLOBAL_MAX_POINT_RAW = float(v)

def _update_global_point_counts(values: list[float]) -> None:
    """
    Update global counts for normalized rubric points observed.
    """
    global GLOBAL_POINT_COUNTS
    for v in values:
        fv = float(v)
        GLOBAL_POINT_COUNTS[fv] = GLOBAL_POINT_COUNTS.get(fv, 0) + 1

def _update_global_point_counts_raw(values: list[float]) -> None:
    """
    Update global counts for raw rubric points observed (before normalization).
    """
    global GLOBAL_POINT_COUNTS_RAW
    for v in values:
        fv = float(v)
        GLOBAL_POINT_COUNTS_RAW[fv] = GLOBAL_POINT_COUNTS_RAW.get(fv, 0) + 1

def _format_sorted_counts(d: dict[float, int]) -> str:
    """
    Return a compact, sorted 'value:count' comma-separated string for counts.
    """
    if not d:
        return "NA"
    items = sorted(d.items(), key=lambda kv: kv[0])
    return ", ".join(f"{k:g}:{v}" for k, v in items)


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


def _normalize_points_list(values: list[float], mode: str) -> list[float]:
    if mode == "none":
        out = [float(v) for v in values]
        _update_global_point_extremes(out)
        return out
    if mode == "ternary":
        mapped = []
        for v in values:
            if v > 0:
                mapped.append(1.0)
            elif v < 0:
                mapped.append(-1.0)
            else:
                mapped.append(0.0)
        _update_global_point_extremes(mapped)
        return mapped
    if mode == "seven":
        # Scale linearly into [-3, 3] using max absolute value per-sample.
        abs_max = max((abs(v) for v in values), default=0.0)
        if abs_max == 0:
            out = [0.0 for _ in values]
            _update_global_point_extremes(out)
            return out
        scale = 3.0 / abs_max
        # First scale and clamp to [-3, 3], then round to nearest int in [-3..3]
        scaled = [max(-3.0, min(3.0, float(v) * scale)) for v in values]
        int_scaled = [float(max(-3, min(3, int(round(x))))) for x in scaled]
        _update_global_point_extremes(int_scaled)
        return int_scaled
    # Fallback: no normalization
    out = [float(v) for v in values]
    _update_global_point_extremes(out)
    return out


def calculate_score_from_rubric_items(
    rubric_items: list[dict[str, Any]],
    norm: str = "none",
) -> float | None:
    """
    Recompute overall score for a single sample based on rubric_items, following
    medical_llm_open_evals/simple-evals/healthbench_eval.py::calculate_score logic:
      - Optional normalization of rubric item points via --norm:
          none      -> use original points
          ternary   -> map points to {-1, 0, 1}
          seven     -> scale points linearly into [-3..3] using per-sample max abs
      - Denominator: sum of normalized points for rubric items with positive normalized points only
      - Numerator: sum of normalized points for items where criteria_met is True (includes negatives)
      - Return numerator / denominator, or None if denominator == 0
    """
    raw_points = [float(item["points"]) for item in rubric_items]
    _update_global_point_extremes_raw(raw_points)
    _update_global_point_counts_raw(raw_points)
    normalized_points = _normalize_points_list(raw_points, norm)
    _update_global_point_counts(normalized_points)
    total_possible_points = sum(p for p in normalized_points if p > 0)
    if total_possible_points == 0:
        return None

    achieved_points = 0.0
    for item, p in zip(rubric_items, normalized_points):
        if bool(item.get("criteria_met", False)) is True:
            achieved_points += p
    return achieved_points / total_possible_points


def recompute_for_file(
    path: Path,
    out_dir: Path | None = None,
    norm: str = "none",
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
        recomputed = calculate_score_from_rubric_items(rubric_items, norm=norm)
        original = sample.get("score", None)

        if recomputed is None:
            positive_denominator_zero += 1

        # Compute delta only if both are present
        delta = None
        if recomputed is not None and original is not None:
            delta = float(recomputed) - float(original)
            # Consider very small floating differences as equal
            if abs(delta) > 1e-9:
                mismatches += 1

        # Count positive items relative to the normalization mode
        normalized_points = _normalize_points_list(
            [float(it["points"]) for it in rubric_items], norm
        )
        num_positive_items = sum(1 for p in normalized_points if p > 0)
        num_positive_met = 0
        for it, p in zip(rubric_items, normalized_points):
            if p > 0 and bool(it.get("criteria_met", False)):
                num_positive_met += 1

        results.append(
            {
                "prompt_id": sample.get("prompt_id"),
                "completion_id": sample.get("completion_id"),
                "recomputed_overall_score": recomputed,
                "original_overall_score": original,
                "delta": delta,
                "num_positive_items": num_positive_items,
                "num_positive_met": num_positive_met,
                "latency_seconds": sample.get("latency_seconds"),
            }
        )

    summary = {
        "file": str(path),
        "num_samples": len(example_level_metadata),
        "num_mismatches_gt_1e-9": mismatches,
        "num_no_positive_points": positive_denominator_zero,
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
    prompt_id_filter: set[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-axis per-model stats.

    Per-axis score = mean of binary criteria_met across rubric items tagged with that axis.
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
            # Per-axis: binary criteria_met
            for ri in rubric_items:
                val = 1.0 if ri.get("criteria_met") else 0.0
                for tag in ri.get("tags", []):
                    if tag.startswith("axis:"):
                        axis_name = tag.replace("axis:", "")
                        axis_values[axis_name][model_name].append(val)

            # Full: recomputed overall score with norm
            score = calculate_score_from_rubric_items(rubric_items, norm=norm)
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


def print_axis_analysis(
    axis_model_stats: dict[str, dict[str, dict[str, float]]],
    out_dir: Path | None = None,
    norm: str = "none",
):
    """Print per-axis metrics table, pairwise agreement matrix, and save heatmap."""
    axes = sorted(k for k in axis_model_stats if k != "overall")
    all_axes = ["overall"] + axes

    # --- Per-axis metrics table (vs overall) ---
    print(f"\n=== Per-axis metrics vs overall (norm={norm}) ===")
    rows = []
    for axis in all_axes:
        stats = axis_model_stats.get(axis, {})
        sep = compute_separability(stats)
        if axis == "overall":
            rows.append({
                "axis": axis,
                "separability": f"{sep['separability']:.1%}",
                "agreement_vs_overall": "-",
                "spearman_vs_overall": "-",
            })
        else:
            agr = compute_agreement(stats, axis_model_stats["overall"])
            spr = compute_spearman(stats, axis_model_stats["overall"])
            rows.append({
                "axis": axis,
                "separability": f"{sep['separability']:.1%}",
                "agreement_vs_overall": f"{agr['agreement']:.2f}",
                "spearman_vs_overall": f"{spr['correlation']:.2f}",
            })
    print(pd.DataFrame(rows).to_markdown(index=False))

    # --- Pairwise agreement matrix ---
    print(f"\n=== Pairwise agreement matrix ===")
    n = len(all_axes)
    agreement_matrix = np.ones((n, n))
    for i, ax_a in enumerate(all_axes):
        for j, ax_b in enumerate(all_axes):
            if i == j:
                continue
            agr = compute_agreement(axis_model_stats[ax_a], axis_model_stats[ax_b])
            agreement_matrix[i, j] = agr["agreement"]

    df_matrix = pd.DataFrame(agreement_matrix, index=all_axes, columns=all_axes)
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
    ax.set_title(f"Pairwise Agreement Between Axes (norm={norm})")
    plt.tight_layout()

    save_dir = out_dir or Path(".")
    save_path = save_dir / f"axis_agreement_heatmap_{norm}.png"
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
        choices=["none", "ternary", "seven"],
        default="none",
        help="Normalization for rubric points: none (raw), ternary (-1,0,1), seven (-3..3).",
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
        "--axis-analysis",
        action="store_true",
        help="Compute per-axis metrics (separability, agreement, spearman) and save agreement heatmap.",
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
        print(f"\n=== Recomputed scores (norm={args.norm}) ===")
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

        # Agreement and Spearman vs original (only meaningful when norm != "none")
        if args.norm != "none":
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
            all_files, input_dir_name, norm=args.norm, prompt_id_filter=prompt_id_filter,
        )
        print_axis_analysis(axis_stats, out_dir=out_dir, norm=args.norm)

    # Print a concise multi-file summary to stdout
    # print(json.dumps({"summaries": summaries}, indent=4))
    # Print global raw rubric point extremes (before normalization)
    print(
        f"Global raw rubric point range: "
        f"min={GLOBAL_MIN_POINT_RAW if GLOBAL_MIN_POINT_RAW is not None else 'NA'}, "
        f"max={GLOBAL_MAX_POINT_RAW if GLOBAL_MAX_POINT_RAW is not None else 'NA'}"
    )
    # Print global normalized rubric point extremes
    print(
        f"Global normalized rubric point range: "
        f"min={GLOBAL_MIN_POINT if GLOBAL_MIN_POINT is not None else 'NA'}, "
        f"max={GLOBAL_MAX_POINT if GLOBAL_MAX_POINT is not None else 'NA'}"
    )
    # Print global counts (raw and normalized)
    print(
        "Global raw rubric point counts (value:count): "
        f"{_format_sorted_counts(GLOBAL_POINT_COUNTS_RAW)}"
    )
    print(
        "Global normalized rubric point counts (value:count): "
        f"{_format_sorted_counts(GLOBAL_POINT_COUNTS)}"
    )


if __name__ == "__main__":
    main()


