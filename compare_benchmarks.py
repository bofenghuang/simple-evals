"""Compare two benchmark result directories (e.g., alternative judge vs default judge).

Usage:
    python -m simple-evals.compare_benchmarks \
        --target outputs/healthbench-llmaj-gemini3-flash_minimal \
        --reference outputs/healthbench

    python -m simple-evals.compare_benchmarks \
        --target outputs/healthbench_hard-llmaj-claude4.5-haiku \
        --reference outputs/healthbench \
        --subset hard
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from .aggregate_subset_scores import compute_subset_score_from_allresults
from .utils.stats import compute_agreement, compute_separability, compute_spearman


def _parse_model_name(fname: str, results_dir: str) -> str:
    pretty_name = Path(fname).stem.replace("_allresults", "")
    pretty_name = re.sub(rf"^{re.escape(results_dir.split('/')[-1])}_", "", pretty_name)
    pretty_name = re.sub(r"^healthbench_", "", pretty_name)
    pretty_name = re.sub(r"_(\d+)_(\d+)$", "", pretty_name)
    return pretty_name


def _build_model_stats(results: dict, results_dir: str) -> dict[str, dict]:
    model_stats = {}
    for fname, stats in results.items():
        name = _parse_model_name(fname, results_dir)
        model_stats[name] = stats
    return model_stats


def _print_scores_table(model_stats: dict[str, dict], label: str):
    rows = []
    for model, stats in model_stats.items():
        rows.append({
            "model": model,
            "score": stats.get("score"),
            "n_samples": stats.get("score:n_samples"),
            "ci_lower": stats.get("ci_lower"),
            "ci_upper": stats.get("ci_upper"),
        })
    if not rows:
        print("  No data.")
        return

    df = pd.DataFrame(rows).sort_values(by="score", ascending=False)
    df["score"] = df.apply(
        lambda r: f"{r['score']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]", axis=1
    )
    df = df.drop(columns=["ci_lower", "ci_upper"])
    df = df[["model", "score", "n_samples"]]
    print(df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark result directories.")
    parser.add_argument("--target", type=str, required=True, help="Target results dir (alternative judge)")
    parser.add_argument("--reference", type=str, required=True, help="Reference results dir (default judge)")
    parser.add_argument(
        "--subset", type=str,
        choices=["hard", "consensus", "pediatric", "pediatric_hard", "pediatric_consensus"],
        default=None,
        help="Subset filter applied to both target and reference",
    )
    args = parser.parse_args()

    # Compute scores for both dirs
    target_results = compute_subset_score_from_allresults(args.target, args.subset)
    ref_results = compute_subset_score_from_allresults(args.reference, args.subset)

    target_stats = _build_model_stats(target_results, args.target)
    ref_stats = _build_model_stats(ref_results, args.reference)

    # Filter to common models
    common_models = sorted(set(target_stats.keys()) & set(ref_stats.keys()))
    if not common_models:
        print("No common models found between target and reference.")
        return

    target_common = {m: target_stats[m] for m in common_models}
    ref_common = {m: ref_stats[m] for m in common_models}

    # Print target
    target_label = args.target.rstrip("/").split("/")[-1]
    ref_label = args.reference.rstrip("/").split("/")[-1]

    print(f"\n=== Target: {target_label} ({len(common_models)} common models) ===")
    _print_scores_table(target_common, target_label)
    sep_target = compute_separability(target_common)
    print(f"Separability: {sep_target['separability']:.1%} ({sep_target['n_separable']}/{sep_target['n_pairs']} pairs)")

    # Print reference
    print(f"\n=== Reference: {ref_label} ({len(common_models)} common models) ===")
    _print_scores_table(ref_common, ref_label)
    sep_ref = compute_separability(ref_common)
    print(f"Separability: {sep_ref['separability']:.1%} ({sep_ref['n_separable']}/{sep_ref['n_pairs']} pairs)")

    # Comparison metrics
    agreement = compute_agreement(target_common, ref_common)
    spearman = compute_spearman(target_common, ref_common)

    print(f"\n=== Comparison: {target_label} vs {ref_label} ===")
    print(f"Agreement: {agreement['agreement']:.2f} ({agreement['n_ref_separable']} reference-separable pairs)")
    print(f"Spearman:  {spearman['correlation']:.2f} (p={spearman['p_value']:.2e})")


if __name__ == "__main__":
    main()
