import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from datasets import load_dataset

from .healthbench_eval import INPUT_PATH_CONSENSUS, INPUT_PATH_HARD, _compute_clipped_stats
from .utils.stats import bootstrap_ci, compute_separability

SubsetName = Literal["hard", "consensus", "pediatric", "pediatric_hard", "pediatric_consensus"]

PEDIATRIC_DATASETS: dict[str, tuple[str, str]] = {
    "pediatric": ("bofenghuang/healthbench-pediatric", "pediatric"),
    "pediatric_hard": ("bofenghuang/healthbench-hard-pediatric", "pediatric"),
    "pediatric_consensus": ("bofenghuang/healthbench-consensus-pediatric", "pediatric"),
}


def _get_subset_prompt_ids(subset_name: SubsetName) -> set[str]:
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

    dataset_name, config_name = PEDIATRIC_DATASETS[subset_name]
    ds = load_dataset(dataset_name, config_name, split="test")
    return set(ds["prompt_id"])


def compute_subset_score_from_allresults(
    results_dir: str,
    subset_name: Optional[SubsetName] = None,
) -> dict[str, dict[str, float]]:
    """
    Read all *_allresults.json files under results_dir, extract per-sample scores
    from metadata.example_level_metadata, filter to a subset by prompt_id, and
    compute clipped stats (mean, n_samples, bootstrap_std) using _compute_clipped_stats.

    Returns a mapping of filename -> {"score", "score:n_samples", "score:bootstrap_std"}.
    Subsets supported: hard, consensus, pediatric, pediatric_hard, pediatric_consensus.
    """
    base = Path(results_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    subset_ids = None if subset_name is None else _get_subset_prompt_ids(subset_name)
    output: dict[str, dict[str, float]] = {}

    for p in sorted(base.glob("*_allresults.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        metadata = data.get("metadata") or {}
        example_meta = metadata.get("example_level_metadata") or []
        # example_meta is a list of dicts, each with keys including "score" and "prompt_id"
        scores = []
        latencies = []
        rubric_counts = []
        for em in example_meta:
            if not isinstance(em, dict):
                continue
            if subset_ids is not None and em.get("prompt_id") not in subset_ids:
                continue

            if em.get("score") is not None:
                scores.append(em.get("score"))

            if isinstance(em.get("latency_seconds"), (int, float)):
                latencies.append(em.get("latency_seconds"))

            rubric_items = em.get("rubric_items")
            if isinstance(rubric_items, list):
                rubric_counts.append(len(rubric_items))

        if len(scores) == 0:
            continue

        try:
            mean_score = _compute_clipped_stats(scores, "mean")
            n = _compute_clipped_stats(scores, "n_samples")
            std = _compute_clipped_stats(scores, "bootstrap_std")
            ci = bootstrap_ci(scores)
        except Exception:
            # if anything odd with values, skip this file
            continue

        output[p.name] = {
            "score": float(mean_score),
            "score:n_samples": float(n),
            "score:bootstrap_std": float(std),
            "ci_lower": ci["lower_bound"],
            "ci_upper": ci["upper_bound"],
            "avg_latency_seconds": (sum(latencies) / len(latencies)) if len(latencies) > 0 else None,
            "n_rubrics": sum(rubric_counts),
        }

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate HealthBench scores from *_allresults.json "
            "(optionally filter to hard/consensus/pediatric subsets)"
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "healthbench"),
        help="Directory containing *_allresults.json files",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["hard", "consensus", "pediatric", "pediatric_hard", "pediatric_consensus"],
        default=None,
        help="Subset to aggregate over; default: no filter",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write aggregated JSON results",
    )
    args = parser.parse_args()

    results = compute_subset_score_from_allresults(results_dir=args.results_dir, subset_name=args.subset)
    if len(results) == 0:
        print("No aggregated results found.")
        return

    # Build per-model stats keyed by pretty name (for separability)
    model_stats: dict[str, dict[str, float]] = {}
    rows = []
    for fname, stats in results.items():
        pretty_name = Path(fname).stem.replace("_allresults", "")
        # If a subset filter is specified, also drop that subset prefix if present
        pretty_name = re.sub(rf"^{re.escape(args.results_dir.split('/')[-1])}_", "", pretty_name)
        # fallback
        pretty_name = re.sub(r"^healthbench_", "", pretty_name)
        # remove date
        pretty_name = re.sub(r"_(\d+)_(\d+)$", "", pretty_name)

        model_stats[pretty_name] = stats
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

    df = pd.DataFrame(rows).sort_values(by="score", ascending=False)
    # Format score as "score [ci_lower, ci_upper]"
    df["score"] = df.apply(
        lambda r: f"{r['score']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]", axis=1
    )
    df = df.drop(columns=["ci_lower", "ci_upper"])
    df["latency_seconds"] = df["latency_seconds"].round(2)
    df = df[["model", "score", "latency_seconds", "n_samples"]]
    print(df.to_markdown(index=False))

    # Compute and display separability
    sep = compute_separability(model_stats)
    print(
        f"\nSeparability: {sep['separability']:.1%} "
        f"({sep['n_separable']}/{sep['n_pairs']} model pairs confidently separated)"
    )

    if args.output_json is not None:
        output = {
            "models": model_stats,
            "separability": sep,
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"Wrote aggregated JSON to {args.output_json}")


if __name__ == "__main__":
    main()
