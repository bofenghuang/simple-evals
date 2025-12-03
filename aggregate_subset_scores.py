import argparse
import json
from pathlib import Path
from typing import Literal, Optional
import urllib.request
import pandas as pd
import re

from .healthbench_eval import _compute_clipped_stats, INPUT_PATH_HARD, INPUT_PATH_CONSENSUS


def _get_subset_prompt_ids(subset_name: Literal["hard", "consensus"]) -> set[str]:
    """
    Fetch prompt_ids for a given HealthBench subset from the public JSONL.
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


def compute_subset_score_from_allresults(
    results_dir: str,
    subset_name: Optional[Literal["hard", "consensus"]] = None,
) -> dict[str, dict[str, float]]:
    """
    Read all *_allresults.json files under results_dir, extract per-sample scores
    from metadata.example_level_metadata, filter to a subset by prompt_id, and
    compute clipped stats (mean, n_samples, bootstrap_std) using _compute_clipped_stats.

    Returns a mapping of filename -> {"score", "score:n_samples", "score:bootstrap_std"}.
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
        scores = [
            em.get("score")
            for em in example_meta
            if isinstance(em, dict)
            and (subset_ids is None or em.get("prompt_id") in subset_ids)
            and em.get("score") is not None
        ]
        latencies = [
            em.get("latency_seconds")
            for em in example_meta
            if isinstance(em, dict)
            and (subset_ids is None or em.get("prompt_id") in subset_ids)
            and isinstance(em.get("latency_seconds"), (int, float))
        ]
        if len(scores) == 0:
            continue

        try:
            mean_score = _compute_clipped_stats(scores, "mean")
            n = _compute_clipped_stats(scores, "n_samples")
            std = _compute_clipped_stats(scores, "bootstrap_std")
        except Exception:
            # if anything odd with values, skip this file
            continue

        output[p.name] = {
            "score": float(mean_score),
            "score:n_samples": float(n),
            "score:bootstrap_std": float(std),
            "avg_latency_seconds": (sum(latencies) / len(latencies)) if len(latencies) > 0 else None,
        }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate HealthBench scores from *_allresults.json (optionally filter to hard/consensus subset)"
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
        choices=["hard", "consensus"],
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

    rows = []
    for fname, stats in results.items():
        pretty_name = Path(fname).stem.replace("_allresults", "")
        # If a subset filter is specified, also drop that subset prefix if present
        # pretty_name = re.sub(r"^healthbench_", "", pretty_name)
        pretty_name = re.sub(rf"^{re.escape(args.results_dir.split('/')[-1])}_", "", pretty_name)
        # remove date
        pretty_name = re.sub(r"_(\d+)_(\d+)$", "", pretty_name)

        rows.append(
            {
                # "file": fname,
                "model": pretty_name,
                "score": stats.get("score"),
                "n_samples": stats.get("score:n_samples"),
                "std": stats.get("score:bootstrap_std"),
                "latency_seconds": stats.get("avg_latency_seconds"),
            }
        )

    df = pd.DataFrame(rows).sort_values(by="score", ascending=False)
    # Format score as "score +- std" and round latency
    df["score"] = df.apply(lambda r: f"{r['score']:.4f} +- {r['std']:.4f}", axis=1)
    df = df.drop(columns=["std"])
    df["latency_seconds"] = df["latency_seconds"].round(2)
    df = df[["model", "score", "latency_seconds", "n_samples"]]
    print(df.to_markdown(index=False))

    if args.output_json is not None:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"Wrote aggregated JSON to {args.output_json}")


if __name__ == "__main__":
    main()
