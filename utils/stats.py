"""Shared statistical utilities for benchmark evaluation and comparison."""

import itertools

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap
from scipy.stats import spearmanr

_rng = np.random.default_rng(seed=42)
_N_BOOTSTRAP_ITERATIONS = 10000


def bootstrap_ci(
    values: list[float],
    confidence_level: float = 0.95,
    n_resamples: int = _N_BOOTSTRAP_ITERATIONS,
) -> dict[str, float]:
    """Compute bootstrap confidence interval using BCa method.

    Returns dict with keys: mean, lower_bound, upper_bound.
    """
    if len(values) == 0:
        return {"mean": 0.0, "lower_bound": 0.0, "upper_bound": 0.0}

    if len(values) == 1:
        value = float(values[0])
        return {"mean": value, "lower_bound": value, "upper_bound": value}

    try:
        result = scipy_bootstrap(
            data=[values],
            statistic=np.mean,
            n_resamples=n_resamples,
            vectorized=True,
            method="BCa",
            random_state=_rng,
            confidence_level=confidence_level,
        )
        mean = float(round(np.mean(result.bootstrap_distribution), 2))
        lower = float(round(result.confidence_interval.low, 2))
        upper = float(round(result.confidence_interval.high, 2))

        if np.isnan(lower):
            lower = mean
        if np.isnan(upper):
            upper = mean

        return {"mean": mean, "lower_bound": lower, "upper_bound": upper}

    except Exception:
        mean_value = float(np.mean(values))
        return {"mean": mean_value, "lower_bound": mean_value, "upper_bound": mean_value}


def compute_separability(model_stats: dict[str, dict[str, float]]) -> dict:
    """Compute separability: fraction of model pairs whose 95% CIs do not overlap.

    Args:
        model_stats: mapping of model_name -> dict with keys "ci_lower" and "ci_upper".

    Returns:
        Dict with separability fraction, counts, and per-pair details.
    """
    models = sorted(model_stats.keys())
    pairs = list(itertools.combinations(models, 2))

    if len(pairs) == 0:
        return {"separability": 0.0, "n_pairs": 0, "n_separable": 0, "pair_details": []}

    pair_details = []
    n_separable = 0
    for model_a, model_b in pairs:
        ci_a = (model_stats[model_a]["ci_lower"], model_stats[model_a]["ci_upper"])
        ci_b = (model_stats[model_b]["ci_lower"], model_stats[model_b]["ci_upper"])
        separable = ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]
        if separable:
            n_separable += 1
        pair_details.append({
            "model_a": model_a,
            "model_b": model_b,
            "ci_a": list(ci_a),
            "ci_b": list(ci_b),
            "separable": separable,
        })

    return {
        "separability": n_separable / len(pairs),
        "n_pairs": len(pairs),
        "n_separable": n_separable,
        "pair_details": pair_details,
    }


def compute_agreement(
    benchmark_a: dict[str, dict[str, float]],
    reference_b: dict[str, dict[str, float]],
) -> dict:
    """Compute agreement of benchmark A with respect to reference B.

    For each model pair that reference B can confidently separate:
    - +1.0 if A also separates them AND rank order agrees with B
    - -1.0 if A separates them AND rank order disagrees with B
    -  0.0 if A cannot separate them

    Agreement = mean of these scores over all B-separable pairs.

    Args:
        benchmark_a: {model_name: {score, ci_lower, ci_upper}} for benchmark A.
        reference_b: {model_name: {score, ci_lower, ci_upper}} for reference B.

    Returns:
        Dict with agreement score, counts, and per-pair details.
    """
    common_models = sorted(set(benchmark_a.keys()) & set(reference_b.keys()))
    pairs = list(itertools.combinations(common_models, 2))

    if len(pairs) == 0:
        return {"agreement": 0.0, "n_ref_separable": 0, "n_pairs": 0, "pair_details": []}

    pair_details = []
    scores = []

    for model_x, model_y in pairs:
        ref_ci_x = (reference_b[model_x]["ci_lower"], reference_b[model_x]["ci_upper"])
        ref_ci_y = (reference_b[model_y]["ci_lower"], reference_b[model_y]["ci_upper"])
        ref_separable = ref_ci_x[1] < ref_ci_y[0] or ref_ci_y[1] < ref_ci_x[0]

        if not ref_separable:
            continue

        # Reference rank order: which model scores higher in B?
        ref_order = reference_b[model_x]["score"] - reference_b[model_y]["score"]

        # Benchmark A separability and rank order
        a_ci_x = (benchmark_a[model_x]["ci_lower"], benchmark_a[model_x]["ci_upper"])
        a_ci_y = (benchmark_a[model_y]["ci_lower"], benchmark_a[model_y]["ci_upper"])
        a_separable = a_ci_x[1] < a_ci_y[0] or a_ci_y[1] < a_ci_x[0]

        if not a_separable:
            pair_score = 0.0
        else:
            a_order = benchmark_a[model_x]["score"] - benchmark_a[model_y]["score"]
            # Same sign means same rank order
            pair_score = 1.0 if (ref_order * a_order > 0) else -1.0

        scores.append(pair_score)
        pair_details.append({
            "model_x": model_x,
            "model_y": model_y,
            "ref_separable": True,
            "a_separable": a_separable,
            "score": pair_score,
        })

    agreement = float(np.mean(scores)) if scores else 0.0
    return {
        "agreement": agreement,
        "n_ref_separable": len(scores),
        "n_pairs": len(pairs),
        "pair_details": pair_details,
    }


def compute_spearman(
    benchmark_a: dict[str, dict[str, float]],
    reference_b: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute Spearman rank correlation between two benchmarks' model scores.

    Args:
        benchmark_a: {model_name: {score, ...}} for benchmark A.
        reference_b: {model_name: {score, ...}} for reference B.

    Returns:
        Dict with correlation and p_value.
    """
    common_models = sorted(set(benchmark_a.keys()) & set(reference_b.keys()))
    if len(common_models) < 3:
        return {"correlation": float("nan"), "p_value": float("nan")}

    scores_a = [benchmark_a[m]["score"] for m in common_models]
    scores_b = [reference_b[m]["score"] for m in common_models]

    corr, pval = spearmanr(scores_a, scores_b)
    return {"correlation": float(corr), "p_value": float(pval)}
