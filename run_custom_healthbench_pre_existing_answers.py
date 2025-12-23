#!/usr/bin/env python3
"""
Run HealthBench evaluation on custom dataset with pre-generated completions.

Usage:
    # Evaluate pre-generated completions
    python -m simple_evals.run_custom_healthbench \
        --data-path healthbench_pediatric_with_completions.jsonl \
        --use-pregenerated

    # Generate new completions with a model
    python -m simple_evals.run_custom_healthbench \
        --data-path healthbench_pediatric_expert_selected.jsonl \
        --models gpt-4.1 gpt-5_medium
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .healthbench_eval import HealthBenchEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionSampler,
)
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .types import SamplerBase
from . import common

class PreGeneratedSampler(SamplerBase):
    """Dummy sampler that uses pre-generated completions from the data."""

    def __call__(self, message_list):
        # This will never actually be called because HealthBenchEval
        # checks for pregenerated completions first
        raise NotImplementedError("This sampler should not be called")


def run_custom_healthbench_eval(
        custom_data_path: str,
        num_examples: int | None = None,
        n_threads: int = 12,
        models_chosen: list[str] | None = None,
        use_pregenerated: bool = False,
):
    """
    Run HealthBench evaluation on custom dataset.

    Args:
        custom_data_path: Path to custom JSONL dataset
        num_examples: Number of examples to run (None = all)
        n_threads: Number of threads for parallel processing
        models_chosen: List of model names to evaluate (ignored if use_pregenerated=True)
        use_pregenerated: If True, evaluate pre-generated completions from the data
    """

    # Define all available models
    models = {
        "gpt-4.1": ChatCompletionSampler(
            model="gpt-4.1",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-5_medium": OChatCompletionSampler(
            model="gpt-5",
        ),
        "gpt-5_high": OChatCompletionSampler(
            model="gpt-5",
            reasoning_effort="high",
        ),
        "gpt-4o": ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "o3-mini": OChatCompletionSampler(
            model="o3-mini",
        ),
    }

    if use_pregenerated:
        # Use dummy sampler for pre-generated completions
        models = {"pregenerated": PreGeneratedSampler()}
    elif models_chosen:
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                print(f"Available models: {', '.join(models.keys())}")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}
    else:
        # Default models
        models = {
            "gpt-4.1": models["gpt-4.1"],
            "gpt-5_medium": models["gpt-5_medium"],
        }

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")

    # Create grading sampler
    grading_sampler = ChatCompletionSampler(
        model="gpt-4.1",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )

    # Verify data file exists
    data_path = Path(custom_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {custom_data_path}")

    print(f"\n{'=' * 70}")
    print(f"HealthBench Custom Evaluation")
    print(f"{'=' * 70}")
    print(f"Dataset: {custom_data_path}")
    print(f"Mode: {'Pre-generated completions' if use_pregenerated else 'Generate new completions'}")
    print(f"Models: {', '.join(models.keys())}")
    print(f"Examples: {num_examples if num_examples else 'all'}")
    print(f"Threads: {n_threads}")
    print(f"{'=' * 70}\n")

    all_results = {}

    # Run evaluation for each model
    for model_name, sampler in models.items():
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 70}\n")

        try:
            # Create custom HealthBenchEval that supports pre-generated completions
            eval_instance = HealthBenchEvalWithPregenerated(
                grader_model=grading_sampler,
                custom_data_path=custom_data_path,
                num_examples=num_examples,
                n_threads=n_threads,
                use_pregenerated=use_pregenerated,
            )

            # Run evaluation
            result = eval_instance(sampler)
            all_results[model_name] = result

            # Save results
            file_stem = f"healthbench_custom_{model_name}_{date_str}"

            # Create output directory
            output_dir = Path("./healthbench_results")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save HTML report
            report_filename = output_dir / f"{file_stem}.html"
            report_filename.write_text(common.make_report(result))
            print(f"\n📊 Report saved to {report_filename}")

            # Save metrics JSON
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            metrics = dict(sorted(metrics.items()))
            result_filename = output_dir / f"{file_stem}.json"
            result_filename.write_text(json.dumps(metrics, indent=2))
            print(f"📈 Metrics saved to {result_filename}")

            # Save full results
            full_result_dict = {
                "score": result.score,
                "metrics": result.metrics,
                "htmls": result.htmls,
                "convos": result.convos,
                "metadata": result.metadata,
            }
            full_result_filename = output_dir / f"{file_stem}_allresults.json"
            full_result_filename.write_text(json.dumps(full_result_dict, indent=2))
            print(f"💾 Full results saved to {full_result_filename}")

            # Print score
            print(f"\n🎯 Score for {model_name}: {result.score:.4f}")

        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print final comparison
    if all_results:
        print(f"\n{'=' * 70}")
        print(f"FINAL RESULTS")
        print(f"{'=' * 70}")
        for model_name, result in all_results.items():
            print(f"{model_name:40s} | Score: {result.score:.4f}")
        print(f"{'=' * 70}\n")

    return all_results


class HealthBenchEvalWithPregenerated(HealthBenchEval):
    """Extended HealthBenchEval that supports pre-generated completions."""

    def __init__(
            self,
            grader_model,
            custom_data_path: str,
            num_examples: int | None = None,
            n_threads: int = 12,
            use_pregenerated: bool = False,
    ):
        self.use_pregenerated = use_pregenerated
        super().__init__(
            grader_model=grader_model,
            subset_name="custom",
            custom_data_path=custom_data_path,
            num_examples=num_examples,
            n_threads=n_threads,
        )

    def __call__(self, sampler):
        def fn(row: dict):
            prompt_messages = row["prompt"]

            if self.use_pregenerated:
                # Use pre-generated completion from the data
                if "pregenerated_completion" not in row:
                    raise ValueError(
                        f"Row {row.get('prompt_id')} missing 'pregenerated_completion' field"
                    )
                response_text = row["pregenerated_completion"]
                response_usage = None
                actual_queried_prompt_messages = prompt_messages
                sampler_latency_seconds = None
            else:
                # Generate new completion
                import time
                t0 = time.time()
                sampler_response = sampler(prompt_messages)
                t1 = time.time()
                response_text = sampler_response.response_text
                response_dict = sampler_response.response_metadata
                actual_queried_prompt_messages = sampler_response.actual_queried_message_list
                response_usage = response_dict.get("usage", None)
                sampler_latency_seconds = t1 - t0

            # Rest is the same as original HealthBenchEval
            metrics, readable_explanation_str, rubric_items_with_grades = self.grade_sample(
                prompt=actual_queried_prompt_messages,
                response_text=response_text,
                rubric_items=row["rubrics"],
                example_tags=row["example_tags"],
            )

            if not metrics or "overall_score" not in metrics:
                return None

            score = metrics["overall_score"]

            # Create HTML
            from .healthbench_eval import HEALTHBENCH_HTML_JINJA, get_usage_dict
            import hashlib
            html = common.jinja_env.from_string(
                HEALTHBENCH_HTML_JINJA.replace(
                    "{{ rubric_grades }}",
                    readable_explanation_str.replace("\n", "<br>"),
                )
            ).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=metrics["overall_score"],
                extracted_answer=response_text,
            )

            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]

            from .types import SingleEvalResult
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
                example_level_metadata={
                    "score": score,
                    "usage": get_usage_dict(response_usage),
                    "rubric_items": rubric_items_with_grades,
                    "prompt": actual_queried_prompt_messages,
                    "completion": [dict(content=response_text, role="assistant")],
                    "prompt_id": row["prompt_id"],
                    "completion_id": hashlib.sha256(
                        (row["prompt_id"] + (response_text if response_text is not None else "")).encode("utf-8")
                    ).hexdigest(),
                    "latency_seconds": sampler_latency_seconds,
                },
            )

        results = common.map_with_progress(
            fn,
            self.examples,
            num_threads=self.n_threads,
            pbar=True,
        )
        results = [r for r in results if r is not None]

        from .healthbench_eval import _aggregate_get_clipped_mean
        final_metrics = _aggregate_get_clipped_mean(results)
        return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run HealthBench evaluation on custom dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to custom JSONL dataset",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=None,
        help="Number of examples to run (default: all)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=12,
        help="Number of threads (default: 12)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--use-pregenerated",
        action="store_true",
        help="Evaluate pre-generated completions from the data",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list_models:
        models = ["gpt-4.1", "gpt-5_medium", "gpt-5_high", "gpt-4o", "o3-mini", "pregenerated"]
        print("Available models:")
        for model in models:
            print(f"  - {model}")
        return

    run_custom_healthbench_eval(
        custom_data_path=args.data_path,
        num_examples=args.examples,
        n_threads=args.n_threads,
        models_chosen=args.models,
        use_pregenerated=args.use_pregenerated,
    )


if __name__ == "__main__":
    main()