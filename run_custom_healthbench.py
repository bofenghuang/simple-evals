#!/usr/bin/env python3
"""
Run HealthBench evaluation on custom dataset for multiple models.

Usage:
    python -m simple_evals.run_custom_healthbench --data-path healthbench_pediatric_expert_selected.jsonl --models gpt-4.1 gpt-5_medium
    python -m simple_evals.run_custom_healthbench --data-path healthbench_pediatric_expert_selected.jsonl --models gpt-4.1 gpt-5_medium --examples 10
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
from . import common


def run_custom_healthbench_eval(
        custom_data_path: str,
        num_examples: int | None = None,
        n_threads: int = 12,
        models_chosen: list[str] | None = None,
        use_gateway: bool = False,
):
    """
    Run HealthBench evaluation on custom dataset for specified models.

    Args:
        custom_data_path: Path to custom JSONL dataset
        num_examples: Number of examples to run (None = all)
        n_threads: Number of threads for parallel processing
        models_chosen: List of model names to evaluate (None = default models)
        use_gateway: If True, use the GenAI Gateway instead of Azure OpenAI
    """

    if use_gateway:
        # Models available through the GenAI Gateway
        models = {
            # Claude models
            "claude-sonnet-4.5": ChatCompletionSampler(
                model="claude-sonnet-4-5-20250929",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=4096,
                use_gateway=True,
            ),
            "claude-haiku-4.5": ChatCompletionSampler(
                model="claude-haiku-4-5-20251001",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=4096,
                use_gateway=True,
            ),
            # Gemini models
            "gemini-2.5-pro": ChatCompletionSampler(
                model="gemini-2-5-pro-20250617",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=8192,
                use_gateway=True,
            ),
            "gemini-2.5-flash": ChatCompletionSampler(
                model="gemini-2.5-flash-20250517",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=8192,
                use_gateway=True,
            ),
        }
    else:
        models = {
            # GPT-4.1 models
            "gpt-4.1": ChatCompletionSampler(
                model="gpt-4.1",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            "gpt-4.1-mini": ChatCompletionSampler(
                model="gpt-4.1-mini-2025-04-14",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            # GPT-5 models (reasoning)
            "gpt-5_medium": OChatCompletionSampler(
                model="gpt-5",
            ),
            "gpt-5_high": OChatCompletionSampler(
                model="gpt-5",
                reasoning_effort="high",
            ),
            "gpt-5_low": OChatCompletionSampler(
                model="gpt-5",
                reasoning_effort="low",
            ),
            # GPT-4o models
            "gpt-4o": ChatCompletionSampler(
                model="gpt-4o",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
            ),
            # o3 models
            "o3-mini": OChatCompletionSampler(
                model="o3-mini",
            ),
            "o3-mini_high": OChatCompletionSampler(
                model="o3-mini",
                reasoning_effort="high",
            ),
            # o1 models
            "o1": OChatCompletionSampler(
                model="o1",
            ),
        }

    # If specific models chosen, filter to those
    if models_chosen:
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                print(f"Available models: {', '.join(models.keys())}")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}
    else:
        # Default to GPT-4.1 and GPT-5 medium
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
            # Create evaluation instance
            eval_instance = HealthBenchEval(
                grader_model=grading_sampler,
                subset_name="custom",
                custom_data_path=custom_data_path,
                num_examples=num_examples,
                n_threads=n_threads,
            )

            # Run evaluation
            result = eval_instance(sampler)
            all_results[model_name] = result

            # Save results (matching original pattern)
            file_stem = f"healthbench_custom_{model_name}_{date_str}"

            # Save HTML report
            report_filename = Path(f"/tmp/{file_stem}.html")
            report_filename.write_text(common.make_report(result))
            print(f"\n📊 Report saved to {report_filename}")

            # Save metrics JSON
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            metrics = dict(sorted(metrics.items()))
            result_filename = Path(f"/tmp/{file_stem}.json")
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
            full_result_filename = Path(f"/tmp/{file_stem}_allresults.json")
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
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS")
    print(f"{'=' * 70}")
    for model_name, result in all_results.items():
        print(f"{model_name:40s} | Score: {result.score:.4f}")
    print(f"{'=' * 70}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run HealthBench evaluation on custom dataset for multiple models."
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
        help="Number of threads for parallel processing (default: 12)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model names to evaluate (default: gpt-4.1 gpt-5_medium)",
    )
    parser.add_argument(
        "--use-gateway",
        action="store_true",
        help="Use GenAI Gateway instead of Azure OpenAI",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )

    args = parser.parse_args()

    if args.list_models:
        if args.use_gateway:
            print("Available models (GenAI Gateway):")
            models = [
                "claude-sonnet-4.5",
                "claude-haiku-4.5",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
            ]
        else:
            print("Available models (Azure OpenAI):")
            models = [
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-5_medium",
                "gpt-5_high",
                "gpt-5_low",
                "gpt-4o",
                "o3-mini",
                "o3-mini_high",
                "o1",
            ]
        for model in models:
            print(f"  - {model}")
        return

    run_custom_healthbench_eval(
        custom_data_path=args.data_path,
        num_examples=args.examples,
        n_threads=args.n_threads,
        models_chosen=args.models,
        use_gateway=args.use_gateway,
    )


if __name__ == "__main__":
    main()