from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.evaluation import _definition_index, _reference_payload
from taxonomy_experiment.inference import _load_records, _select_smoke
from taxonomy_experiment.io import iter_jsonl, read_json, write_json
from taxonomy_experiment.models import trajectory_for_prompt
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.taxonomy import (
    CONDITION_ORDER,
    Condition,
    load_taxonomy_set,
    taxonomy_block,
    taxonomy_for_condition,
)
from taxonomy_experiment.token_count import TokenCounter


def _gold_index(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    processed = resolve_path(config, "processed_data")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_gold.jsonl"):
            index[(dataset, row["trajectory_id"])] = row
    return index


def estimate_api_budget(
    config_path: str | Path = "config/experiment.yaml", mode: str = "full"
) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    config = load_config(config_path)
    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset in experiment_datasets(config):
        records = _load_records(config, dataset)
        if mode == "smoke":
            records = _select_smoke(
                records,
                int(config["sampling"]["smoke_examples_per_dataset"]),
                int(config["experiment"]["seed"]),
                dataset,
            )
        datasets[dataset] = records

    taxonomies = load_taxonomy_set(
        resolve_path(config, "taxonomy_a"),
        resolve_path(config, "taxonomy_b"),
        resolve_path(config, "merged_taxonomy"),
    )
    attr_model = config["models"]["attribution"]
    eval_model = config["models"]["simple_evaluator"]
    merge_model = config["models"]["merge"]
    attr_output_ceiling = max(
        int(attr_model["max_output_tokens"]),
        int(config["execution"]["attribution_truncation_retry_max_output_tokens"]),
    )
    attr_counter = TokenCounter.for_model(attr_model["name"], attr_model.get("tokenizer_encoding"))
    eval_counter = TokenCounter.for_model(eval_model["name"], eval_model.get("tokenizer_encoding"))
    merge_counter = TokenCounter.for_model(merge_model["name"], merge_model.get("tokenizer_encoding"))
    attr_system = read_prompt(config["_root"], "attribution_system.txt")
    attr_template = read_prompt(config["_root"], "attribution_user.txt")
    eval_system = read_prompt(config["_root"], "simple_evaluator_system.txt")
    eval_template = read_prompt(config["_root"], "simple_evaluator_user.txt")
    merge_template = read_prompt(config["_root"], "taxonomy_merge.txt")

    attribution_rows: list[dict[str, Any]] = []
    unavailable_conditions: set[str] = set()
    for dataset, records in datasets.items():
        for record in records:
            trajectory = trajectory_for_prompt(record)
            for condition in CONDITION_ORDER:
                try:
                    taxonomy = taxonomy_for_condition(dataset, condition, taxonomies)
                except FileNotFoundError:
                    unavailable_conditions.add(condition.value)
                    continue
                user = format_template(
                    attr_template,
                    taxonomy_block=taxonomy_block(taxonomy),
                    trajectory=trajectory,
                )
                attribution_rows.append(
                    {
                        "dataset": dataset,
                        "trajectory_id": record["trajectory_id"],
                        "condition": condition.value,
                        "input_tokens": attr_counter.count(attr_system + "\n" + user),
                    }
                )

    gold = _gold_index(config)
    candidate_placeholder = {
        "error_type": "<model error type>",
        "overall_root_cause": "<model root cause>",
        "explanation": "<model explanation>",
        "evidence": ["<trajectory evidence>"],
    }
    evaluator_lower_bound = 0
    example_count = sum(len(records) for records in datasets.values())
    definitions = _definition_index(config)
    for dataset, records in datasets.items():
        for record in records:
            reference = gold[(dataset, record["trajectory_id"])]
            reference_types = reference.get("failure_types") or [reference["failure_type"]]
            reference_payload = {
                "accepted_failure_types": [
                    {
                        "name": failure_type,
                        "definition": definitions.get((dataset, failure_type)),
                    }
                    for failure_type in reference_types
                ],
                "reference_reasoning": reference.get("reference_reasoning") or None,
            }
            for _condition in CONDITION_ORDER:
                user = format_template(
                    eval_template,
                    trajectory=trajectory_for_prompt(record),
                    reference=json.dumps(reference_payload, ensure_ascii=False, indent=2),
                    prediction=json.dumps(candidate_placeholder, ensure_ascii=False, indent=2),
                )
                evaluator_lower_bound += eval_counter.count(eval_system + "\n" + user)

    taxonomy_a = read_json(resolve_path(config, "taxonomy_a"))
    taxonomy_b = read_json(resolve_path(config, "taxonomy_b"))
    from taxonomy_experiment.taxonomy import format_taxonomy

    merge_user = format_template(
        merge_template,
        taxonomy_a=format_taxonomy(taxonomy_a),
        taxonomy_b=format_taxonomy(taxonomy_b),
    )
    merge_tokens = merge_counter.count(
        "Merge only the two supplied taxonomy definitions. Do not use dataset knowledge.\n"
        + merge_user
    )
    planned_attribution_calls = example_count * len(CONDITION_ORDER)
    planned_evaluator_calls = example_count * len(CONDITION_ORDER)
    known_attribution_tokens = sum(row["input_tokens"] for row in attribution_rows)
    if Condition.MERGED_TAXONOMY.value in unavailable_conditions:
        baseline_tokens = sum(
            row["input_tokens"]
            for row in attribution_rows
            if row["condition"] == Condition.NO_TAXONOMY.value
        )
        conservative_merged_tokens = baseline_tokens + example_count * (
            int(merge_model["max_output_tokens"]) + 256
        )
        attribution_input_upper = known_attribution_tokens + conservative_merged_tokens
        merged_input_policy = (
            "Merged taxonomy is not frozen; reserve assumes every merge output token plus "
            "256 wrapper tokens is repeated in every merged-condition request."
        )
    else:
        conservative_merged_tokens = 0
        attribution_input_upper = known_attribution_tokens
        merged_input_policy = "Frozen merged taxonomy was token-counted directly."

    evaluator_input_upper = evaluator_lower_bound + (
        planned_evaluator_calls * attr_output_ceiling
    )
    safety = float(config["cost_budget"]["input_token_safety_multiplier"])

    def role_cost(
        model_config: dict[str, Any], input_tokens: int, output_tokens: int
    ) -> dict[str, float | int]:
        prices = model_config["pricing_usd_per_million"]
        guarded_input = math.ceil(input_tokens * safety)
        input_cost = guarded_input * float(prices["input"]) / 1_000_000
        output_cost = output_tokens * float(prices["output"]) / 1_000_000
        return {
            "guarded_input_tokens": guarded_input,
            "maximum_output_tokens": output_tokens,
            "input_usd": input_cost,
            "output_usd": output_cost,
            "total_usd": input_cost + output_cost,
        }

    cost_roles = {
        "merge": role_cost(merge_model, merge_tokens, int(merge_model["max_output_tokens"])),
        "attribution": role_cost(
            attr_model,
            attribution_input_upper,
            planned_attribution_calls * attr_output_ceiling,
        ),
        "evaluator": role_cost(
            eval_model,
            evaluator_input_upper,
            planned_evaluator_calls * int(eval_model["max_output_tokens"]),
        ),
    }
    inference_upper = sum(float(item["total_usd"]) for item in cost_roles.values())
    ledger_path = Path(config["cost_budget"]["ledger_path"])
    if not ledger_path.is_absolute():
        ledger_path = config["_root"] / ledger_path
    incurred_ledger = (
        sum(float(row.get("cost_usd", 0.0)) for row in iter_jsonl(ledger_path))
        if ledger_path.exists()
        else 0.0
    )
    projected_cumulative_inference = inference_upper
    fee = max(
        projected_cumulative_inference
        * float(config["cost_budget"]["credit_purchase_fee_rate"]),
        float(config["cost_budget"]["minimum_credit_purchase_fee_usd"]),
    )
    charged_upper = projected_cumulative_inference + fee
    inference_cap = float(config["cost_budget"]["max_inference_usd"])
    charge_cap = float(config["cost_budget"]["max_total_charge_usd"])
    within_budget = (
        projected_cumulative_inference <= inference_cap and charged_upper <= charge_cap
    )
    report = {
        "mode": mode,
        "example_counts": {name: len(records) for name, records in datasets.items()},
        "planned_calls": {
            "merge": 1,
            "attribution": planned_attribution_calls,
            "evaluator": planned_evaluator_calls,
            "total": 1 + planned_attribution_calls + planned_evaluator_calls,
        },
        "known_attribution_requests": len(attribution_rows),
        "unavailable_until_merge": sorted(unavailable_conditions),
        "tokenizers": {
            "attribution": attr_counter.encoding_name,
            "merge": merge_counter.encoding_name,
            "evaluator": eval_counter.encoding_name,
        },
        "input_tokens": {
            "merge": merge_tokens,
            "attribution_known_conditions": known_attribution_tokens,
            "attribution_known_max_request": max(row["input_tokens"] for row in attribution_rows),
            "attribution_conservative_merged_reserve": conservative_merged_tokens,
            "attribution_upper_bound": attribution_input_upper,
            "evaluator_lower_bound_all_examples": evaluator_lower_bound,
            "evaluator_upper_bound_with_maximum_predictions": evaluator_input_upper,
            "note": merged_input_policy,
        },
        "maximum_configured_output_tokens": {
            "merge": int(merge_model["max_output_tokens"]),
            "attribution_all_calls": planned_attribution_calls * attr_output_ceiling,
            "evaluator_all_calls": planned_evaluator_calls * int(eval_model["max_output_tokens"]),
        },
        "cost_budget": {
            "currency": "USD",
            "pricing_snapshot": {
                role: {
                    "model": model["name"],
                    "usd_per_million": model["pricing_usd_per_million"],
                }
                for role, model in (
                    ("merge", merge_model),
                    ("attribution", attr_model),
                    ("evaluator", eval_model),
                )
            },
            "input_token_safety_multiplier": safety,
            "roles": cost_roles,
            "incurred_ledger_inference_usd": incurred_ledger,
            "hard_runtime_max_total_charge_usd": inference_cap
            + max(
                inference_cap
                * float(config["cost_budget"]["credit_purchase_fee_rate"]),
                float(config["cost_budget"]["minimum_credit_purchase_fee_usd"]),
            ),
            "projected_maximum_inference_usd": projected_cumulative_inference,
            "projected_credit_purchase_fee_usd": fee,
            "projected_maximum_total_charge_usd": charged_upper,
            "configured_inference_cap_usd": inference_cap,
            "configured_total_charge_cap_usd": charge_cap,
            "headroom_to_inference_cap_usd": inference_cap
            - projected_cumulative_inference,
            "within_configured_budget": within_budget,
            "scope": (
                "One complete theoretical run, with every attribution using the truncation "
                "retry ceiling and every condition independently judged by the simple evaluator. "
                "The runtime ledger independently limits inference to the configured cap."
            ),
        },
    }
    processed = resolve_path(config, "processed_data")
    write_json(processed / f"api_budget_{mode}.json", report)
    if not within_budget:
        raise ValueError(
            f"Configured experiment exceeds the hard cost budget: projected inference "
            f"${projected_cumulative_inference:.4f} / ${inference_cap:.2f}, "
            f"projected total charge "
            f"${charged_upper:.4f} / ${charge_cap:.2f}. See api_budget_{mode}.json."
        )
    return report
