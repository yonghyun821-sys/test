from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
from taxonomy_experiment.models import BinaryEvaluation, trajectory_for_prompt
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.token_count import TokenCounter


def _index_processed(config: dict[str, Any], suffix: str) -> dict[tuple[str, str], dict[str, Any]]:
    processed = resolve_path(config, "processed_data")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_{suffix}.jsonl"):
            index[(dataset, row["trajectory_id"])] = row
    return index


def _definition_index(config: dict[str, Any]) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for dataset, path_key in zip(experiment_datasets(config), ("taxonomy_a", "taxonomy_b")):
        taxonomy = read_json(resolve_path(config, path_key))
        for category in taxonomy["categories"]:
            index[(dataset, category["name"])] = category["definition"]
    return index


def _reference_payload(
    gold: dict[str, Any], definitions: dict[tuple[str, str], str] | None = None
) -> dict[str, Any]:
    failure_types = gold.get("failure_types") or [gold["failure_type"]]
    payload = {
        "failure_type": gold["failure_type"],
        "accepted_failure_types": failure_types,
        "reference_reasoning": gold.get("reference_reasoning") or None,
    }
    if definitions is not None:
        payload["accepted_failure_definitions"] = [
            {
                "name": failure_type,
                "definition": definitions.get((gold["dataset"], failure_type)),
            }
            for failure_type in failure_types
        ]
    if gold.get("annotation_kind"):
        payload["annotation_kind"] = gold["annotation_kind"]
    return payload


def _prediction_payload(row: dict[str, Any]) -> dict[str, Any]:
    prediction = row["prediction"]
    primary = prediction["predicted_errors"][0]
    return {
        "error_type": primary["error_type"],
        "explanation": primary["explanation"],
        "evidence": primary["evidence"],
        "overall_root_cause": prediction["overall_root_cause"],
    }


def _build_prompt(
    template: str,
    trajectory: dict[str, Any],
    gold: dict[str, Any],
    prediction: dict[str, Any],
    definitions: dict[tuple[str, str], str] | None = None,
) -> str:
    return format_template(
        template,
        trajectory=trajectory_for_prompt(trajectory),
        reference=json.dumps(
            _reference_payload(gold, definitions), ensure_ascii=False, indent=2
        ),
        prediction=json.dumps(_prediction_payload(prediction), ensure_ascii=False, indent=2),
    )


def estimate_simple_evaluation_budget(
    prediction_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    path = Path(prediction_path)
    if not path.is_absolute():
        path = config["_root"] / path
    predictions = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row
        for row in iter_jsonl(path)
    }
    trajectories = _index_processed(config, "inputs")
    gold = _index_processed(config, "gold")
    definitions = _definition_index(config)
    system = read_prompt(config["_root"], "simple_evaluator_system.txt")
    template = read_prompt(config["_root"], "simple_evaluator_user.txt")
    model = config["models"]["simple_evaluator"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    input_tokens = sum(
        counter.count(
            system
            + "\n"
            + _build_prompt(
                template, trajectories[key[:2]], gold[key[:2]], row, definitions
            )
        )
        for key, row in predictions.items()
        if row.get("status") == "ok"
    )
    calls = len(predictions)
    safety = float(config["cost_budget"]["input_token_safety_multiplier"])
    guarded_input = math.ceil(input_tokens * safety)
    max_output = calls * int(model["max_output_tokens"])
    prices = model["pricing_usd_per_million"]
    projected_new = (
        guarded_input * float(prices["input"])
        + max_output * float(prices["output"])
    ) / 1_000_000
    ledger_path = Path(config["cost_budget"]["ledger_path"])
    if not ledger_path.is_absolute():
        ledger_path = config["_root"] / ledger_path
    lifetime_incurred = (
        sum(float(row.get("cost_usd", 0.0)) for row in iter_jsonl(ledger_path))
        if ledger_path.exists()
        else 0.0
    )
    baseline = float(config["cost_budget"].get("ledger_baseline_usd", 0.0))
    incurred = max(0.0, lifetime_incurred - baseline)
    cumulative = incurred + projected_new
    fee = max(
        cumulative * float(config["cost_budget"]["credit_purchase_fee_rate"]),
        float(config["cost_budget"]["minimum_credit_purchase_fee_usd"]),
    )
    report = {
        "model": model["name"],
        "calls": calls,
        "input_tokens": input_tokens,
        "guarded_input_tokens": guarded_input,
        "maximum_output_tokens": max_output,
        "incurred_inference_usd": incurred,
        "lifetime_ledger_inference_usd": lifetime_incurred,
        "ledger_baseline_usd": baseline,
        "budget_epoch_name": config["cost_budget"].get("budget_epoch_name"),
        "projected_new_inference_usd": projected_new,
        "projected_cumulative_inference_usd": cumulative,
        "projected_fee_inclusive_total_usd": cumulative + fee,
        "within_budget": cumulative <= float(config["cost_budget"]["max_inference_usd"])
        and cumulative + fee <= float(config["cost_budget"]["max_total_charge_usd"]),
    }
    write_json(resolve_path(config, "processed_data") / "simple_evaluation_budget.json", report)
    return report


def evaluate_predictions_simple(
    prediction_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
) -> Path:
    config = load_config(config_path)
    prediction_path = Path(prediction_path)
    if not prediction_path.is_absolute():
        prediction_path = config["_root"] / prediction_path
    prediction_index = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row
        for row in iter_jsonl(prediction_path)
    }
    if not prediction_index:
        raise ValueError("Prediction file is empty")
    run_id = next(iter(prediction_index.values()))["run_id"]
    budget = estimate_simple_evaluation_budget(prediction_path, config_path)
    if not budget["within_budget"]:
        raise ValueError(
            "Simple evaluation conservative projection exceeds the configured cumulative budget: "
            + json.dumps(budget, ensure_ascii=False)
        )

    trajectories = _index_processed(config, "inputs")
    gold = _index_processed(config, "gold")
    definitions = _definition_index(config)
    system = read_prompt(config["_root"], "simple_evaluator_system.txt")
    template = read_prompt(config["_root"], "simple_evaluator_user.txt")
    model = config["models"]["simple_evaluator"]
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "prompt_version": config["evaluation"]["simple_prompt_version"],
                "system": system,
                "template": template,
                "response_schema": BinaryEvaluation.model_json_schema(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    evaluator_run_id = f"{run_id}--simple-eval-{fingerprint}"
    output_path = resolve_path(config, "results") / "evaluations_simple" / f"{evaluator_run_id}.jsonl"
    completed = set()
    if output_path.exists():
        completed = {
            (row["dataset"], row["trajectory_id"], row["condition"])
            for row in iter_jsonl(output_path)
            if row.get("status") in {"ok", "prediction_error"}
        }
    evaluator = CachedLLM(config, resolve_path(config, "cache") / "simple_evaluator")
    for key in sorted(prediction_index):
        if key in completed:
            continue
        prediction = prediction_index[key]
        base = {
            "run_id": run_id,
            "evaluator_run_id": evaluator_run_id,
            "evaluator_prompt_version": config["evaluation"]["simple_prompt_version"],
            "dataset": key[0],
            "domain": prediction["domain"],
            "trajectory_id": key[1],
            "condition": key[2],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if prediction.get("status") != "ok":
            append_jsonl(
                output_path,
                {
                    **base,
                    "status": "prediction_error",
                    "evaluation": {
                        "correct": False,
                        "reason": "Prediction did not contain a valid attribution.",
                    },
                },
            )
            continue
        user = _build_prompt(
            template, trajectories[key[:2]], gold[key[:2]], prediction, definitions
        )
        try:
            call_kwargs = {
                "model": model["name"],
                "system_prompt": system,
                "user_prompt": user,
                "temperature": model.get("temperature"),
                "reasoning_effort": model.get("reasoning_effort"),
                "seed": int(config["experiment"]["seed"]),
                "provider": model.get("provider"),
                "response_model": BinaryEvaluation,
            }
            try:
                result, call = evaluator.call(
                    **call_kwargs,
                    max_output_tokens=int(model["max_output_tokens"]),
                )
            except TruncatedResponseError:
                result, call = evaluator.call(
                    **call_kwargs,
                    max_output_tokens=max(500, int(model["max_output_tokens"])),
                )
            row = {
                **base,
                "status": "ok",
                "evaluation": result.model_dump(mode="json"),
                "evaluator_model": model["name"],
                "cache_key": call["cache_key"],
                "cache_path": str(Path(call["cache_path"]).relative_to(config["_root"])),
                "cache_hit": call["cache_hit"],
                "response_metadata": call["response_metadata"],
            }
        except Exception as exc:
            append_jsonl(
                output_path,
                {**base, "status": "evaluator_error", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        append_jsonl(output_path, row)
    return output_path
