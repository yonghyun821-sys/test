from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json
from taxonomy_experiment.llm import CachedLLM
from taxonomy_experiment.models import ComparativeEvaluation, trajectory_for_prompt
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.taxonomy import CONDITION_ORDER


def _definition_index(config: dict[str, Any]) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for dataset, path_key in zip(experiment_datasets(config), ("taxonomy_a", "taxonomy_b")):
        taxonomy = read_json(resolve_path(config, path_key))
        for category in taxonomy["categories"]:
            index[(dataset, category["name"])] = category["definition"]
    return index


def _reference_payload(gold: dict[str, Any], failure_definition: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "failure_definition": failure_definition,
        "reference_reasoning": gold.get("reference_reasoning") or None,
        "reference_step_index": gold.get("critical_failure_step"),
    }
    if gold.get("failure_summary"):
        payload["failure_summary"] = gold["failure_summary"]
    root = gold.get("root_failure") or {}
    if root:
        payload["root_failure_step_reason"] = root.get("step_reason")
        payload["root_failure_category_reason"] = root.get("category_reason")
    return payload


def _candidate_payload(candidate_id: str, prediction_row: dict[str, Any]) -> dict[str, Any]:
    prediction = prediction_row["prediction"]
    primary = prediction["predicted_errors"][0]
    return {
        "candidate_id": candidate_id,
        "overall_root_cause": prediction["overall_root_cause"],
        "explanation": primary["explanation"],
        "evidence": primary["evidence"],
        "failure_step": primary["failure_step"],
    }


def _derived_evaluation(assessment: dict[str, Any]) -> dict[str, Any]:
    match = assessment["mechanism_match"]
    support = assessment["trajectory_support"]
    # Resolve cross-field disagreement conservatively instead of asking the LLM
    # to produce the same paid response repeatedly. These rules are deterministic.
    if support == "contradicted":
        score = 1
    elif support == "none":
        score = 2
    elif assessment["conflicting_primary_cause"] or assessment["symptom_only"]:
        score = 3 if match in {"full", "substantial", "partial"} else 2
    elif match == "full":
        score = 5 if support == "strong" else 4
    elif match == "substantial":
        score = 4
    elif match == "partial":
        score = 3
    else:
        score = 1 if support == "contradicted" else 2
    return {"correct": score >= 4, "score": score, "reason": assessment["reason"]}


def _index_processed(config: dict[str, Any], suffix: str) -> dict[tuple[str, str], dict[str, Any]]:
    processed = resolve_path(config, "processed_data")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_{suffix}.jsonl"):
            index[(dataset, row["trajectory_id"])] = row
    return index


def evaluate_predictions(
    prediction_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
) -> Path:
    config = load_config(config_path)
    prediction_path = Path(prediction_path)
    if not prediction_path.is_absolute():
        prediction_path = config["_root"] / prediction_path

    prediction_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(prediction_path):
        prediction_index[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    predictions = list(prediction_index.values())
    if not predictions:
        raise ValueError(f"No predictions in {prediction_path}")
    run_id = predictions[0]["run_id"]
    if any(row["run_id"] != run_id for row in predictions):
        raise ValueError("Prediction file mixes multiple run IDs")

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        grouped[(row["dataset"], row["trajectory_id"])][row["condition"]] = row
    expected_conditions = {condition.value for condition in CONDITION_ORDER}
    incomplete = {
        key: sorted(expected_conditions - set(rows))
        for key, rows in grouped.items()
        if set(rows) != expected_conditions
    }
    if incomplete:
        raise ValueError(f"Every trajectory must contain all four conditions: {incomplete}")

    from taxonomy_experiment.budget import estimate_api_budget

    estimate_api_budget(config_path, mode="smoke" if run_id.startswith("smoke-") else "full")

    # Gold is loaded only after condition predictions already exist on disk.
    trajectories = _index_processed(config, "inputs")
    gold = _index_processed(config, "gold")
    definitions = _definition_index(config)
    system_prompt = read_prompt(config["_root"], "evaluator_system.txt")
    user_template = read_prompt(config["_root"], "evaluator_user.txt")
    model_config = config["models"]["evaluator"]
    evaluator_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": model_config,
                "prompt_version": config["evaluation"].get("prompt_version"),
                "system_prompt": system_prompt,
                "user_template": user_template,
                "comparison_design": "four-way-condition-blind-v1",
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    evaluator_run_id = f"{run_id}--eval-{evaluator_fingerprint}"
    output_path = resolve_path(config, "results") / "evaluations" / f"{evaluator_run_id}.jsonl"

    completed_groups: set[tuple[str, str]] = set()
    if output_path.exists():
        completion: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in iter_jsonl(output_path):
            if row.get("status") in {"ok", "prediction_error"}:
                completion[(row["dataset"], row["trajectory_id"])].add(row["condition"])
        completed_groups = {
            key for key, conditions in completion.items() if conditions == expected_conditions
        }

    evaluator = CachedLLM(config, resolve_path(config, "cache") / "evaluator")
    seed = int(config["experiment"]["seed"])
    for key in sorted(grouped):
        if key in completed_groups:
            continue
        condition_rows = grouped[key]
        base = {
            "run_id": run_id,
            "evaluator_run_id": evaluator_run_id,
            "evaluator_prompt_version": config["evaluation"].get("prompt_version"),
            "dataset": key[0],
            "domain": next(iter(condition_rows.values()))["domain"],
            "trajectory_id": key[1],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        failed_conditions = [
            condition for condition, row in condition_rows.items() if row.get("status") != "ok"
        ]
        if failed_conditions:
            for condition in sorted(condition_rows):
                append_jsonl(
                    output_path,
                    {
                        **base,
                        "condition": condition,
                        "status": "prediction_error",
                        "evaluation": {
                            "correct": False,
                            "score": 1,
                            "reason": "At least one condition lacked a valid structured attribution.",
                        },
                        "failed_conditions": sorted(failed_conditions),
                    },
                )
            continue

        shuffled_conditions = sorted(condition_rows)
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(shuffled_conditions)
        candidate_ids = ("A", "B", "C", "D")
        candidate_to_condition = dict(zip(candidate_ids, shuffled_conditions, strict=True))
        candidates = [
            _candidate_payload(candidate_id, condition_rows[condition])
            for candidate_id, condition in candidate_to_condition.items()
        ]
        reference = gold[key]
        failure_definition = definitions.get((key[0], reference["failure_type"]))
        user_prompt = format_template(
            user_template,
            trajectory=trajectory_for_prompt(trajectories[key]),
            reference=json.dumps(
                _reference_payload(reference, failure_definition), ensure_ascii=False, indent=2
            ),
            candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        )
        blind_prompt_sha = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        try:
            result, call = evaluator.call(
                model=model_config["name"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=model_config.get("temperature"),
                reasoning_effort=model_config.get("reasoning_effort"),
                seed=seed,
                provider=model_config.get("provider"),
                max_output_tokens=int(model_config["max_output_tokens"]),
                response_model=ComparativeEvaluation,
            )
            parsed = result.model_dump(mode="json")
            assessments = {item["candidate_id"]: item for item in parsed["candidates"]}
            common = {
                **base,
                "status": "ok",
                "reference_consistency": parsed["reference_consistency"],
                "reference_consistency_reason": parsed["reference_consistency_reason"],
                "evaluator_model": model_config["name"],
                "blind_prompt_sha256": blind_prompt_sha,
                "cache_key": call["cache_key"],
                "cache_path": str(Path(call["cache_path"]).relative_to(config["_root"])),
                "cache_hit": call["cache_hit"],
                "response_metadata": call["response_metadata"],
            }
            for candidate_id, condition in candidate_to_condition.items():
                assessment = assessments[candidate_id]
                append_jsonl(
                    output_path,
                    {
                        **common,
                        "condition": condition,
                        "anonymous_candidate_id": candidate_id,
                        "structured_assessment": assessment,
                        "evaluation": _derived_evaluation(assessment),
                    },
                )
        except Exception as exc:
            append_jsonl(
                output_path,
                {
                    **base,
                    "status": "evaluator_error",
                    "blind_prompt_sha256": blind_prompt_sha,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
    return output_path
