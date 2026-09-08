from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
from taxonomy_experiment.models import NativeRelabelResult
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.token_count import TokenCounter


NATIVE_TERM_MASK_VERSION = "native-category-terms-v1"
NATIVE_TERM_MASK_TOKEN = "[FAILURE_CATEGORY_TERM]"


def _prediction_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (row["dataset"], row["trajectory_id"], row["condition"])
        if key in index:
            raise ValueError(f"Duplicate prediction key: {key}")
        index[key] = row
    return index


def _gold_index(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    processed = resolve_path(config, "processed_data")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_gold.jsonl"):
            key = (dataset, row["trajectory_id"])
            if key in index:
                raise ValueError(f"Duplicate gold key: {key}")
            index[key] = row
    return index


def _native_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        dataset: read_json(resolve_path(config, path_key))
        for dataset, path_key in zip(
            experiment_datasets(config), ("taxonomy_a", "taxonomy_b")
        )
    }


def _taxonomy_prompt(taxonomy: dict[str, Any]) -> str:
    return json.dumps(
        [
            {"name": item["name"], "definition": item["definition"]}
            for item in taxonomy["categories"]
        ],
        ensure_ascii=False,
        indent=2,
    )


def candidate_narrative(row: dict[str, Any]) -> dict[str, Any]:
    """Return only semantic claims; deliberately exclude label, step, and condition."""
    primary = row["prediction"]["predicted_errors"][0]
    return {
        "explanation": primary["explanation"],
        "evidence": primary["evidence"],
        "overall_root_cause": row["prediction"]["overall_root_cause"],
    }


def _category_term_pattern(category_name: str) -> re.Pattern[str]:
    """Match a category name while tolerating punctuation/spacing variants."""
    tokens = re.findall(r"[^\W_]+", category_name.casefold(), flags=re.UNICODE)
    if not tokens:
        raise ValueError(f"Native category has no matchable tokens: {category_name!r}")
    body = r"[\W_]+".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<!\w){body}(?!\w)", flags=re.IGNORECASE | re.UNICODE)


def mask_native_category_terms(
    narrative: dict[str, Any], taxonomy: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Mask explicit native category names without altering semantic paraphrases."""
    patterns = [
        (item["name"], _category_term_pattern(item["name"]))
        for item in sorted(
            taxonomy["categories"], key=lambda item: len(item["name"]), reverse=True
        )
    ]
    matched: set[str] = set()

    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [transform(item) for item in value]
        if not isinstance(value, str):
            return value
        masked = value
        for name, pattern in patterns:
            masked, count = pattern.subn(NATIVE_TERM_MASK_TOKEN, masked)
            if count:
                matched.add(name)
        return masked

    return transform(narrative), sorted(matched)


def _build_prompt(
    template: str,
    taxonomy: dict[str, Any],
    prediction: dict[str, Any],
    *,
    mask_native_terms: bool = False,
) -> str:
    narrative = candidate_narrative(prediction)
    if mask_native_terms:
        narrative, _ = mask_native_category_terms(narrative, taxonomy)
    return format_template(
        template,
        native_taxonomy=_taxonomy_prompt(taxonomy),
        candidate_narrative=json.dumps(
            narrative, ensure_ascii=False, indent=2
        ),
    )


def _normalize_native_label(value: str, taxonomy: dict[str, Any]) -> tuple[str, bool]:
    names = [item["name"] for item in taxonomy["categories"]]
    raw = value.strip()
    if raw in names:
        return raw, False
    folded = [name for name in names if name.casefold() == raw.casefold()]
    if len(folded) == 1:
        return folded[0], True
    raise ValueError(f"Relabeler returned a label outside the native taxonomy: {value!r}")


def _accepted_gold_labels(gold: dict[str, Any]) -> list[str]:
    return list(gold.get("failure_types") or [gold["failure_type"]])


def estimate_native_evaluation_budget(
    prediction_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
    *,
    mask_native_terms: bool = False,
    evaluator_role: str = "simple_evaluator",
) -> dict[str, Any]:
    config = load_config(config_path)
    path = Path(prediction_path)
    if not path.is_absolute():
        path = config["_root"] / path
    predictions = _prediction_index(path)
    taxonomies = _native_taxonomies(config)
    system = read_prompt(config["_root"], "native_relabel_system.txt")
    template = read_prompt(config["_root"], "native_relabel_user.txt")
    if evaluator_role not in config["models"]:
        raise ValueError(f"Unknown evaluator model role: {evaluator_role}")
    model = config["models"][evaluator_role]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    input_tokens = sum(
        counter.count(
            system
            + "\n"
            + _build_prompt(
                template,
                taxonomies[key[0]],
                row,
                mask_native_terms=mask_native_terms,
            )
        )
        for key, row in predictions.items()
        if row.get("status") == "ok"
    )
    calls = sum(row.get("status") == "ok" for row in predictions.values())
    safety = float(config["cost_budget"]["input_token_safety_multiplier"])
    guarded_input = math.ceil(input_tokens * safety)
    maximum_output = calls * int(model["max_output_tokens"])
    prices = model["pricing_usd_per_million"]
    projected_new = (
        guarded_input * float(prices["input"])
        + maximum_output * float(prices["output"])
    ) / 1_000_000
    ledger_path = Path(config["cost_budget"]["ledger_path"])
    if not ledger_path.is_absolute():
        ledger_path = config["_root"] / ledger_path
    incurred = (
        sum(float(row.get("cost_usd", 0.0)) for row in iter_jsonl(ledger_path))
        if ledger_path.exists()
        else 0.0
    )
    cumulative = incurred + projected_new
    fee = max(
        cumulative * float(config["cost_budget"]["credit_purchase_fee_rate"]),
        float(config["cost_budget"]["minimum_credit_purchase_fee_usd"]),
    )
    report = {
        "evaluation_method": (
            "condition_source_label_and_native_term_masked_relabeling"
            if mask_native_terms
            else "condition_and_source_label_blind_native_relabeling"
        ),
        "candidate_narrative_transform": (
            NATIVE_TERM_MASK_VERSION if mask_native_terms else "none"
        ),
        "model": model["name"],
        "evaluator_role": evaluator_role,
        "calls": calls,
        "input_tokens": input_tokens,
        "guarded_input_tokens": guarded_input,
        "maximum_output_tokens": maximum_output,
        "incurred_inference_usd": incurred,
        "projected_new_inference_usd": projected_new,
        "projected_cumulative_inference_usd": cumulative,
        "projected_fee_inclusive_total_usd": cumulative + fee,
        "within_budget": cumulative <= float(config["cost_budget"]["max_inference_usd"])
        and cumulative + fee <= float(config["cost_budget"]["max_total_charge_usd"]),
        "blind_fields": [
            "trajectory",
            "gold_annotation",
            "experimental_condition",
            "source_taxonomy_id",
            "source_predicted_error_type",
            "predicted_failure_step",
        ],
    }
    budget_variant = "masked" if mask_native_terms else "primary"
    role_variant = (
        "native_evaluation"
        if evaluator_role == "simple_evaluator"
        else f"native_evaluation_{evaluator_role}"
    )
    budget_name = f"{role_variant}_{budget_variant}_budget.json"
    # Preserve the established primary budget path for existing tooling.
    if evaluator_role == "simple_evaluator" and not mask_native_terms:
        budget_name = "native_evaluation_budget.json"
    elif evaluator_role == "simple_evaluator" and mask_native_terms:
        budget_name = "native_evaluation_masked_budget.json"
    write_json(resolve_path(config, "processed_data") / budget_name, report)
    return report


def _compact(path: Path) -> None:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        latest[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    write_jsonl(path, latest.values())


def evaluate_predictions_native_blind(
    prediction_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
    *,
    mask_native_terms: bool = False,
    evaluator_role: str = "simple_evaluator",
) -> Path:
    config = load_config(config_path)
    path = Path(prediction_path)
    if not path.is_absolute():
        path = config["_root"] / path
    predictions = _prediction_index(path)
    if not predictions:
        raise ValueError("Prediction file is empty")
    run_id = next(iter(predictions.values()))["run_id"]
    budget = estimate_native_evaluation_budget(
        path,
        config_path,
        mask_native_terms=mask_native_terms,
        evaluator_role=evaluator_role,
    )
    if not budget["within_budget"]:
        raise ValueError("Native blind evaluation exceeds the configured budget")

    gold = _gold_index(config)
    taxonomies = _native_taxonomies(config)
    system = read_prompt(config["_root"], "native_relabel_system.txt")
    template = read_prompt(config["_root"], "native_relabel_user.txt")
    model = config["models"][evaluator_role]
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": model,
                "evaluator_role": evaluator_role,
                "prompt_version": config["evaluation"]["native_blind_prompt_version"],
                "system": system,
                "template": template,
                "candidate_narrative_transform": (
                    NATIVE_TERM_MASK_VERSION if mask_native_terms else "none"
                ),
                "response_schema": NativeRelabelResult.model_json_schema(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    if evaluator_role == "simple_evaluator":
        evaluator_variant = (
            "native-blind-masked" if mask_native_terms else "native-blind"
        )
        output_directory = (
            "evaluations_native_blind_masked"
            if mask_native_terms
            else "evaluations_native_blind"
        )
        cache_directory = "native_blind"
    else:
        evaluator_variant = (
            f"native-blind-{evaluator_role}"
            + ("-masked" if mask_native_terms else "")
        )
        output_directory = f"evaluations_native_blind_{evaluator_role}"
        cache_directory = f"native_blind_{evaluator_role}"
    evaluator_run_id = f"{run_id}--{evaluator_variant}-{fingerprint}"
    output = (
        resolve_path(config, "results")
        / output_directory
        / f"{evaluator_run_id}.jsonl"
    )
    completed = set()
    if output.exists():
        completed = {
            (row["dataset"], row["trajectory_id"], row["condition"])
            for row in iter_jsonl(output)
            if row.get("status") in {"ok", "prediction_error"}
        }
    progress_label = (
        f"native-{evaluator_role}"
        + ("-masked" if mask_native_terms else "")
    )
    completed_count = len(completed)
    total_count = len(predictions)
    print(
        f"[{progress_label}] starting/resuming at {completed_count}/{total_count}",
        flush=True,
    )
    evaluator = CachedLLM(
        config, resolve_path(config, "cache") / cache_directory
    )
    for key in sorted(predictions):
        if key in completed:
            continue
        prediction = predictions[key]
        base = {
            "run_id": run_id,
            "evaluator_run_id": evaluator_run_id,
            "evaluator_prompt_version": config["evaluation"]["native_blind_prompt_version"],
            "candidate_narrative_transform": (
                NATIVE_TERM_MASK_VERSION if mask_native_terms else "none"
            ),
            "evaluator_role": evaluator_role,
            "dataset": key[0],
            "domain": prediction["domain"],
            "trajectory_id": key[1],
            "condition": key[2],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if prediction.get("status") != "ok":
            append_jsonl(output, {**base, "status": "prediction_error", "correct": False})
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_count:
                print(
                    f"[{progress_label}] {completed_count}/{total_count}",
                    flush=True,
                )
            continue
        taxonomy = taxonomies[key[0]]
        narrative = candidate_narrative(prediction)
        matched_native_terms: list[str] = []
        if mask_native_terms:
            narrative, matched_native_terms = mask_native_category_terms(
                narrative, taxonomy
            )
        user = format_template(
            template,
            native_taxonomy=_taxonomy_prompt(taxonomy),
            candidate_narrative=json.dumps(narrative, ensure_ascii=False, indent=2),
        )
        try:
            kwargs = {
                "model": model["name"],
                "system_prompt": system,
                "user_prompt": user,
                "temperature": model.get("temperature"),
                "reasoning_effort": model.get("reasoning_effort"),
                "seed": int(config["experiment"]["seed"]),
                "provider": model.get("provider"),
                "response_model": NativeRelabelResult,
            }
            try:
                result, call = evaluator.call(
                    **kwargs, max_output_tokens=int(model["max_output_tokens"])
                )
            except TruncatedResponseError:
                result, call = evaluator.call(**kwargs, max_output_tokens=500)
            native_label, normalized = _normalize_native_label(
                result.native_label, taxonomy
            )
            accepted = _accepted_gold_labels(gold[key[:2]])
            row = {
                **base,
                "status": "ok",
                "native_taxonomy_id": taxonomy["taxonomy_id"],
                "native_label": native_label,
                "raw_native_label": result.native_label if normalized else None,
                "mapping_reason": result.reason,
                "accepted_gold_labels": accepted,
                "correct": native_label in accepted,
                "blind_fields": budget["blind_fields"],
                "masked_native_terms": matched_native_terms,
                "evaluator_model": model["name"],
                "cache_key": call["cache_key"],
                "cache_path": str(Path(call["cache_path"]).relative_to(config["_root"])),
                "cache_hit": call["cache_hit"],
                "response_metadata": call["response_metadata"],
            }
        except Exception as exc:
            append_jsonl(
                output,
                {**base, "status": "evaluator_error", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        append_jsonl(output, row)
        completed_count += 1
        if completed_count % 10 == 0 or completed_count == total_count:
            print(
                f"[{progress_label}] {completed_count}/{total_count} "
                f"dataset={key[0]} condition={key[2]}",
                flush=True,
            )
    _compact(output)
    return output
