from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
from taxonomy_experiment.models import (
    AttributionResult,
    CompactAttributionResult,
    trajectory_for_prompt,
)
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.taxonomy import (
    CONDITION_ORDER,
    Condition,
    load_taxonomy_set,
    taxonomy_block,
    taxonomy_digest,
    taxonomy_for_condition,
)
from taxonomy_experiment.token_count import TokenCounter


PREDICTION_POSTPROCESSING_VERSION = "v3-exact-taxonomy-name"
TRUNCATION_RECOVERY_VERSION = "v1-explicit-bounded-schema"


def _truncation_recovery_system_prompt(system_prompt: str) -> str:
    return (
        system_prompt
        + "\n\nTRUNCATION RECOVERY: A previous response copied long tool output and was cut off. "
        "Return only a compact attribution. Paraphrase evidence; never reproduce a tool log. "
        "Keep explanation under 100 words, each evidence item under 30 words, and the root-cause "
        "sentence under 60 words."
    )


def _normalize_taxonomy_error_type(
    value: str, taxonomy: dict[str, Any] | None
) -> tuple[str, bool]:
    """Canonicalize only unambiguous formatting variants of supplied category names."""
    if taxonomy is None:
        return value, False
    raw = value.strip()
    folded = raw.casefold()
    matches: list[str] = []
    for category in taxonomy["categories"]:
        name = category["name"]
        module = category.get("module")
        variants = {name.casefold()}
        if module:
            variants.update(
                {
                    f"{module}.{name}".casefold(),
                    f"{module} {name}".casefold(),
                    f"{module}: {name}".casefold(),
                    f"{module}_{name}".casefold(),
                }
            )
        if folded in variants:
            matches.append(name)
    if len(matches) == 1 and matches[0] != raw:
        return matches[0], True
    return raw, False


def _select_smoke(records: list[dict[str, Any]], count: int, seed: int, dataset: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["domain"]].append(record)
    for domain, rows in groups.items():
        random.Random(f"{seed}:{dataset}:{domain}").shuffle(rows)
    selected: list[dict[str, Any]] = []
    domains = sorted(groups)
    while len(selected) < min(count, len(records)):
        progressed = False
        for domain in domains:
            if groups[domain] and len(selected) < count:
                selected.append(groups[domain].pop())
                progressed = True
        if not progressed:
            break
    return sorted(selected, key=lambda item: (item["domain"], item["trajectory_id"]))


def _load_records(config: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    processed = resolve_path(config, "processed_data")
    input_path = processed / f"{dataset}_inputs.jsonl"
    ids_path = processed / f"{dataset}_attribution_ids.json"
    if not input_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Run scripts/inspect_datasets.py before inference")
    eligible_ids = set(read_json(ids_path))
    return [record for record in iter_jsonl(input_path) if record["trajectory_id"] in eligible_ids]


def _compact_prediction_rows(path: Path) -> None:
    """Keep the latest row per experiment key after a successfully resumed run."""
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        latest[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    write_jsonl(path, latest.values())


def _run_id(
    config: dict[str, Any],
    mode: str,
    selected: dict[str, list[dict[str, Any]]],
    taxonomies: dict[str, dict[str, Any]],
    system_prompt: str,
    user_template: str,
) -> str:
    model = config["models"]["attribution"]
    payload = {
        "experiment": config["experiment"],
        "mode": mode,
        "model": model,
        "api": config["api"],
        "selected_ids": {
            dataset: [row["trajectory_id"] for row in rows] for dataset, rows in selected.items()
        },
        "selected_input_digests": {
            dataset: hashlib.sha256(
                json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for dataset, rows in selected.items()
        },
        "taxonomy_digests": {key: taxonomy_digest(value) for key, value in taxonomies.items()},
        "system_prompt": system_prompt,
        "user_template": user_template,
        "prediction_postprocessing_version": PREDICTION_POSTPROCESSING_VERSION,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{mode}-{digest}"


def _preflight(
    config: dict[str, Any],
    selected: dict[str, list[dict[str, Any]]],
    conditions: Iterable[Condition],
    taxonomies: dict[str, dict[str, Any]],
    system_prompt: str,
    user_template: str,
) -> dict[str, Any]:
    model = config["models"]["attribution"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    context_limit = int(model["context_window"]) - int(model["max_output_tokens"])
    rows: list[dict[str, Any]] = []
    for dataset, records in selected.items():
        for record in records:
            trajectory = trajectory_for_prompt(record)
            for condition in conditions:
                taxonomy = taxonomy_for_condition(dataset, condition, taxonomies)
                user_prompt = format_template(
                    user_template,
                    taxonomy_block=taxonomy_block(taxonomy),
                    trajectory=trajectory,
                )
                tokens = counter.count(system_prompt + "\n" + user_prompt)
                rows.append(
                    {
                        "dataset": dataset,
                        "trajectory_id": record["trajectory_id"],
                        "condition": condition.value,
                        "input_tokens": tokens,
                        "within_context": tokens <= context_limit,
                    }
                )
    failures = [row for row in rows if not row["within_context"]]
    return {
        "model": model["name"],
        "tokenizer": counter.encoding_name,
        "exact_model_tokenizer": counter.exact_tokenizer_available,
        "configured_context_window": model["context_window"],
        "reserved_output_tokens": model["max_output_tokens"],
        "max_allowed_input_tokens": context_limit,
        "request_count": len(rows),
        "minimum_input_tokens": min(row["input_tokens"] for row in rows),
        "maximum_input_tokens": max(row["input_tokens"] for row in rows),
        "context_failures": failures,
        "requests": rows,
    }


def run_attribution(
    config_path: str | Path = "config/experiment.yaml",
    *,
    mode: str = "smoke",
    datasets: Iterable[str] | None = None,
    conditions: Iterable[Condition] = CONDITION_ORDER,
) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    config = load_config(config_path)
    from taxonomy_experiment.budget import estimate_api_budget

    estimate_api_budget(config_path, mode=mode)
    dataset_names = tuple(datasets) if datasets is not None else experiment_datasets(config)
    conditions = tuple(conditions)
    taxonomy_a = resolve_path(config, "taxonomy_a")
    taxonomy_b = resolve_path(config, "taxonomy_b")
    merged = resolve_path(config, "merged_taxonomy")
    taxonomies = load_taxonomy_set(taxonomy_a, taxonomy_b, merged)
    if Condition.MERGED_TAXONOMY in conditions and not merged.exists():
        raise FileNotFoundError("Create and freeze the merged taxonomy before running merged conditions")

    selected: dict[str, list[dict[str, Any]]] = {}
    for dataset in dataset_names:
        records = _load_records(config, dataset)
        if mode == "smoke":
            records = _select_smoke(
                records,
                int(config["sampling"]["smoke_examples_per_dataset"]),
                int(config["experiment"]["seed"]),
                dataset,
            )
        selected[dataset] = records

    system_prompt = read_prompt(config["_root"], "attribution_system.txt")
    user_template = read_prompt(config["_root"], "attribution_user.txt")
    run_id = _run_id(config, mode, selected, taxonomies, system_prompt, user_template)
    result_root = resolve_path(config, "results")
    prediction_path = result_root / "raw_predictions" / f"{run_id}.jsonl"
    manifest_path = result_root / "manifests" / f"{run_id}.json"
    preflight = _preflight(
        config, selected, conditions, taxonomies, system_prompt, user_template
    )
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "datasets": {
            dataset: [record["trajectory_id"] for record in records]
            for dataset, records in selected.items()
        },
        "conditions": [condition.value for condition in conditions],
        "preflight": preflight,
        "prediction_path": str(prediction_path.relative_to(config["_root"])),
    }
    write_json(manifest_path, manifest)
    if preflight["context_failures"]:
        raise ValueError(
            f"{len(preflight['context_failures'])} requests exceed the configured context window; see {manifest_path}"
        )

    completed: set[tuple[str, str, str]] = set()
    previously_truncated: set[tuple[str, str, str]] = set()
    if prediction_path.exists():
        previous_rows = list(iter_jsonl(prediction_path))
        completed = {
            (row["dataset"], row["trajectory_id"], row["condition"])
            for row in previous_rows
            if row.get("status") == "ok"
        }
        previously_truncated = {
            (row["dataset"], row["trajectory_id"], row["condition"])
            for row in previous_rows
            if row.get("status") == "error"
            and "TruncatedResponseError" in str(row.get("error", ""))
        }
    llm = CachedLLM(config, resolve_path(config, "cache") / "attribution")
    model_config = config["models"]["attribution"]
    for dataset, records in selected.items():
        for record in records:
            trajectory = trajectory_for_prompt(record)
            for condition in conditions:
                result_key = (dataset, record["trajectory_id"], condition.value)
                if result_key in completed:
                    continue
                taxonomy = taxonomy_for_condition(dataset, condition, taxonomies)
                user_prompt = format_template(
                    user_template,
                    taxonomy_block=taxonomy_block(taxonomy),
                    trajectory=trajectory,
                )
                try:
                    request_max_output_tokens = int(model_config["max_output_tokens"])
                    if result_key in previously_truncated:
                        request_max_output_tokens = int(
                            config["execution"][
                                "attribution_truncation_retry_max_output_tokens"
                            ]
                        )
                    recovery_mode = False
                    try:
                        prediction, call = llm.call(
                            model=model_config["name"],
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=model_config.get("temperature"),
                            reasoning_effort=model_config.get("reasoning_effort"),
                            seed=int(config["experiment"]["seed"]),
                            provider=model_config.get("provider"),
                            max_output_tokens=request_max_output_tokens,
                            response_model=AttributionResult,
                        )
                    except TruncatedResponseError:
                        retry_limit = int(
                            config["execution"][
                                "attribution_truncation_retry_max_output_tokens"
                            ]
                        )
                        try:
                            if retry_limit <= request_max_output_tokens:
                                raise TruncatedResponseError(
                                    "Previously truncated request requires compact recovery"
                                )
                            request_max_output_tokens = retry_limit
                            prediction, call = llm.call(
                                model=model_config["name"],
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                temperature=model_config.get("temperature"),
                                reasoning_effort=model_config.get("reasoning_effort"),
                                seed=int(config["experiment"]["seed"]),
                                provider=model_config.get("provider"),
                                max_output_tokens=request_max_output_tokens,
                                response_model=AttributionResult,
                            )
                        except TruncatedResponseError:
                            recovery_mode = True
                            request_max_output_tokens = int(model_config["max_output_tokens"])
                            prediction, call = llm.call(
                                model=model_config["name"],
                                system_prompt=_truncation_recovery_system_prompt(system_prompt),
                                user_prompt=user_prompt,
                                temperature=model_config.get("temperature"),
                                reasoning_effort=model_config.get("reasoning_effort"),
                                seed=int(config["experiment"]["seed"]),
                                provider=model_config.get("provider"),
                                max_output_tokens=request_max_output_tokens,
                                response_model=CompactAttributionResult,
                            )
                    prediction_payload = prediction.model_dump(mode="json")
                    original_error_type = prediction_payload["predicted_errors"][0]["error_type"]
                    normalized_error_type, normalized = _normalize_taxonomy_error_type(
                        original_error_type, taxonomy
                    )
                    prediction_payload["predicted_errors"][0]["error_type"] = normalized_error_type
                    row = {
                        "run_id": run_id,
                        "dataset": dataset,
                        "domain": record["domain"],
                        "trajectory_id": record["trajectory_id"],
                        "condition": condition.value,
                        "taxonomy_id": taxonomy["taxonomy_id"] if taxonomy else None,
                        "taxonomy_sha256": taxonomy_digest(taxonomy) if taxonomy else None,
                        "model": model_config["name"],
                        "prompt_version": config["experiment"]["prompt_version"],
                        "prediction_postprocessing_version": PREDICTION_POSTPROCESSING_VERSION,
                        "request_max_output_tokens": request_max_output_tokens,
                        "truncation_recovery_version": TRUNCATION_RECOVERY_VERSION if recovery_mode else None,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "ok",
                        "prediction": prediction_payload,
                        "raw_error_type": original_error_type if normalized else None,
                        "cache_key": call["cache_key"],
                        "cache_path": str(Path(call["cache_path"]).relative_to(config["_root"])),
                        "cache_hit": call["cache_hit"],
                        "response_metadata": call["response_metadata"],
                    }
                except Exception as exc:
                    row = {
                        "run_id": run_id,
                        "dataset": dataset,
                        "domain": record["domain"],
                        "trajectory_id": record["trajectory_id"],
                        "condition": condition.value,
                        "taxonomy_id": taxonomy["taxonomy_id"] if taxonomy else None,
                        "model": model_config["name"],
                        "prompt_version": config["experiment"]["prompt_version"],
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    append_jsonl(prediction_path, row)
                    raise
                append_jsonl(prediction_path, row)
    _compact_prediction_rows(prediction_path)
    return prediction_path
