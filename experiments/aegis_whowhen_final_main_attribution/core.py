from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import load_config
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json
from taxonomy_experiment.token_count import TokenCounter

from experiments.aegis_whowhen_main_fixed_guidance.audits.task_independence import (
    verify_frozen_identity,
)
from experiments.aegis_whowhen_main_fixed_guidance import core as frozen


HERE = Path(__file__).resolve().parent
DATASETS = ("aegis", "whowhen")
CONDITIONS = (
    "label_only",
    "own_taxonomy",
    "foreign_taxonomy",
    "merged_taxonomy",
)
CONTRASTS = (
    ("own_taxonomy", "label_only"),
    ("foreign_taxonomy", "label_only"),
    ("merged_taxonomy", "label_only"),
    ("merged_taxonomy", "foreign_taxonomy"),
)
TRANSITIONS = (
    ("label_only", "own_taxonomy"),
    ("label_only", "foreign_taxonomy"),
    ("label_only", "merged_taxonomy"),
    ("foreign_taxonomy", "merged_taxonomy"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def resolve(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else config["_root"] / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path)) if path.exists() else []


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not names:
            return
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Frozen run artifact differs: {path}")
        return
    write_json(path, value)


def parse_sample_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["coverage_repair"] = row["coverage_repair"].casefold() == "true"
            row["sampling_seed"] = int(row["sampling_seed"])
            rows.append(row)
    return rows


def load_frozen_sample(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_root = resolve(config, "frozen_sample_root")
    manifest = read_json(sample_root / "FINAL_MAIN_SAMPLE_MANIFEST.json")
    inputs = read_jsonl(sample_root / "FINAL_MAIN_INPUTS.jsonl")
    gold_rows = read_jsonl(sample_root / "FINAL_MAIN_GOLD.jsonl")
    gold_index = {
        (row["dataset"], row["trajectory_id"]): row for row in gold_rows
    }
    records = []
    for row in inputs:
        key = (row["dataset"], row["trajectory_id"])
        gold = gold_index.get(key)
        if gold is None or gold["task_hash"] != row["task_hash"]:
            raise ValueError(f"Input/gold identity mismatch: {key}")
        records.append({**row, "gold_category": gold["gold_category"]})
    if len(records) != len(gold_rows):
        raise ValueError("Input/gold row counts differ")
    return records, manifest


def load_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = resolve(config, "frozen_implementation_root")
    return {
        "aegis": read_json(root / "aegis_taxonomy.json"),
        "whowhen": read_json(root / "whowhen_taxonomy.json"),
        "merged": read_json(root / "merged_taxonomy.json"),
    }


def _sample_hashes(config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, str]:
    sample_root = resolve(config, "frozen_sample_root")
    sample_rows = parse_sample_csv(sample_root / "FINAL_MAIN_SAMPLE_MANIFEST.csv")
    by_dataset = {
        dataset: [row for row in sample_rows if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    inputs = read_jsonl(sample_root / "FINAL_MAIN_INPUTS.jsonl")
    gold = read_jsonl(sample_root / "FINAL_MAIN_GOLD.jsonl")
    if len(records) != len(inputs):
        raise ValueError("Frozen sample record count changed")
    return {
        "aegis_sample": sha256_json(by_dataset["aegis"]),
        "whowhen_sample": sha256_json(by_dataset["whowhen"]),
        "combined_main_input": sha256_json(inputs),
        "combined_gold": sha256_json(gold),
    }


def _harness_hashes(config: dict[str, Any]) -> dict[str, str]:
    paths = [HERE / "core.py", HERE / "run.py", Path(config["_config_path"])]
    return {
        str(path.relative_to(config["_root"])): sha256_file(path) for path in paths
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    expected = config["experiment"]
    run_root = resolve(config, "run_root")
    run_root.mkdir(parents=True, exist_ok=True)
    identity = verify_frozen_identity()
    records, sample_manifest = load_frozen_sample(config)
    implementation_manifest = read_json(
        resolve(config, "frozen_implementation_root") / "implementation_manifest.json"
    )
    frozen_config = load_config(resolve(config, "frozen_prediction_config"))
    taxonomies = load_taxonomies(config)
    observed_hashes = _sample_hashes(config, records)
    expected_hashes = config["expected_hashes"]

    counts = Counter(row["dataset"] for row in records)
    task_sets = {
        dataset: {row["task_hash"] for row in records if row["dataset"] == dataset}
        for dataset in DATASETS
    }
    categories = {
        dataset: {row["gold_category"] for row in records if row["dataset"] == dataset}
        for dataset in DATASETS
    }
    leakage = frozen.leakage_audit(records)
    frozen_model_config = implementation_manifest["model_config"]
    request_config_matches = (
        frozen_config["model"] == frozen_model_config["model"]
        and frozen_config["api"]["base_url"] == frozen_model_config["api"]["base_url"]
        and frozen_config["api"]["sdk_attempts"] == frozen_model_config["api"]["sdk_attempts"]
        and frozen_config["api"]["provider"] == frozen_model_config["api_provider"]
        and frozen_config["retry_policy"] == frozen_model_config["retry_policy"]
    )
    checks = {
        "prediction_implementation_identity": identity["passed"]
        and identity["implementation_run_id"]
        == expected["prediction_implementation_id"],
        "sample_id": sample_manifest["main_sample_id"] == expected["main_sample_id"],
        "run_namespace": sample_manifest["final_run_namespace"]
        == expected["run_namespace"]
        and run_root.name == expected["run_namespace"],
        "sample_hashes": observed_hashes == expected_hashes
        and sample_manifest["sample_hashes"]
        == {
            "aegis": expected_hashes["aegis_sample"],
            "whowhen": expected_hashes["whowhen_sample"],
            "combined_main_input": expected_hashes["combined_main_input"],
            "combined_gold": expected_hashes["combined_gold"],
        },
        "dataset_counts": counts
        == Counter(
            {
                "aegis": int(expected["expected_aegis_n"]),
                "whowhen": int(expected["expected_whowhen_n"]),
            }
        ),
        "total_count": len(records) == int(expected["expected_total_n"]),
        "unique_task_hashes": all(
            len(task_sets[dataset]) == counts[dataset] for dataset in DATASETS
        ),
        "unique_trajectory_ids": all(
            len(
                {
                    row["trajectory_id"]
                    for row in records
                    if row["dataset"] == dataset
                }
            )
            == counts[dataset]
            for dataset in DATASETS
        ),
        "all_14_native_classes": all(len(categories[dataset]) == 14 for dataset in DATASETS),
        "pilot_task_overlap_zero": sample_manifest["validation"]["pilot_task_hash_overlap_zero"],
        "smoke_task_overlap_zero": sample_manifest["validation"]["smoke_task_hash_overlap_zero"],
        "cross_dataset_task_overlap_zero": not (task_sets["aegis"] & task_sets["whowhen"]),
        "leakage_zero": leakage["leakage_count"] == 0,
        "prediction_results_not_used": not sample_manifest["accuracy_or_pilot_results_used"],
        "four_conditions": tuple(expected["conditions"]) == CONDITIONS,
        "planned_rows": len(records) * len(CONDITIONS)
        == int(expected["expected_prediction_rows"]),
        "frozen_model": frozen_config["model"]["name"]
        == "google/gemini-2.5-flash-lite"
        and float(frozen_config["model"]["temperature"]) == 0,
        "request_config_matches_frozen_manifest": request_config_matches,
        "exact_native_enum_schema": all(
            frozen.response_schema(frozen.native_ids(dataset, taxonomies))["json_schema"]["schema"]["properties"]["failure_category"]["enum"]
            == frozen.native_ids(dataset, taxonomies)
            for dataset in DATASETS
        ),
    }

    counter = TokenCounter.for_model(
        frozen_config["model"]["name"],
        frozen_config["model"].get("tokenizer_encoding"),
    )
    prompt_tokens = []
    for record in records:
        for condition in CONDITIONS:
            built = frozen.build_prompt(record, condition, taxonomies)
            prompt_tokens.append(
                counter.count(built["system_prompt"] + "\n" + built["user_prompt"])
            )
    max_prompt = max(prompt_tokens, default=0)
    checks["context_window"] = (
        max_prompt + int(frozen_config["model"]["max_output_tokens"])
        < int(frozen_config["model"]["context_window"])
    )
    price = frozen_config["model"]["pricing_usd_per_million"]
    guarded_input = sum(prompt_tokens) * float(
        frozen_config["budget"]["input_token_safety_multiplier"]
    )
    estimated_reserve = (
        guarded_input * float(price["input"])
        + len(prompt_tokens)
        * int(frozen_config["model"]["max_output_tokens"])
        * float(price["output"])
    ) / 1_000_000
    checks["budget_reserve_below_hard_cap"] = estimated_reserve < float(
        config["execution"]["hard_cost_cap_usd"]
    )
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"STOP_BEFORE_INFERENCE: preflight failed: {failed}")

    planned = frozen.planned_keys(records, int(frozen_config["experiment"]["main_seed"]))
    planned_key_hash = sha256_json(planned)
    audit = {
        "status": "PREFLIGHT_READY_FOR_MAIN_INFERENCE",
        "offline_only": True,
        "api_calls": 0,
        "prediction_implementation_id": expected["prediction_implementation_id"],
        "main_sample_id": expected["main_sample_id"],
        "run_namespace": expected["run_namespace"],
        "checks": checks,
        "observed_sample_hashes": observed_hashes,
        "dataset_counts": dict(sorted(counts.items())),
        "total_trajectories": len(records),
        "planned_logical_rows": len(planned),
        "planned_key_hash": planned_key_hash,
        "leakage_count": leakage["leakage_count"],
        "prompt_token_estimate": {
            "total": sum(prompt_tokens),
            "maximum": max_prompt,
            "guarded_cost_reserve_usd": estimated_reserve,
            "hard_cap_usd": float(config["execution"]["hard_cost_cap_usd"]),
        },
    }
    write_json(run_root / "preflight_audit.json", audit)
    run_manifest = {
        "research_question": expected["research_question"],
        "prediction_implementation_id": expected["prediction_implementation_id"],
        "main_sample_id": expected["main_sample_id"],
        "run_namespace": expected["run_namespace"],
        "sample_hashes": observed_hashes,
        "implementation_hashes": {
            "code_hash": implementation_manifest["code_hash"],
            "parser_hash": implementation_manifest["parser_hash"],
            "serializer_hash": implementation_manifest["serializer_hash"],
            "prompt_hashes": implementation_manifest["prompt_hashes"],
            "schema_hashes": implementation_manifest["schema_hashes"],
            "taxonomy_hashes": implementation_manifest["taxonomy_hashes"],
            "model_config_hash": implementation_manifest["model_config_hash"],
        },
        "model": frozen_config["model"],
        "provider": frozen_config["api"]["provider"],
        "retry_policy": frozen_config["retry_policy"],
        "conditions": list(CONDITIONS),
        "planned_trajectories": len(records),
        "planned_logical_rows": len(planned),
        "planned_key_hash": planned_key_hash,
        "bootstrap": config["analysis"],
        "execution_budget_hard_cap_usd": float(config["execution"]["hard_cost_cap_usd"]),
        "harness_hashes": _harness_hashes(config),
        "prediction_outputs_imported_from_other_namespaces": False,
    }
    freeze_json(run_root / "main_run_manifest.json", run_manifest)
    status_path = run_root / "main_run_status.json"
    current = read_json(status_path) if status_path.exists() else {}
    if current.get("status") != "FINAL_MAIN_EXPERIMENT_COMPLETE":
        write_json(
            status_path,
            {
                "status": "PREFLIGHT_READY_FOR_MAIN_INFERENCE",
                "planned_logical_rows": len(planned),
                "completed_logical_rows": len(
                    frozen.prediction_index(run_root / "main_predictions.jsonl")
                ),
                "attribution_inference_started": bool(
                    frozen.prediction_index(run_root / "main_predictions.jsonl")
                ),
            },
        )
    return {
        "audit": audit,
        "records": records,
        "taxonomies": taxonomies,
        "frozen_config": frozen_config,
        "planned_keys": planned,
    }


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _logical_ledger(
    path: Path, dataset: str, trajectory_id: str, condition: str
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, row)
        for index, row in enumerate(_ledger_rows(path), start=1)
        if row.get("purpose") == "final_main_attribution"
        and row.get("dataset") == dataset
        and row.get("trajectory_id") == trajectory_id
        and row.get("condition") == condition
    ]


def _usage(events: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    actual = [row for _, row in events if row.get("actual_request")]
    usages = [row.get("reported_usage") or {} for row in actual]
    return {
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usages),
        "response_tokens": sum(int(item.get("completion_tokens") or 0) for item in usages),
        "provider_transport_retry_count": sum(
            row.get("accounting_type") == "provider_transport_retry" for row in actual
        ),
        "schema_parser_retry_count": sum(
            row.get("accounting_type") == "schema_parser_retry" for row in actual
        ),
        "actual_api_calls": len(actual),
        "billed_responses": sum(bool(row.get("billed")) for row in actual),
        "cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in actual),
        "api_call_ledger_references": [index for index, _ in events],
    }


def run_predictions(config: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    run_root = resolve(config, "run_root")
    prediction_path = run_root / "main_predictions.jsonl"
    ledger_path = run_root / "main_api_call_ledger.jsonl"
    status_path = run_root / "main_run_status.json"
    records = prepared["records"]
    taxonomies = prepared["taxonomies"]
    planned = prepared["planned_keys"]
    record_map = {(row["dataset"], row["trajectory_id"]): row for row in records}
    existing = frozen.prediction_index(prediction_path)

    runtime_config = copy.deepcopy(prepared["frozen_config"])
    runtime_config["paths"]["runtime"] = str(run_root)
    runtime_config["api"]["api_key_envs"] = list(config["execution"]["api_key_envs"])
    runtime_config["budget"]["max_inference_usd"] = float(
        config["execution"]["hard_cost_cap_usd"]
    )
    runtime_config["budget"]["max_total_charge_usd"] = float(
        config["execution"]["hard_cost_cap_usd"]
    )
    client = frozen.AuditedOpenRouter(runtime_config)
    client.cache_dir = run_root / "cache"
    client.cache_dir.mkdir(parents=True, exist_ok=True)
    client.ledger = ledger_path

    consecutive_system_failures = 0
    progress_every = int(config["execution"]["progress_every"])
    systematic_limit = int(
        config["execution"]["systematic_failure_consecutive_rows"]
    )
    for ordinal, (dataset, trajectory_id, condition) in enumerate(planned, start=1):
        key = (dataset, trajectory_id, condition)
        old = existing.get(key)
        if old and old.get("status") == "complete" and old.get("terminal_reason") != "terminal_parser_or_transport_error":
            continue
        record = record_map[(dataset, trajectory_id)]
        result = frozen.execute_prediction(
            config=runtime_config,
            client=client,
            record=record,
            condition=condition,
            taxonomies=taxonomies,
            purpose="final_main_attribution",
        )
        events = _logical_ledger(ledger_path, dataset, trajectory_id, condition)
        usage = _usage(events)
        result.update(
            {
                "task_hash": record["task_hash"],
                "source_trajectory_id": record["source_trajectory_id"],
                "gold_native_category": record["gold_category"],
                "predicted_native_category": result["failure_category"],
                "correct": int(
                    result["final_compliant"]
                    and result["failure_category"] == record["gold_category"]
                ),
                **usage,
            }
        )
        append_jsonl(prediction_path, result)
        existing[key] = result
        budget_failure = any(
            "RuntimeBudgetExceeded" in str(attempt.get("error", ""))
            for attempt in result.get("attempts", [])
        )
        if result.get("terminal_reason") == "terminal_parser_or_transport_error":
            consecutive_system_failures += 1
        else:
            consecutive_system_failures = 0
        completed = sum(
            row.get("status") == "complete"
            and row.get("terminal_reason") != "terminal_parser_or_transport_error"
            for row in existing.values()
        )
        write_json(
            status_path,
            {
                "status": "RUNNING",
                "planned_logical_rows": len(planned),
                "completed_logical_rows": completed,
                "last_planned_ordinal": ordinal,
                "attribution_inference_started": True,
                "actual_api_calls": sum(
                    bool(row.get("actual_request")) for row in _ledger_rows(ledger_path)
                ),
                "recorded_cost_usd": sum(
                    float(row.get("cost_usd") or 0.0)
                    for row in _ledger_rows(ledger_path)
                    if row.get("actual_request")
                ),
            },
        )
        if ordinal % progress_every == 0 or ordinal == len(planned):
            print(
                f"[final-main] completed={completed}/{len(planned)} "
                f"dataset={dataset} condition={condition}",
                flush=True,
            )
        if budget_failure:
            status = read_json(status_path)
            status["status"] = "STOPPED_BUDGET_HARD_CAP"
            status["budget_hard_cap_usd"] = float(
                config["execution"]["hard_cost_cap_usd"]
            )
            write_json(status_path, status)
            raise RuntimeError(
                "Budget hard cap reached; no further attribution calls were attempted"
            )
        if consecutive_system_failures >= systematic_limit:
            status = read_json(status_path)
            status["status"] = "STOPPED_SYSTEMATIC_PROVIDER_OR_PARSER_FAILURE"
            write_json(status_path, status)
            raise RuntimeError(
                f"Stopped after {systematic_limit} consecutive provider/parser terminal rows"
            )

    latest = frozen.prediction_index(prediction_path)
    missing = [key for key in planned if key not in latest]
    transient = [
        key
        for key in planned
        if latest.get(key, {}).get("terminal_reason")
        == "terminal_parser_or_transport_error"
    ]
    if missing or transient:
        write_json(
            status_path,
            {
                "status": "INCOMPLETE_RETRY_SAME_COMMAND",
                "planned_logical_rows": len(planned),
                "completed_logical_rows": len(planned) - len(missing) - len(transient),
                "missing_rows": len(missing),
                "transient_terminal_rows": len(transient),
                "attribution_inference_started": True,
            },
        )
        raise RuntimeError(
            f"Run incomplete: missing={len(missing)}, transient={len(transient)}; rerun the same command"
        )
    return {"latest": latest, "ledger_path": ledger_path}


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def paired_bootstrap_ci(
    differences: list[int], *, resamples: int, seed: int
) -> tuple[float, float]:
    if not differences:
        raise ValueError("Paired bootstrap requires observations")
    rng = random.Random(seed)
    n = len(differences)
    draws = []
    for _ in range(resamples):
        draws.append(sum(rng.choices(differences, k=n)) / n)
    draws.sort()
    low = draws[int(0.025 * (resamples - 1))]
    high = draws[int(0.975 * (resamples - 1))]
    return low, high


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    ordered = sorted(range(m), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(ordered):
        candidate = min(1.0, p_values[index] * (m - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def class_metrics(
    rows: list[dict[str, Any]], native_ids: list[str]
) -> tuple[list[dict[str, Any]], float, dict[str, dict[str, int]]]:
    matrix: dict[str, Counter[str]] = {
        gold: Counter() for gold in native_ids
    }
    for row in rows:
        predicted = row["predicted_native_category"] or "__TERMINAL_INVALID__"
        matrix[row["gold_native_category"]][predicted] += 1
    metrics = []
    for category in native_ids:
        tp = matrix[category][category]
        support = sum(matrix[category].values())
        predicted_count = sum(matrix[gold][category] for gold in native_ids)
        fp = predicted_count - tp
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics.append(
            {
                "category": category,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return metrics, sum(item["f1"] for item in metrics) / len(metrics), {
        gold: dict(sorted(counter.items())) for gold, counter in matrix.items()
    }


def _csv_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]


def analyze_results(
    config: dict[str, Any], prepared: dict[str, Any]
) -> dict[str, Any]:
    run_root = resolve(config, "run_root")
    prediction_path = run_root / "main_predictions.jsonl"
    ledger_path = run_root / "main_api_call_ledger.jsonl"
    latest = frozen.prediction_index(prediction_path)
    planned = prepared["planned_keys"]
    if set(latest) != set(planned):
        raise RuntimeError(
            f"Cannot analyze incomplete run: {len(latest)}/{len(planned)} logical rows"
        )
    if any(
        latest[key].get("terminal_reason") == "terminal_parser_or_transport_error"
        for key in planned
    ):
        raise RuntimeError("Provider/parser terminal rows must be retried before analysis")
    rows = [latest[key] for key in planned]
    records = prepared["records"]
    record_index = {(row["dataset"], row["trajectory_id"]): row for row in records}
    if any(
        row["task_hash"]
        != record_index[(row["dataset"], row["trajectory_id"])]["task_hash"]
        for row in rows
    ):
        raise RuntimeError("Prediction/sample task identity mismatch")
    if any(
        sum(
            row["dataset"] == dataset and row["trajectory_id"] == trajectory_id
            for row in rows
        )
        != 4
        for dataset, trajectory_id in record_index
    ):
        raise RuntimeError("Every frozen trajectory must have exactly four condition rows")
    write_csv(run_root / "main_predictions.csv", _csv_safe_rows(rows))

    taxonomies = prepared["taxonomies"]
    accuracy_rows = []
    macro_rows = []
    per_class_rows = []
    confusion: dict[str, Any] = {}
    major_confusions: list[dict[str, Any]] = []
    frequency_rows = []
    compliance_rows = []
    retry_rows = []
    token_rows = []
    for dataset in DATASETS:
        native_ids = frozen.native_ids(dataset, taxonomies)
        confusion[dataset] = {}
        for condition in CONDITIONS:
            subset = [
                row
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            correct = sum(int(row["correct"]) for row in subset)
            accuracy = correct / len(subset)
            accuracy_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(subset),
                    "correct": correct,
                    "accuracy": accuracy,
                    "accuracy_percent": 100 * accuracy,
                }
            )
            metrics, macro_f1, matrix = class_metrics(subset, native_ids)
            macro_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(subset),
                    "macro_f1": macro_f1,
                }
            )
            for metric in metrics:
                per_class_rows.append(
                    {"dataset": dataset, "condition": condition, **metric}
                )
            confusion[dataset][condition] = matrix
            off_diagonal = sorted(
                (
                    (count, gold, predicted)
                    for gold, predictions in matrix.items()
                    for predicted, count in predictions.items()
                    if predicted != gold and count
                ),
                key=lambda item: (-item[0], item[1], item[2]),
            )
            for rank, (count, gold, predicted) in enumerate(
                off_diagonal[:5], start=1
            ):
                major_confusions.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "rank": rank,
                        "gold_category": gold,
                        "predicted_category": predicted,
                        "count": count,
                    }
                )
            frequencies = Counter(
                row["predicted_native_category"] or "__TERMINAL_INVALID__"
                for row in subset
            )
            for category, count in sorted(frequencies.items()):
                frequency_rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "predicted_category": category,
                        "count": count,
                        "percent": 100 * count / len(subset),
                    }
                )
            compliance_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(subset),
                    "initial_compliant": sum(row["initial_compliant"] for row in subset),
                    "initial_compliance_rate": sum(row["initial_compliant"] for row in subset) / len(subset),
                    "final_compliant": sum(row["final_compliant"] for row in subset),
                    "final_compliance_rate": sum(row["final_compliant"] for row in subset) / len(subset),
                    "terminal_invalid": sum(row["terminal_invalid"] for row in subset),
                    "terminal_invalid_rate": sum(row["terminal_invalid"] for row in subset) / len(subset),
                }
            )
            retry_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(subset),
                    "formatting_retries": sum(row["formatting_retry_count"] for row in subset),
                    "truncation_retries": sum(row["truncation_retry_count"] for row in subset),
                    "provider_transport_retries": sum(row["provider_transport_retry_count"] for row in subset),
                    "schema_parser_retries": sum(row["schema_parser_retry_count"] for row in subset),
                    "rows_with_any_retry": sum(
                        bool(
                            row["formatting_retry_count"]
                            or row["truncation_retry_count"]
                            or row["provider_transport_retry_count"]
                            or row["schema_parser_retry_count"]
                        )
                        for row in subset
                    ),
                }
            )
            token_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(subset),
                    "prompt_tokens_total": sum(row["prompt_tokens"] for row in subset),
                    "prompt_tokens_mean": sum(row["prompt_tokens"] for row in subset) / len(subset),
                    "prompt_tokens_max": max(row["prompt_tokens"] for row in subset),
                    "response_tokens_total": sum(row["response_tokens"] for row in subset),
                    "response_tokens_mean": sum(row["response_tokens"] for row in subset) / len(subset),
                    "response_tokens_max": max(row["response_tokens"] for row in subset),
                }
            )

    outcomes = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row
        for row in rows
    }
    contrast_rows = []
    bootstrap_rows = []
    mcnemar_rows = []
    resamples = int(config["analysis"]["bootstrap_resamples"])
    bootstrap_seed = int(config["analysis"]["bootstrap_seed"])
    for dataset in DATASETS:
        identities = sorted(
            (row["trajectory_id"], row["task_hash"])
            for row in records
            if row["dataset"] == dataset
        )
        for condition_a, condition_b in CONTRASTS:
            paired = [
                (
                    outcomes[(dataset, trajectory_id, condition_a)],
                    outcomes[(dataset, trajectory_id, condition_b)],
                )
                for trajectory_id, _ in identities
            ]
            improved = sum(not a[1]["correct"] and a[0]["correct"] for a in paired)
            regressed = sum(a[1]["correct"] and not a[0]["correct"] for a in paired)
            both_correct = sum(a[1]["correct"] and a[0]["correct"] for a in paired)
            both_incorrect = len(paired) - improved - regressed - both_correct
            acc_a = sum(a[0]["correct"] for a in paired) / len(paired)
            acc_b = sum(a[1]["correct"] for a in paired) / len(paired)
            differences = [int(a[0]["correct"]) - int(a[1]["correct"]) for a in paired]
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{bootstrap_seed}:{dataset}:{condition_a}:{condition_b}".encode("utf-8")
                ).digest()[:8],
                "big",
            )
            ci_low, ci_high = paired_bootstrap_ci(
                differences, resamples=resamples, seed=derived_seed
            )
            raw_p = exact_mcnemar_p(improved, regressed)
            base = {
                "dataset": dataset,
                "contrast": f"{condition_a}_minus_{condition_b}",
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n": len(paired),
                "condition_a_accuracy": acc_a,
                "condition_b_accuracy": acc_b,
                "difference": acc_a - acc_b,
                "difference_percentage_points": 100 * (acc_a - acc_b),
                "incorrect_to_correct": improved,
                "correct_to_incorrect": regressed,
                "both_correct": both_correct,
                "both_incorrect": both_incorrect,
                "mcnemar_exact_two_sided_p": raw_p,
                "bootstrap_resamples": resamples,
                "bootstrap_seed": derived_seed,
                "paired_ci_95_low": ci_low,
                "paired_ci_95_high": ci_high,
                "paired_ci_95_low_percentage_points": 100 * ci_low,
                "paired_ci_95_high_percentage_points": 100 * ci_high,
            }
            contrast_rows.append(base)
            mcnemar_rows.append(
                {
                    key: base[key]
                    for key in (
                        "dataset",
                        "contrast",
                        "n",
                        "incorrect_to_correct",
                        "correct_to_incorrect",
                        "both_correct",
                        "both_incorrect",
                        "mcnemar_exact_two_sided_p",
                    )
                }
            )
            bootstrap_rows.append(
                {
                    key: base[key]
                    for key in (
                        "dataset",
                        "contrast",
                        "n",
                        "difference",
                        "difference_percentage_points",
                        "bootstrap_resamples",
                        "bootstrap_seed",
                        "paired_ci_95_low",
                        "paired_ci_95_high",
                        "paired_ci_95_low_percentage_points",
                        "paired_ci_95_high_percentage_points",
                    )
                }
            )
    adjusted = holm_adjust(
        [row["mcnemar_exact_two_sided_p"] for row in contrast_rows]
    )
    holm_rows = []
    for row, adjusted_p in zip(contrast_rows, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        row["holm_reject_alpha_0_05"] = adjusted_p <= float(config["analysis"]["alpha"])
        holm_rows.append(
            {
                "dataset": row["dataset"],
                "contrast": row["contrast"],
                "raw_p": row["mcnemar_exact_two_sided_p"],
                "holm_adjusted_p": adjusted_p,
                "alpha": float(config["analysis"]["alpha"]),
                "reject": row["holm_reject_alpha_0_05"],
            }
        )

    transition_rows = []
    for dataset in DATASETS:
        identities = sorted(
            row["trajectory_id"] for row in records if row["dataset"] == dataset
        )
        for source, target in TRANSITIONS:
            pairs = [
                (
                    outcomes[(dataset, identity, source)],
                    outcomes[(dataset, identity, target)],
                )
                for identity in identities
            ]
            transition_rows.append(
                {
                    "dataset": dataset,
                    "transition": f"{source}_to_{target}",
                    "n": len(pairs),
                    "incorrect_to_correct": sum(not a[0]["correct"] and a[1]["correct"] for a in pairs),
                    "correct_to_incorrect": sum(a[0]["correct"] and not a[1]["correct"] for a in pairs),
                    "correct_to_correct": sum(a[0]["correct"] and a[1]["correct"] for a in pairs),
                    "incorrect_to_same_incorrect": sum(
                        not a[0]["correct"]
                        and not a[1]["correct"]
                        and a[0]["predicted_native_category"] == a[1]["predicted_native_category"]
                        for a in pairs
                    ),
                    "incorrect_to_different_incorrect": sum(
                        not a[0]["correct"]
                        and not a[1]["correct"]
                        and a[0]["predicted_native_category"] != a[1]["predicted_native_category"]
                        for a in pairs
                    ),
                    "prediction_changes": sum(
                        a[0]["predicted_native_category"] != a[1]["predicted_native_category"]
                        for a in pairs
                    ),
                    "prediction_change_rate": sum(
                        a[0]["predicted_native_category"] != a[1]["predicted_native_category"]
                        for a in pairs
                    ) / len(pairs),
                }
            )

    sensitivity_rows = []
    for dataset in DATASETS:
        counts_pattern = Counter()
        identities = sorted(
            row["trajectory_id"] for row in records if row["dataset"] == dataset
        )
        for identity in identities:
            p = {
                condition: outcomes[(dataset, identity, condition)]["predicted_native_category"]
                for condition in CONDITIONS
            }
            if len(set(p.values())) == 1:
                pattern = "all_four_same"
            elif p["label_only"] == p["foreign_taxonomy"] == p["merged_taxonomy"] != p["own_taxonomy"]:
                pattern = "only_own_changed"
            elif p["label_only"] == p["own_taxonomy"] == p["merged_taxonomy"] != p["foreign_taxonomy"]:
                pattern = "only_foreign_changed"
            elif p["label_only"] == p["own_taxonomy"] == p["foreign_taxonomy"] != p["merged_taxonomy"]:
                pattern = "only_merged_changed"
            else:
                pattern = "multiple_conditions_changed"
            counts_pattern[pattern] += 1
        for pattern in (
            "all_four_same",
            "only_own_changed",
            "only_foreign_changed",
            "only_merged_changed",
            "multiple_conditions_changed",
        ):
            sensitivity_rows.append(
                {
                    "dataset": dataset,
                    "pattern": pattern,
                    "count": counts_pattern[pattern],
                    "percent": 100 * counts_pattern[pattern] / len(identities),
                }
            )

    framework_rows = []
    for dataset in DATASETS:
        frameworks = sorted(
            {row["framework"] for row in records if row["dataset"] == dataset}
        )
        for framework in frameworks:
            identities = {
                row["trajectory_id"]
                for row in records
                if row["dataset"] == dataset and row["framework"] == framework
            }
            result = {"dataset": dataset, "framework": framework, "n": len(identities)}
            for condition in CONDITIONS:
                result[f"{condition}_accuracy"] = sum(
                    outcomes[(dataset, identity, condition)]["correct"]
                    for identity in identities
                ) / len(identities)
            framework_rows.append(result)

    ledger = _ledger_rows(ledger_path)
    actual = [row for row in ledger if row.get("actual_request")]
    accounting_types = (
        "base_attribution",
        "formatting_retry",
        "truncation_retry",
        "schema_parser_retry",
        "provider_transport_retry",
        "merge_adjudication",
    )
    ledger_counts = {
        name: sum(row.get("accounting_type") == name for row in actual)
        for name in accounting_types
    }
    cost_summary = {
        "prediction_implementation_id": config["experiment"]["prediction_implementation_id"],
        "main_sample_id": config["experiment"]["main_sample_id"],
        "planned_logical_rows": len(planned),
        "completed_logical_rows": len(rows),
        "actual_api_calls": len(actual),
        "billed_responses": sum(bool(row.get("billed")) for row in actual),
        "failed_network_attempts": sum(not bool(row.get("billed")) for row in actual),
        "cache_events": sum(not bool(row.get("actual_request")) for row in ledger),
        "calls_by_accounting_type": ledger_counts,
        "ledger_equation_sum": sum(ledger_counts.values()),
        "ledger_reconciled": sum(ledger_counts.values()) == len(actual),
        "prompt_tokens": sum(
            int((row.get("reported_usage") or {}).get("prompt_tokens") or 0)
            for row in actual
        ),
        "response_tokens": sum(
            int((row.get("reported_usage") or {}).get("completion_tokens") or 0)
            for row in actual
        ),
        "total_recorded_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in actual),
        "hard_cost_cap_usd": float(config["execution"]["hard_cost_cap_usd"]),
    }
    if not cost_summary["ledger_reconciled"]:
        raise RuntimeError("API call ledger failed reconciliation")

    write_csv(run_root / "main_accuracy.csv", accuracy_rows)
    write_csv(run_root / "main_macro_f1.csv", macro_rows)
    write_csv(run_root / "main_per_class_metrics.csv", per_class_rows)
    write_json(run_root / "main_confusion_matrices.json", confusion)
    write_csv(run_root / "main_primary_contrasts.csv", contrast_rows)
    write_csv(run_root / "main_mcnemar_tests.csv", mcnemar_rows)
    write_csv(run_root / "main_holm_adjustment.csv", holm_rows)
    write_csv(run_root / "main_bootstrap_cis.csv", bootstrap_rows)
    write_csv(run_root / "main_transitions.csv", transition_rows)
    write_csv(run_root / "main_guidance_sensitivity.csv", sensitivity_rows)
    write_csv(run_root / "main_category_frequencies.csv", frequency_rows)
    write_csv(run_root / "main_framework_results.csv", framework_rows)
    write_csv(run_root / "main_output_compliance.csv", compliance_rows)
    write_csv(run_root / "main_retry_audit.csv", retry_rows)
    write_csv(run_root / "main_token_audit.csv", token_rows)
    write_json(run_root / "main_cost_summary.json", cost_summary)

    accuracy_index = {
        (row["dataset"], row["condition"]): row for row in accuracy_rows
    }
    macro_index = {(row["dataset"], row["condition"]): row for row in macro_rows}
    lines = [
        "# Final Main Attribution Experiment",
        "",
        "Status: `FINAL_MAIN_EXPERIMENT_COMPLETE`",
        "",
        "## Experiment identity",
        "",
        f"- Research question: **{config['experiment']['research_question']}**",
        f"- Prediction implementation ID: `{config['experiment']['prediction_implementation_id']}`",
        f"- Main sample ID: `{config['experiment']['main_sample_id']}`",
        f"- Namespace: `{config['experiment']['run_namespace']}`",
        f"- Attribution model: `{prepared['frozen_config']['model']['name']}`; temperature=0",
        "- Conditions: Label-Only, Own, Foreign, Merged",
        "",
        "## Sample",
        "",
        "- AEGIS N = 97",
        "- Who&When N = 500",
        "- **One trajectory per unique original task was used in the primary analysis.**",
        "- All 14 native classes are represented in each dataset; very small supports are interpreted descriptively.",
        "",
        "### Native-category support",
        "",
        "| Dataset | Native category | Support |",
        "|---|---|---:|",
    ]
    support_counts = Counter(
        (row["dataset"], row["gold_category"]) for row in records
    )
    for (dataset, category), support in sorted(support_counts.items()):
        lines.append(f"| {dataset} | {category} | {support} |")
    lines.extend(
        [
        "",
        "## Output compliance",
        "",
        "| Dataset | Condition | Initial valid | Formatting retries | Truncation retries | Final valid | Terminal invalid |",
        "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in compliance_rows:
        retry = next(
            item
            for item in retry_rows
            if item["dataset"] == row["dataset"] and item["condition"] == row["condition"]
        )
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {100*row['initial_compliance_rate']:.2f}% | "
            f"{retry['formatting_retries']} | {retry['truncation_retries']} | "
            f"{100*row['final_compliance_rate']:.2f}% | {row['terminal_invalid']} |"
        )
    lines.extend(
        [
            "",
            "## Primary condition accuracy",
            "",
            "| Dataset | Label-Only | Own | Foreign | Merged |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | "
            + " | ".join(
                f"{100*accuracy_index[(dataset, condition)]['accuracy']:.2f}%"
                for condition in CONDITIONS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Primary paired contrasts",
            "",
            "| Dataset | Contrast | Difference (pp) | Improved | Regressed | Exact McNemar p | Holm p | Paired 95% CI (pp) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrast_rows:
        lines.append(
            f"| {row['dataset']} | {row['condition_a']} − {row['condition_b']} | "
            f"{row['difference_percentage_points']:.2f} | {row['incorrect_to_correct']} | "
            f"{row['correct_to_incorrect']} | {row['mcnemar_exact_two_sided_p']:.6g} | "
            f"{row['holm_adjusted_p']:.6g} | [{row['paired_ci_95_low_percentage_points']:.2f}, "
            f"{row['paired_ci_95_high_percentage_points']:.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Secondary metrics",
            "",
            "| Dataset | Label-Only Macro-F1 | Own | Foreign | Merged |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        lines.append(
            f"| {dataset} | "
            + " | ".join(
                f"{macro_index[(dataset, condition)]['macro_f1']:.4f}"
                for condition in CONDITIONS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Major confusion patterns",
            "",
            "| Dataset | Condition | Gold | Predicted | Count |",
            "|---|---|---|---|---:|",
        ]
    )
    for item in major_confusions:
        lines.append(
            f"| {item['dataset']} | {item['condition']} | {item['gold_category']} | "
            f"{item['predicted_category']} | {item['count']} |"
        )
    lines.extend(
        [
            "",
            "Full per-class precision, recall, F1, and support are reported in `main_per_class_metrics.csv`; full confusion matrices are in `main_confusion_matrices.json`.",
            "",
            "## Guidance sensitivity",
            "",
        ]
    )
    for row in transition_rows:
        lines.append(
            f"- {row['dataset']} {row['transition']}: improved={row['incorrect_to_correct']}, "
            f"regressed={row['correct_to_incorrect']}, prediction-change={100*row['prediction_change_rate']:.2f}%."
        )
    lines.extend(
        [
            "",
            "### Four-condition prediction patterns",
            "",
            "| Dataset | Pattern | Count | Percent |",
            "|---|---|---:|---:|",
        ]
    )
    for item in sensitivity_rows:
        lines.append(
            f"| {item['dataset']} | {item['pattern']} | {item['count']} | {item['percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Framework analysis",
            "",
            "Framework results are descriptive only; no confirmatory framework-level tests were performed.",
            "",
            "| Dataset | Framework | N | Label-Only | Own | Foreign | Merged |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in framework_rows:
        lines.append(
            f"| {item['dataset']} | {item['framework']} | {item['n']} | "
            f"{100*item['label_only_accuracy']:.2f}% | "
            f"{100*item['own_taxonomy_accuracy']:.2f}% | "
            f"{100*item['foreign_taxonomy_accuracy']:.2f}% | "
            f"{100*item['merged_taxonomy_accuracy']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Execution",
            "",
            f"- Planned logical rows: {len(planned)}",
            f"- Completed logical rows: {len(rows)}",
            f"- Actual API calls: {cost_summary['actual_api_calls']}",
            f"- Formatting retries: {ledger_counts['formatting_retry']}",
            f"- Truncation retries: {ledger_counts['truncation_retry']}",
            f"- Schema/parser retries: {ledger_counts['schema_parser_retry']}",
            f"- Provider/transport retries: {ledger_counts['provider_transport_retry']}",
            f"- Prompt tokens: {cost_summary['prompt_tokens']}",
            f"- Response tokens: {cost_summary['response_tokens']}",
            f"- Total recorded cost: ${cost_summary['total_recorded_cost_usd']:.8f}",
            f"- API ledger reconciled: {cost_summary['ledger_reconciled']}",
            "",
            "## Interpretation",
            "",
        ]
    )
    for dataset in DATASETS:
        base = accuracy_index[(dataset, "label_only")]["accuracy"]
        changes = {
            condition: 100 * (accuracy_index[(dataset, condition)]["accuracy"] - base)
            for condition in ("own_taxonomy", "foreign_taxonomy", "merged_taxonomy")
        }
        significant = sum(
            row["holm_reject_alpha_0_05"]
            for row in contrast_rows
            if row["dataset"] == dataset
        )
        lines.append(
            f"- {dataset}: relative to Label-Only, Own changed accuracy by {changes['own_taxonomy']:.2f} pp, "
            f"Foreign by {changes['foreign_taxonomy']:.2f} pp, and Merged by {changes['merged_taxonomy']:.2f} pp. "
            f"{significant}/4 pre-specified paired contrasts remained significant after Holm adjustment."
        )
    lines.extend(["", "These results answer only how diagnostic taxonomy guidance changed exact native-label attribution accuracy while the native output label space was held constant. Mixed or null effects are retained as experimental results. AEGIS estimates have limited power for small effects; categories with support 1–3 are not used for strong class-specific claims.", ""])
    (run_root / "MAIN_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    write_json(
        run_root / "main_run_status.json",
        {
            "status": "FINAL_MAIN_EXPERIMENT_COMPLETE",
            "planned_logical_rows": len(planned),
            "completed_logical_rows": len(rows),
            "frozen_trajectories": len(records),
            "all_four_conditions_accounted_for": True,
            "task_hashes_unique_within_dataset": True,
            "leakage_count": prepared["audit"]["leakage_count"],
            "api_ledger_reconciled": cost_summary["ledger_reconciled"],
            "actual_api_calls": cost_summary["actual_api_calls"],
            "total_recorded_cost_usd": cost_summary["total_recorded_cost_usd"],
            "additional_experiment_launched": False,
        },
    )
    return {
        "status": "FINAL_MAIN_EXPERIMENT_COMPLETE",
        "results": str(run_root / "MAIN_RESULTS.md"),
        "completed_logical_rows": len(rows),
        "actual_api_calls": cost_summary["actual_api_calls"],
        "total_recorded_cost_usd": cost_summary["total_recorded_cost_usd"],
    }


def run_workflow(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
    analyze_only: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    prepared = preflight(config)
    if preflight_only:
        return {
            "status": prepared["audit"]["status"],
            "api_calls": 0,
            "prediction_implementation_id": config["experiment"]["prediction_implementation_id"],
            "main_sample_id": config["experiment"]["main_sample_id"],
            "planned_logical_rows": prepared["audit"]["planned_logical_rows"],
            "estimated_guarded_cost_usd": prepared["audit"]["prompt_token_estimate"]["guarded_cost_reserve_usd"],
            "preflight_audit": str(resolve(config, "run_root") / "preflight_audit.json"),
        }
    if analyze_only:
        return analyze_results(config, prepared)
    run_predictions(config, prepared)
    return analyze_results(config, prepared)
