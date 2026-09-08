from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.token_count import TokenCounter

from experiments.aegis_whowhen_final_main_attribution import core as final_harness
from experiments.aegis_whowhen_main_fixed_guidance import core as prediction_impl
from experiments.aegis_whowhen_main_fixed_guidance.audits.task_independence import (
    verify_frozen_identity,
)
from experiments.aegis_whowhen_pilot_fixed_guidance import core as merge_impl


HERE = Path(__file__).resolve().parent
DATASETS = ("aegis", "whowhen")
CONDITIONS = (
    "label_only",
    "own_taxonomy",
    "foreign_taxonomy",
    "merged_taxonomy",
)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, key)


def merge_ledger_path(config: dict[str, Any]) -> Path:
    path = Path(config["cost_budget"]["ledger_path"])
    return path if path.is_absolute() else config["_root"] / path


def freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Frozen artifact differs: {path}")
        return
    write_json(path, value)


def source_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = resolve(config, "frozen_implementation_root")
    return {
        "aegis": read_json(root / "aegis_taxonomy.json"),
        "whowhen": read_json(root / "whowhen_taxonomy.json"),
    }


def merge_source_payload(
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        dataset: {
            "taxonomy_id": taxonomy["taxonomy_id"],
            "name": taxonomy["name"],
            "categories": [
                {
                    "id": category["id"],
                    "name": category["name"],
                    "definition": category["definition"],
                    "module": category["module"],
                }
                for category in taxonomy["categories"]
            ],
        }
        for dataset, taxonomy in taxonomies.items()
    }


def ledger_summary(path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(path)) if path.exists() else []
    actual = [row for row in rows if row.get("actual_request", True)]
    return {
        "rows": rows,
        "actual_api_calls": len(actual),
        "cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in actual),
        "prompt_tokens": sum(
            int((row.get("reported_usage") or {}).get("prompt_tokens") or 0)
            for row in actual
        ),
        "response_tokens": sum(
            int((row.get("reported_usage") or {}).get("completion_tokens") or 0)
            for row in actual
        ),
    }


def attribution_config(config: dict[str, Any]) -> dict[str, Any]:
    base = load_config(resolve(config, "frozen_prediction_config"))
    base["model"] = copy.deepcopy(config["attribution_model"])
    base["experiment"]["main_seed"] = int(config["experiment"]["main_seed"])
    base["api"]["api_key_envs"] = list(config["execution"]["api_key_envs"])
    base["api"]["app_title"] = config["api"]["app_title"]
    return base


def offline_preflight(config: dict[str, Any]) -> dict[str, Any]:
    identity = verify_frozen_identity()
    records, sample_manifest = final_harness.load_frozen_sample(config)
    sources = source_taxonomies(config)
    observed_sample_hashes = final_harness._sample_hashes(config, records)
    expected = config["expected_hashes"]
    counts = {
        dataset: sum(row["dataset"] == dataset for row in records)
        for dataset in DATASETS
    }
    task_sets = {
        dataset: {
            row["task_hash"] for row in records if row["dataset"] == dataset
        }
        for dataset in DATASETS
    }
    categories = {
        dataset: {
            row["gold_category"] for row in records if row["dataset"] == dataset
        }
        for dataset in DATASETS
    }
    leakage = prediction_impl.leakage_audit(records)
    prior_status = read_json(resolve(config, "prior_final_status"))
    prior_status_hash = sha256_file(resolve(config, "prior_final_status"))
    checks = {
        "base_implementation_verified": identity["passed"],
        "sample_id": sample_manifest["main_sample_id"]
        == config["experiment"]["main_sample_id"],
        "sample_hashes": observed_sample_hashes
        == {
            "aegis_sample": expected["aegis_sample"],
            "whowhen_sample": expected["whowhen_sample"],
            "combined_main_input": expected["combined_main_input"],
            "combined_gold": expected["combined_gold"],
        },
        "dataset_counts": counts
        == {
            "aegis": int(config["experiment"]["expected_aegis_n"]),
            "whowhen": int(config["experiment"]["expected_whowhen_n"]),
        },
        "total_count": len(records)
        == int(config["experiment"]["expected_total_n"]),
        "task_independence": all(
            len(task_sets[dataset]) == counts[dataset] for dataset in DATASETS
        ),
        "cross_dataset_overlap_zero": not (task_sets["aegis"] & task_sets["whowhen"]),
        "all_14_classes": all(len(categories[dataset]) == 14 for dataset in DATASETS),
        "pilot_overlap_zero": sample_manifest["validation"][
            "pilot_task_hash_overlap_zero"
        ],
        "smoke_overlap_zero": sample_manifest["validation"][
            "smoke_task_hash_overlap_zero"
        ],
        "leakage_zero": leakage["leakage_count"] == 0,
        "aegis_taxonomy_hash": sha256_json(sources["aegis"])
        == expected["aegis_taxonomy"],
        "whowhen_taxonomy_hash": sha256_json(sources["whowhen"])
        == expected["whowhen_taxonomy"],
        "conditions": tuple(config["experiment"]["conditions"]) == CONDITIONS,
        "planned_rows": len(records) * len(CONDITIONS)
        == int(config["experiment"]["expected_prediction_rows"]),
        "merge_model": config["models"]["merge"]["name"] == "openai/gpt-5",
        "attribution_model": config["attribution_model"]["name"]
        == "google/gemini-2.5-flash",
        "attribution_temperature_zero": float(
            config["attribution_model"]["temperature"]
        )
        == 0,
        "prior_final_complete_and_preserved": prior_status["status"]
        == "FINAL_MAIN_EXPERIMENT_COMPLETE",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"STOP_BEFORE_API: offline preflight failed: {failed}")

    merge_model = config["models"]["merge"]
    payload = merge_source_payload(sources)
    merge_system = merge_impl._read_prompt("merge_system.txt").strip()
    merge_user = merge_impl._read_prompt("merge_user.txt").format(
        source_taxonomies=merge_impl._json_text(payload), repair_context=""
    )
    merge_tokens = TokenCounter.for_model(
        merge_model["name"], merge_model.get("tokenizer_encoding")
    ).count(merge_system + "\n" + merge_user)
    merge_guarded = int(
        merge_tokens
        * float(config["cost_budget"]["input_token_safety_multiplier"])
        + 0.999999
    )
    merge_price = merge_model["pricing_usd_per_million"]
    merge_one_call_reserve = (
        merge_guarded * float(merge_price["input"])
        + int(merge_model["max_output_tokens"]) * float(merge_price["output"])
    ) / 1_000_000

    proxy_taxonomies = {
        **sources,
        "merged": read_json(resolve(config, "frozen_implementation_root") / "merged_taxonomy.json"),
    }
    attr = attribution_config(config)
    counter = TokenCounter.for_model(
        attr["model"]["name"], attr["model"].get("tokenizer_encoding")
    )
    proxy_prompt_tokens = []
    for record in records:
        for condition in CONDITIONS:
            built = prediction_impl.build_prompt(record, condition, proxy_taxonomies)
            proxy_prompt_tokens.append(
                counter.count(built["system_prompt"] + "\n" + built["user_prompt"])
            )
    attr_price = attr["model"]["pricing_usd_per_million"]
    attr_guarded = sum(proxy_prompt_tokens) * float(
        attr["budget"]["input_token_safety_multiplier"]
    )
    attr_proxy_reserve = (
        attr_guarded * float(attr_price["input"])
        + len(proxy_prompt_tokens)
        * int(attr["model"]["max_output_tokens"])
        * float(attr_price["output"])
    ) / 1_000_000
    audit = {
        "status": "READY_FOR_COMBINED_PIPELINE",
        "api_calls": 0,
        "checks": checks,
        "sample_id": config["experiment"]["main_sample_id"],
        "sample_hashes": observed_sample_hashes,
        "dataset_counts": counts,
        "total_trajectories": len(records),
        "planned_logical_rows": len(records) * len(CONDITIONS),
        "models": {
            "taxonomy_merge": merge_model["name"],
            "attribution": attr["model"]["name"],
        },
        "cost_estimate": {
            "gpt5_merge_one_call_reserve_usd": merge_one_call_reserve,
            "gemini25flash_attribution_proxy_reserve_usd": attr_proxy_reserve,
            "provisional_total_usd": merge_one_call_reserve + attr_proxy_reserve,
            "hard_cap_usd": float(config["execution"]["hard_cost_cap_usd"]),
            "note": "Attribution proxy uses the old merge only to estimate prompt length; an exact gate is recomputed after the GPT-5 merge is frozen.",
        },
        "prior_final_status_hash_before": prior_status_hash,
    }
    artifacts = resolve(config, "artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    write_json(HERE / "OFFLINE_PREFLIGHT.json", audit)
    return {
        "audit": audit,
        "records": records,
        "sources": sources,
        "attribution_config": attr,
        "prior_status_hash": prior_status_hash,
    }


def load_or_generate_merge(
    config: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    *,
    allow_generation: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_path = resolve(config, "artifacts") / "pilot_merged_taxonomy.json"
    if not allow_generation and not artifact_path.exists():
        raise RuntimeError("No GPT-5 merge exists; analyze-only cannot generate it")
    merged, audit = merge_impl.obtain_merged_taxonomy(
        config, sources, allow_generation=allow_generation
    )
    if merged is None:
        raise RuntimeError("GPT-5 merge was not generated and validated")
    if merged.get("generated_by", {}).get("model") != "openai/gpt-5":
        raise RuntimeError("Refusing a merged taxonomy generated by a non-GPT-5 model")
    merge_impl._validate_merged_taxonomy(merged, sources)
    return merged, audit


def derive_implementation_id(
    config: dict[str, Any], merged: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    base_manifest = read_json(
        resolve(config, "frozen_implementation_root") / "implementation_manifest.json"
    )
    identity_payload = {
        "pipeline_harness_hashes": {
            "core.py": sha256_file(HERE / "core.py"),
            "run.py": sha256_file(HERE / "run.py"),
            "experiment.yaml": sha256_file(HERE / "experiment.yaml"),
        },
        "base_code_hash": base_manifest["code_hash"],
        "base_parser_hash": base_manifest["parser_hash"],
        "base_serializer_hash": base_manifest["serializer_hash"],
        "base_prompt_hashes": base_manifest["prompt_hashes"],
        "base_schema_hashes": base_manifest["schema_hashes"],
        "source_taxonomy_hashes": {
            "aegis": config["expected_hashes"]["aegis_taxonomy"],
            "whowhen": config["expected_hashes"]["whowhen_taxonomy"],
        },
        "merged_taxonomy_hash": sha256_json(merged),
        "merge_model": config["models"]["merge"],
        "attribution_model": config["attribution_model"],
        "conditions": list(CONDITIONS),
        "scoring": "exact_native_id_only_no_relabeler",
    }
    implementation_id = f"main-{sha256_json(identity_payload)[:16]}"
    return implementation_id, identity_payload


def exact_attribution_reserve(
    config: dict[str, Any],
    prepared: dict[str, Any],
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attr = prepared["attribution_config"]
    counter = TokenCounter.for_model(
        attr["model"]["name"], attr["model"].get("tokenizer_encoding")
    )
    prompt_tokens = []
    for record in prepared["records"]:
        for condition in CONDITIONS:
            built = prediction_impl.build_prompt(record, condition, taxonomies)
            prompt_tokens.append(
                counter.count(built["system_prompt"] + "\n" + built["user_prompt"])
            )
    maximum = max(prompt_tokens)
    if maximum + int(attr["model"]["max_output_tokens"]) >= int(
        attr["model"]["context_window"]
    ):
        raise RuntimeError("STOP_BEFORE_ATTRIBUTION: context window exceeded")
    prices = attr["model"]["pricing_usd_per_million"]
    guarded_input = sum(prompt_tokens) * float(
        attr["budget"]["input_token_safety_multiplier"]
    )
    reserve = (
        guarded_input * float(prices["input"])
        + len(prompt_tokens)
        * int(attr["model"]["max_output_tokens"])
        * float(prices["output"])
    ) / 1_000_000
    return {
        "logical_rows": len(prompt_tokens),
        "total_prompt_tokens": sum(prompt_tokens),
        "maximum_prompt_tokens": maximum,
        "guarded_cost_reserve_usd": reserve,
    }


def prepare_run(
    config: dict[str, Any],
    prepared: dict[str, Any],
    merged: dict[str, Any],
    merge_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation_id, identity_payload = derive_implementation_id(config, merged)
    namespace = (
        f"{implementation_id}--{config['experiment']['main_sample_id']}"
    )
    run_root = resolve(config, "run_parent") / namespace
    run_root.mkdir(parents=True, exist_ok=True)
    taxonomies = {**prepared["sources"], "merged": merged}
    reserve = exact_attribution_reserve(config, prepared, taxonomies)
    merge_ledger = ledger_summary(merge_ledger_path(config))
    hard_cap = float(config["execution"]["hard_cost_cap_usd"])
    if merge_ledger["cost_usd"] + reserve["guarded_cost_reserve_usd"] >= hard_cap:
        raise RuntimeError(
            "STOP_BEFORE_ATTRIBUTION: merge spend plus attribution reserve exceeds $50"
        )
    if sha256_file(resolve(config, "prior_final_status")) != prepared["prior_status_hash"]:
        raise RuntimeError("Prior final experiment status changed during this pipeline")

    runtime_config = copy.deepcopy(config)
    runtime_config["experiment"]["prediction_implementation_id"] = implementation_id
    runtime_config["experiment"]["run_namespace"] = namespace
    runtime_config["paths"]["run_root"] = str(run_root)
    runtime_config["execution"]["hard_cost_cap_usd"] = (
        hard_cap - merge_ledger["cost_usd"]
    )
    attr_config = copy.deepcopy(prepared["attribution_config"])
    planned = prediction_impl.planned_keys(
        prepared["records"], int(config["experiment"]["main_seed"])
    )
    prediction_prepared = {
        "audit": {
            **prepared["audit"],
            "leakage_count": 0,
        },
        "records": prepared["records"],
        "taxonomies": taxonomies,
        "frozen_config": attr_config,
        "planned_keys": planned,
    }
    manifest = {
        "research_question": config["experiment"]["research_question"],
        "prediction_implementation_id": implementation_id,
        "main_sample_id": config["experiment"]["main_sample_id"],
        "run_namespace": namespace,
        "base_prediction_implementation_id": "main-95e2d9ee2ba7c1fd",
        "identity_payload": identity_payload,
        "merge_model": config["models"]["merge"],
        "attribution_model": config["attribution_model"],
        "merged_taxonomy_sha256": sha256_json(merged),
        "sample_hashes": prepared["audit"]["sample_hashes"],
        "planned_trajectories": len(prepared["records"]),
        "planned_logical_rows": len(planned),
        "planned_key_hash": sha256_json(planned),
        "attribution_cost_preflight": reserve,
        "merge_cost_before_attribution": {
            key: value for key, value in merge_ledger.items() if key != "rows"
        },
        "hard_cost_cap_usd": hard_cap,
        "previous_experiment_outputs_imported": False,
    }
    freeze_json(run_root / "main_run_manifest.json", manifest)
    freeze_json(run_root / "merged_taxonomy.json", merged)
    freeze_json(run_root / "merge_audit.json", merge_audit)
    write_json(
        run_root / "preflight_audit.json",
        {
            "status": "READY_FOR_GEMINI_2_5_FLASH_ATTRIBUTION",
            "merge_validated_and_frozen": True,
            "attribution_cost_preflight": reserve,
            "merge_recorded_cost_usd": merge_ledger["cost_usd"],
            "remaining_hard_cap_usd": runtime_config["execution"][
                "hard_cost_cap_usd"
            ],
        },
    )
    return runtime_config, prediction_prepared


def finalize_pipeline_accounting(
    config: dict[str, Any],
    runtime_config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    run_root = Path(runtime_config["paths"]["run_root"])
    merge = ledger_summary(merge_ledger_path(config))
    attribution = ledger_summary(run_root / "main_api_call_ledger.jsonl")
    total_calls = merge["actual_api_calls"] + attribution["actual_api_calls"]
    total_cost = merge["cost_usd"] + attribution["cost_usd"]
    hard_cap = float(config["execution"]["hard_cost_cap_usd"])
    if total_cost > hard_cap:
        raise RuntimeError("Combined pipeline cost exceeded its hard cap")

    combined_rows = [
        {"pipeline_stage": "gpt5_taxonomy_merge", "source_row": index, **row}
        for index, row in enumerate(merge["rows"], start=1)
    ]
    combined_rows.extend(
        {
            "pipeline_stage": "gemini25flash_attribution",
            "source_row": index,
            **row,
        }
        for index, row in enumerate(attribution["rows"], start=1)
    )
    write_jsonl(run_root / "pipeline_api_call_ledger.jsonl", combined_rows)
    combined = {
        "status": "GPT5_MERGE_GEMINI25FLASH_MAIN_COMPLETE",
        "merge_model": "openai/gpt-5",
        "attribution_model": "google/gemini-2.5-flash",
        "merge_actual_api_calls": merge["actual_api_calls"],
        "attribution_actual_api_calls": attribution["actual_api_calls"],
        "total_actual_api_calls": total_calls,
        "merge_prompt_tokens": merge["prompt_tokens"],
        "merge_response_tokens": merge["response_tokens"],
        "attribution_prompt_tokens": attribution["prompt_tokens"],
        "attribution_response_tokens": attribution["response_tokens"],
        "merge_cost_usd": merge["cost_usd"],
        "attribution_cost_usd": attribution["cost_usd"],
        "total_recorded_cost_usd": total_cost,
        "hard_cost_cap_usd": hard_cap,
        "ledger_reconciled": len(combined_rows)
        == len(merge["rows"]) + len(attribution["rows"]),
    }
    write_json(run_root / "pipeline_cost_summary.json", combined)

    main_cost_path = run_root / "main_cost_summary.json"
    main_cost = read_json(main_cost_path)
    main_cost["attribution_actual_api_calls"] = main_cost["actual_api_calls"]
    main_cost["merge_actual_api_calls"] = merge["actual_api_calls"]
    main_cost["actual_api_calls"] = total_calls
    main_cost["attribution_recorded_cost_usd"] = main_cost[
        "total_recorded_cost_usd"
    ]
    main_cost["merge_recorded_cost_usd"] = merge["cost_usd"]
    main_cost["total_recorded_cost_usd"] = total_cost
    main_cost["hard_cost_cap_usd"] = hard_cap
    main_cost["pipeline_ledger_reconciled"] = combined["ledger_reconciled"]
    write_json(main_cost_path, main_cost)

    status_path = run_root / "main_run_status.json"
    status = read_json(status_path)
    status.update(
        {
            "status": "FINAL_MAIN_EXPERIMENT_COMPLETE",
            "pipeline_status": "GPT5_MERGE_GEMINI25FLASH_MAIN_COMPLETE",
            "merge_actual_api_calls": merge["actual_api_calls"],
            "attribution_actual_api_calls": attribution["actual_api_calls"],
            "actual_api_calls": total_calls,
            "merge_cost_usd": merge["cost_usd"],
            "attribution_cost_usd": attribution["cost_usd"],
            "total_recorded_cost_usd": total_cost,
            "prior_completed_experiment_modified": False,
        }
    )
    write_json(status_path, status)

    report_path = run_root / "MAIN_RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "- Attribution model: `google/gemini-2.5-flash`; temperature=0",
        "- Merge model: `openai/gpt-5`; reasoning effort=medium\n"
        "- Attribution model: `google/gemini-2.5-flash`; temperature=0",
    )
    report = report.replace(
        f"- Actual API calls: {attribution['actual_api_calls']}",
        f"- Actual API calls, combined: {total_calls}\n"
        f"- GPT-5 merge API calls: {merge['actual_api_calls']}\n"
        f"- Gemini 2.5 Flash attribution API calls: {attribution['actual_api_calls']}",
    )
    report = report.replace(
        f"- Total recorded cost: ${attribution['cost_usd']:.8f}",
        f"- GPT-5 merge cost: ${merge['cost_usd']:.8f}\n"
        f"- Gemini 2.5 Flash attribution cost: ${attribution['cost_usd']:.8f}\n"
        f"- Total recorded pipeline cost: ${total_cost:.8f}",
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")
    return {
        **result,
        "status": "GPT5_MERGE_GEMINI25FLASH_MAIN_COMPLETE",
        "run_directory": str(run_root),
        "actual_api_calls": total_calls,
        "total_recorded_cost_usd": total_cost,
    }


def run_workflow(
    config_path: str | Path,
    *,
    preflight_only: bool = False,
    analyze_only: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    prepared = offline_preflight(config)
    if preflight_only:
        return prepared["audit"]

    merged, merge_audit = load_or_generate_merge(
        config,
        prepared["sources"],
        allow_generation=not analyze_only,
    )
    runtime_config, prediction_prepared = prepare_run(
        config, prepared, merged, merge_audit
    )
    if analyze_only:
        result = final_harness.analyze_results(runtime_config, prediction_prepared)
    else:
        final_harness.run_predictions(runtime_config, prediction_prepared)
        result = final_harness.analyze_results(runtime_config, prediction_prepared)
    return finalize_pipeline_accounting(config, runtime_config, result)
