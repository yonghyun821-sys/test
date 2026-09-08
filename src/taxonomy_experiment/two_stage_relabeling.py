from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, create_model

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
from taxonomy_experiment.models import (
    NativeRelabelDraft,
    NativeRelabelDraftLongReason,
    NativeRelabelVerification,
    NativeRelabelVerificationLongReason,
)
from taxonomy_experiment.native_evaluation import (
    _accepted_gold_labels,
    _gold_index,
    _native_taxonomies,
    _normalize_native_label,
    _prediction_index,
    _taxonomy_prompt,
    candidate_narrative,
)
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.reporting import DISPLAY_NAMES
from taxonomy_experiment.token_count import TokenCounter


METHOD = "condition_blind_two_stage_native_relabeling_with_self_verification"
BLIND_FIELDS = [
    "trajectory",
    "gold_annotation",
    "experimental_condition",
    "source_taxonomy_id",
    "source_predicted_error_type",
    "predicted_failure_step",
]


def _resolve(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else config["_root"] / path


def _roles(config: dict[str, Any]) -> list[str]:
    roles = list(config["experiment"]["relabeler_roles"])
    if len(roles) != 3 or len(set(roles)) != 3:
        raise ValueError("Exactly three distinct relabeler roles are required")
    missing = [role for role in roles if role not in config["models"]]
    if missing:
        raise ValueError(f"Missing model configurations: {missing}")
    return roles


def _load_predictions(
    prediction_path: str | Path, config: dict[str, Any]
) -> tuple[Path, dict[tuple[str, str, str], dict[str, Any]]]:
    path = _resolve(config, prediction_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    predictions = _prediction_index(path)
    if not predictions:
        raise ValueError("Prediction file is empty")
    non_ok = [key for key, row in predictions.items() if row.get("status") != "ok"]
    if non_ok:
        raise ValueError(f"Prediction file contains {len(non_ok)} non-ok rows")
    run_ids = {row["run_id"] for row in predictions.values()}
    if len(run_ids) != 1:
        raise ValueError(f"Prediction file contains multiple run IDs: {sorted(run_ids)}")
    return path, predictions


def _candidate_json(
    ordered_labels: list[str], initial_label: str, definitions: dict[str, str]
) -> str:
    return json.dumps(
        [
            {
                "candidate_id": candidate_id,
                "name": label,
                "definition": definitions[label],
                "was_stage1_initial_selection": label == initial_label,
            }
            for candidate_id, label in zip(("A", "B", "C"), ordered_labels)
        ],
        ensure_ascii=False,
        indent=2,
    )


def _ordered_candidates(
    *,
    key: tuple[str, str, str],
    role: str,
    labels: list[str],
    seed: int,
) -> list[str]:
    ordered = list(labels)
    digest = hashlib.sha256(
        f"{seed}:{role}:{key[0]}:{key[1]}:{key[2]}".encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(ordered)
    return ordered


def _stage1_prompt(
    template: str, taxonomy: dict[str, Any], prediction: dict[str, Any]
) -> str:
    return format_template(
        template,
        native_taxonomy=_taxonomy_prompt(taxonomy),
        candidate_narrative=json.dumps(
            candidate_narrative(prediction), ensure_ascii=False, indent=2
        ),
    )


def _stage2_prompt(
    template: str,
    prediction: dict[str, Any],
    ordered_labels: list[str],
    initial_label: str,
    definitions: dict[str, str],
) -> str:
    return format_template(
        template,
        candidate_narrative=json.dumps(
            candidate_narrative(prediction), ensure_ascii=False, indent=2
        ),
        candidates=_candidate_json(ordered_labels, initial_label, definitions),
    )


def _normalize_draft(
    draft: NativeRelabelDraft, taxonomy: dict[str, Any]
) -> tuple[str, list[str]]:
    initial, _ = _normalize_native_label(draft.native_label, taxonomy)
    alternatives = [
        _normalize_native_label(label, taxonomy)[0]
        for label in draft.alternative_labels
    ]
    labels = [initial, *alternatives]
    if len({label.casefold() for label in labels}) != 3:
        raise ValueError("Stage 1 did not return three distinct taxonomy labels")
    return initial, alternatives


def _stage1_response_model(model: dict[str, Any]) -> type[NativeRelabelDraft]:
    if model.get("extended_reason_schema"):
        return NativeRelabelDraftLongReason
    return NativeRelabelDraft


def _stage2_response_model(
    model: dict[str, Any],
) -> type[NativeRelabelVerification]:
    if model.get("extended_reason_schema"):
        return NativeRelabelVerificationLongReason
    return NativeRelabelVerification


def _taxonomy_constrained_draft_model(
    taxonomy: dict[str, Any],
    base_model: type[NativeRelabelDraft] = NativeRelabelDraft,
) -> type[NativeRelabelDraft]:
    """Build a strict response schema whose labels are native-taxonomy enums."""
    names = tuple(item["name"] for item in taxonomy["categories"])
    if len(names) < 3:
        raise ValueError("Two-stage relabeling requires at least three categories")
    label_type = Literal.__getitem__(names)
    safe_taxonomy_id = "".join(
        character if character.isalnum() else "_"
        for character in str(taxonomy["taxonomy_id"])
    )
    model = create_model(
        f"NativeRelabelDraft_{safe_taxonomy_id}",
        __base__=base_model,
        native_label=(label_type, Field()),
        alternative_labels=(list[label_type], Field(min_length=2, max_length=2)),
    )
    return cast(type[NativeRelabelDraft], model)


def _repair_prompt(first_prompt: str, draft: NativeRelabelDraft) -> str:
    return (
        first_prompt
        + "\n\n<SCHEMA_CORRECTION>\n"
        + "The previous response used at least one label that was not an exact "
        + "category name in TARGET_NATIVE_TAXONOMY. Re-evaluate the mapping and "
        + "return three distinct labels permitted by the response schema. Do not "
        + "invent, paraphrase, singularize, or pluralize category names.\n"
        + "Previous response: "
        + json.dumps(draft.model_dump(mode="json"), ensure_ascii=False)
        + "\n</SCHEMA_CORRECTION>"
    )


def _longest_candidate_labels(taxonomy: dict[str, Any]) -> list[str]:
    return [
        item["name"]
        for item in sorted(
            taxonomy["categories"],
            key=lambda item: len(item["name"]) + len(item["definition"]),
            reverse=True,
        )[:3]
    ]


def estimate_two_stage_budget(
    prediction_path: str | Path,
    config_path: str | Path = "config/two_stage_relabeler.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    prediction_file, predictions = _load_predictions(prediction_path, config)
    taxonomies = _native_taxonomies(config)
    definitions = {
        dataset: {
            item["name"]: item["definition"]
            for item in taxonomy["categories"]
        }
        for dataset, taxonomy in taxonomies.items()
    }
    stage1_system = read_prompt(config["_root"], "two_stage_relabel_stage1_system.txt")
    stage1_template = read_prompt(config["_root"], "two_stage_relabel_stage1_user.txt")
    stage2_system = read_prompt(config["_root"], "two_stage_relabel_stage2_system.txt")
    stage2_template = read_prompt(config["_root"], "two_stage_relabel_stage2_user.txt")
    safety = float(config["cost_budget"]["input_token_safety_multiplier"])
    model_reports: list[dict[str, Any]] = []
    total_projected = 0.0
    for role in _roles(config):
        model = config["models"][role]
        counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
        stage1_tokens = 0
        stage2_tokens = 0
        stage1_max_single = 0
        stage2_max_single = 0
        for key, prediction in predictions.items():
            taxonomy = taxonomies[key[0]]
            first = _stage1_prompt(stage1_template, taxonomy, prediction)
            first_tokens = counter.count(stage1_system + "\n" + first)
            stage1_tokens += first_tokens
            stage1_max_single = max(stage1_max_single, first_tokens)

            worst_labels = _longest_candidate_labels(taxonomy)
            second = _stage2_prompt(
                stage2_template,
                prediction,
                worst_labels,
                worst_labels[0],
                definitions[key[0]],
            )
            second_tokens = counter.count(stage2_system + "\n" + second)
            stage2_tokens += second_tokens
            stage2_max_single = max(stage2_max_single, second_tokens)

        guarded_stage1 = math.ceil(stage1_tokens * safety)
        guarded_stage2 = math.ceil(stage2_tokens * safety)
        maximum_output = len(predictions) * (
            int(model["stage1_max_output_tokens"])
            + int(model["stage2_max_output_tokens"])
        )
        prices = model["pricing_usd_per_million"]
        projected = (
            (guarded_stage1 + guarded_stage2) * float(prices["input"])
            + maximum_output * float(prices["output"])
        ) / 1_000_000
        total_projected += projected
        context = int(model["context_window"])
        stage1_limit = context - int(model["stage1_max_output_tokens"])
        stage2_limit = context - int(model["stage2_max_output_tokens"])
        model_reports.append(
            {
                "role": role,
                "model": model["name"],
                "stage1_calls": len(predictions),
                "stage2_calls": len(predictions),
                "total_calls": len(predictions) * 2,
                "stage1_input_tokens": stage1_tokens,
                "stage2_worst_case_input_tokens": stage2_tokens,
                "guarded_total_input_tokens": guarded_stage1 + guarded_stage2,
                "maximum_output_tokens": maximum_output,
                "maximum_stage1_input_tokens": stage1_max_single,
                "maximum_stage2_input_tokens": stage2_max_single,
                "within_context": stage1_max_single <= stage1_limit
                and stage2_max_single <= stage2_limit,
                "projected_inference_usd": projected,
            }
        )

    ledger_path = _resolve(config, config["cost_budget"]["ledger_path"])
    incurred = (
        sum(float(row.get("cost_usd", 0.0)) for row in iter_jsonl(ledger_path))
        if ledger_path.exists()
        else 0.0
    )
    cumulative = incurred + total_projected
    fee = max(
        cumulative * float(config["cost_budget"]["credit_purchase_fee_rate"]),
        float(config["cost_budget"]["minimum_credit_purchase_fee_usd"]),
    )
    total_calls = len(predictions) * 2 * len(model_reports)
    report = {
        "status": "ready",
        "method": METHOD,
        "run_id": next(iter(predictions.values()))["run_id"],
        "rows": len(predictions),
        "models": len(model_reports),
        "stage1_calls": len(predictions) * len(model_reports),
        "stage2_calls": len(predictions) * len(model_reports),
        "total_calls": total_calls,
        "fresh_cache_namespace": str(resolve_path(config, "cache").relative_to(config["_root"])),
        "consensus_included": False,
        "model_estimates": model_reports,
        "incurred_inference_usd": incurred,
        "projected_new_inference_usd": total_projected,
        "projected_cumulative_inference_usd": cumulative,
        "projected_fee_inclusive_total_usd": cumulative + fee,
        "within_context": all(item["within_context"] for item in model_reports),
        "within_budget": cumulative <= float(config["cost_budget"]["max_inference_usd"])
        and cumulative + fee <= float(config["cost_budget"]["max_total_charge_usd"]),
        "blind_fields": BLIND_FIELDS,
        "prediction_file": str(prediction_file.relative_to(config["_root"])),
        "prediction_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
    }
    write_json(
        resolve_path(config, "processed_data") / "two_stage_relabeler_budget.json",
        report,
    )
    return report


def _compact(path: Path) -> None:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        latest[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    write_jsonl(path, latest.values())


def _index_if_exists(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        (row["dataset"], row["trajectory_id"], row["condition"]): row
        for row in iter_jsonl(path)
        if row.get("status") == "ok"
    }


def run_two_stage_relabeler(
    prediction_path: str | Path,
    role: str,
    config_path: str | Path = "config/two_stage_relabeler.yaml",
) -> Path:
    config = load_config(config_path)
    if role not in _roles(config):
        raise ValueError(f"Unknown two-stage relabeler role: {role}")
    budget = estimate_two_stage_budget(prediction_path, config_path)
    if not budget["within_context"]:
        raise ValueError("At least one two-stage prompt exceeds its context window")
    if not budget["within_budget"]:
        raise ValueError("Two-stage experiment exceeds the configured budget")

    prediction_file, predictions = _load_predictions(prediction_path, config)
    taxonomies = _native_taxonomies(config)
    definitions = {
        dataset: {
            item["name"]: item["definition"]
            for item in taxonomy["categories"]
        }
        for dataset, taxonomy in taxonomies.items()
    }
    gold = _gold_index(config)
    model = config["models"][role]
    stage1_response_model = _stage1_response_model(model)
    stage2_response_model = _stage2_response_model(model)
    stage1_system = read_prompt(config["_root"], "two_stage_relabel_stage1_system.txt")
    stage1_template = read_prompt(config["_root"], "two_stage_relabel_stage1_user.txt")
    stage2_system = read_prompt(config["_root"], "two_stage_relabel_stage2_system.txt")
    stage2_template = read_prompt(config["_root"], "two_stage_relabel_stage2_user.txt")
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "method": METHOD,
                "role": role,
                "model": model,
                "prompt_version": config["evaluation"]["two_stage_prompt_version"],
                "stage1_system": stage1_system,
                "stage1_template": stage1_template,
                "stage2_system": stage2_system,
                "stage2_template": stage2_template,
                "stage1_schema": stage1_response_model.model_json_schema(),
                "stage2_schema": stage2_response_model.model_json_schema(),
                "prediction_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    run_id = next(iter(predictions.values()))["run_id"]
    evaluator_run_id = f"{run_id}--two-stage-{role}-{fingerprint}"
    results = resolve_path(config, "results") / "two_stage_relabeler"
    stage1_path = results / "stage1" / role / f"{evaluator_run_id}--stage1.jsonl"
    final_path = results / "final" / role / f"{evaluator_run_id}--final.jsonl"
    stage1_rows = _index_if_exists(stage1_path)
    final_rows = _index_if_exists(final_path)
    completed = len(final_rows)
    print(
        f"[two-stage:{role}] starting/resuming final={completed}/{len(predictions)}; "
        f"stage1={len(stage1_rows)}/{len(predictions)}",
        flush=True,
    )
    stage1_llm = CachedLLM(config, resolve_path(config, "cache") / role / "stage1")
    stage2_llm = CachedLLM(config, resolve_path(config, "cache") / role / "stage2")

    for key in sorted(predictions):
        if key in final_rows:
            continue
        prediction = predictions[key]
        taxonomy = taxonomies[key[0]]
        stage1_row = stage1_rows.get(key)
        if stage1_row is None:
            first_prompt = _stage1_prompt(stage1_template, taxonomy, prediction)
            kwargs = {
                "model": model["name"],
                "system_prompt": stage1_system,
                "user_prompt": first_prompt,
                "temperature": model.get("temperature"),
                "reasoning_effort": model.get("reasoning_effort"),
                "seed": int(config["experiment"]["seed"]),
                "provider": model.get("provider"),
                "response_model": stage1_response_model,
            }
            repair_used = False
            original_invalid_response: dict[str, Any] | None = None
            initial_attempt_call: dict[str, Any] | None = None
            try:
                try:
                    draft, call1 = stage1_llm.call(
                        **kwargs,
                        max_output_tokens=int(model["stage1_max_output_tokens"]),
                    )
                except TruncatedResponseError:
                    draft, call1 = stage1_llm.call(
                        **kwargs,
                        max_output_tokens=int(model["stage1_max_output_tokens"]) * 2,
                    )
                try:
                    initial, alternatives = _normalize_draft(draft, taxonomy)
                except ValueError:
                    repair_used = True
                    original_invalid_response = draft.model_dump(mode="json")
                    initial_attempt_call = call1
                    repair_kwargs = {
                        **kwargs,
                        "user_prompt": _repair_prompt(first_prompt, draft),
                        "response_model": _taxonomy_constrained_draft_model(
                            taxonomy, stage1_response_model
                        ),
                    }
                    try:
                        draft, call1 = stage1_llm.call(
                            **repair_kwargs,
                            max_output_tokens=int(model["stage1_max_output_tokens"]),
                        )
                    except TruncatedResponseError:
                        draft, call1 = stage1_llm.call(
                            **repair_kwargs,
                            max_output_tokens=int(model["stage1_max_output_tokens"]) * 2,
                        )
                    initial, alternatives = _normalize_draft(draft, taxonomy)
            except Exception as exc:
                append_jsonl(
                    stage1_path,
                    {
                        "run_id": run_id,
                        "evaluator_run_id": evaluator_run_id,
                        "dataset": key[0],
                        "trajectory_id": key[1],
                        "condition": key[2],
                        "status": "stage1_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
            stage1_row = {
                "run_id": run_id,
                "evaluator_run_id": evaluator_run_id,
                "evaluator_role": role,
                "evaluator_model": model["name"],
                "dataset": key[0],
                "domain": prediction["domain"],
                "trajectory_id": key[1],
                "condition": key[2],
                "status": "ok",
                "native_taxonomy_id": taxonomy["taxonomy_id"],
                "stage1_native_label": initial,
                "stage1_alternative_labels": alternatives,
                "stage1_reason": draft.reason,
                "stage1_repair_used": repair_used,
                "stage1_original_invalid_response": original_invalid_response,
                "stage1_initial_attempt_cache_key": (
                    initial_attempt_call["cache_key"] if initial_attempt_call else None
                ),
                "blind_fields": BLIND_FIELDS,
                "cache_key": call1["cache_key"],
                "cache_path": str(Path(call1["cache_path"]).relative_to(config["_root"])),
                "cache_hit": call1["cache_hit"],
                "response_metadata": call1["response_metadata"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(stage1_path, stage1_row)
            stage1_rows[key] = stage1_row
        initial = stage1_row["stage1_native_label"]
        alternatives = list(stage1_row["stage1_alternative_labels"])
        ordered = _ordered_candidates(
            key=key,
            role=role,
            labels=[initial, *alternatives],
            seed=int(config["experiment"]["seed"]),
        )
        second_prompt = _stage2_prompt(
            stage2_template,
            prediction,
            ordered,
            initial,
            definitions[key[0]],
        )
        kwargs2 = {
            "model": model["name"],
            "system_prompt": stage2_system,
            "user_prompt": second_prompt,
            "temperature": model.get("temperature"),
            "reasoning_effort": model.get("reasoning_effort"),
            "seed": int(config["experiment"]["seed"]),
            "provider": model.get("provider"),
            "response_model": stage2_response_model,
        }
        try:
            try:
                verification, call2 = stage2_llm.call(
                    **kwargs2,
                    max_output_tokens=int(model["stage2_max_output_tokens"]),
                )
            except TruncatedResponseError:
                verification, call2 = stage2_llm.call(
                    **kwargs2,
                    max_output_tokens=int(model["stage2_max_output_tokens"]) * 2,
                )
            selected_index = {"A": 0, "B": 1, "C": 2}[verification.choice]
            final_label = ordered[selected_index]
            accepted = _accepted_gold_labels(gold[key[:2]])
            final_row = {
                "run_id": run_id,
                "evaluator_run_id": evaluator_run_id,
                "evaluator_prompt_version": config["evaluation"]["two_stage_prompt_version"],
                "candidate_narrative_transform": "none",
                "evaluator_role": f"two_stage_{role}",
                "evaluator_model": model["name"],
                "dataset": key[0],
                "domain": prediction["domain"],
                "trajectory_id": key[1],
                "condition": key[2],
                "status": "ok",
                "native_taxonomy_id": taxonomy["taxonomy_id"],
                "native_label": final_label,
                "raw_native_label": None,
                "mapping_reason": verification.reason,
                "accepted_gold_labels": accepted,
                "correct": any(
                    final_label.casefold() == label.casefold() for label in accepted
                ),
                "blind_fields": BLIND_FIELDS,
                "masked_native_terms": [],
                "stage1_native_label": initial,
                "stage1_alternative_labels": alternatives,
                "stage1_reason": stage1_row["stage1_reason"],
                "stage2_candidate_order": {
                    candidate_id: label
                    for candidate_id, label in zip(("A", "B", "C"), ordered)
                },
                "stage2_choice": verification.choice,
                "stage2_revised": final_label != initial,
                "cache_key": call2["cache_key"],
                "cache_path": str(Path(call2["cache_path"]).relative_to(config["_root"])),
                "cache_hit": call2["cache_hit"],
                "response_metadata": call2["response_metadata"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            append_jsonl(
                final_path,
                {
                    "run_id": run_id,
                    "evaluator_run_id": evaluator_run_id,
                    "dataset": key[0],
                    "trajectory_id": key[1],
                    "condition": key[2],
                    "status": "stage2_error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        append_jsonl(final_path, final_row)
        final_rows[key] = final_row
        completed += 1
        if completed % 10 == 0 or completed == len(predictions):
            print(
                f"[two-stage:{role}] final={completed}/{len(predictions)} "
                f"dataset={key[0]} condition={key[2]}",
                flush=True,
            )
    _compact(stage1_path)
    _compact(final_path)
    return final_path


def generate_two_stage_comparison(
    final_paths: dict[str, str | Path],
    config_path: str | Path = "config/two_stage_relabeler.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
    roles = _roles(config)
    if set(final_paths) != set(roles):
        raise ValueError("Final paths must contain exactly the three configured roles")
    indexed: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for role in roles:
        path = _resolve(config, final_paths[role])
        rows = {
            (row["dataset"], row["trajectory_id"], row["condition"]): row
            for row in iter_jsonl(path)
        }
        if any(row.get("status") != "ok" for row in rows.values()):
            raise ValueError(f"{role} final evaluation contains non-ok rows")
        indexed[role] = rows
    key_sets = {frozenset(rows) for rows in indexed.values()}
    if len(key_sets) != 1:
        raise ValueError("Three two-stage final evaluations have different keys")
    keys = next(iter(key_sets))
    run_ids = {
        row["run_id"] for rows in indexed.values() for row in rows.values()
    }
    if len(run_ids) != 1:
        raise ValueError(f"Final evaluation run IDs do not match: {sorted(run_ids)}")
    run_id = next(iter(run_ids))

    summary_rows: list[dict[str, Any]] = []
    for dataset in experiment_datasets(config):
        for condition in (
            "no_taxonomy",
            "own_taxonomy",
            "foreign_taxonomy",
            "merged_taxonomy",
        ):
            cell_keys = sorted(
                key for key in keys if key[0] == dataset and key[2] == condition
            )
            row: dict[str, Any] = {
                "dataset": dataset,
                "condition": condition,
                "n": len(cell_keys),
            }
            for role in roles:
                values = [indexed[role][key] for key in cell_keys]
                correct = sum(bool(value["correct"]) for value in values)
                revised = sum(bool(value["stage2_revised"]) for value in values)
                row[f"{role}_correct"] = correct
                row[f"{role}_accuracy"] = correct / len(values)
                row[f"{role}_revised"] = revised
                row[f"{role}_revision_rate"] = revised / len(values)
            for left_index, left in enumerate(roles):
                for right in roles[left_index + 1 :]:
                    agreement = sum(
                        indexed[left][key]["native_label"].casefold()
                        == indexed[right][key]["native_label"].casefold()
                        for key in cell_keys
                    )
                    row[f"{left}_vs_{right}_label_agreement"] = agreement / len(cell_keys)
            summary_rows.append(row)

    output_dir = resolve_path(config, "results") / "two_stage_relabeler"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_three_model_two_stage_comparison"
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    write_json(
        json_path,
        {"run_id": run_id, "method": METHOD, "results": summary_rows},
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# Three-Model Two-Stage Relabeler Comparison",
        "",
        f"Run: `{run_id}`",
        "",
        "| Dataset | Condition | N | GPT-5 Nano accuracy / revision | Gemini 2.5 Flash Lite accuracy / revision | GPT-4.1-mini accuracy / revision |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        cells = []
        for role in roles:
            cells.append(
                f"{row[f'{role}_accuracy']:.2%} / {row[f'{role}_revision_rate']:.2%}"
            )
        lines.append(
            f"| {row['dataset']} | {DISPLAY_NAMES[row['condition']]} | {row['n']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "Each cell reports final native-label accuracy followed by the percentage of stage-1 labels changed during stage 2.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"markdown": markdown_path, "json": json_path, "csv": csv_path}
