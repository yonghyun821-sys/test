from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
from taxonomy_experiment.models import CandidateAdjudicationResult
from taxonomy_experiment.native_evaluation import candidate_narrative
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.reporting import DISPLAY_NAMES
from taxonomy_experiment.token_count import TokenCounter


ADJUDICATION_METHOD = "two_relabeler_agreement_then_anonymous_disagreement_adjudication"


def _resolve(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else config["_root"] / path


def _index(path: Path, *, kind: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (row["dataset"], row["trajectory_id"], row["condition"])
        if key in result:
            raise ValueError(f"Duplicate {kind} key: {key}")
        result[key] = row
    return result


def _same_label(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


def _gold_index(config: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    processed = resolve_path(config, "processed_data")
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_gold.jsonl"):
            result[(dataset, row["trajectory_id"])] = list(
                row.get("failure_types") or [row["failure_type"]]
            )
    return result


def _native_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        dataset: read_json(resolve_path(config, path_key))
        for dataset, path_key in zip(
            experiment_datasets(config), ("taxonomy_a", "taxonomy_b")
        )
    }


def _definition_index(
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        dataset: {item["name"]: item["definition"] for item in taxonomy["categories"]}
        for dataset, taxonomy in taxonomies.items()
    }


def _candidate_order(
    key: tuple[str, str, str],
    primary_label: str,
    alternate_label: str,
    seed: int,
) -> tuple[tuple[str, str], tuple[str, str]]:
    digest = hashlib.sha256(
        f"{seed}:{key[0]}:{key[1]}:{key[2]}".encode("utf-8")
    ).digest()
    primary = ("primary", primary_label)
    alternate = ("alternate", alternate_label)
    return (primary, alternate) if digest[0] % 2 == 0 else (alternate, primary)


def _prompt(
    template: str,
    prediction: dict[str, Any],
    candidate_a: tuple[str, str],
    candidate_b: tuple[str, str],
    definitions: dict[str, str],
) -> str:
    return format_template(
        template,
        candidate_narrative=json.dumps(
            candidate_narrative(prediction), ensure_ascii=False, indent=2
        ),
        candidate_a_name=candidate_a[1],
        candidate_a_definition=definitions[candidate_a[1]],
        candidate_b_name=candidate_b[1],
        candidate_b_definition=definitions[candidate_b[1]],
    )


def _load_inputs(
    prediction_path: str | Path,
    primary_evaluation_path: str | Path,
    alternate_evaluation_path: str | Path,
    config: dict[str, Any],
) -> tuple[
    Path,
    Path,
    Path,
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    prediction_file = _resolve(config, prediction_path)
    primary_file = _resolve(config, primary_evaluation_path)
    alternate_file = _resolve(config, alternate_evaluation_path)
    predictions = _index(prediction_file, kind="prediction")
    primary = _index(primary_file, kind="primary evaluation")
    alternate = _index(alternate_file, kind="alternate evaluation")
    if not predictions:
        raise ValueError("Prediction file is empty")
    if set(predictions) != set(primary) or set(predictions) != set(alternate):
        raise ValueError("Prediction and relabeler evaluation keys do not match")
    non_ok = [
        key
        for key in predictions
        if predictions[key].get("status") != "ok"
        or primary[key].get("status") != "ok"
        or alternate[key].get("status") != "ok"
    ]
    if non_ok:
        raise ValueError(f"Inputs contain {len(non_ok)} non-ok experiment keys")
    run_ids = {row["run_id"] for row in predictions.values()}
    run_ids.update(row["run_id"] for row in primary.values())
    run_ids.update(row["run_id"] for row in alternate.values())
    if len(run_ids) != 1:
        raise ValueError(f"Input run IDs do not match: {sorted(run_ids)}")
    return (
        prediction_file,
        primary_file,
        alternate_file,
        predictions,
        primary,
        alternate,
    )


def estimate_adjudication_budget(
    prediction_path: str | Path,
    primary_evaluation_path: str | Path,
    alternate_evaluation_path: str | Path,
    config_path: str | Path = "config/relabeler_adjudication.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    (
        prediction_file,
        primary_file,
        alternate_file,
        predictions,
        primary,
        alternate,
    ) = _load_inputs(
        prediction_path,
        primary_evaluation_path,
        alternate_evaluation_path,
        config,
    )
    system = read_prompt(config["_root"], "native_label_adjudicator_system.txt")
    template = read_prompt(config["_root"], "native_label_adjudicator_user.txt")
    taxonomies = _native_taxonomies(config)
    definitions = _definition_index(taxonomies)
    model = config["models"]["adjudicator"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    disagreement_keys: list[tuple[str, str, str]] = []
    input_tokens = 0
    maximum_single_input = 0
    counts: Counter[str] = Counter()
    for key in sorted(predictions):
        primary_label = primary[key]["native_label"]
        alternate_label = alternate[key]["native_label"]
        if _same_label(primary_label, alternate_label):
            continue
        disagreement_keys.append(key)
        counts[f"{key[0]}:{key[2]}"] += 1
        candidate_a, candidate_b = _candidate_order(
            key,
            primary_label,
            alternate_label,
            int(config["experiment"]["seed"]),
        )
        user_prompt = _prompt(
            template,
            predictions[key],
            candidate_a,
            candidate_b,
            definitions[key[0]],
        )
        tokens = counter.count(system + "\n" + user_prompt)
        input_tokens += tokens
        maximum_single_input = max(maximum_single_input, tokens)
    calls = len(disagreement_keys)
    safety = float(config["cost_budget"]["input_token_safety_multiplier"])
    guarded_input = math.ceil(input_tokens * safety)
    maximum_output = calls * int(model["max_output_tokens"])
    prices = model["pricing_usd_per_million"]
    projected_new = (
        guarded_input * float(prices["input"])
        + maximum_output * float(prices["output"])
    ) / 1_000_000
    ledger_path = _resolve(config, config["cost_budget"]["ledger_path"])
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
    context_limit = int(model["context_window"]) - int(model["max_output_tokens"])
    report = {
        "method": ADJUDICATION_METHOD,
        "run_id": next(iter(predictions.values()))["run_id"],
        "primary_evaluator_models": sorted(
            {row.get("evaluator_model") for row in primary.values()}
        ),
        "alternate_evaluator_models": sorted(
            {row.get("evaluator_model") for row in alternate.values()}
        ),
        "adjudicator_model": model["name"],
        "total_rows": len(predictions),
        "agreement_rows": len(predictions) - calls,
        "disagreement_rows": calls,
        "disagreement_rate": calls / len(predictions),
        "disagreements_by_dataset_condition": dict(sorted(counts.items())),
        "calls": calls,
        "input_tokens": input_tokens,
        "guarded_input_tokens": guarded_input,
        "maximum_output_tokens": maximum_output,
        "maximum_single_input_tokens": maximum_single_input,
        "context_limit": context_limit,
        "within_context": maximum_single_input <= context_limit,
        "incurred_inference_usd": incurred,
        "projected_new_inference_usd": projected_new,
        "projected_cumulative_inference_usd": cumulative,
        "projected_fee_inclusive_total_usd": cumulative + fee,
        "within_budget": cumulative <= float(config["cost_budget"]["max_inference_usd"])
        and cumulative + fee <= float(config["cost_budget"]["max_total_charge_usd"]),
        "candidate_order_policy": "deterministic SHA-256 randomization; model identities hidden",
        "adjudicator_visible_fields": [
            "candidate_narrative.explanation",
            "candidate_narrative.evidence",
            "candidate_narrative.overall_root_cause",
            "anonymous_candidate_A.name_and_definition",
            "anonymous_candidate_B.name_and_definition",
        ],
        "adjudicator_hidden_fields": [
            "trajectory",
            "gold_annotation",
            "experimental_condition",
            "source_taxonomy_id",
            "relabeler_model_identities",
            "predicted_failure_step",
        ],
        "input_files": {
            "predictions": str(prediction_file.relative_to(config["_root"])),
            "primary_evaluation": str(primary_file.relative_to(config["_root"])),
            "alternate_evaluation": str(alternate_file.relative_to(config["_root"])),
        },
        "input_sha256": {
            "predictions": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
            "primary_evaluation": hashlib.sha256(primary_file.read_bytes()).hexdigest(),
            "alternate_evaluation": hashlib.sha256(alternate_file.read_bytes()).hexdigest(),
        },
    }
    write_json(
        resolve_path(config, "processed_data") / "relabeler_adjudication_budget.json",
        report,
    )
    return report


def _compact(path: Path) -> None:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        latest[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    write_jsonl(path, latest.values())


def run_disagreement_adjudication(
    prediction_path: str | Path,
    primary_evaluation_path: str | Path,
    alternate_evaluation_path: str | Path,
    config_path: str | Path = "config/relabeler_adjudication.yaml",
) -> Path:
    config = load_config(config_path)
    budget = estimate_adjudication_budget(
        prediction_path,
        primary_evaluation_path,
        alternate_evaluation_path,
        config_path,
    )
    if not budget["within_context"]:
        raise ValueError("At least one adjudicator prompt exceeds its context limit")
    if not budget["within_budget"]:
        raise ValueError("Disagreement adjudication exceeds the configured budget")
    (
        _,
        _,
        _,
        predictions,
        primary,
        alternate,
    ) = _load_inputs(
        prediction_path,
        primary_evaluation_path,
        alternate_evaluation_path,
        config,
    )
    gold = _gold_index(config)
    taxonomies = _native_taxonomies(config)
    definitions = _definition_index(taxonomies)
    system = read_prompt(config["_root"], "native_label_adjudicator_system.txt")
    template = read_prompt(config["_root"], "native_label_adjudicator_user.txt")
    model = config["models"]["adjudicator"]
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "method": ADJUDICATION_METHOD,
                "model": model,
                "prompt_version": config["evaluation"]["adjudicator_prompt_version"],
                "system": system,
                "template": template,
                "response_schema": CandidateAdjudicationResult.model_json_schema(),
                "input_sha256": budget["input_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    run_id = budget["run_id"]
    evaluator_run_id = f"{run_id}--consensus-adjudicated-{fingerprint}"
    output = (
        resolve_path(config, "results")
        / "evaluations_consensus_adjudicated"
        / f"{evaluator_run_id}.jsonl"
    )
    completed: set[tuple[str, str, str]] = set()
    if output.exists():
        completed = {
            (row["dataset"], row["trajectory_id"], row["condition"])
            for row in iter_jsonl(output)
            if row.get("status") == "ok"
        }
    completed_count = len(completed)
    total = len(predictions)
    adjudicated_done = sum(
        not _same_label(
            primary[key]["native_label"], alternate[key]["native_label"]
        )
        for key in completed
    )
    print(
        f"[consensus-adjudication] rows={completed_count}/{total}; "
        f"disagreements={adjudicated_done}/{budget['disagreement_rows']}",
        flush=True,
    )
    llm = CachedLLM(
        config, resolve_path(config, "cache") / "adjudicator"
    )
    for key in sorted(predictions):
        if key in completed:
            continue
        primary_label = primary[key]["native_label"]
        alternate_label = alternate[key]["native_label"]
        accepted = gold[key[:2]]
        base = {
            "run_id": run_id,
            "evaluator_run_id": evaluator_run_id,
            "evaluator_prompt_version": config["evaluation"][
                "adjudicator_prompt_version"
            ],
            "candidate_narrative_transform": "none",
            "evaluator_role": "consensus_adjudicator",
            "dataset": key[0],
            "domain": predictions[key]["domain"],
            "trajectory_id": key[1],
            "condition": key[2],
            "primary_native_label": primary_label,
            "alternate_native_label": alternate_label,
            "accepted_gold_labels": accepted,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "adjudicator_hidden_fields": budget["adjudicator_hidden_fields"],
        }
        if _same_label(primary_label, alternate_label):
            row = {
                **base,
                "status": "ok",
                "decision_method": "two_relabeler_agreement",
                "native_label": primary_label,
                "correct": any(
                    _same_label(primary_label, label) for label in accepted
                ),
                "evaluator_model": None,
                "candidate_a_source": None,
                "candidate_b_source": None,
                "adjudicator_choice": None,
                "adjudicator_reason": None,
                "cache_key": None,
                "cache_path": None,
                "cache_hit": None,
                "response_metadata": None,
            }
        else:
            candidate_a, candidate_b = _candidate_order(
                key,
                primary_label,
                alternate_label,
                int(config["experiment"]["seed"]),
            )
            user_prompt = _prompt(
                template,
                predictions[key],
                candidate_a,
                candidate_b,
                definitions[key[0]],
            )
            try:
                kwargs = {
                    "model": model["name"],
                    "system_prompt": system,
                    "user_prompt": user_prompt,
                    "temperature": model.get("temperature"),
                    "reasoning_effort": model.get("reasoning_effort"),
                    "seed": int(config["experiment"]["seed"]),
                    "provider": model.get("provider"),
                    "response_model": CandidateAdjudicationResult,
                }
                try:
                    result, call = llm.call(
                        **kwargs, max_output_tokens=int(model["max_output_tokens"])
                    )
                except TruncatedResponseError:
                    result, call = llm.call(**kwargs, max_output_tokens=320)
                selected = candidate_a if result.choice == "A" else candidate_b
                row = {
                    **base,
                    "status": "ok",
                    "decision_method": "anonymous_two_candidate_adjudication",
                    "native_label": selected[1],
                    "correct": any(
                        _same_label(selected[1], label) for label in accepted
                    ),
                    "evaluator_model": model["name"],
                    "candidate_a_source": candidate_a[0],
                    "candidate_b_source": candidate_b[0],
                    "adjudicator_choice": result.choice,
                    "adjudicator_selected_source": selected[0],
                    "adjudicator_reason": result.reason,
                    "cache_key": call["cache_key"],
                    "cache_path": str(
                        Path(call["cache_path"]).relative_to(config["_root"])
                    ),
                    "cache_hit": call["cache_hit"],
                    "response_metadata": call["response_metadata"],
                }
                adjudicated_done += 1
            except Exception as exc:
                append_jsonl(
                    output,
                    {
                        **base,
                        "status": "adjudicator_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
        append_jsonl(output, row)
        completed_count += 1
        if (
            completed_count % 25 == 0
            or completed_count == total
            or (
                row["decision_method"] == "anonymous_two_candidate_adjudication"
                and adjudicated_done % 10 == 0
            )
        ):
            print(
                f"[consensus-adjudication] rows={completed_count}/{total}; "
                f"disagreements={adjudicated_done}/{budget['disagreement_rows']}",
                flush=True,
            )
    _compact(output)
    return output


def generate_adjudication_report(
    consensus_evaluation_path: str | Path,
    config_path: str | Path = "config/relabeler_adjudication.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
    path = _resolve(config, consensus_evaluation_path)
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError("Consensus evaluation is empty")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("Consensus evaluation contains non-ok rows")
    run_id = rows[0]["run_id"]
    disagreements = [
        row
        for row in rows
        if row["decision_method"] == "anonymous_two_candidate_adjudication"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in disagreements:
        grouped[(row["dataset"], row["condition"])].append(row)
    detail_rows: list[dict[str, Any]] = []
    for dataset in experiment_datasets(config):
        for condition in (
            "no_taxonomy",
            "own_taxonomy",
            "foreign_taxonomy",
            "merged_taxonomy",
        ):
            items = grouped[(dataset, condition)]
            primary_selected = sum(
                row["adjudicator_selected_source"] == "primary" for row in items
            )
            alternate_selected = sum(
                row["adjudicator_selected_source"] == "alternate" for row in items
            )
            a_choices = sum(row["adjudicator_choice"] == "A" for row in items)
            correct = sum(bool(row["correct"]) for row in items)
            detail_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "disagreements": len(items),
                    "primary_selected": primary_selected,
                    "alternate_selected": alternate_selected,
                    "candidate_a_selected": a_choices,
                    "candidate_b_selected": len(items) - a_choices,
                    "adjudicated_correct": correct,
                    "adjudicated_accuracy": correct / len(items) if items else None,
                }
            )
    output_dir = resolve_path(config, "results") / "relabeler_adjudication"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_disagreement_adjudication"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    summary = {
        "run_id": run_id,
        "method": ADJUDICATION_METHOD,
        "total_rows": len(rows),
        "agreement_rows": len(rows) - len(disagreements),
        "disagreement_rows": len(disagreements),
        "adjudicator_models": sorted(
            {
                row["evaluator_model"]
                for row in disagreements
                if row.get("evaluator_model")
            }
        ),
        "adjudicator_selected_primary": sum(
            row["adjudicator_selected_source"] == "primary"
            for row in disagreements
        ),
        "adjudicator_selected_alternate": sum(
            row["adjudicator_selected_source"] == "alternate"
            for row in disagreements
        ),
        "candidate_a_selected": sum(
            row["adjudicator_choice"] == "A" for row in disagreements
        ),
        "candidate_b_selected": sum(
            row["adjudicator_choice"] == "B" for row in disagreements
        ),
        "by_dataset_condition": detail_rows,
    }
    write_json(json_path, summary)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    lines = [
        "# Two-Relabeler Consensus with Disagreement Adjudication",
        "",
        f"Run: `{run_id}`",
        "",
        f"Agreement rows: {summary['agreement_rows']}/{summary['total_rows']}; adjudicated disagreements: {summary['disagreement_rows']}.",
        "",
        f"Adjudicator model(s): `{', '.join(summary['adjudicator_models'])}`.",
        "",
        "| Dataset | Condition | Disagreements | Primary selected | Alternate selected | A selected | B selected | Adjudicated correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detail_rows:
        accuracy = row["adjudicated_accuracy"]
        accuracy_display = (
            "N/A"
            if accuracy is None
            else f"{row['adjudicated_correct']}/{row['disagreements']} ({accuracy:.2%})"
        )
        lines.append(
            f"| {row['dataset']} | {DISPLAY_NAMES[row['condition']]} | "
            f"{row['disagreements']} | {row['primary_selected']} | "
            f"{row['alternate_selected']} | {row['candidate_a_selected']} | "
            f"{row['candidate_b_selected']} | {accuracy_display} |"
        )
    lines.extend(
        [
            "",
            "Candidate A/B order was deterministically randomized per experiment key, and relabeler model identities were hidden from the adjudicator.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"markdown": markdown_path, "json": json_path, "csv": csv_path}
