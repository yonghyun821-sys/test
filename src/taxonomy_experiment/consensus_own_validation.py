from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import write_json
from taxonomy_experiment.relabeler_validation import (
    _evaluation_index,
    _gold_index,
    _in_gold,
    _prediction_index,
    _same_label,
)


RELABELERS = (
    ("gpt_4_1_mini", "GPT-4.1-mini"),
    ("gemini_2_5_flash_lite", "Gemini 2.5 Flash Lite"),
    ("consensus", "Consensus"),
)


def _resolve(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else config["_root"] / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_and_validate(
    prediction_path: str | Path,
    primary_evaluation_path: str | Path,
    alternate_evaluation_path: str | Path,
    consensus_evaluation_path: str | Path,
    config_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, dict[tuple[str, str, str], dict[str, Any]]],
    dict[tuple[str, str], list[str]],
    dict[str, Path],
]:
    config = load_config(config_path)
    paths = {
        "predictions": _resolve(config, prediction_path),
        "gpt_4_1_mini": _resolve(config, primary_evaluation_path),
        "gemini_2_5_flash_lite": _resolve(config, alternate_evaluation_path),
        "consensus": _resolve(config, consensus_evaluation_path),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")

    predictions = _prediction_index(paths["predictions"])
    evaluations = {
        name: _evaluation_index(paths[name])
        for name, _ in RELABELERS
    }
    keys = set(predictions)
    if not keys:
        raise ValueError("Prediction file is empty")
    for name, rows in evaluations.items():
        if set(rows) != keys:
            raise ValueError(f"{name} evaluation keys do not match predictions")

    all_rows = list(predictions.values())
    for rows in evaluations.values():
        all_rows.extend(rows.values())
    run_ids = {row["run_id"] for row in all_rows}
    if len(run_ids) != 1:
        raise ValueError(f"Input run IDs do not match: {sorted(run_ids)}")

    non_ok = [
        (name, key)
        for name, rows in (("predictions", predictions), *evaluations.items())
        for key, row in rows.items()
        if row.get("status") != "ok"
    ]
    if non_ok:
        raise ValueError(f"Inputs contain {len(non_ok)} non-ok rows")

    gold = _gold_index(config)
    for key in sorted(keys):
        accepted = gold[key[:2]]
        primary_label = evaluations["gpt_4_1_mini"][key]["native_label"]
        alternate_label = evaluations["gemini_2_5_flash_lite"][key]["native_label"]
        consensus = evaluations["consensus"][key]
        consensus_label = consensus["native_label"]
        for name, rows in evaluations.items():
            computed = _in_gold(rows[key]["native_label"], accepted)
            if bool(rows[key]["correct"]) != computed:
                raise ValueError(f"Stored correctness mismatch for {name}: {key}")
        if _same_label(primary_label, alternate_label):
            if not _same_label(consensus_label, primary_label):
                raise ValueError(f"Consensus changed an agreed label: {key}")
        elif not (
            _same_label(consensus_label, primary_label)
            or _same_label(consensus_label, alternate_label)
        ):
            raise ValueError(f"Consensus is not one of the two candidates: {key}")

    return config, predictions, evaluations, gold, paths


def inspect_consensus_own_inputs(
    prediction_path: str | Path,
    primary_evaluation_path: str | Path,
    alternate_evaluation_path: str | Path,
    consensus_evaluation_path: str | Path,
    config_path: str | Path = "config/relabeler_adjudication.yaml",
) -> dict[str, Any]:
    config, predictions, evaluations, _, paths = _load_and_validate(
        prediction_path,
        primary_evaluation_path,
        alternate_evaluation_path,
        consensus_evaluation_path,
        config_path,
    )
    own_counts = {
        dataset: sum(
            key[0] == dataset and key[2] == "own_taxonomy"
            for key in predictions
        )
        for dataset in experiment_datasets(config)
    }
    return {
        "status": "ready",
        "api_calls_required": 0,
        "run_id": next(iter(predictions.values()))["run_id"],
        "total_experiment_rows": len(predictions),
        "own_rows": sum(own_counts.values()),
        "own_rows_by_dataset": own_counts,
        "relabeler_models": {
            name: sorted(
                {
                    row.get("evaluator_model")
                    for row in evaluations[name].values()
                    if row.get("evaluator_model")
                }
            )
            for name, _ in RELABELERS
        },
        "input_files": {
            name: str(path.relative_to(config["_root"]))
            for name, path in paths.items()
        },
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    n = len(rows)
    source_correct = sum(row["source_label_gold_correct"] for row in rows)
    source_wrong = n - source_correct
    agreements = sum(row[f"{prefix}_label_agreement"] for row in rows)
    relabel_correct = sum(row[f"{prefix}_gold_correct"] for row in rows)
    losses = sum(row[f"{prefix}_source_correct_relabel_wrong"] for row in rows)
    corrections = sum(row[f"{prefix}_source_wrong_relabel_correct"] for row in rows)
    return {
        "n": n,
        "label_agreement": agreements,
        "label_agreement_rate": agreements / n,
        "source_label_gold_correct": source_correct,
        "source_label_gold_accuracy": source_correct / n,
        "relabel_gold_correct": relabel_correct,
        "relabel_gold_accuracy": relabel_correct / n,
        "source_correct_relabel_wrong": losses,
        "loss_rate_all_own": losses / n,
        "loss_rate_among_source_correct": losses / source_correct
        if source_correct
        else None,
        "source_wrong_relabel_correct": corrections,
        "correction_rate_all_own": corrections / n,
        "correction_rate_among_source_wrong": corrections / source_wrong
        if source_wrong
        else None,
        "net_correctness_change": corrections - losses,
    }


def generate_consensus_own_comparison(
    prediction_path: str | Path,
    primary_evaluation_path: str | Path,
    alternate_evaluation_path: str | Path,
    consensus_evaluation_path: str | Path,
    config_path: str | Path = "config/relabeler_adjudication.yaml",
) -> dict[str, Path]:
    config, predictions, evaluations, gold, _ = _load_and_validate(
        prediction_path,
        primary_evaluation_path,
        alternate_evaluation_path,
        consensus_evaluation_path,
        config_path,
    )
    run_id = next(iter(predictions.values()))["run_id"]
    details: list[dict[str, Any]] = []
    for key in sorted(predictions):
        if key[2] != "own_taxonomy":
            continue
        prediction = predictions[key]
        source_label = prediction["prediction"]["predicted_errors"][0]["error_type"]
        accepted = gold[key[:2]]
        source_correct = _in_gold(source_label, accepted)
        detail: dict[str, Any] = {
            "run_id": run_id,
            "dataset": key[0],
            "domain": prediction["domain"],
            "trajectory_id": key[1],
            "source_own_label": source_label,
            "source_label_gold_correct": source_correct,
            "accepted_gold_labels": " | ".join(accepted),
        }
        for name, _ in RELABELERS:
            evaluation = evaluations[name][key]
            correct = bool(evaluation["correct"])
            detail[f"{name}_label"] = evaluation["native_label"]
            detail[f"{name}_label_agreement"] = _same_label(
                source_label, evaluation["native_label"]
            )
            detail[f"{name}_gold_correct"] = correct
            detail[f"{name}_source_correct_relabel_wrong"] = (
                source_correct and not correct
            )
            detail[f"{name}_source_wrong_relabel_correct"] = (
                not source_correct and correct
            )
        detail["consensus_decision_method"] = evaluations["consensus"][key].get(
            "decision_method"
        )
        details.append(detail)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[row["dataset"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    metric_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in experiment_datasets(config):
        for name, display_name in RELABELERS:
            metrics = _metrics(grouped[dataset], name)
            metric_lookup[(dataset, name)] = metrics
            summary_rows.append(
                {
                    "dataset": dataset,
                    "relabeler": name,
                    "relabeler_display_name": display_name,
                    **metrics,
                }
            )

    comparison_rows: list[dict[str, Any]] = []
    for dataset in experiment_datasets(config):
        consensus = metric_lookup[(dataset, "consensus")]
        for reference, reference_display in RELABELERS[:2]:
            baseline = metric_lookup[(dataset, reference)]
            consensus_loss_rate = consensus["loss_rate_among_source_correct"]
            baseline_loss_rate = baseline["loss_rate_among_source_correct"]
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "reference_relabeler": reference,
                    "reference_display_name": reference_display,
                    "consensus_minus_reference_label_agreement_rate": (
                        consensus["label_agreement_rate"]
                        - baseline["label_agreement_rate"]
                    ),
                    "consensus_minus_reference_relabel_gold_accuracy": (
                        consensus["relabel_gold_accuracy"]
                        - baseline["relabel_gold_accuracy"]
                    ),
                    "reference_minus_consensus_loss_count": (
                        baseline["source_correct_relabel_wrong"]
                        - consensus["source_correct_relabel_wrong"]
                    ),
                    "consensus_minus_reference_loss_rate_among_source_correct": (
                        None
                        if consensus_loss_rate is None or baseline_loss_rate is None
                        else consensus_loss_rate - baseline_loss_rate
                    ),
                }
            )

    output_dir = resolve_path(config, "results") / "relabeler_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_consensus_own_label_comparison"
    markdown_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    summary_csv = output_dir / f"{stem}_summary.csv"
    comparison_csv = output_dir / f"{stem}_deltas.csv"
    details_csv = output_dir / f"{stem}_details.csv"

    write_json(
        json_path,
        {
            "run_id": run_id,
            "scope": "own_taxonomy condition only",
            "denominators": {
                "label_agreement_rate": "all Own-condition rows in each dataset",
                "loss_rate_among_source_correct": (
                    "Own-condition rows whose original attribution label is gold-correct"
                ),
            },
            "summary": summary_rows,
            "consensus_deltas": comparison_rows,
        },
    )
    for path, rows in (
        (summary_csv, summary_rows),
        (comparison_csv, comparison_rows),
        (details_csv, details),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Consensus Own-Label Consistency Validation",
        "",
        f"Run: `{run_id}`",
        "",
        "Label agreement uses every Own-condition row as its denominator. The loss rate uses only rows where the original Own attribution label was gold-correct.",
        "",
        "| Dataset | Relabeler | N | Own-label agreement | Relabel gold accuracy | Correct-to-wrong losses | Loss rate among source-correct | Wrong-to-correct recoveries |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        loss_rate = row["loss_rate_among_source_correct"]
        lines.append(
            f"| {row['dataset']} | {row['relabeler_display_name']} | {row['n']} | "
            f"{row['label_agreement']}/{row['n']} ({row['label_agreement_rate']:.2%}) | "
            f"{row['relabel_gold_correct']}/{row['n']} ({row['relabel_gold_accuracy']:.2%}) | "
            f"{row['source_correct_relabel_wrong']} | "
            f"{'N/A' if loss_rate is None else f'{loss_rate:.2%}'} | "
            f"{row['source_wrong_relabel_correct']} |"
        )
    lines.extend(
        [
            "",
            "## Consensus change relative to each single relabeler",
            "",
            "| Dataset | Reference | Agreement-rate change | Relabel-accuracy change | Losses reduced | Source-correct loss-rate change |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        loss_delta = row[
            "consensus_minus_reference_loss_rate_among_source_correct"
        ]
        lines.append(
            f"| {row['dataset']} | {row['reference_display_name']} | "
            f"{row['consensus_minus_reference_label_agreement_rate']:+.2%} | "
            f"{row['consensus_minus_reference_relabel_gold_accuracy']:+.2%} | "
            f"{row['reference_minus_consensus_loss_count']:+d} | "
            f"{'N/A' if loss_delta is None else f'{loss_delta:+.2%}'} |"
        )
    lines.extend(
        [
            "",
            "For MAST, gold correctness means membership in the official positive multi-label set. Therefore, two different native labels can both be gold-correct.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "markdown": markdown_path,
        "json": json_path,
        "summary_csv": summary_csv,
        "deltas_csv": comparison_csv,
        "details_csv": details_csv,
    }
