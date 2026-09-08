from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, write_json
from taxonomy_experiment.native_reporting import (
    exact_mcnemar_p_value,
    paired_bootstrap_interval,
)
from taxonomy_experiment.reporting import DISPLAY_NAMES


def _resolve(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else config["_root"] / path


def _evaluation_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (row["dataset"], row["trajectory_id"], row["condition"])
        if key in rows:
            raise ValueError(f"Duplicate evaluation key: {key}")
        rows[key] = row
    return rows


def _prediction_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (row["dataset"], row["trajectory_id"], row["condition"])
        if key in rows:
            raise ValueError(f"Duplicate prediction key: {key}")
        rows[key] = row
    return rows


def _gold_index(config: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    processed = resolve_path(config, "processed_data")
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_gold.jsonl"):
            result[(dataset, row["trajectory_id"])] = list(
                row.get("failure_types") or [row["failure_type"]]
            )
    return result


def _same_label(left: str, right: str) -> bool:
    return left.strip().casefold() == right.strip().casefold()


def _in_gold(label: str, accepted: list[str]) -> bool:
    return any(_same_label(label, gold_label) for gold_label in accepted)


def generate_own_label_consistency(
    prediction_path: str | Path,
    evaluation_path: str | Path,
    config_path: str | Path = "config/relabeler_validation.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
    predictions = _prediction_index(_resolve(config, prediction_path))
    evaluations = _evaluation_index(_resolve(config, evaluation_path))
    if set(predictions) != set(evaluations):
        raise ValueError("Prediction and evaluation keys do not match")
    gold = _gold_index(config)
    run_id = next(iter(predictions.values()))["run_id"]
    roles = {
        row.get("evaluator_role", "simple_evaluator")
        for row in evaluations.values()
    }
    if len(roles) != 1:
        raise ValueError(f"Mixed evaluator roles: {sorted(roles)}")
    evaluator_role = next(iter(roles))
    models = sorted(
        {
            row.get("evaluator_model")
            for row in evaluations.values()
            if row.get("evaluator_model")
        }
    )

    details: list[dict[str, Any]] = []
    for key in sorted(predictions):
        if key[2] != "own_taxonomy":
            continue
        prediction = predictions[key]
        evaluation = evaluations[key]
        if prediction.get("status") != "ok" or evaluation.get("status") != "ok":
            continue
        source_label = prediction["prediction"]["predicted_errors"][0]["error_type"]
        relabel = evaluation["native_label"]
        accepted = gold[key[:2]]
        source_correct = _in_gold(source_label, accepted)
        relabel_correct = bool(evaluation["correct"])
        details.append(
            {
                "run_id": run_id,
                "evaluator_role": evaluator_role,
                "dataset": key[0],
                "domain": prediction["domain"],
                "trajectory_id": key[1],
                "source_own_label": source_label,
                "blind_relabel": relabel,
                "label_agreement": _same_label(source_label, relabel),
                "source_label_gold_correct": source_correct,
                "blind_relabel_gold_correct": relabel_correct,
                "source_correct_relabel_wrong": source_correct
                and not relabel_correct,
                "source_wrong_relabel_correct": (not source_correct)
                and relabel_correct,
                "accepted_gold_labels": " | ".join(accepted),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[row["dataset"]].append(row)
    summary_rows: list[dict[str, Any]] = []
    for dataset in experiment_datasets(config):
        rows = grouped[dataset]
        n = len(rows)
        agreements = sum(row["label_agreement"] for row in rows)
        source_correct = sum(row["source_label_gold_correct"] for row in rows)
        relabel_correct = sum(row["blind_relabel_gold_correct"] for row in rows)
        losses = sum(row["source_correct_relabel_wrong"] for row in rows)
        corrections = sum(row["source_wrong_relabel_correct"] for row in rows)
        summary_rows.append(
            {
                "dataset": dataset,
                "n": n,
                "label_agreement": agreements,
                "label_agreement_rate": agreements / n,
                "source_label_gold_correct": source_correct,
                "source_label_gold_accuracy": source_correct / n,
                "blind_relabel_gold_correct": relabel_correct,
                "blind_relabel_gold_accuracy": relabel_correct / n,
                "source_correct_relabel_wrong": losses,
                "loss_rate_among_source_correct": (
                    losses / source_correct if source_correct else None
                ),
                "source_wrong_relabel_correct": corrections,
            }
        )

    output_dir = resolve_path(config, "results") / "relabeler_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_{evaluator_role}_own_label_consistency"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    details_csv = output_dir / f"{stem}_details.csv"
    markdown_path = output_dir / f"{stem}.md"
    write_json(
        json_path,
        {
            "run_id": run_id,
            "evaluator_role": evaluator_role,
            "evaluator_models": models,
            "scope": "own_taxonomy condition only",
            "results": summary_rows,
        },
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with details_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    lines = [
        "# Own-Condition Source Label vs Blind Relabel Consistency",
        "",
        f"Run: `{run_id}`",
        "",
        f"Relabeler role: `{evaluator_role}`; model(s): `{', '.join(models)}`.",
        "",
        "| Dataset | N | Label agreement | Source-label gold accuracy | Blind-relabel gold accuracy | Source correct → relabel wrong | Loss rate among source-correct | Source wrong → relabel correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        loss_rate = row["loss_rate_among_source_correct"]
        lines.append(
            f"| {row['dataset']} | {row['n']} | "
            f"{row['label_agreement']}/{row['n']} ({row['label_agreement_rate']:.2%}) | "
            f"{row['source_label_gold_correct']}/{row['n']} ({row['source_label_gold_accuracy']:.2%}) | "
            f"{row['blind_relabel_gold_correct']}/{row['n']} ({row['blind_relabel_gold_accuracy']:.2%}) | "
            f"{row['source_correct_relabel_wrong']} | "
            f"{'N/A' if loss_rate is None else f'{loss_rate:.2%}'} | "
            f"{row['source_wrong_relabel_correct']} |"
        )
    lines.extend(
        [
            "",
            "For MAST, both source and relabeled predictions are counted correct when they belong to the official positive multi-label set; two different labels can therefore both be correct.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "markdown": markdown_path,
        "json": json_path,
        "summary_csv": csv_path,
        "details_csv": details_csv,
    }


def _cohens_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa inputs must have the same non-zero length")
    n = len(left)
    observed = sum(_same_label(a, b) for a, b in zip(left, right)) / n
    left_counts = Counter(value.casefold() for value in left)
    right_counts = Counter(value.casefold() for value in right)
    expected = sum(
        left_counts[label] / n * right_counts[label] / n
        for label in set(left_counts) | set(right_counts)
    )
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def compare_relabelers(
    primary_evaluation_path: str | Path,
    alternate_evaluation_path: str | Path,
    config_path: str | Path = "config/relabeler_validation.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
    primary = _evaluation_index(_resolve(config, primary_evaluation_path))
    alternate = _evaluation_index(_resolve(config, alternate_evaluation_path))
    if set(primary) != set(alternate):
        raise ValueError("Primary and alternate evaluation keys do not match")
    run_ids = {row["run_id"] for row in primary.values()} | {
        row["run_id"] for row in alternate.values()
    }
    if len(run_ids) != 1:
        raise ValueError(f"Evaluation run IDs do not match: {sorted(run_ids)}")
    run_id = next(iter(run_ids))
    samples = int(config["evaluation"]["bootstrap_samples"])
    confidence = float(config["evaluation"]["confidence_level"])
    cell_rows: list[dict[str, Any]] = []
    accuracy: dict[tuple[str, str, str], float] = {}
    for dataset in experiment_datasets(config):
        for condition in (
            "no_taxonomy",
            "own_taxonomy",
            "foreign_taxonomy",
            "merged_taxonomy",
        ):
            keys = sorted(
                key
                for key in primary
                if key[0] == dataset and key[2] == condition
            )
            valid = [
                key
                for key in keys
                if primary[key].get("status") == "ok"
                and alternate[key].get("status") == "ok"
            ]
            primary_labels = [primary[key]["native_label"] for key in valid]
            alternate_labels = [alternate[key]["native_label"] for key in valid]
            primary_correct = [bool(primary[key]["correct"]) for key in valid]
            alternate_correct = [bool(alternate[key]["correct"]) for key in valid]
            agreement = sum(
                _same_label(left, right)
                for left, right in zip(primary_labels, alternate_labels)
            )
            improved = sum(
                (not left) and right
                for left, right in zip(primary_correct, alternate_correct)
            )
            regressed = sum(
                left and (not right)
                for left, right in zip(primary_correct, alternate_correct)
            )
            primary_acc = sum(primary_correct) / len(valid)
            alternate_acc = sum(alternate_correct) / len(valid)
            ci_low, ci_high = paired_bootstrap_interval(
                primary_correct,
                alternate_correct,
                samples=samples,
                confidence=confidence,
                seed=(
                    f"{config['experiment']['seed']}:{dataset}:{condition}:"
                    "alternate-vs-primary-relabeler"
                ),
            )
            accuracy[("primary", dataset, condition)] = primary_acc
            accuracy[("alternate", dataset, condition)] = alternate_acc
            cell_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(valid),
                    "native_label_agreement": agreement,
                    "native_label_agreement_rate": agreement / len(valid),
                    "cohens_kappa": _cohens_kappa(
                        primary_labels, alternate_labels
                    ),
                    "primary_accuracy": primary_acc,
                    "alternate_accuracy": alternate_acc,
                    "alternate_minus_primary": alternate_acc - primary_acc,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "alternate_improved": improved,
                    "alternate_regressed": regressed,
                    "mcnemar_exact_p_value": exact_mcnemar_p_value(
                        improved, regressed
                    ),
                }
            )

    direction_rows: list[dict[str, Any]] = []
    for dataset in experiment_datasets(config):
        values: dict[str, Any] = {"dataset": dataset}
        for role in ("primary", "alternate"):
            baseline = accuracy[(role, dataset, "no_taxonomy")]
            foreign = accuracy[(role, dataset, "foreign_taxonomy")]
            merged = accuracy[(role, dataset, "merged_taxonomy")]
            values[f"{role}_foreign_minus_baseline"] = foreign - baseline
            values[f"{role}_merged_minus_foreign"] = merged - foreign
        values["foreign_below_baseline_same_direction"] = (
            values["primary_foreign_minus_baseline"] < 0
            and values["alternate_foreign_minus_baseline"] < 0
        )
        values["merged_above_foreign_same_direction"] = (
            values["primary_merged_minus_foreign"] > 0
            and values["alternate_merged_minus_foreign"] > 0
        )
        direction_rows.append(values)

    output_dir = resolve_path(config, "results") / "relabeler_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_primary_vs_alternate_relabeler"
    json_path = output_dir / f"{stem}.json"
    cells_csv = output_dir / f"{stem}_cells.csv"
    directions_csv = output_dir / f"{stem}_directions.csv"
    markdown_path = output_dir / f"{stem}.md"
    write_json(
        json_path,
        {
            "run_id": run_id,
            "cell_comparisons": cell_rows,
            "research_direction_comparisons": direction_rows,
        },
    )
    with cells_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    with directions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(direction_rows[0]))
        writer.writeheader()
        writer.writerows(direction_rows)
    lines = [
        "# Primary vs Alternate Relabeler Sensitivity",
        "",
        f"Run: `{run_id}`",
        "",
        "| Dataset | Condition | N | Native-label agreement | Cohen's κ | Primary accuracy | Alternate accuracy | Alternate − primary | 95% paired-bootstrap CI | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cell_rows:
        kappa = row["cohens_kappa"]
        lines.append(
            f"| {row['dataset']} | {DISPLAY_NAMES[row['condition']]} | "
            f"{row['n']} | {row['native_label_agreement_rate']:.2%} | "
            f"{'N/A' if kappa is None else f'{kappa:.4f}'} | "
            f"{row['primary_accuracy']:.4f} | {row['alternate_accuracy']:.4f} | "
            f"{row['alternate_minus_primary']:+.4f} | "
            f"[{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}] | "
            f"{row['mcnemar_exact_p_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Research-direction stability",
            "",
            "| Dataset | Primary foreign − baseline | Alternate foreign − baseline | Both foreign < baseline | Primary merged − foreign | Alternate merged − foreign | Both merged > foreign |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in direction_rows:
        lines.append(
            f"| {row['dataset']} | "
            f"{row['primary_foreign_minus_baseline']:+.4f} | "
            f"{row['alternate_foreign_minus_baseline']:+.4f} | "
            f"{row['foreign_below_baseline_same_direction']} | "
            f"{row['primary_merged_minus_foreign']:+.4f} | "
            f"{row['alternate_merged_minus_foreign']:+.4f} | "
            f"{row['merged_above_foreign_same_direction']} |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "markdown": markdown_path,
        "json": json_path,
        "cells_csv": cells_csv,
        "directions_csv": directions_csv,
    }

