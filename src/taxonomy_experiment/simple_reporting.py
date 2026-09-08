from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, write_json
from taxonomy_experiment.reporting import DISPLAY_NAMES, _canonical_predicted_type
from taxonomy_experiment.taxonomy import CONDITION_ORDER


DATASET_NAMES = {
    "agentrx": "AgentRx",
    "mast": "MAST (Magentic × GAIA)",
}


def generate_simple_report(
    prediction_path: str | Path,
    evaluation_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
) -> dict[str, Path]:
    """Report the independent binary judgment for every dataset/condition row."""
    config = load_config(config_path)
    dataset_order = experiment_datasets(config)
    root = config["_root"]
    prediction_path = Path(prediction_path)
    evaluation_path = Path(evaluation_path)
    if not prediction_path.is_absolute():
        prediction_path = root / prediction_path
    if not evaluation_path.is_absolute():
        evaluation_path = root / evaluation_path

    predictions = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row
        for row in iter_jsonl(prediction_path)
    }
    evaluations = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row
        for row in iter_jsonl(evaluation_path)
    }
    if not predictions:
        raise ValueError("Prediction file is empty")
    run_id = next(iter(predictions.values()))["run_id"]
    gold: dict[tuple[str, str], dict[str, Any]] = {}
    processed = resolve_path(config, "processed_data")
    for dataset in dataset_order:
        for row in iter_jsonl(processed / f"{dataset}_gold.jsonl"):
            gold[(dataset, row["trajectory_id"])] = row

    keys_by_group: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in predictions:
        keys_by_group[(key[0], key[2])].append(key)

    rows: list[dict[str, Any]] = []
    for dataset in dataset_order:
        for condition in CONDITION_ORDER:
            keys = keys_by_group.get((dataset, condition.value), [])
            if not keys:
                continue
            correct = 0
            evaluated = 0
            step_correct = 0
            step_evaluated = 0
            native_correct = 0
            native_evaluated = 0
            prediction_failures = 0
            evaluator_failures = 0
            for key in keys:
                prediction = predictions[key]
                if prediction.get("status") != "ok":
                    prediction_failures += 1
                else:
                    primary = prediction["prediction"]["predicted_errors"][0]
                    gold_row = gold[key[:2]]
                    predicted_step = primary.get("failure_step")
                    if gold_row.get("critical_failure_step") is not None:
                        step_evaluated += 1
                        step_correct += int(
                            predicted_step == gold_row["critical_failure_step"]
                        )
                    if condition.value == "own_taxonomy":
                        native_evaluated += 1
                        accepted_types = set(
                            gold_row.get("failure_types") or [gold_row["failure_type"]]
                        )
                        native_correct += int(
                            _canonical_predicted_type(dataset, primary.get("error_type"))
                            in accepted_types
                        )
                evaluation = evaluations.get(key)
                if not evaluation or evaluation.get("status") not in {"ok", "prediction_error"}:
                    evaluator_failures += 1
                    continue
                evaluated += 1
                correct += int(bool((evaluation.get("evaluation") or {}).get("correct")))
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition.value,
                    "n": len(keys),
                    "evaluated": evaluated,
                    "correct": correct,
                    "accuracy": correct / evaluated if evaluated else None,
                    "step_exact": step_correct,
                    "step_evaluated": step_evaluated,
                    "step_accuracy": step_correct / step_evaluated if step_evaluated else None,
                    "native_category_exact": native_correct if native_evaluated else None,
                    "native_category_evaluated": native_evaluated if native_evaluated else None,
                    "native_category_accuracy": (
                        native_correct / native_evaluated if native_evaluated else None
                    ),
                    "prediction_failures": prediction_failures,
                    "evaluator_failures": evaluator_failures,
                }
            )

    accuracy = {(row["dataset"], row["condition"]): row["accuracy"] for row in rows}
    differences: list[dict[str, Any]] = []
    for dataset in dataset_order:
        baseline = accuracy.get((dataset, "no_taxonomy"))
        for condition in ("own_taxonomy", "foreign_taxonomy", "merged_taxonomy"):
            value = accuracy.get((dataset, condition))
            differences.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "minus_no_taxonomy": (
                        value - baseline if value is not None and baseline is not None else None
                    ),
                }
            )

    summary = {
        "run_id": run_id,
        "evaluation_method": "independent_binary_reference_match",
        "evaluator_model": config["models"]["simple_evaluator"]["name"],
        "prediction_rows": len(predictions),
        "evaluation_rows": len(evaluations),
        "results": rows,
        "differences_from_no_taxonomy": differences,
    }
    output_dir = resolve_path(config, "results") / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_id}_simple_summary.json"
    csv_path = output_dir / f"{run_id}_simple_results.csv"
    markdown_path = output_dir / f"{run_id}_simple_results.md"
    write_json(summary_path, summary)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Simple Independent Evaluation Results",
        "",
        f"Run: `{run_id}`",
        "",
        "Each prediction was judged independently against its dataset gold annotation. "
        "All examples are included; no reference filtering or cross-condition comparison was used by the evaluator.",
        "",
        f"Evaluator: `{config['models']['simple_evaluator']['name']}`",
        "",
        "| Dataset | Condition | N | Semantic correct | Semantic accuracy | Step exact | Step accuracy | Native category accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        accuracy_text = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        step_text = "N/A" if row["step_accuracy"] is None else f"{row['step_accuracy']:.4f}"
        native_text = (
            "N/A"
            if row["native_category_accuracy"] is None
            else f"{row['native_category_accuracy']:.4f}"
        )
        lines.append(
            f"| {DATASET_NAMES[row['dataset']]} | {DISPLAY_NAMES[row['condition']]} | "
            f"{row['n']} | {row['correct']} | {accuracy_text} | "
            f"{row['step_exact']} | {step_text} | {native_text} |"
        )
    lines.extend(
        [
            "",
            "## Difference from no-taxonomy baseline",
            "",
            "| Dataset | Condition | Accuracy difference |",
            "|---|---|---:|",
        ]
    )
    for row in differences:
        difference = row["minus_no_taxonomy"]
        difference_text = "N/A" if difference is None else f"{difference:+.4f} ({difference * 100:+.2f} pp)"
        lines.append(
            f"| {DATASET_NAMES[row['dataset']]} | {DISPLAY_NAMES[row['condition']]} | "
            f"{difference_text} |"
        )
    lines.extend(
        [
            "",
            "Semantic correctness judges only whether the candidate identifies the same underlying root cause as the gold reference; step equality is not part of that judgment. "
            "Step accuracy is exact-match and native category accuracy is reported only for the own-taxonomy condition.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"markdown": markdown_path, "csv": csv_path, "summary": summary_path}
