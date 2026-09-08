from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, read_json, write_json
from taxonomy_experiment.reporting import DISPLAY_NAMES
from taxonomy_experiment.taxonomy import CONDITION_ORDER


DATASET_NAMES = {"agentrx": "AgentRx", "mast": "MAST (Magentic × GAIA)"}


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_interval(
    baseline: list[bool],
    candidate: list[bool],
    *,
    samples: int,
    confidence: float,
    seed: str,
) -> tuple[float, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired arrays must have the same non-zero length")
    differences = [int(right) - int(left) for left, right in zip(baseline, candidate)]
    rng = random.Random(seed)
    estimates = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    ]
    alpha = (1 - confidence) / 2
    return _quantile(estimates, alpha), _quantile(estimates, 1 - alpha)


def exact_mcnemar_p_value(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def _paired_comparison(
    reference_by_id: dict[str, bool],
    candidate_by_id: dict[str, bool],
    *,
    dataset: str,
    reference_condition: str,
    candidate_condition: str,
    samples: int,
    confidence: float,
    seed: str,
) -> dict[str, Any]:
    ids = sorted(set(reference_by_id) & set(candidate_by_id))
    reference = [reference_by_id[item] for item in ids]
    candidate = [candidate_by_id[item] for item in ids]
    improved = sum(
        (not left) and right for left, right in zip(reference, candidate)
    )
    regressed = sum(
        left and (not right) for left, right in zip(reference, candidate)
    )
    both_correct = sum(left and right for left, right in zip(reference, candidate))
    both_wrong = len(ids) - improved - regressed - both_correct
    difference = sum(int(value) for value in candidate) / len(candidate) - sum(
        int(value) for value in reference
    ) / len(reference)
    ci_low, ci_high = paired_bootstrap_interval(
        reference,
        candidate,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "dataset": dataset,
        "reference_condition": reference_condition,
        "candidate_condition": candidate_condition,
        "contrast": f"{candidate_condition}_minus_{reference_condition}",
        "n": len(ids),
        "accuracy_difference": difference,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_samples": samples,
        "confidence_level": confidence,
        "improved": improved,
        "regressed": regressed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(improved, regressed),
    }


def _source_taxonomy_names(config: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for path_key in ("taxonomy_a", "taxonomy_b", "merged_taxonomy"):
        taxonomy = read_json(resolve_path(config, path_key))
        result[taxonomy["taxonomy_id"]] = {
            item["name"].casefold() for item in taxonomy["categories"]
        }
    return result


def generate_native_blind_report(
    prediction_path: str | Path,
    evaluation_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
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
    if len(predictions) != len(evaluations):
        raise ValueError(
            f"Prediction/evaluation size mismatch: {len(predictions)} != {len(evaluations)}"
        )
    if set(predictions) != set(evaluations):
        raise ValueError("Prediction and evaluation keys do not match")
    run_id = next(iter(predictions.values()))["run_id"]
    evaluator_run_id = next(iter(evaluations.values()))["evaluator_run_id"]
    transforms = {
        row.get("candidate_narrative_transform", "none")
        for row in evaluations.values()
    }
    if len(transforms) != 1:
        raise ValueError(f"Mixed candidate narrative transforms: {sorted(transforms)}")
    narrative_transform = next(iter(transforms))
    evaluator_roles = {
        row.get("evaluator_role", "simple_evaluator")
        for row in evaluations.values()
    }
    if len(evaluator_roles) != 1:
        raise ValueError(f"Mixed evaluator roles: {sorted(evaluator_roles)}")
    evaluator_role = next(iter(evaluator_roles))
    evaluator_models = {
        row.get("evaluator_model")
        for row in evaluations.values()
        if row.get("evaluator_model")
    }
    source_names = _source_taxonomy_names(config)
    grouped: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in predictions:
        grouped[(key[0], key[2])].append(key)

    rows: list[dict[str, Any]] = []
    correctness: dict[tuple[str, str], dict[str, bool]] = {}
    for dataset in experiment_datasets(config):
        for condition in CONDITION_ORDER:
            keys = sorted(grouped[(dataset, condition.value)], key=lambda key: key[1])
            valid = [
                key
                for key in keys
                if predictions[key].get("status") == "ok"
                and evaluations[key].get("status") == "ok"
            ]
            correct = sum(bool(evaluations[key]["correct"]) for key in valid)
            correctness[(dataset, condition.value)] = {
                key[1]: bool(evaluations[key]["correct"]) for key in valid
            }
            adherence_total = 0
            adherence_hits = 0
            for key in valid:
                prediction = predictions[key]
                taxonomy_id = prediction.get("taxonomy_id")
                if taxonomy_id is None:
                    continue
                adherence_total += 1
                error_type = prediction["prediction"]["predicted_errors"][0]["error_type"]
                adherence_hits += int(error_type.casefold() in source_names[taxonomy_id])
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition.value,
                    "n": len(keys),
                    "evaluated": len(valid),
                    "native_correct": correct,
                    "native_accuracy": correct / len(valid) if valid else None,
                    "source_taxonomy_adherence_correct": (
                        adherence_hits if adherence_total else None
                    ),
                    "source_taxonomy_adherence_evaluated": (
                        adherence_total if adherence_total else None
                    ),
                    "source_taxonomy_adherence": (
                        adherence_hits / adherence_total if adherence_total else None
                    ),
                    "prediction_errors": len(keys) - sum(
                        predictions[key].get("status") == "ok" for key in keys
                    ),
                    "evaluator_errors": len(keys) - len(valid),
                }
            )

    paired: list[dict[str, Any]] = []
    samples = int(config["evaluation"]["bootstrap_samples"])
    confidence = float(config["evaluation"]["confidence_level"])
    for dataset in experiment_datasets(config):
        baseline_by_id = correctness[(dataset, "no_taxonomy")]
        for condition in ("own_taxonomy", "foreign_taxonomy", "merged_taxonomy"):
            candidate_by_id = correctness[(dataset, condition)]
            comparison = _paired_comparison(
                baseline_by_id,
                candidate_by_id,
                dataset=dataset,
                reference_condition="no_taxonomy",
                candidate_condition=condition,
                samples=samples,
                confidence=confidence,
                # Preserve the canonical baseline-comparison bootstrap stream.
                seed=f"{config['experiment']['seed']}:{dataset}:{condition}",
            )
            # Backward-compatible field used by the existing figure and reports.
            comparison["condition"] = condition
            paired.append(comparison)

    merged_vs_foreign = [
        _paired_comparison(
            correctness[(dataset, "foreign_taxonomy")],
            correctness[(dataset, "merged_taxonomy")],
            dataset=dataset,
            reference_condition="foreign_taxonomy",
            candidate_condition="merged_taxonomy",
            samples=samples,
            confidence=confidence,
            seed=(
                f"{config['experiment']['seed']}:{dataset}:"
                "merged_taxonomy:foreign_taxonomy"
            ),
        )
        for dataset in experiment_datasets(config)
    ]

    agentrx_split_rows: list[dict[str, Any]] = []
    agentrx_domains = sorted(
        {
            predictions[key]["domain"]
            for key in predictions
            if key[0] == "agentrx"
        }
    )
    for domain in agentrx_domains:
        for condition in CONDITION_ORDER:
            keys = sorted(
                (
                    key
                    for key in grouped[("agentrx", condition.value)]
                    if predictions[key]["domain"] == domain
                ),
                key=lambda key: key[1],
            )
            valid = [
                key
                for key in keys
                if predictions[key].get("status") == "ok"
                and evaluations[key].get("status") == "ok"
            ]
            correct = sum(bool(evaluations[key]["correct"]) for key in valid)
            agentrx_split_rows.append(
                {
                    "dataset": "agentrx",
                    "domain": domain,
                    "condition": condition.value,
                    "n": len(keys),
                    "evaluated": len(valid),
                    "native_correct": correct,
                    "native_accuracy": correct / len(valid) if valid else None,
                }
            )

    summary = {
        "run_id": run_id,
        "evaluator_run_id": evaluator_run_id,
        "primary_evaluation_method": (
            "condition_source_label_and_native_term_masked_relabeling"
            if narrative_transform != "none"
            else "condition_and_source_label_blind_native_relabeling"
        ),
        "candidate_narrative_transform": narrative_transform,
        "evaluator_role": evaluator_role,
        "evaluator_models": sorted(evaluator_models),
        "candidate_fields_used": ["explanation", "evidence", "overall_root_cause"],
        "fields_hidden_from_relabeler": [
            "trajectory",
            "gold_annotation",
            "experimental_condition",
            "source_taxonomy_id",
            "source_predicted_error_type",
            "predicted_failure_step",
        ],
        "mast_gold_provenance": "official_release_llm_judge_multilabel_pseudo_gold",
        "results": rows,
        "paired_comparisons_to_no_taxonomy": paired,
        "paired_comparisons_merged_vs_foreign": merged_vs_foreign,
        "agentrx_split_results": agentrx_split_rows,
    }
    output_dir = resolve_path(config, "results") / "tables_native_blind"
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_suffix = (
        "" if narrative_transform == "none" else "_masked_native_terms"
    )
    role_suffix = (
        "" if evaluator_role == "simple_evaluator" else f"_{evaluator_role}"
    )
    stem = f"{run_id}_native_blind{role_suffix}{transform_suffix}"
    summary_path = output_dir / f"{stem}_summary.json"
    result_csv = output_dir / f"{stem}_results.csv"
    paired_csv = output_dir / f"{stem}_paired.csv"
    merged_foreign_csv = output_dir / f"{stem}_merged_vs_foreign.csv"
    agentrx_split_csv = output_dir / f"{stem}_agentrx_splits.csv"
    markdown_path = output_dir / f"{stem}_results.md"
    write_json(summary_path, summary)
    with result_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paired_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    with merged_foreign_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_vs_foreign[0]))
        writer.writeheader()
        writer.writerows(merged_vs_foreign)
    with agentrx_split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agentrx_split_rows[0]))
        writer.writeheader()
        writer.writerows(agentrx_split_rows)

    lines = [
        "# Corrected Label-Blind Native Evaluation",
        "",
        f"Run: `{run_id}`",
        "",
        f"Relabeler role: `{evaluator_role}`; model(s): `{', '.join(sorted(evaluator_models))}`.",
        "",
        "The relabeler received only the candidate explanation, evidence, overall root cause, and the target dataset's native taxonomy. It did not receive the source prediction label, condition, trajectory, or gold annotation.",
        "",
        "| Dataset | Condition | N | Native correct | Native accuracy | Source taxonomy adherence |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        adherence = row["source_taxonomy_adherence"]
        lines.append(
            f"| {DATASET_NAMES[row['dataset']]} | {DISPLAY_NAMES[row['condition']]} | "
            f"{row['n']} | {row['native_correct']} | {row['native_accuracy']:.4f} | "
            f"{'N/A' if adherence is None else f'{adherence:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparisons against no-taxonomy",
            "",
            "| Dataset | Condition | Difference | 95% paired-bootstrap CI | Improved | Regressed | McNemar exact p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {DATASET_NAMES[row['dataset']]} | {DISPLAY_NAMES[row['condition']]} | "
            f"{row['accuracy_difference']:+.4f} | "
            f"[{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}] | "
            f"{row['improved']} | {row['regressed']} | {row['mcnemar_exact_p_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Primary transfer contrast: merged taxonomy against foreign taxonomy",
            "",
            "| Dataset | Contrast | Difference | 95% paired-bootstrap CI | Foreign wrong → merged correct | Foreign correct → merged wrong | McNemar exact p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in merged_vs_foreign:
        lines.append(
            f"| {DATASET_NAMES[row['dataset']]} | LLM-Merged − Foreign | "
            f"{row['accuracy_difference']:+.4f} | "
            f"[{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}] | "
            f"{row['improved']} | {row['regressed']} | "
            f"{row['mcnemar_exact_p_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## AgentRx split-level descriptive sensitivity",
            "",
            "| Split | Condition | N | Native correct | Native accuracy |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in agentrx_split_rows:
        lines.append(
            f"| {row['domain']} | {DISPLAY_NAMES[row['condition']]} | "
            f"{row['n']} | {row['native_correct']} | "
            f"{row['native_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "AgentRx accuracy is exact match to the single critical native category. MAST accuracy is top-1 membership in the official positive multi-label set. MAST labels are LLM-annotated pseudo-gold and do not identify a single critical root cause.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "markdown": markdown_path,
        "results_csv": result_csv,
        "paired_csv": paired_csv,
        "merged_vs_foreign_csv": merged_foreign_csv,
        "agentrx_split_csv": agentrx_split_csv,
        "summary": summary_path,
    }
