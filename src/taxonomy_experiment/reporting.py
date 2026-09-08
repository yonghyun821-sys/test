from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.datasets.agenterrorbench import canonicalize_failure_type
from taxonomy_experiment.datasets.agentrx import canonicalize_category
from taxonomy_experiment.datasets.mast import canonicalize_mast_category
from taxonomy_experiment.io import iter_jsonl, write_json
from taxonomy_experiment.taxonomy import (
    CONDITION_ORDER,
    Condition,
    load_taxonomy_set,
    taxonomy_for_condition,
)


DISPLAY_NAMES = {
    Condition.NO_TAXONOMY.value: "No Taxonomy",
    Condition.OWN_TAXONOMY.value: "Own Taxonomy",
    Condition.FOREIGN_TAXONOMY.value: "Foreign Taxonomy",
    Condition.MERGED_TAXONOMY.value: "LLM-Merged Taxonomy",
}

COMPARISONS = [
    (Condition.OWN_TAXONOMY.value, Condition.NO_TAXONOMY.value),
    (Condition.FOREIGN_TAXONOMY.value, Condition.NO_TAXONOMY.value),
    (Condition.MERGED_TAXONOMY.value, Condition.NO_TAXONOMY.value),
    (Condition.MERGED_TAXONOMY.value, Condition.FOREIGN_TAXONOMY.value),
    (Condition.OWN_TAXONOMY.value, Condition.FOREIGN_TAXONOMY.value),
    (Condition.OWN_TAXONOMY.value, Condition.MERGED_TAXONOMY.value),
]


def _primary_prediction(row: dict[str, Any]) -> dict[str, Any]:
    errors = (row.get("prediction") or {}).get("predicted_errors") or []
    return errors[0] if errors else {}


def _canonical_predicted_type(dataset: str, value: Any) -> str:
    raw = str(value or "").strip()
    if dataset == "mast":
        return canonicalize_mast_category(raw)
    if dataset == "agenterrorbench":
        if "." in raw:
            raw = raw.rsplit(".", 1)[-1]
        return canonicalize_failure_type(raw)
    numeric_names = {
        "1": "Instruction/Plan Adherence Failure",
        "2": "Invention of New Information",
        "3": "Invalid Invocation",
        "4": "Misinterpretation of Tool Output / Handoff Failure",
        "5": "Intent-Plan Misalignment",
        "6": "Underspecified User Intent",
        "7": "Intent Not Supported",
        "8": "Guardrails Triggered",
        "9": "System Failure",
        "10": "Inconclusive (USE SPARINGLY)",
    }
    return canonicalize_category(numeric_names.get(raw, raw))


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    pairs: list[tuple[float, float]], samples: int, confidence: float, seed: int
) -> dict[str, Any]:
    if not pairs:
        return {"n_pairs": 0, "difference": None, "ci_low": None, "ci_high": None}
    differences = [left - right for left, right in pairs]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    estimates = [
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(samples)
    ]
    alpha = 1 - confidence
    return {
        "n_pairs": len(pairs),
        "difference": observed,
        "ci_low": _percentile(estimates, alpha / 2),
        "ci_high": _percentile(estimates, 1 - alpha / 2),
    }


def generate_report(
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
    predictions = list(iter_jsonl(prediction_path))
    evaluations = list(iter_jsonl(evaluation_path))
    if not predictions:
        raise ValueError("Prediction file is empty")
    run_id = predictions[0]["run_id"]

    processed = resolve_path(config, "processed_data")
    gold: dict[tuple[str, str], dict[str, Any]] = {}
    trajectories: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_inputs.jsonl"):
            trajectories[(dataset, row["trajectory_id"])] = row
        for row in iter_jsonl(processed / f"{dataset}_gold.jsonl"):
            gold[(dataset, row["trajectory_id"])] = row

    taxonomies = load_taxonomy_set(
        resolve_path(config, "taxonomy_a"),
        resolve_path(config, "taxonomy_b"),
        resolve_path(config, "merged_taxonomy"),
    )

    pred_index = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row for row in predictions
    }
    eval_index = {
        (row["dataset"], row["trajectory_id"], row["condition"]): row
        for row in evaluations
        if row.get("condition") in {condition.value for condition in CONDITION_ORDER}
    }
    grouped_keys: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in pred_index:
        grouped_keys[(key[0], key[2])].append(key)

    summaries: list[dict[str, Any]] = []
    for dataset in experiment_datasets(config):
        for condition in CONDITION_ORDER:
            keys = grouped_keys.get((dataset, condition.value), [])
            if not keys:
                continue
            correct_values: list[float] = []
            score_values: list[float] = []
            official_correct_values: list[float] = []
            official_score_values: list[float] = []
            localization_hits: list[float] = []
            native_hits: list[float] = []
            taxonomy_adherence_hits: list[float] = []
            parse_failures = 0
            evaluator_failures = 0
            for key in keys:
                prediction_row = pred_index[key]
                if prediction_row.get("status") != "ok":
                    parse_failures += 1
                elif condition != Condition.NO_TAXONOMY:
                    supplied_taxonomy = taxonomy_for_condition(dataset, condition, taxonomies)
                    valid_names = {item["name"] for item in supplied_taxonomy["categories"]}
                    taxonomy_adherence_hits.append(
                        float(_primary_prediction(prediction_row).get("error_type") in valid_names)
                    )
                evaluation_row = eval_index.get(key)
                if not evaluation_row or evaluation_row.get("status") == "evaluator_error":
                    evaluator_failures += 1
                    continue
                evaluation = evaluation_row["evaluation"]
                official_correct_values.append(float(bool(evaluation["correct"])))
                official_score_values.append(float(evaluation["score"]))
                if evaluation_row.get("reference_consistency") != "consistent":
                    continue
                correct_values.append(float(bool(evaluation["correct"])))
                score_values.append(float(evaluation["score"]))
                primary = _primary_prediction(prediction_row)
                predicted_step = primary.get("failure_step")
                gold_row = gold[(dataset, key[1])]
                if gold_row.get("critical_failure_step") is not None:
                    localization_hits.append(
                        float(predicted_step == gold_row["critical_failure_step"])
                    )
                if condition == Condition.OWN_TAXONOMY:
                    predicted_type = _canonical_predicted_type(dataset, primary.get("error_type"))
                    accepted_types = set(
                        gold_row.get("failure_types") or [gold_row["failure_type"]]
                    )
                    native_hits.append(float(predicted_type in accepted_types))
            summaries.append(
                {
                    "dataset": dataset,
                    "condition": condition.value,
                    "n_planned": len(keys),
                    "n_evaluated": len(correct_values),
                    "attribution_accuracy": sum(correct_values) / len(correct_values) if correct_values else None,
                    "mean_semantic_score": sum(score_values) / len(score_values) if score_values else None,
                    "n_evaluated_all": len(official_correct_values),
                    "official_set_attribution_accuracy": sum(official_correct_values) / len(official_correct_values) if official_correct_values else None,
                    "official_set_mean_semantic_score": sum(official_score_values) / len(official_score_values) if official_score_values else None,
                    "reference_issue_rate": (len(official_correct_values) - len(correct_values)) / len(official_correct_values) if official_correct_values else None,
                    "native_label_accuracy": sum(native_hits) / len(native_hits) if native_hits else None,
                    "taxonomy_output_adherence": sum(taxonomy_adherence_hits) / len(taxonomy_adherence_hits) if taxonomy_adherence_hits else None,
                    "localization_accuracy": sum(localization_hits) / len(localization_hits) if localization_hits else None,
                    "parsing_failure_rate": parse_failures / len(keys),
                    "evaluator_failure_rate": evaluator_failures / len(keys),
                }
            )

    paired: dict[str, dict[str, Any]] = {}
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    confidence = float(config["evaluation"]["confidence_level"])
    base_seed = int(config["experiment"]["seed"])
    for dataset in experiment_datasets(config):
        paired[dataset] = {}
        ids = sorted({key[1] for key in pred_index if key[0] == dataset})
        for left, right in COMPARISONS:
            name = f"{left}_minus_{right}"
            accuracy_pairs: list[tuple[float, float]] = []
            score_pairs: list[tuple[float, float]] = []
            for trajectory_id in ids:
                left_row = eval_index.get((dataset, trajectory_id, left))
                right_row = eval_index.get((dataset, trajectory_id, right))
                if not left_row or not right_row:
                    continue
                if left_row.get("status") == "evaluator_error" or right_row.get("status") == "evaluator_error":
                    continue
                if left_row.get("reference_consistency") != "consistent" or right_row.get("reference_consistency") != "consistent":
                    continue
                left_eval = left_row["evaluation"]
                right_eval = right_row["evaluation"]
                accuracy_pairs.append((float(left_eval["correct"]), float(right_eval["correct"])))
                score_pairs.append((float(left_eval["score"]), float(right_eval["score"])))
            comparison_seed = base_seed + int(
                hashlib_sha256_int(f"{dataset}:{name}") % 1_000_000
            )
            paired[dataset][name] = {
                "accuracy": _paired_bootstrap(
                    accuracy_pairs, bootstrap_samples, confidence, comparison_seed
                ),
                "semantic_score": _paired_bootstrap(
                    score_pairs, bootstrap_samples, confidence, comparison_seed + 1
                ),
            }

    transitions: dict[str, Any] = {}
    transition_specs: dict[str, tuple[str, bool, str, bool]] = {
        "baseline_correct_foreign_incorrect": ("no_taxonomy", True, "foreign_taxonomy", False),
        "foreign_incorrect_merged_correct": ("foreign_taxonomy", False, "merged_taxonomy", True),
        "baseline_incorrect_foreign_correct": ("no_taxonomy", False, "foreign_taxonomy", True),
        "foreign_correct_merged_incorrect": ("foreign_taxonomy", True, "merged_taxonomy", False),
    }
    for dataset in experiment_datasets(config):
        transitions[dataset] = {}
        ids = sorted({key[1] for key in pred_index if key[0] == dataset})
        for name, (left, left_value, right, right_value) in transition_specs.items():
            matches: list[str] = []
            for trajectory_id in ids:
                left_row = eval_index.get((dataset, trajectory_id, left))
                right_row = eval_index.get((dataset, trajectory_id, right))
                if not left_row or not right_row:
                    continue
                if left_row.get("status") == "evaluator_error" or right_row.get("status") == "evaluator_error":
                    continue
                if left_row.get("reference_consistency") != "consistent" or right_row.get("reference_consistency") != "consistent":
                    continue
                if bool(left_row["evaluation"]["correct"]) == left_value and bool(
                    right_row["evaluation"]["correct"]
                ) == right_value:
                    matches.append(trajectory_id)
            representative_cases: list[dict[str, Any]] = []
            for trajectory_id in matches[:3]:
                gold_row = gold[(dataset, trajectory_id)]
                left_prediction = pred_index[(dataset, trajectory_id, left)]
                right_prediction = pred_index[(dataset, trajectory_id, right)]
                representative_cases.append(
                    {
                        "trajectory_id": trajectory_id,
                        "trajectory": trajectories[(dataset, trajectory_id)],
                        "reference": {
                            "failure_type": gold_row["failure_type"],
                            "failure_module": gold_row.get("failure_module"),
                            "critical_failure_step": gold_row.get("critical_failure_step"),
                            "reference_reasoning": gold_row.get("reference_reasoning"),
                        },
                        "left": {
                            "condition": left,
                            "prediction": left_prediction.get("prediction"),
                            "evaluation": eval_index[(dataset, trajectory_id, left)]["evaluation"],
                        },
                        "right": {
                            "condition": right,
                            "prediction": right_prediction.get("prediction"),
                            "evaluation": eval_index[(dataset, trajectory_id, right)]["evaluation"],
                        },
                    }
                )
            transitions[dataset][name] = {
                "count": len(matches),
                "example_ids": matches[:10],
                "representative_cases": representative_cases,
            }

    reference_audit: list[dict[str, Any]] = []
    seen_audit: set[tuple[str, str]] = set()
    for row in evaluations:
        key = (row.get("dataset", ""), row.get("trajectory_id", ""))
        consistency = row.get("reference_consistency")
        if consistency not in {"uncertain", "conflicting"} or key in seen_audit:
            continue
        seen_audit.add(key)
        gold_row = gold.get(key, {})
        reference_audit.append(
            {
                "dataset": key[0],
                "trajectory_id": key[1],
                "reference_consistency": consistency,
                "reason": row.get("reference_consistency_reason"),
                "failure_type": gold_row.get("failure_type"),
                "reference_reasoning": gold_row.get("reference_reasoning"),
            }
        )

    table_dir = resolve_path(config, "results") / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = table_dir / f"{run_id}_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    paired_path = table_dir / f"{run_id}_paired_differences.json"
    transitions_path = table_dir / f"{run_id}_transitions.json"
    reference_audit_path = table_dir / f"{run_id}_reference_audit.json"
    write_json(paired_path, paired)
    write_json(transitions_path, transitions)
    write_json(reference_audit_path, reference_audit)
    report_path = table_dir / f"{run_id}_results.md"
    report_path.write_text(
        _markdown_results(run_id, summaries, paired, transitions), encoding="utf-8", newline="\n"
    )
    summary_path = table_dir / f"{run_id}_summary.json"
    write_json(
        summary_path,
        {
            "run_id": run_id,
            "summaries": summaries,
            "paired_differences": paired,
            "transitions": transitions,
            "reference_audit": reference_audit,
        },
    )
    figure_paths = _generate_figures(run_id, summaries, paired, resolve_path(config, "results"))
    outputs = {
        "summary_csv": summary_csv,
        "paired_differences": paired_path,
        "transitions": transitions_path,
        "reference_audit": reference_audit_path,
        "report": report_path,
        "summary_json": summary_path,
    }
    outputs.update(figure_paths)
    return outputs


def hashlib_sha256_int(value: str) -> int:
    import hashlib

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _markdown_results(
    run_id: str,
    summaries: list[dict[str, Any]],
    paired: dict[str, dict[str, Any]],
    transitions: dict[str, Any],
) -> str:
    lines = [f"# Experiment Results: {run_id}", ""]
    dataset_order = list(dict.fromkeys(row["dataset"] for row in summaries))
    for dataset in dataset_order:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "Clean-reference metrics exclude evaluator-flagged uncertain or conflicting annotations; official-set metrics retain every evaluated annotation.",
                "",
                "| Condition | Clean N | Clean Accuracy | Official N | Official Accuracy | Clean Score | Reference Issue Rate | Taxonomy Output Adherence | Native Label Accuracy | Localization Accuracy | Parsing Failure Rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summaries:
            if row["dataset"] != dataset:
                continue
            lines.append(
                "| {name} | {n} | {acc} | {official_n} | {official_acc} | {score} | {issue} | {adherence} | {native} | {loc} | {parse} |".format(
                    name=DISPLAY_NAMES[row["condition"]],
                    n=row["n_evaluated"],
                    acc=_fmt(row["attribution_accuracy"]),
                    official_n=row["n_evaluated_all"],
                    official_acc=_fmt(row["official_set_attribution_accuracy"]),
                    score=_fmt(row["mean_semantic_score"]),
                    issue=_fmt(row["reference_issue_rate"]),
                    adherence=_fmt(row["taxonomy_output_adherence"]),
                    native=_fmt(row["native_label_accuracy"]),
                    loc=_fmt(row["localization_accuracy"]),
                    parse=_fmt(row["parsing_failure_rate"]),
                )
            )
        lines.extend(["", "### Important paired differences", ""])
        for name in (
            "foreign_taxonomy_minus_no_taxonomy",
            "merged_taxonomy_minus_foreign_taxonomy",
            "own_taxonomy_minus_no_taxonomy",
            "merged_taxonomy_minus_no_taxonomy",
        ):
            item = paired[dataset].get(name, {})
            accuracy = item.get("accuracy", {})
            lines.append(
                f"- {name}: accuracy difference={_fmt(accuracy.get('difference'))}, "
                f"95% CI=[{_fmt(accuracy.get('ci_low'))}, {_fmt(accuracy.get('ci_high'))}], "
                f"n={accuracy.get('n_pairs', 0)}"
            )
        lines.extend(["", "### Transition cases", ""])
        for name, item in transitions[dataset].items():
            lines.append(f"- {name}: {item['count']}")
        lines.append("")
    return "\n".join(lines)


def _escape_svg(value: str) -> str:
    import html

    return html.escape(value, quote=True)


def _bar_svg(title: str, labels: list[str], values: list[float], maximum: float) -> str:
    width, height = 760, 460
    left, top, plot_width, plot_height = 80, 60, 630, 300
    slot = plot_width / max(len(values), 1)
    bar_width = slot * 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{_escape_svg(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height - (value / maximum) * plot_height
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>'
        )
    colors = ["#64748b", "#2563eb", "#dc2626", "#16a34a"]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + slot * index + (slot - bar_width) / 2
        bar_height = max(0, min(value / maximum, 1)) * plot_height
        y = top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[index % len(colors)]}"/>'
        )
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.3f}</text>'
        )
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{top+plot_height+25}" text-anchor="middle" font-family="sans-serif" font-size="12">{_escape_svg(label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _difference_svg(title: str, labels: list[str], values: list[float]) -> str:
    width, height = 840, 480
    left, top, plot_width, plot_height = 80, 60, 700, 320
    zero_y = top + plot_height / 2
    slot = plot_width / max(len(values), 1)
    bar_width = slot * 0.55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{_escape_svg(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{zero_y}" x2="{left+plot_width}" y2="{zero_y}" stroke="#111" stroke-width="2"/>',
    ]
    for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = zero_y - tick * (plot_height / 2)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.1f}</text>'
        )
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + slot * index + (slot - bar_width) / 2
        height_value = abs(max(-1.0, min(value, 1.0))) * plot_height / 2
        y = zero_y - height_value if value >= 0 else zero_y
        color = "#16a34a" if value >= 0 else "#dc2626"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height_value:.1f}" fill="{color}"/>'
        )
        label_y = y - 7 if value >= 0 else y + height_value + 16
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{label_y:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:+.3f}</text>'
        )
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{top+plot_height+25}" text-anchor="middle" font-family="sans-serif" font-size="11">{_escape_svg(label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _generate_figures(
    run_id: str,
    summaries: list[dict[str, Any]],
    paired: dict[str, dict[str, Any]],
    results_root: Path,
) -> dict[str, Path]:
    figure_dir = results_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    labels = ["No Taxonomy", "Own", "Foreign", "Merged"]
    dataset_order = list(dict.fromkeys(row["dataset"] for row in summaries))
    for dataset in dataset_order:
        by_condition = {
            row["condition"]: row for row in summaries if row["dataset"] == dataset
        }
        if not by_condition:
            continue
        values = [
            float(by_condition[condition.value]["attribution_accuracy"] or 0)
            for condition in CONDITION_ORDER
        ]
        path = figure_dir / f"{run_id}_{dataset}_accuracy.svg"
        path.write_text(
            _bar_svg(f"{dataset}: Attribution Accuracy", labels, values, 1.0),
            encoding="utf-8",
            newline="\n",
        )
        outputs[f"{dataset}_accuracy_figure"] = path
    difference_labels = [
        f"{dataset} Foreign-Base"
        if comparison == "foreign_taxonomy_minus_no_taxonomy"
        else f"{dataset} Merged-Foreign"
        for dataset in dataset_order
        for comparison in (
            "foreign_taxonomy_minus_no_taxonomy",
            "merged_taxonomy_minus_foreign_taxonomy",
        )
    ]
    differences = [
        float(
            paired[dataset][comparison]["accuracy"]["difference"] or 0
        )
        for dataset in dataset_order
        for comparison in (
            "foreign_taxonomy_minus_no_taxonomy",
            "merged_taxonomy_minus_foreign_taxonomy",
        )
    ]
    difference_path = figure_dir / f"{run_id}_important_differences.svg"
    difference_path.write_text(
        _difference_svg(
            "Important Accuracy Differences",
            difference_labels,
            differences,
        ),
        encoding="utf-8",
        newline="\n",
    )
    outputs["important_differences_figure"] = difference_path
    return outputs
