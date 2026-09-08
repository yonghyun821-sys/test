from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.datasets import load_agentrx, load_mast
from taxonomy_experiment.io import read_json, write_json, write_jsonl
from taxonomy_experiment.models import trajectory_for_prompt
from taxonomy_experiment.token_count import TokenCounter


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _token_statistics(
    records: list[dict[str, Any]], counter: TokenCounter, context_window: int
) -> dict[str, Any]:
    counts = [
        {
            "trajectory_id": record["trajectory_id"],
            "domain": record["domain"],
            "tokens": counter.count(trajectory_for_prompt(record)),
        }
        for record in records
    ]
    values = [item["tokens"] for item in counts]
    over = [item for item in counts if item["tokens"] >= context_window]
    largest = sorted(counts, key=lambda item: item["tokens"], reverse=True)[:20]
    return {
        "model": counter.model,
        "encoding": counter.encoding_name,
        "exact_model_tokenizer": counter.exact_tokenizer_available,
        "context_window": context_window,
        "minimum": min(values) if values else 0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "maximum": max(values) if values else 0,
        "mean": (sum(values) / len(values)) if values else 0,
        "at_or_over_context_count": len(over),
        "at_or_over_context": over,
        "largest_20": largest,
    }


def _category_statistics(gold: list[dict[str, Any]]) -> dict[str, Any]:
    raw = Counter(
        label
        for item in gold
        for label in (item.get("failure_types_raw") or [item.get("failure_type_raw") or "<MISSING>"])
    )
    canonical = Counter(
        label
        for item in gold
        for label in (item.get("failure_types") or [item.get("failure_type") or "<MISSING>"])
    )
    modules_raw = Counter(item["failure_module_raw"] for item in gold if item["failure_module_raw"])
    modules = Counter(item["failure_module"] for item in gold if item["failure_module"])
    return {
        "unique_raw_gold_categories": dict(sorted(raw.items())),
        "canonicalized_gold_categories": dict(sorted(canonical.items())),
        "raw_modules": dict(sorted(modules_raw.items())),
        "canonicalized_modules": dict(sorted(modules.items())),
    }


def _taxonomy_summary(path: Path) -> dict[str, Any]:
    taxonomy = read_json(path)
    categories = taxonomy.get("categories") or []
    declared = taxonomy.get("category_count")
    if declared != len(categories):
        raise ValueError(f"Taxonomy count mismatch in {path}: declared={declared}, actual={len(categories)}")
    ids = [item["id"] for item in categories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate taxonomy category IDs in {path}")
    return {
        "taxonomy_id": taxonomy["taxonomy_id"],
        "category_count": len(categories),
        "categories": [
            {"id": item["id"], "module": item.get("module"), "name": item["name"]}
            for item in categories
        ],
        "source": taxonomy.get("source"),
        "representation_policy": taxonomy.get("representation_policy"),
        "excluded_implementation_entries": taxonomy.get("excluded_implementation_entries", []),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Inspection Report",
        "",
        "This report is generated deterministically before any LLM inference. Gold files are used only for matching and inspection; processed inference inputs are written separately from processed gold annotations.",
        "",
    ]
    for dataset in report["dataset_order"]:
        section = report["datasets"][dataset]
        stats = section["counts"]
        tokens = section["trajectory_token_lengths"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Total trajectories: {stats['total_trajectories']}",
                f"- Total official annotations: {stats['total_official_annotations']}",
                f"- Matched pairs: {stats['matched_pairs']}",
                f"- Attribution-eligible pairs: {stats['eligible_attribution_pairs']}",
                f"- Missing gold failure types: {stats['missing_gold_failure_types']}",
                f"- Missing gold reasoning: {stats['missing_gold_reasoning']}",
                f"- Unmatched annotation IDs: {len(stats['unmatched_annotation_ids'])}",
                f"- Unmatched trajectory IDs: {len(stats['unmatched_trajectory_ids'])}",
                "",
                "### Normalized trajectory token lengths",
                "",
                f"Tokenizer: `{tokens['encoding']}` for `{tokens['model']}`; exact model mapping: `{tokens['exact_model_tokenizer']}`.",
                "",
                f"Min={tokens['minimum']}, p50={tokens['p50']:.1f}, p90={tokens['p90']:.1f}, p95={tokens['p95']:.1f}, mean={tokens['mean']:.1f}, max={tokens['maximum']}.",
                "",
                f"Trajectories at or over configured context window: {tokens['at_or_over_context_count']}.",
                "",
                "### Gold category counts",
                "",
                "```json",
                json.dumps(section["categories"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Canonical taxonomies",
            "",
            *[
                f"- {item['taxonomy_id']} categories: {item['category_count']}"
                for item in report["taxonomies"].values()
            ],
            "",
            "The complete category lists and provenance are available in `inspection_report.json` and the canonical taxonomy JSON files.",
            "",
        ]
    )
    return "\n".join(lines)


def run_inspection(config_path: str | Path = "config/experiment.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    root = config["_root"]
    rx_inputs, rx_gold, rx_stats = load_agentrx(
        resolve_path(config, "agentrx_tau_trajectories"),
        resolve_path(config, "agentrx_tau_annotations"),
        resolve_path(config, "agentrx_magentic_trajectories"),
        resolve_path(config, "agentrx_magentic_annotations"),
        root,
    )
    mast_config = config["mast"]
    mast_inputs, mast_gold, mast_stats = load_mast(
        resolve_path(config, "mast_dataset"),
        root,
        mas_name=str(mast_config["mas_name"]),
        benchmark_name=str(mast_config["benchmark_name"]),
        only_annotated_failures=bool(mast_config["only_annotated_failures"]),
    )
    dataset_payloads = {
        "agentrx": (rx_inputs, rx_gold, rx_stats),
        "mast": (mast_inputs, mast_gold, mast_stats),
    }
    dataset_order = experiment_datasets(config)
    processed = resolve_path(config, "processed_data")
    for dataset in dataset_order:
        inputs, gold, _ = dataset_payloads[dataset]
        write_jsonl(processed / f"{dataset}_inputs.jsonl", inputs)
        write_jsonl(processed / f"{dataset}_gold.jsonl", gold)
        write_json(
            processed / f"{dataset}_attribution_ids.json",
            [item["trajectory_id"] for item in gold if item["eligible_for_attribution"]],
        )

    attribution_model = config["models"]["attribution"]
    counter = TokenCounter.for_model(
        attribution_model["name"], attribution_model.get("tokenizer_encoding")
    )
    context_window = int(attribution_model["context_window"])
    report = {
        "inspection_version": 1,
        "inference_started": False,
        "dataset_order": list(dataset_order),
        "datasets": {
            dataset: {
                "counts": dataset_payloads[dataset][2],
                "categories": _category_statistics(dataset_payloads[dataset][1]),
                "trajectory_token_lengths": _token_statistics(
                    dataset_payloads[dataset][0], counter, context_window
                ),
            }
            for dataset in dataset_order
        },
        "taxonomies": {
            item["taxonomy_id"]: item
            for item in (
                _taxonomy_summary(resolve_path(config, "taxonomy_a")),
                _taxonomy_summary(resolve_path(config, "taxonomy_b")),
            )
        },
    }
    write_json(processed / "inspection_report.json", report)
    markdown_path = processed / "inspection_report.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    return report
