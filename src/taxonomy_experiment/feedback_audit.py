from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, read_json, write_json
from taxonomy_experiment.native_evaluation import candidate_narrative
from taxonomy_experiment.reporting import DISPLAY_NAMES
from taxonomy_experiment.token_count import TokenCounter


def _narrative_text(row: dict[str, Any]) -> str:
    narrative = candidate_narrative(row)
    return "\n".join(
        [
            str(narrative["explanation"]),
            *[str(item) for item in narrative["evidence"]],
            str(narrative["overall_root_cause"]),
        ]
    )


def _normalized_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _exact_occurs(text: str, label: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w){re.escape(label)}(?!\w)",
            text,
            flags=re.IGNORECASE | re.UNICODE,
        )
    )


def _normalized_occurs(text: str, label: str) -> bool:
    normalized_text = f" {_normalized_phrase(text)} "
    normalized_label = _normalized_phrase(label)
    return bool(normalized_label) and f" {normalized_label} " in normalized_text


def run_lexical_leakage_audit(
    prediction_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
    root = config["_root"]
    prediction_path = Path(prediction_path)
    if not prediction_path.is_absolute():
        prediction_path = root / prediction_path
    predictions = list(iter_jsonl(prediction_path))
    if not predictions:
        raise ValueError("Prediction file is empty")
    run_id = predictions[0]["run_id"]

    native_taxonomies = {
        dataset: read_json(resolve_path(config, path_key))
        for dataset, path_key in zip(
            experiment_datasets(config), ("taxonomy_a", "taxonomy_b")
        )
    }
    native_names = {
        dataset: [item["name"] for item in taxonomy["categories"]]
        for dataset, taxonomy in native_taxonomies.items()
    }
    gold: dict[tuple[str, str], list[str]] = {}
    processed = resolve_path(config, "processed_data")
    for dataset in experiment_datasets(config):
        for row in iter_jsonl(processed / f"{dataset}_gold.jsonl"):
            gold[(dataset, row["trajectory_id"])] = list(
                row.get("failure_types") or [row["failure_type"]]
            )

    detail_rows: list[dict[str, Any]] = []
    for row in predictions:
        if row.get("status") != "ok":
            continue
        dataset = row["dataset"]
        text = _narrative_text(row)
        accepted = gold[(dataset, row["trajectory_id"])]
        source_label = row["prediction"]["predicted_errors"][0]["error_type"]
        exact_native = [
            name for name in native_names[dataset] if _exact_occurs(text, name)
        ]
        normalized_native = [
            name for name in native_names[dataset] if _normalized_occurs(text, name)
        ]
        exact_gold = [name for name in accepted if _exact_occurs(text, name)]
        normalized_gold = [
            name for name in accepted if _normalized_occurs(text, name)
        ]
        detail_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "domain": row["domain"],
                "trajectory_id": row["trajectory_id"],
                "condition": row["condition"],
                "any_native_exact": bool(exact_native),
                "any_native_normalized": bool(normalized_native),
                "gold_label_exact": bool(exact_gold),
                "gold_label_normalized": bool(normalized_gold),
                "source_label_exact": _exact_occurs(text, source_label),
                "source_label_normalized": _normalized_occurs(text, source_label),
                "matched_native_exact": " | ".join(exact_native),
                "matched_native_normalized": " | ".join(normalized_native),
                "matched_gold_exact": " | ".join(exact_gold),
                "matched_gold_normalized": " | ".join(normalized_gold),
                "source_label": source_label,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    metric_names = (
        "any_native_exact",
        "any_native_normalized",
        "gold_label_exact",
        "gold_label_normalized",
        "source_label_exact",
        "source_label_normalized",
    )
    summary_rows: list[dict[str, Any]] = []
    for dataset in experiment_datasets(config):
        for condition in (
            "no_taxonomy",
            "own_taxonomy",
            "foreign_taxonomy",
            "merged_taxonomy",
        ):
            rows = grouped[(dataset, condition)]
            summary: dict[str, Any] = {
                "dataset": dataset,
                "condition": condition,
                "n": len(rows),
            }
            for metric in metric_names:
                count = sum(bool(row[metric]) for row in rows)
                summary[f"{metric}_count"] = count
                summary[f"{metric}_rate"] = count / len(rows) if rows else None
            summary_rows.append(summary)

    output_dir = resolve_path(config, "results") / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_lexical_leakage_audit"
    summary_path = output_dir / f"{stem}.json"
    summary_csv = output_dir / f"{stem}_summary.csv"
    details_csv = output_dir / f"{stem}_details.csv"
    markdown_path = output_dir / f"{stem}.md"
    write_json(
        summary_path,
        {
            "run_id": run_id,
            "audit_scope": "candidate explanation, evidence, and overall_root_cause",
            "exact_definition": "case-insensitive native category string occurrence",
            "normalized_definition": "case- and punctuation/spacing-normalized full category phrase occurrence; not a semantic similarity test",
            "summary": summary_rows,
        },
    )
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with details_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    lines = [
        "# Lexical Leakage Audit",
        "",
        f"Run: `{run_id}`",
        "",
        "This audit checks whether candidate explanation, evidence, or overall root cause directly repeats target native-category phrases. Normalized matching tolerates case, punctuation, hyphen, slash, and whitespace differences; it is not a semantic-similarity test.",
        "",
        "| Dataset | Condition | N | Any native exact | Any native normalized | Gold exact | Gold normalized | Source label exact | Source label normalized |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {DISPLAY_NAMES[row['condition']]} | {row['n']} | "
            f"{row['any_native_exact_rate']:.2%} | "
            f"{row['any_native_normalized_rate']:.2%} | "
            f"{row['gold_label_exact_rate']:.2%} | "
            f"{row['gold_label_normalized_rate']:.2%} | "
            f"{row['source_label_exact_rate']:.2%} | "
            f"{row['source_label_normalized_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "`Any native` searches every category in the evaluation dataset's native taxonomy. `Gold` searches only the accepted gold category or categories. `Source label` checks whether the hidden `error_type` string was repeated in the visible narrative.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "markdown": markdown_path,
        "summary_json": summary_path,
        "summary_csv": summary_csv,
        "details_csv": details_csv,
    }


def generate_taxonomy_characteristics(
    config_path: str | Path = "config/experiment.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
    model = config["models"]["attribution"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    taxonomy_specs = (
        ("AgentRx", "taxonomy_a"),
        ("MAST", "taxonomy_b"),
        ("LLM-Merged", "merged_taxonomy"),
    )
    rows: list[dict[str, Any]] = []
    for display_name, path_key in taxonomy_specs:
        taxonomy = read_json(resolve_path(config, path_key))
        prompt_projection = json.dumps(
            [
                {
                    "id": item["id"],
                    "module": item.get("module"),
                    "name": item["name"],
                    "definition": item["definition"],
                }
                for item in taxonomy["categories"]
            ],
            ensure_ascii=False,
            indent=2,
        )
        rows.append(
            {
                "taxonomy": display_name,
                "taxonomy_id": taxonomy["taxonomy_id"],
                "categories": len(taxonomy["categories"]),
                "prompt_characters": len(prompt_projection),
                "prompt_tokens": counter.count(prompt_projection),
                "definition_characters": sum(
                    len(item["definition"]) for item in taxonomy["categories"]
                ),
                "tokenizer": counter.encoding_name,
                "exact_model_tokenizer": counter.exact_tokenizer_available,
            }
        )
    output_dir = resolve_path(config, "results") / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "taxonomy_characteristics.json"
    csv_path = output_dir / "taxonomy_characteristics.csv"
    markdown_path = output_dir / "taxonomy_characteristics.md"
    write_json(json_path, {"attribution_model": model["name"], "taxonomies": rows})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Taxonomy Characteristics",
        "",
        f"Tokenizer configured for attribution model: `{model['name']}` (`{counter.encoding_name}`).",
        "",
        "| Taxonomy | Categories | Prompt characters | Prompt tokens | Definition characters |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['taxonomy']} | {row['categories']} | "
            f"{row['prompt_characters']} | {row['prompt_tokens']} | "
            f"{row['definition_characters']} |"
        )
    lines.extend(
        [
            "",
            "Prompt token counts use the configured tokenizer approximation and document taxonomy-length confounding; they are not provider billing totals.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"markdown": markdown_path, "json": json_path, "csv": csv_path}

