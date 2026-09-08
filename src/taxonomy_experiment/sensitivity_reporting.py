from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import experiment_datasets, load_config, resolve_path
from taxonomy_experiment.io import iter_jsonl, write_json
from taxonomy_experiment.native_reporting import (
    exact_mcnemar_p_value,
    paired_bootstrap_interval,
)
from taxonomy_experiment.reporting import DISPLAY_NAMES


def compare_native_evaluations(
    primary_evaluation_path: str | Path,
    masked_evaluation_path: str | Path,
    config_path: str | Path = "config/experiment.yaml",
) -> dict[str, Path]:
    config = load_config(config_path)
    root = config["_root"]

    def load(path_value: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
        path = Path(path_value)
        if not path.is_absolute():
            path = root / path
        return {
            (row["dataset"], row["trajectory_id"], row["condition"]): row
            for row in iter_jsonl(path)
        }

    primary = load(primary_evaluation_path)
    masked = load(masked_evaluation_path)
    if set(primary) != set(masked):
        raise ValueError("Primary and masked evaluation keys do not match")
    run_ids = {row["run_id"] for row in primary.values()} | {
        row["run_id"] for row in masked.values()
    }
    if len(run_ids) != 1:
        raise ValueError(f"Evaluation run IDs do not match: {sorted(run_ids)}")
    run_id = next(iter(run_ids))
    samples = int(config["evaluation"]["bootstrap_samples"])
    confidence = float(config["evaluation"]["confidence_level"])
    rows: list[dict[str, Any]] = []
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
                and masked[key].get("status") == "ok"
            ]
            primary_correct = [bool(primary[key]["correct"]) for key in valid]
            masked_correct = [bool(masked[key]["correct"]) for key in valid]
            improved = sum(
                (not left) and right
                for left, right in zip(primary_correct, masked_correct)
            )
            regressed = sum(
                left and (not right)
                for left, right in zip(primary_correct, masked_correct)
            )
            primary_accuracy = sum(primary_correct) / len(valid)
            masked_accuracy = sum(masked_correct) / len(valid)
            ci_low, ci_high = paired_bootstrap_interval(
                primary_correct,
                masked_correct,
                samples=samples,
                confidence=confidence,
                seed=(
                    f"{config['experiment']['seed']}:{dataset}:{condition}:"
                    "masked-vs-primary"
                ),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(valid),
                    "primary_accuracy": primary_accuracy,
                    "masked_accuracy": masked_accuracy,
                    "masked_minus_primary": masked_accuracy - primary_accuracy,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "improved_after_masking": improved,
                    "regressed_after_masking": regressed,
                    "mcnemar_exact_p_value": exact_mcnemar_p_value(
                        improved, regressed
                    ),
                }
            )
    output_dir = resolve_path(config, "results") / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}_masked_native_terms_sensitivity"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    write_json(
        json_path,
        {
            "run_id": run_id,
            "contrast": "native-category-term-masked relabeling minus primary relabeling",
            "results": rows,
        },
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Native Category-Term Masking Sensitivity",
        "",
        f"Run: `{run_id}`",
        "",
        "| Dataset | Condition | N | Primary | Masked | Masked − primary | 95% paired-bootstrap CI | Improved | Regressed | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {DISPLAY_NAMES[row['condition']]} | "
            f"{row['n']} | {row['primary_accuracy']:.4f} | "
            f"{row['masked_accuracy']:.4f} | "
            f"{row['masked_minus_primary']:+.4f} | "
            f"[{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}] | "
            f"{row['improved_after_masking']} | "
            f"{row['regressed_after_masking']} | "
            f"{row['mcnemar_exact_p_value']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This sensitivity changes only the relabeler input by replacing explicit target native-category phrases with `[FAILURE_CATEGORY_TERM]`; attribution predictions are unchanged.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"markdown": markdown_path, "json": json_path, "csv": csv_path}

