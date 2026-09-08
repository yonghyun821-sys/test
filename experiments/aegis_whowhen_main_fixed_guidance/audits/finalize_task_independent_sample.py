from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taxonomy_experiment.config import load_config

from experiments.aegis_whowhen_main_fixed_guidance.audits.task_independence import (
    verify_frozen_identity,
)
from experiments.aegis_whowhen_main_fixed_guidance.core import leakage_audit
from experiments.aegis_whowhen_pilot_fixed_guidance.core import (
    _load_aegis_candidates,
    _load_whowhen_candidates,
)


SEED = 20260907
TARGET_N = 500
ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "aegis_whowhen_main_fixed_guidance"
RUN = EXPERIMENT / "implementation_runs" / "proportional-stratified-v4"
FROZEN = RUN / "frozen"
FINAL = FROZEN / "final_task_independent_sample"
CONFIG_PATH = EXPERIMENT / "experiment.yaml"
PILOT_INPUTS = (
    ROOT
    / "experiments"
    / "aegis_whowhen_pilot_fixed_guidance"
    / "prepared"
    / "pilot_inputs.jsonl"
)
SMOKE_MANIFEST = RUN / "smoke" / "smoke_sampling_manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def task_hash(task_query: str) -> str:
    if not isinstance(task_query, str):
        raise TypeError("task_query must be an exact string")
    return sha256_bytes(task_query.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def freeze_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise FileExistsError(f"Frozen artifact differs: {path}")
        return
    path.write_bytes(value)


def load_original_candidates(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    aegis, _ = _load_aegis_candidates(config)
    whowhen, _ = _load_whowhen_candidates(config)
    datasets = {"aegis": aegis, "whowhen": whowhen}
    for dataset, rows in datasets.items():
        counts = Counter(str(row["trajectory_id"]) for row in rows)
        for row in rows:
            source_id = str(row["trajectory_id"])
            row["source_trajectory_id"] = source_id
            if counts[source_id] > 1:
                row["trajectory_id"] = (
                    f"{source_id}::split={row['split']}::payload="
                    f"{sha256_json(row['trajectory_payload'])[:12]}"
                )
            query = row["trajectory_payload"].get("task_query")
            row["task_hash"] = task_hash(query)
        resolved = [row["trajectory_id"] for row in rows]
        if len(resolved) != len(set(resolved)):
            raise ValueError(f"Non-unique resolved trajectory IDs in {dataset}")
    return datasets


def hamilton_quotas(
    counts: dict[str, int], *, target: int, namespace: str
) -> tuple[dict[str, int], dict[str, float]]:
    total = sum(counts.values())
    if target > total or target < 0 or not counts:
        raise ValueError(f"Invalid Hamilton allocation: target={target}, total={total}")
    quotas = {key: target * value // total for key, value in counts.items()}
    expected = {key: target * value / total for key, value in counts.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (
            -((target * counts[key]) % total),
            sha256_bytes(f"{SEED}:{namespace}:{key}".encode("utf-8")),
            key,
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    if sum(quotas.values()) != target:
        raise ValueError("Hamilton quotas do not sum to target")
    return quotas, expected


def enforce_category_coverage(
    quotas: dict[str, int],
    expected: dict[str, float],
    *,
    namespace: str,
) -> list[dict[str, Any]]:
    adjustments = []
    zero_categories = sorted(key for key, value in quotas.items() if value == 0)
    for category in zero_categories:
        donors = [key for key, value in quotas.items() if value > 1]
        if not donors:
            raise ValueError("Cannot restore one-per-class coverage")
        donor = sorted(
            donors,
            key=lambda key: (
                -(quotas[key] - expected[key]),
                -quotas[key],
                sha256_bytes(
                    f"{SEED}:{namespace}:coverage-donor:{category}:{key}".encode(
                        "utf-8"
                    )
                ),
                key,
            ),
        )[0]
        quotas[donor] -= 1
        quotas[category] = 1
        adjustments.append(
            {
                "category_given_minimum_one": category,
                "donor_category": donor,
                "reason": "Hamilton quota was zero for an observed native class",
            }
        )
    return adjustments


def representative_pool(
    dataset: str,
    rows: list[dict[str, Any]],
    native_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["task_hash"]].append(row)
    representatives: dict[str, dict[str, Any]] = {}
    selection_hashes: dict[tuple[str, str], str] = {}
    for task_id, members in sorted(clusters.items()):
        ranked = []
        for row in members:
            candidate_hash = sha256_bytes(
                f"{SEED}:{dataset}:{task_id}:{row['trajectory_id']}".encode("utf-8")
            )
            selection_hashes[(task_id, row["trajectory_id"])] = candidate_hash
            ranked.append((candidate_hash, row["trajectory_id"], row))
        representatives[task_id] = min(ranked)[2]

    initial_counts = Counter(
        row["gold_category"] for row in representatives.values()
    )
    missing = sorted(set(native_ids) - set(initial_counts))
    repairs = []
    if missing:
        options: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        for category in missing:
            choices = []
            for task_id, members in clusters.items():
                current = representatives[task_id]
                for row in members:
                    if row["gold_category"] != category:
                        continue
                    repair_hash = sha256_bytes(
                        (
                            f"{SEED}:{dataset}:coverage-repair:{category}:"
                            f"{task_id}:{row['trajectory_id']}"
                        ).encode("utf-8")
                    )
                    choices.append((repair_hash, task_id, row))
            options[category] = sorted(choices, key=lambda item: (item[0], item[1]))

        ordered_missing = sorted(missing, key=lambda key: (len(options[key]), key))
        solution: dict[str, tuple[str, dict[str, Any]]] | None = None

        def search(
            index: int,
            chosen: dict[str, tuple[str, dict[str, Any]]],
            used_tasks: set[str],
        ) -> None:
            nonlocal solution
            if solution is not None:
                return
            if index == len(ordered_missing):
                removed = Counter(
                    representatives[task_id]["gold_category"]
                    for task_id, _ in chosen.values()
                )
                if all(initial_counts[key] - removed[key] > 0 for key in initial_counts):
                    solution = dict(chosen)
                return
            category = ordered_missing[index]
            for _, task_id, row in options[category]:
                if task_id in used_tasks:
                    continue
                chosen[category] = (task_id, row)
                used_tasks.add(task_id)
                search(index + 1, chosen, used_tasks)
                used_tasks.remove(task_id)
                chosen.pop(category)
                if solution is not None:
                    return

        search(0, {}, set())
        if solution is None:
            raise ValueError(
                f"Cannot restore all native categories with minimum swaps in {dataset}"
            )
        for category in sorted(solution):
            task_id, replacement = solution[category]
            previous = representatives[task_id]
            representatives[task_id] = replacement
            repairs.append(
                {
                    "task_hash": task_id,
                    "from_trajectory_id": previous["trajectory_id"],
                    "from_category": previous["gold_category"],
                    "to_trajectory_id": replacement["trajectory_id"],
                    "to_category": category,
                    "repair_selection_hash": sha256_bytes(
                        (
                            f"{SEED}:{dataset}:coverage-repair:{category}:"
                            f"{task_id}:{replacement['trajectory_id']}"
                        ).encode("utf-8")
                    ),
                }
            )

    final_counts = Counter(row["gold_category"] for row in representatives.values())
    if set(final_counts) != set(native_ids):
        raise ValueError(f"Native class coverage failed for {dataset}")
    pool = []
    for task_id, row in sorted(representatives.items()):
        enriched = dict(row)
        enriched["representative_selection_hash"] = selection_hashes[
            (task_id, row["trajectory_id"])
        ]
        enriched["coverage_repair"] = any(
            repair["task_hash"] == task_id for repair in repairs
        )
        pool.append(enriched)
    return pool, {
        "dataset": dataset,
        "input_eligible_trajectories_after_task_exclusions": len(rows),
        "unique_task_representatives": len(pool),
        "initial_category_counts": dict(sorted(initial_counts.items())),
        "initial_missing_categories": missing,
        "coverage_repairs": repairs,
        "coverage_repair_count": len(repairs),
        "final_category_counts": dict(sorted(final_counts.items())),
        "all_native_categories_represented": set(final_counts) == set(native_ids),
    }


def sample_unique_pool(
    dataset: str,
    pool: list[dict[str, Any]],
    native_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_n = min(TARGET_N, len(pool))
    if len(pool) <= TARGET_N:
        chosen = list(pool)
        audit = {
            "dataset": dataset,
            "unique_task_pool_size": len(pool),
            "target_n": TARGET_N,
            "final_n_before_cross_dataset_overlap": len(chosen),
            "method": "all_unique_task_representatives",
            "category_coverage_adjustments": [],
            "sampling_without_replacement": True,
        }
    else:
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            by_category[row["gold_category"]].append(row)
        category_counts = {
            key: len(value) for key, value in sorted(by_category.items())
        }
        category_quotas, category_expected = hamilton_quotas(
            category_counts,
            target=final_n,
            namespace=f"final:{dataset}:failure-mode",
        )
        adjustments = enforce_category_coverage(
            category_quotas,
            category_expected,
            namespace=f"final:{dataset}:failure-mode",
        )
        chosen = []
        strata = []
        for category in sorted(by_category):
            by_framework: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in by_category[category]:
                by_framework[row["framework"]].append(row)
            framework_counts = {
                key: len(value) for key, value in sorted(by_framework.items())
            }
            framework_quotas, framework_expected = hamilton_quotas(
                framework_counts,
                target=category_quotas[category],
                namespace=f"final:{dataset}:{category}:framework",
            )
            for framework in sorted(by_framework):
                ranked = sorted(
                    by_framework[framework],
                    key=lambda row: (
                        sha256_bytes(
                            (
                                f"{SEED}:{dataset}:final:{category}:{framework}:"
                                f"{row['task_hash']}:{row['trajectory_id']}"
                            ).encode("utf-8")
                        ),
                        row["trajectory_id"],
                    ),
                )
                quota = framework_quotas[framework]
                selected = ranked[:quota]
                chosen.extend(selected)
                strata.append(
                    {
                        "failure_mode": category,
                        "framework": framework,
                        "eligible_unique_tasks": framework_counts[framework],
                        "expected_sample_count": framework_expected[framework],
                        "allocated_quota": quota,
                        "selected_count": len(selected),
                    }
                )
        audit = {
            "dataset": dataset,
            "unique_task_pool_size": len(pool),
            "target_n": TARGET_N,
            "final_n_before_cross_dataset_overlap": len(chosen),
            "method": "deterministic_hierarchical_proportional_stratified_without_replacement",
            "primary_stratum": "gold_failure_mode",
            "secondary_stratum": "framework",
            "quota_allocation": "hamilton_largest_remainder",
            "proportional_reference_population": "one_representative_per_task_pool",
            "category_counts": category_counts,
            "category_expected_sample_counts": category_expected,
            "category_quotas": category_quotas,
            "category_coverage_adjustments": adjustments,
            "framework_strata": strata,
            "sampling_without_replacement": True,
        }
    if len(chosen) != final_n:
        raise ValueError(f"Final sample size mismatch in {dataset}")
    if len({row["task_hash"] for row in chosen}) != len(chosen):
        raise ValueError(f"Duplicate task in final {dataset} sample")
    counts = Counter(row["gold_category"] for row in chosen)
    if set(counts) != set(native_ids):
        raise ValueError(f"Final sample lost native class coverage in {dataset}")
    audit["final_category_counts_before_cross_dataset_overlap"] = dict(
        sorted(counts.items())
    )
    audit["all_native_categories_represented"] = True
    return chosen, audit


def public_pool_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "trajectory_id": row["trajectory_id"],
        "source_trajectory_id": row["source_trajectory_id"],
        "task_hash": row["task_hash"],
        "gold_failure_mode": row["gold_category"],
        "framework": row["framework"],
        "benchmark": row["benchmark"],
        "source_split": row["split"],
        "representative_selection_hash": row["representative_selection_hash"],
        "coverage_repair": row["coverage_repair"],
    }


def final_row(row: dict[str, Any]) -> dict[str, Any]:
    selection_hash = sha256_bytes(
        (
            f"{SEED}:{row['dataset']}:final:{row['gold_category']}:"
            f"{row['framework']}:{row['task_hash']}:{row['trajectory_id']}"
        ).encode("utf-8")
    )
    return {
        **public_pool_row(row),
        "sampling_seed": SEED,
        "final_sampling_selection_hash": selection_hash,
        "final_sampling_stratum": (
            f"{row['gold_category']}|{row['framework']}"
        ),
    }


def main() -> None:
    identity = verify_frozen_identity()
    if not identity["passed"] or identity["implementation_run_id"] != "main-95e2d9ee2ba7c1fd":
        raise RuntimeError("Prediction implementation identity is not frozen as expected")
    config = load_config(CONFIG_PATH)
    candidates = load_original_candidates(config)
    native_taxonomies = {
        dataset: read_json(FROZEN / f"{dataset}_taxonomy.json")
        for dataset in ("aegis", "whowhen")
    }
    native_ids = {
        dataset: [item["id"] for item in taxonomy["categories"]]
        for dataset, taxonomy in native_taxonomies.items()
    }

    pilot_rows = read_jsonl(PILOT_INPUTS)
    pilot_hashes_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in pilot_rows:
        pilot_hashes_by_dataset[row["dataset"]].add(
            task_hash(row["trajectory_payload"]["task_query"])
        )
    smoke_manifest = read_json(SMOKE_MANIFEST)
    candidate_lookup = {
        (row["dataset"], row["trajectory_id"]): row
        for rows in candidates.values()
        for row in rows
    }
    smoke_hashes_by_dataset: dict[str, set[str]] = defaultdict(set)
    for item in smoke_manifest["rows"]:
        row = candidate_lookup[(item["dataset"], item["trajectory_id"])]
        smoke_hashes_by_dataset[item["dataset"]].add(row["task_hash"])
    excluded_hashes = set().union(
        *pilot_hashes_by_dataset.values(), *smoke_hashes_by_dataset.values()
    )

    remaining: dict[str, list[dict[str, Any]]] = {}
    removed: dict[str, list[dict[str, Any]]] = {}
    for dataset, rows in candidates.items():
        remaining[dataset] = [row for row in rows if row["task_hash"] not in excluded_hashes]
        removed[dataset] = [row for row in rows if row["task_hash"] in excluded_hashes]
    exclusions = {
        "task_hash_definition": "SHA256(exact UTF-8 bytes of task_query); no normalization",
        "pilot_trajectory_ids": len(pilot_rows),
        "pilot_unique_task_hashes": len(set().union(*pilot_hashes_by_dataset.values())),
        "pilot_by_dataset": {
            dataset: {
                "trajectory_ids": sum(row["dataset"] == dataset for row in pilot_rows),
                "unique_task_hashes": len(pilot_hashes_by_dataset[dataset]),
            }
            for dataset in ("aegis", "whowhen")
        },
        "smoke_trajectory_ids": len(smoke_manifest["rows"]),
        "smoke_unique_task_hashes": len(set().union(*smoke_hashes_by_dataset.values())),
        "smoke_by_dataset": {
            dataset: {
                "trajectory_ids": sum(
                    row["dataset"] == dataset for row in smoke_manifest["rows"]
                ),
                "unique_task_hashes": len(smoke_hashes_by_dataset[dataset]),
            }
            for dataset in ("aegis", "whowhen")
        },
        "pilot_smoke_union_unique_task_hashes": len(excluded_hashes),
        "eligible_trajectories_removed": {
            dataset: len(removed[dataset]) for dataset in ("aegis", "whowhen")
        },
        "eligible_unique_tasks_removed": {
            dataset: len({row["task_hash"] for row in removed[dataset]})
            for dataset in ("aegis", "whowhen")
        },
        "excluded_task_hashes": sorted(excluded_hashes),
    }

    pools = {}
    representative_audits = {}
    samples = {}
    sampling_audits = {}
    for dataset in ("aegis", "whowhen"):
        pools[dataset], representative_audits[dataset] = representative_pool(
            dataset, remaining[dataset], native_ids[dataset]
        )
        samples[dataset], sampling_audits[dataset] = sample_unique_pool(
            dataset, pools[dataset], native_ids[dataset]
        )

    overlap = sorted(
        {row["task_hash"] for row in samples["aegis"]}
        & {row["task_hash"] for row in samples["whowhen"]}
    )
    if overlap:
        overlap_set = set(overlap)
        for dataset in ("aegis", "whowhen"):
            samples[dataset] = [
                row for row in samples[dataset] if row["task_hash"] not in overlap_set
            ]
    final_category_counts = {
        dataset: dict(
            sorted(Counter(row["gold_category"] for row in samples[dataset]).items())
        )
        for dataset in ("aegis", "whowhen")
    }
    for dataset in ("aegis", "whowhen"):
        if set(final_category_counts[dataset]) != set(native_ids[dataset]):
            raise ValueError(
                f"Cross-dataset overlap removal lost class coverage in {dataset}"
            )

    combined_records = samples["aegis"] + samples["whowhen"]
    leak = leakage_audit(combined_records)
    pilot_union = set().union(*pilot_hashes_by_dataset.values())
    smoke_union = set().union(*smoke_hashes_by_dataset.values())
    validation = {
        "trajectory_ids_unique_by_dataset": all(
            len(samples[dataset])
            == len({row["trajectory_id"] for row in samples[dataset]})
            for dataset in ("aegis", "whowhen")
        ),
        "task_hashes_unique_by_dataset": all(
            len(samples[dataset])
            == len({row["task_hash"] for row in samples[dataset]})
            for dataset in ("aegis", "whowhen")
        ),
        "one_trajectory_per_task_hash": all(
            len(samples[dataset])
            == len({row["task_hash"] for row in samples[dataset]})
            for dataset in ("aegis", "whowhen")
        ),
        "all_14_native_classes_represented": all(
            len(final_category_counts[dataset]) == 14
            for dataset in ("aegis", "whowhen")
        ),
        "pilot_task_hash_overlap_zero": not any(
            row["task_hash"] in pilot_union for row in combined_records
        ),
        "smoke_task_hash_overlap_zero": not any(
            row["task_hash"] in smoke_union for row in combined_records
        ),
        "cross_dataset_task_hash_overlap_zero": not overlap,
        "gold_leakage_zero": leak["leakage_count"] == 0,
        "prediction_or_result_data_used": False,
        "sampling_without_replacement": True,
        "rare_class_oversampling_only_for_minimum_coverage": True,
    }
    if not all(value is True for key, value in validation.items() if key != "prediction_or_result_data_used"):
        raise ValueError(f"Final sample validation failed: {validation}")
    if validation["prediction_or_result_data_used"] is not False:
        raise ValueError("Prediction/result data must not be used")

    final_rows = {
        dataset: sorted(
            [final_row(row) for row in samples[dataset]],
            key=lambda row: (row["task_hash"], row["trajectory_id"]),
        )
        for dataset in ("aegis", "whowhen")
    }
    inputs = sorted(
        [
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "source_trajectory_id": row["source_trajectory_id"],
                "task_hash": row["task_hash"],
                "framework": row["framework"],
                "benchmark": row["benchmark"],
                "source_split": row["split"],
                "trajectory_payload": row["trajectory_payload"],
                "trajectory_payload_sha256": sha256_json(row["trajectory_payload"]),
            }
            for row in combined_records
        ],
        key=lambda row: (row["dataset"], row["task_hash"], row["trajectory_id"]),
    )
    gold = sorted(
        [
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "source_trajectory_id": row["source_trajectory_id"],
                "task_hash": row["task_hash"],
                "gold_category": row["gold_category"],
            }
            for row in combined_records
        ],
        key=lambda row: (row["dataset"], row["task_hash"], row["trajectory_id"]),
    )
    sample_hashes = {
        "aegis": sha256_json(final_rows["aegis"]),
        "whowhen": sha256_json(final_rows["whowhen"]),
        "combined_main_input": sha256_json(inputs),
        "combined_gold": sha256_json(gold),
    }
    implementation_manifest = read_json(FROZEN / "implementation_manifest.json")
    sample_identity_material = {
        "prediction_implementation_id": identity["implementation_run_id"],
        "sample_hashes": sample_hashes,
        "taxonomy_hashes": implementation_manifest["taxonomy_hashes"],
        "prompt_hashes": implementation_manifest["prompt_hashes"],
        "schema_hashes": implementation_manifest["schema_hashes"],
        "model_config_hash": implementation_manifest["model_config_hash"],
    }
    main_sample_id = "sample-" + sha256_json(sample_identity_material)[:16]
    final_run_namespace = f"{identity['implementation_run_id']}--{main_sample_id}"
    framework_counts = {
        dataset: dict(
            sorted(Counter(row["framework"] for row in samples[dataset]).items())
        )
        for dataset in ("aegis", "whowhen")
    }
    manifest = {
        "status": "READY_FOR_MAIN_INFERENCE",
        "prediction_implementation_id": identity["implementation_run_id"],
        "main_sample_id": main_sample_id,
        "final_run_namespace": final_run_namespace,
        "sampling_seed": SEED,
        "target_n_per_dataset": TARGET_N,
        "independence_unit": "one exact agent-visible task_query hash per dataset",
        "task_hash": "SHA256(exact UTF-8 bytes of task_query); no normalization",
        "dataset_counts": {
            dataset: {
                "original_eligible_trajectories": len(candidates[dataset]),
                "eligible_trajectories_after_task_exclusions": len(remaining[dataset]),
                "eligible_unique_tasks": len(pools[dataset]),
                "final_n": len(samples[dataset]),
            }
            for dataset in ("aegis", "whowhen")
        },
        "total_main_trajectories": len(combined_records),
        "conditions": [
            "label_only",
            "own_taxonomy",
            "foreign_taxonomy",
            "merged_taxonomy",
        ],
        "planned_prediction_rows": 4 * len(combined_records),
        "category_counts": final_category_counts,
        "framework_counts": framework_counts,
        "task_level_exclusions": exclusions,
        "cross_dataset_task_hash_overlap": {
            "overlap_before_removal": len(overlap),
            "removed_from_each_dataset": len(overlap),
            "final_overlap": 0,
            "overlapping_hashes": overlap,
        },
        "leakage_count": leak["leakage_count"],
        "sample_hashes": sample_hashes,
        "sample_identity_material": sample_identity_material,
        "validation": validation,
        "representative_selection_audits": representative_audits,
        "sampling_audits": sampling_audits,
        "api_calls": 0,
        "attribution_inference_started": False,
        "accuracy_or_pilot_results_used": False,
        "old_duplicated_sample_retained": True,
        "old_duplicated_sample_status": "SUPERSEDED_BY_TASK_INDEPENDENT_SAMPLE",
    }

    freeze_bytes(FINAL / "aegis_unique_task_pool.jsonl", jsonl_bytes([
        public_pool_row(row) for row in pools["aegis"]
    ]))
    freeze_bytes(FINAL / "whowhen_unique_task_pool.jsonl", jsonl_bytes([
        public_pool_row(row) for row in pools["whowhen"]
    ]))
    freeze_bytes(FINAL / "final_task_level_exclusions.json", json_bytes(exclusions))
    freeze_bytes(
        FINAL / "final_sampling_audit.json",
        json_bytes(
            {
                "representative_selection": representative_audits,
                "final_sampling": sampling_audits,
                "final_validation": validation,
                "cross_dataset_overlap": manifest["cross_dataset_task_hash_overlap"],
            }
        ),
    )
    freeze_bytes(FINAL / "final_leakage_audit.json", json_bytes(leak))
    freeze_bytes(FINAL / "FINAL_MAIN_INPUTS.jsonl", jsonl_bytes(inputs))
    freeze_bytes(FINAL / "FINAL_MAIN_GOLD.jsonl", jsonl_bytes(gold))
    freeze_bytes(FINAL / "FINAL_MAIN_SAMPLE_MANIFEST.json", json_bytes(manifest))

    csv_buffer = io.StringIO(newline="")
    fields = list(final_rows["aegis"][0])
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for row in final_rows["aegis"] + final_rows["whowhen"]:
        writer.writerow(row)
    freeze_bytes(
        FINAL / "FINAL_MAIN_SAMPLE_MANIFEST.csv",
        csv_buffer.getvalue().encode("utf-8"),
    )
    report_lines = [
        "# Final Task-Independent Main Sample Audit",
        "",
        "Status: `READY_FOR_MAIN_INFERENCE`",
        "",
        f"- Prediction implementation ID: `{identity['implementation_run_id']}`",
        f"- Main sample ID: `{main_sample_id}`",
        f"- Final run namespace: `{final_run_namespace}`",
        f"- AEGIS: eligible trajectories after exclusions={len(remaining['aegis'])}, unique tasks={len(pools['aegis'])}, final N={len(samples['aegis'])}",
        f"- Who&When: eligible trajectories after exclusions={len(remaining['whowhen'])}, unique tasks={len(pools['whowhen'])}, final N={len(samples['whowhen'])}",
        f"- Pilot unique task hashes excluded: {exclusions['pilot_unique_task_hashes']}",
        f"- Smoke unique task hashes excluded: {exclusions['smoke_unique_task_hashes']}",
        "- Cross-dataset task-hash overlap: 0",
        f"- Leakage count: {leak['leakage_count']}",
        f"- Planned prediction rows: {manifest['planned_prediction_rows']}",
        "- Attribution API calls: 0",
        "- Attribution inference started: no",
        "",
        "## Sample hashes",
        "",
        *[f"- {key}: `{value}`" for key, value in sample_hashes.items()],
        "",
        "All trajectory IDs and task hashes are unique within dataset, all 14 native classes remain represented, and no pilot/smoke task hash is present.",
        "",
    ]
    freeze_bytes(
        FINAL / "FINAL_MAIN_SAMPLE_AUDIT.md",
        ("\n".join(report_lines) + "\n").encode("utf-8"),
    )
    freeze_bytes(
        FROZEN / "main_sample" / "SUPERSEDED_BY_TASK_INDEPENDENT_SAMPLE.md",
        (
            "# SUPERSEDED_BY_TASK_INDEPENDENT_SAMPLE\n\n"
            "The original duplicated 1,000 + 1,000 sample is retained for audit only "
            f"and must not be used for inference. Authoritative sample: `../final_task_independent_sample/` ({main_sample_id}).\n"
        ).encode("utf-8"),
    )
    run_namespace_dir = RUN / "main_runs" / final_run_namespace
    freeze_bytes(
        run_namespace_dir / "RUN_NAMESPACE.json",
        json_bytes(
            {
                "prediction_implementation_id": identity["implementation_run_id"],
                "main_sample_id": main_sample_id,
                "namespace": final_run_namespace,
                "prediction_status": "NOT_STARTED",
                "planned_prediction_rows": manifest["planned_prediction_rows"],
                "sample_manifest": str(
                    (FINAL / "FINAL_MAIN_SAMPLE_MANIFEST.json").relative_to(ROOT)
                ),
            }
        ),
    )
    active_pointer = json_bytes(
        {
            "status": "READY_FOR_MAIN_INFERENCE",
            "active_namespace": "implementation_runs/proportional-stratified-v4",
            "prediction_implementation_id": identity["implementation_run_id"],
            "main_sample_id": main_sample_id,
            "final_run_namespace": str(run_namespace_dir.relative_to(EXPERIMENT)),
            "main_inference_started": False,
            "main_inference_authorized": True,
            "planned_prediction_rows": manifest["planned_prediction_rows"],
            "authoritative_sample_manifest": str(
                (FINAL / "FINAL_MAIN_SAMPLE_MANIFEST.json").relative_to(EXPERIMENT)
            ),
            "note": "Do not use the superseded duplicated 1,000 + 1,000 sample.",
        }
    )
    # This is the sole mutable status pointer, not a frozen sample artifact.
    (EXPERIMENT / "ACTIVE_IMPLEMENTATION.json").write_bytes(active_pointer)
    print("status: READY_FOR_MAIN_INFERENCE")
    print(f"prediction_implementation_id: {identity['implementation_run_id']}")
    print(f"main_sample_id: {main_sample_id}")
    print(f"aegis: remaining={len(remaining['aegis'])} unique={len(pools['aegis'])} final={len(samples['aegis'])}")
    print(f"whowhen: remaining={len(remaining['whowhen'])} unique={len(pools['whowhen'])} final={len(samples['whowhen'])}")
    print(f"planned_prediction_rows: {manifest['planned_prediction_rows']}")
    print(f"manifest: {FINAL / 'FINAL_MAIN_SAMPLE_MANIFEST.json'}")


if __name__ == "__main__":
    main()
