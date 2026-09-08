from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUN = (
    ROOT
    / "experiments"
    / "aegis_whowhen_main_fixed_guidance"
    / "implementation_runs"
    / "proportional-stratified-v4"
)
FROZEN = RUN / "frozen"
INPUTS = FROZEN / "main_sample" / "main_inputs.jsonl"
OUTPUT = FROZEN / "main_task_independence_audit.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_frozen_identity() -> dict[str, Any]:
    manifest = read_json(FROZEN / "implementation_manifest.json")
    ready = read_json(FROZEN / "READY_FOR_MAIN_EXPERIMENT.json")
    sample = read_json(FROZEN / "main_sample" / "main_sampling_manifest.json")
    code_checks = {
        relative: sha256_file(ROOT / relative) == expected
        for relative, expected in manifest["code_files"].items()
    }
    prompt_checks = {
        name: sha256_file(
            ROOT
            / "experiments"
            / "aegis_whowhen_main_fixed_guidance"
            / "prompts"
            / name
        )
        == expected
        for name, expected in manifest["prompt_hashes"].items()
    }
    taxonomy_checks = {
        name: sha256_json(read_json(FROZEN / filename))
        == manifest["taxonomy_hashes"][name]
        for name, filename in {
            "aegis": "aegis_taxonomy.json",
            "whowhen": "whowhen_taxonomy.json",
            "merged": "merged_taxonomy.json",
        }.items()
    }
    rows = read_jsonl(INPUTS)
    gold = read_jsonl(FROZEN / "main_sample" / "main_gold.jsonl")
    ids = {
        manifest["implementation_run_id"],
        ready["implementation_run_id"],
        sample["implementation_run_id"],
    }
    checks = {
        "implementation_ids_identical": len(ids) == 1,
        "all_code_hashes_match": all(code_checks.values()),
        "all_prompt_hashes_match": all(prompt_checks.values()),
        "all_taxonomy_hashes_match": all(taxonomy_checks.values()),
        "config_hash_matches": sha256_file(
            ROOT
            / "experiments"
            / "aegis_whowhen_main_fixed_guidance"
            / "experiment.yaml"
        )
        == manifest["experiment_config_file_hash"],
        "main_input_content_hash_matches": sha256_json(rows) == sample["input_hash"],
        "main_gold_content_hash_matches": sha256_json(gold) == sample["gold_hash"],
    }
    return {
        "implementation_run_id": manifest["implementation_run_id"],
        "checks": checks,
        "code_checks": code_checks,
        "prompt_checks": prompt_checks,
        "taxonomy_checks": taxonomy_checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    rows = read_jsonl(INPUTS)
    identity = verify_frozen_identity()
    if not identity["passed"]:
        raise RuntimeError("Frozen prediction-relevant identity no longer matches")

    groups: dict[str, dict[str, list[dict[str, str]]]] = {
        "aegis": defaultdict(list),
        "whowhen": defaultdict(list),
    }
    missing_or_non_string = []
    for row in rows:
        task_query = row["trajectory_payload"].get("task_query")
        if not isinstance(task_query, str):
            missing_or_non_string.append(
                {"dataset": row["dataset"], "trajectory_id": row["trajectory_id"]}
            )
            continue
        task_hash = sha256_bytes(task_query.encode("utf-8"))
        groups[row["dataset"]][task_hash].append(
            {
                "trajectory_id": row["trajectory_id"],
                "source_trajectory_id": row["source_trajectory_id"],
            }
        )
    if missing_or_non_string:
        raise ValueError(f"Rows without string task_query: {missing_or_non_string}")

    datasets = {}
    for dataset in ("aegis", "whowhen"):
        clusters = groups[dataset]
        duplicate_clusters = [
            {
                "task_hash": task_hash,
                "cluster_size": len(members),
                "trajectories": sorted(
                    members, key=lambda item: item["trajectory_id"]
                ),
            }
            for task_hash, members in sorted(clusters.items())
            if len(members) > 1
        ]
        total = sum(len(members) for members in clusters.values())
        in_duplicates = sum(item["cluster_size"] for item in duplicate_clusters)
        datasets[dataset] = {
            "total_sampled_trajectories": total,
            "unique_task_hashes": len(clusters),
            "duplicate_task_clusters": len(duplicate_clusters),
            "trajectories_in_duplicate_task_clusters": in_duplicates,
            "duplicate_cluster_trajectory_fraction": in_duplicates / total,
            "maximum_cluster_size": max(map(len, clusters.values()), default=0),
            "clusters": duplicate_clusters,
        }

    overlap = sorted(set(groups["aegis"]) & set(groups["whowhen"]))
    cross_dataset = {
        "overlapping_unique_task_hashes": len(overlap),
        "aegis_trajectories_with_overlapping_hashes": sum(
            len(groups["aegis"][task_hash]) for task_hash in overlap
        ),
        "whowhen_trajectories_with_overlapping_hashes": sum(
            len(groups["whowhen"][task_hash]) for task_hash in overlap
        ),
        "task_hashes": overlap,
    }

    substantial = (
        datasets["aegis"]["duplicate_cluster_trajectory_fraction"] >= 0.10
        or datasets["whowhen"]["duplicate_cluster_trajectory_fraction"] >= 0.10
    )
    audit = {
        "audit": "main_task_independence",
        "audit_version": "exact-visible-task-query-sha256-v1",
        "offline_only": True,
        "api_calls": 0,
        "accuracy_or_pilot_results_used": False,
        "implementation_run_id": identity["implementation_run_id"],
        "task_hash_construction": {
            "source": "trajectory_payload.task_query from frozen main_inputs.jsonl",
            "algorithm": "SHA-256 over the exact UTF-8 bytes of the task/query string",
            "normalization": "none",
            "included": ["original task/query content visible to the agent"],
            "excluded": [
                "gold failure category",
                "failure step",
                "injected agent",
                "injection metadata",
                "reference/correct answers",
                "prediction/results",
            ],
        },
        "input_artifact": str(INPUTS.relative_to(ROOT)),
        "input_file_sha256": sha256_file(INPUTS),
        "frozen_identity_verification": identity,
        "datasets": datasets,
        "cross_dataset_exact_task_hash_overlap": cross_dataset,
        "decision_rule": (
            "Substantial duplication is flagged when at least 10% of sampled "
            "trajectories in either dataset belong to duplicate-task clusters."
        ),
        "substantial_task_duplication": substantial,
        "sample_preserved_without_resampling": True,
        "prediction_relevant_artifacts_modified": False,
        "implementation_id_retained": True,
        "main_inference_started": False,
        "status": (
            "STOP_BEFORE_INFERENCE_SUBSTANTIAL_TASK_DUPLICATION"
            if substantial
            else "READY_FOR_MAIN_INFERENCE"
        ),
    }
    OUTPUT.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"status: {audit['status']}")
    print(f"implementation_run_id: {audit['implementation_run_id']}")
    for dataset in ("aegis", "whowhen"):
        item = datasets[dataset]
        print(
            dataset,
            item["total_sampled_trajectories"],
            item["unique_task_hashes"],
            item["duplicate_task_clusters"],
            item["trajectories_in_duplicate_task_clusters"],
            item["maximum_cluster_size"],
        )
    print(
        "cross_dataset_overlap:",
        cross_dataset["overlapping_unique_task_hashes"],
    )
    print(f"output: {OUTPUT}")


if __name__ == "__main__":
    main()
