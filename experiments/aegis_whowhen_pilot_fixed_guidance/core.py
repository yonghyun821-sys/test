from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from datasets import load_from_disk
from pydantic import BaseModel, ConfigDict, Field

from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.io import (
    append_jsonl,
    iter_jsonl,
    read_json,
    write_json,
    write_jsonl,
)
from taxonomy_experiment.llm import CachedLLM, TruncatedResponseError
from taxonomy_experiment.token_count import TokenCounter


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASETS = ("aegis", "whowhen")
CONDITIONS = (
    "label_only",
    "own_taxonomy",
    "foreign_taxonomy",
    "merged_taxonomy",
)
AEGIS_IDS = tuple(
    [f"FM-1.{index}" for index in range(1, 6)]
    + [f"FM-2.{index}" for index in range(1, 7)]
    + [f"FM-3.{index}" for index in range(1, 4)]
)
WHOWHEN_OBSERVED_IDS = (
    "A.1",
    "A.2",
    "A.3",
    "A.4",
    "R.1",
    "R.2",
    "R.3",
    "R.4",
    "V.1",
    "V.2",
    "C.1",
    "C.2",
    "C.3",
    "PL.1",
)
TRANSITIONS = (
    ("label_only", "own_taxonomy"),
    ("label_only", "foreign_taxonomy"),
    ("label_only", "merged_taxonomy"),
    ("foreign_taxonomy", "merged_taxonomy"),
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttributionOutput(StrictModel):
    # Keep the advertised JSON schema limits unchanged, but tolerate providers that
    # return otherwise-valid structured JSON slightly beyond maxLength. Category
    # validity is enforced separately by exact native-ID comparison and one retry;
    # explanation/evidence length never affects accuracy.
    failure_category: str = Field(
        min_length=1, json_schema_extra={"maxLength": 40}
    )
    explanation: str = Field(
        min_length=1, json_schema_extra={"maxLength": 700}
    )
    evidence: str = Field(
        min_length=1, json_schema_extra={"maxLength": 700}
    )


class ProvenanceRef(StrictModel):
    source_taxonomy_id: str = Field(min_length=1, max_length=80)
    source_category_id: str = Field(min_length=1, max_length=40)


class MergedCategory(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=160)
    definition: str = Field(min_length=1, max_length=1200)
    module: str = Field(min_length=1, max_length=120)
    source_categories: list[ProvenanceRef] = Field(min_length=1)


class MergedTaxonomyOutput(StrictModel):
    categories: list[MergedCategory] = Field(min_length=1, max_length=28)


class MergeProvenanceDecision(StrictModel):
    selected_merged_category_id: str = Field(min_length=1, max_length=40)
    reason: str = Field(min_length=1, max_length=700)


class InvalidOutputLabelError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_prompt(name: str) -> str:
    return (EXPERIMENT_DIR / "prompts" / name).read_text(encoding="utf-8")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    _write_text(path, _csv_text(rows, fields))


def _freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Frozen artifact differs; refusing overwrite: {path}")
        return
    write_json(path, value)


def _freeze_text(path: Path, value: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        expected = value.replace("\r\n", "\n")
        if existing != expected:
            raise FileExistsError(f"Frozen artifact differs; refusing overwrite: {path}")
        return
    _write_text(path, value)


def _freeze_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        if list(iter_jsonl(path)) != rows:
            raise FileExistsError(f"Frozen artifact differs; refusing overwrite: {path}")
        return
    write_jsonl(path, rows)


def _freeze_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(f"Frozen run artifact differs: {destination}")
        return
    shutil.copyfile(source, destination)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _aegis_group(category_id: str) -> str:
    if category_id.startswith("FM-1."):
        return "specification_issues"
    if category_id.startswith("FM-2."):
        return "inter_agent_misalignment"
    if category_id.startswith("FM-3."):
        return "task_verification"
    raise ValueError(f"Unknown AEGIS category ID: {category_id}")


def _whowhen_group(category_id: str) -> str:
    prefix = category_id.split(".", 1)[0]
    groups = {
        "P": "perception",
        "R": "reasoning",
        "PL": "planning",
        "A": "action",
        "V": "verification",
        "C": "coordination",
    }
    if prefix not in groups:
        raise ValueError(f"Unknown Who&When category ID: {category_id}")
    return groups[prefix]


def build_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    aegis_source = resolve_path(config, "aegis_official_source")
    source_text = aegis_source.read_text(encoding="utf-8")
    matches = re.findall(
        r'^\s+FM_\d_\d\s*=\s*"(FM-\d\.\d)"\s*#\s*(.+?)\s*$',
        source_text,
        flags=re.MULTILINE,
    )
    aegis_names = {category_id: name.strip() for category_id, name in matches}
    if set(aegis_names) != set(AEGIS_IDS):
        raise ValueError(
            "AEGIS official taxonomy IDs did not resolve exactly: "
            f"{sorted(aegis_names)}"
        )
    mast_path = resolve_path(config, "mast_taxonomy")
    mast = read_json(mast_path)
    mast_categories = {str(item["id"]): item for item in mast["categories"]}
    if set(mast_categories) != {item.removeprefix("FM-") for item in AEGIS_IDS}:
        raise ValueError("Local official MAST definitions do not cover all AEGIS IDs")
    aegis_categories = []
    for category_id in AEGIS_IDS:
        mast_item = mast_categories[category_id.removeprefix("FM-")]
        aegis_categories.append(
            {
                "id": category_id,
                "name": aegis_names[category_id],
                "definition": str(mast_item["definition"]),
                "module": _aegis_group(category_id),
                "name_source": (
                    "official AEGIS FMErrorType/_fm_descriptions implementation"
                ),
                "definition_source": "official MAST taxonomy definition for matching ID",
            }
        )
    aegis = {
        "taxonomy_id": "aegis_mast_14",
        "name": "AEGIS MAST-derived Failure Modes",
        "category_count": 14,
        "source": {
            "aegis_repository": "https://github.com/kfq20/AEGIS",
            "aegis_source_file": str(aegis_source.relative_to(config["_root"])),
            "aegis_source_sha256": _sha256_file(aegis_source),
            "mast_source_file": str(mast_path.relative_to(config["_root"])),
            "mast_source_sha256": _sha256_file(mast_path),
        },
        "categories": aegis_categories,
    }

    whowhen_path = resolve_path(config, "whowhen_official_taxonomy")
    whowhen_raw = yaml.safe_load(whowhen_path.read_text(encoding="utf-8")) or {}
    unresolved = [item for item in WHOWHEN_OBSERVED_IDS if item not in whowhen_raw]
    if unresolved:
        raise ValueError(
            f"Observed Who&When IDs missing from official taxonomy: {unresolved}"
        )
    whowhen_categories = []
    for category_id in WHOWHEN_OBSERVED_IDS:
        source = whowhen_raw[category_id]
        name = str(source.get("name") or "").strip()
        definition = re.sub(r"\s+", " ", str(source.get("description") or "")).strip()
        if not name or not definition:
            raise ValueError(f"Incomplete official Who&When category: {category_id}")
        whowhen_categories.append(
            {
                "id": category_id,
                "name": name,
                "definition": definition,
                "module": _whowhen_group(category_id),
            }
        )
    whowhen = {
        "taxonomy_id": "whowhen_pro_text_observed_14",
        "name": "Who&When Pro Text Observed Error Modes",
        "category_count": 14,
        "selection_policy": (
            "Only the 14 IDs observed in the local text split are retained; "
            "P.1, P.2, and PL.2 are not observed and are excluded from this pilot."
        ),
        "source": {
            "dataset": "Leoxx/whowhen_pro",
            "repository": "https://github.com/ag2ai/whowhen_pro",
            "source_file": str(whowhen_path.relative_to(config["_root"])),
            "source_sha256": _sha256_file(whowhen_path),
        },
        "categories": whowhen_categories,
    }
    return {"aegis": aegis, "whowhen": whowhen}


def _load_aegis_candidates(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_from_disk(str(resolve_path(config, "aegis_data")))
    candidates: list[dict[str, Any]] = []
    total = 0
    single = 0
    unsuccessful_single = 0
    for split_name, split in dataset.items():
        for row in split:
            total += 1
            metadata = _maybe_json(row.get("metadata") or {})
            output = _maybe_json(row.get("output") or {})
            ground_truth = _maybe_json(row.get("ground_truth") or {})
            faulty_agents = output.get("faulty_agents") or []
            if metadata.get("num_injected_agents") == 1 and len(faulty_agents) == 1:
                single += 1
                if ground_truth.get("is_injection_successful") is not True:
                    unsuccessful_single += 1
                    continue
                faulty = _maybe_json(faulty_agents[0])
                category_id = str(faulty.get("error_type") or "")
                payload_source = _maybe_json(row.get("input") or {})
                payload = {
                    "task_query": payload_source.get("query"),
                    "conversation_history": payload_source.get("conversation_history"),
                }
                if payload_source.get("final_output") is not None:
                    payload["final_output"] = payload_source["final_output"]
                candidates.append(
                    {
                        "dataset": "aegis",
                        "trajectory_id": str(row["id"]),
                        "split": str(split_name),
                        "framework": str(metadata.get("framework") or "unknown"),
                        "benchmark": str(metadata.get("benchmark") or "unknown"),
                        "gold_category": category_id,
                        "gold_agent": str(faulty.get("agent_name") or ""),
                        "gold_step": None,
                        "trajectory_payload": payload,
                    }
                )
    audit = {
        "total_rows": total,
        "single_injection_rows": single,
        "eligible_single_injection_success_rows": len(candidates),
        "unsuccessful_single_injection_rows": unsuccessful_single,
        "injection_success_rate_among_single": (
            len(candidates) / single if single else None
        ),
        "native_category_counts": dict(
            sorted(Counter(row["gold_category"] for row in candidates).items())
        ),
    }
    return candidates, audit


def _load_whowhen_candidates(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_from_disk(str(resolve_path(config, "whowhen_data")))
    candidates: list[dict[str, Any]] = []
    missing_ground_truth = 0
    multi_mode_rows = 0
    for row in dataset:
        ground_truth = _maybe_json(row.get("ground_truth"))
        if not ground_truth:
            missing_ground_truth += 1
            continue
        mode = ground_truth.get("mode")
        if isinstance(mode, list):
            multi_mode_rows += 1
            continue
        task = _maybe_json(row.get("task"))
        trajectory = _maybe_json(row.get("trajectory"))
        payload = {
            "task_query": task.get("query"),
            "trajectory": trajectory,
        }
        candidates.append(
            {
                "dataset": "whowhen",
                "trajectory_id": str(row["id"]),
                "split": "text",
                "framework": str(row.get("framework") or "unknown"),
                "benchmark": str(row.get("benchmark") or "unknown"),
                "gold_category": str(mode),
                "gold_agent": ground_truth.get("agent"),
                "gold_step": ground_truth.get("step"),
                "trajectory_payload": payload,
            }
        )
    audit = {
        "total_rows": len(dataset),
        "missing_ground_truth": missing_ground_truth,
        "multi_mode_rows": multi_mode_rows,
        "eligible_rows": len(candidates),
        "native_category_counts": dict(
            sorted(Counter(row["gold_category"] for row in candidates).items())
        ),
    }
    return candidates, audit


def _sample_category(
    rows: list[dict[str, Any]], *, dataset: str, category_id: str, seed: int, count: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: item["trajectory_id"])
    category_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{dataset}:{category_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    random.Random(category_seed).shuffle(ordered)
    if len(ordered) < count:
        raise ValueError(
            f"Not enough rows for {dataset}/{category_id}: {len(ordered)} < {count}"
        )
    selected: list[dict[str, Any]] = []
    seen_frameworks: set[str] = set()
    for row in ordered:
        if row["framework"] not in seen_frameworks:
            selected.append(row)
            seen_frameworks.add(row["framework"])
            if len(selected) == count:
                return selected
    selected_ids = {row["trajectory_id"] for row in selected}
    for row in ordered:
        if row["trajectory_id"] not in selected_ids:
            selected.append(row)
            if len(selected) == count:
                return selected
    raise AssertionError("Deterministic sampler failed to fill category quota")


def sample_pilot(
    candidates: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    count = int(config["experiment"]["samples_per_category"])
    seed = int(config["experiment"]["pilot_seed"])
    expected = {"aegis": AEGIS_IDS, "whowhen": WHOWHEN_OBSERVED_IDS}
    selected: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for category_id in expected[dataset]:
            pool = [
                row for row in candidates[dataset] if row["gold_category"] == category_id
            ]
            picked = _sample_category(
                pool,
                dataset=dataset,
                category_id=category_id,
                seed=seed,
                count=count,
            )
            for rank, row in enumerate(picked, start=1):
                selected.append({**row, "category_sample_rank": rank})
    return selected


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).casefold())
            keys.update(_collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_collect_keys(item))
    return keys


def leakage_audit(selected: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_anywhere = {
        "aegis": {
            "faulty_agents",
            "error_type",
            "num_injected_agents",
            "is_injection_successful",
            "ground_truth",
            "injected_agents",
            "correct_answer",
            "reference_answer",
            "expected_answer",
            "injection_strategy",
            "construction_annotations",
        },
        "whowhen": {
            "ground_truth",
            "mode",
            "gold_step",
            "gold_agent",
            "reference_answer",
            "expected_answer",
            "injection_metadata",
            "construction_annotations",
        },
    }
    rows = []
    for row in selected:
        payload = row["trajectory_payload"]
        serialized = _json_text(payload)
        key_hits = sorted(
            _collect_keys(payload) & forbidden_anywhere[row["dataset"]]
        )
        top_level_allowed = (
            {"task_query", "conversation_history", "final_output"}
            if row["dataset"] == "aegis"
            else {"task_query", "trajectory"}
        )
        unexpected_top_level_keys = sorted(set(payload) - top_level_allowed)
        gold_literal_present = row["gold_category"] in serialized
        rows.append(
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "gold_category_literal_in_trajectory": gold_literal_present,
                "forbidden_structural_keys": key_hits,
                "unexpected_top_level_keys": unexpected_top_level_keys,
                "passed": (
                    not key_hits
                    and not unexpected_top_level_keys
                    and not gold_literal_present
                ),
            }
        )
    failures = [row for row in rows if not row["passed"]]
    return {
        "policy": (
            "Scan only the serialized TRAJECTORY_DATA payload, before output labels "
            "or guidance are added. Enforce dataset-specific top-level allowlists and "
            "scan for unambiguous gold/construction keys. Generic nested execution "
            "fields such as ledger.answer are retained because they are trajectory "
            "evidence, not the excluded source-level task.answer reference field."
        ),
        "serializer_allowlists": {
            "aegis": ["input.query", "input.conversation_history", "input.final_output"],
            "whowhen": ["task.query", "trajectory"],
        },
        "explicitly_excluded_source_objects": {
            "aegis": ["metadata", "output", "ground_truth"],
            "whowhen": ["task.answer", "ground_truth", "extras"],
        },
        "rows_scanned": len(rows),
        "leakage_count": len(failures),
        "passed": not failures,
        "failures": failures,
        "rows": rows,
    }


def _sanitized_config(config: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    output = {
        "run_id": run_id,
        "experiment": config["experiment"],
        "models": config["models"],
        "api": {
            "api_key_env": config["api"]["api_key_env"],
            "base_url": config["api"]["base_url"],
            "app_title": config["api"]["app_title"],
            "provider_defaults": config["api"]["provider_defaults"],
        },
        "cost_budget": config["cost_budget"],
        "execution": config["execution"],
        "paths": config["paths"],
    }
    return output


def _prompt_templates_artifact() -> dict[str, Any]:
    return {
        "prompts": {
            path.name: {
                "sha256": _sha256_file(path),
                "text": path.read_text(encoding="utf-8"),
            }
            for path in sorted((EXPERIMENT_DIR / "prompts").glob("*.txt"))
        },
        "attribution_response_schema": AttributionOutput.model_json_schema(),
        "merge_response_schema": MergedTaxonomyOutput.model_json_schema(),
        "merge_adjudication_response_schema": (
            MergeProvenanceDecision.model_json_schema()
        ),
        "condition_invariant": (
            "Model, system prompt, output schema, trajectory serialization, task "
            "instruction, and target-native ID/name output list are identical. "
            "Only DIAGNOSTIC_GUIDANCE differs across conditions."
        ),
    }


def prepare_pilot(config: dict[str, Any]) -> dict[str, Any]:
    prepared = resolve_path(config, "prepared")
    taxonomies = build_taxonomies(config)
    aegis_candidates, aegis_audit = _load_aegis_candidates(config)
    whowhen_candidates, whowhen_audit = _load_whowhen_candidates(config)
    candidates = {"aegis": aegis_candidates, "whowhen": whowhen_candidates}
    selected = sample_pilot(candidates, config)

    input_rows = []
    gold_rows = []
    sample_rows = []
    for row in selected:
        payload_hash = _sha256_json(row["trajectory_payload"])
        input_rows.append(
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "split": row["split"],
                "framework": row["framework"],
                "benchmark": row["benchmark"],
                "trajectory_payload": row["trajectory_payload"],
                "trajectory_payload_sha256": payload_hash,
            }
        )
        gold_rows.append(
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "gold_category": row["gold_category"],
                "gold_source_field": (
                    'output["faulty_agents"][0]["error_type"]'
                    if row["dataset"] == "aegis"
                    else 'ground_truth["mode"]'
                ),
            }
        )
        sample_rows.append(
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "split": row["split"],
                "framework": row["framework"],
                "benchmark": row["benchmark"],
                "gold_category": row["gold_category"],
                "category_sample_rank": row["category_sample_rank"],
                "trajectory_payload_sha256": payload_hash,
            }
        )
    input_rows.sort(key=lambda item: (item["dataset"], item["trajectory_id"]))
    gold_rows.sort(key=lambda item: (item["dataset"], item["trajectory_id"]))
    sample_rows.sort(
        key=lambda item: (
            item["dataset"],
            item["gold_category"],
            item["category_sample_rank"],
        )
    )

    category_summary = {}
    for dataset in DATASETS:
        selected_for_dataset = [row for row in sample_rows if row["dataset"] == dataset]
        category_summary[dataset] = {
            category_id: {
                "sampled_n": sum(
                    row["gold_category"] == category_id
                    for row in selected_for_dataset
                ),
                "distinct_frameworks": len(
                    {
                        row["framework"]
                        for row in selected_for_dataset
                        if row["gold_category"] == category_id
                    }
                ),
                "frameworks": sorted(
                    {
                        row["framework"]
                        for row in selected_for_dataset
                        if row["gold_category"] == category_id
                    }
                ),
            }
            for category_id in (
                AEGIS_IDS if dataset == "aegis" else WHOWHEN_OBSERVED_IDS
            )
        }
    manifest = {
        "pilot_seed": int(config["experiment"]["pilot_seed"]),
        "sampling_policy": (
            "Within each native category, deterministic seeded shuffling followed "
            "by one-per-framework selection where available, then deterministic fill. "
            "No proportional or result-dependent sampling."
        ),
        "samples_per_category": int(config["experiment"]["samples_per_category"]),
        "total_sampled_trajectories": len(sample_rows),
        "planned_prediction_rows": len(sample_rows) * len(CONDITIONS),
        "dataset_audits": {
            "aegis": aegis_audit,
            "whowhen": whowhen_audit,
        },
        "category_summary": category_summary,
        "sampled_rows": sample_rows,
        "pilot_ids_must_be_excluded_from_future_full_sampling": True,
        "future_full_experiment_seed": int(
            config["experiment"]["final_seed_reserved"]
        ),
    }
    leakage = leakage_audit(selected)

    _freeze_json(prepared / "aegis_taxonomy.json", taxonomies["aegis"])
    _freeze_json(prepared / "whowhen_taxonomy.json", taxonomies["whowhen"])
    _freeze_json(prepared / "pilot_sampling_manifest.json", manifest)
    _freeze_text(
        prepared / "pilot_sampling_manifest.csv",
        _csv_text(
            sample_rows,
            [
                "dataset",
                "trajectory_id",
                "split",
                "framework",
                "benchmark",
                "gold_category",
                "category_sample_rank",
                "trajectory_payload_sha256",
            ],
        ),
    )
    _freeze_jsonl(prepared / "pilot_inputs.jsonl", input_rows)
    _freeze_jsonl(prepared / "pilot_gold.jsonl", gold_rows)
    # This audit may legitimately change while pre-API implementation bugs are fixed.
    # The validated version is frozen into the content-addressed run directory before
    # the first attribution call.
    write_json(prepared / "pilot_leakage_audit.json", leakage)
    # Prompt artifacts may change only while fixing implementation failures before
    # attribution begins. The final version is frozen into the run directory.
    write_json(prepared / "pilot_prompt_templates.json", _prompt_templates_artifact())
    _freeze_json(prepared / "pilot_config.json", _sanitized_config(config))
    return {
        "inputs": input_rows,
        "gold": {
            (row["dataset"], row["trajectory_id"]): row["gold_category"]
            for row in gold_rows
        },
        "taxonomies": taxonomies,
        "manifest": manifest,
        "leakage": leakage,
    }


def _taxonomy_guidance(taxonomy: dict[str, Any] | None) -> str:
    if taxonomy is None:
        return "[NONE]"
    return _json_text(
        {
            "taxonomy_id": taxonomy["taxonomy_id"],
            "name": taxonomy["name"],
            "categories": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "definition": item["definition"],
                    "module": item["module"],
                }
                for item in taxonomy["categories"]
            ],
        }
    )


def native_ids(dataset: str, taxonomies: dict[str, dict[str, Any]]) -> list[str]:
    return [str(item["id"]) for item in taxonomies[dataset]["categories"]]


def output_label_block(dataset: str, taxonomies: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        f"- {item['id']} -> {item['name']}"
        for item in taxonomies[dataset]["categories"]
    )


def guidance_for(
    dataset: str, condition: str, taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if condition == "label_only":
        return None
    if condition == "own_taxonomy":
        return taxonomies[dataset]
    if condition == "foreign_taxonomy":
        other = "whowhen" if dataset == "aegis" else "aegis"
        return taxonomies[other]
    if condition == "merged_taxonomy":
        return taxonomies["merged"]
    raise ValueError(f"Unknown condition: {condition}")


def build_prompt(
    record: dict[str, Any],
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dataset = record["dataset"]
    trajectory = _json_text(record["trajectory_payload"])
    guidance_taxonomy = guidance_for(dataset, condition, taxonomies)
    guidance = _taxonomy_guidance(guidance_taxonomy)
    labels = native_ids(dataset, taxonomies)
    return {
        "system_prompt": _read_prompt("attribution_system.txt").strip(),
        "user_prompt": _read_prompt("attribution_user.txt").format(
            output_labels=output_label_block(dataset, taxonomies),
            diagnostic_guidance=guidance,
            trajectory=trajectory,
        ),
        "trajectory": trajectory,
        "diagnostic_guidance": guidance,
        "guidance_taxonomy_id": (
            guidance_taxonomy["taxonomy_id"] if guidance_taxonomy else None
        ),
        "output_ids": labels,
        "output_label_block": output_label_block(dataset, taxonomies),
    }


def _preview_merged_taxonomy(taxonomies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    categories = []
    for dataset in DATASETS:
        for item in taxonomies[dataset]["categories"]:
            categories.append(
                {
                    "id": f"PREVIEW-{dataset}-{item['id']}",
                    "name": item["name"],
                    "definition": item["definition"],
                    "module": item["module"],
                }
            )
    return {
        "taxonomy_id": "pilot_merged_preview_conservative",
        "name": "Conservative preflight preview containing all source entries",
        "categories": categories,
    }


def planned_keys(
    inputs: list[dict[str, Any]], seed: int
) -> list[tuple[str, str, str]]:
    keys = []
    for row in sorted(inputs, key=lambda item: (item["dataset"], item["trajectory_id"])):
        conditions = list(CONDITIONS)
        digest = hashlib.sha256(
            f"{seed}:{row['dataset']}:{row['trajectory_id']}".encode("utf-8")
        ).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(conditions)
        keys.extend(
            (row["dataset"], row["trajectory_id"], condition)
            for condition in conditions
        )
    return keys


def _without_guidance(prompt: str) -> str:
    return re.sub(
        r"<DIAGNOSTIC_GUIDANCE>\n.*?\n</DIAGNOSTIC_GUIDANCE>",
        "<DIAGNOSTIC_GUIDANCE>\n[ONLY_ALLOWED_DIFFERENCE]\n</DIAGNOSTIC_GUIDANCE>",
        prompt,
        flags=re.DOTALL,
    )


def validate_pre_inference(
    config: dict[str, Any], prepared: dict[str, Any], *, merged_available: bool
) -> dict[str, Any]:
    inputs = prepared["inputs"]
    manifest = prepared["manifest"]
    taxonomies = prepared["taxonomies"]
    validation_taxonomies = dict(taxonomies)
    if "merged" not in validation_taxonomies:
        validation_taxonomies["merged"] = _preview_merged_taxonomy(taxonomies)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    audits = manifest["dataset_audits"]
    add(
        "source_dataset_counts",
        audits["aegis"]["total_rows"] == 9533
        and audits["aegis"]["eligible_single_injection_success_rows"] == 1690
        and audits["aegis"]["injection_success_rate_among_single"] == 1.0
        and audits["whowhen"]["total_rows"] == 6257
        and audits["whowhen"]["missing_ground_truth"] == 0
        and audits["whowhen"]["multi_mode_rows"] == 0,
        "AEGIS=9,533/eligible=1,690/success=100%; Who&When text=6,257/missing=0/multi-mode=0.",
    )
    add(
        "sample_size_and_rows",
        len(inputs) == 112
        and manifest["total_sampled_trajectories"] == 112
        and manifest["planned_prediction_rows"] == 448
        and len(planned_keys(inputs, int(config["experiment"]["pilot_seed"]))) == 448,
        "56 trajectories per dataset and four conditions produce 448 rows.",
    )
    sample_counts = Counter(
        (row["dataset"], row["gold_category"])
        for row in manifest["sampled_rows"]
    )
    add(
        "four_per_native_category",
        len(sample_counts) == 28 and set(sample_counts.values()) == {4},
        f"represented_categories={len(sample_counts)}; per_category={sorted(set(sample_counts.values()))}",
    )
    add(
        "unique_sample_ids",
        len({(row["dataset"], row["trajectory_id"]) for row in inputs}) == 112,
        "No duplicate trajectory within a dataset.",
    )
    add(
        "taxonomy_resolution",
        native_ids("aegis", taxonomies) == list(AEGIS_IDS)
        and native_ids("whowhen", taxonomies) == list(WHOWHEN_OBSERVED_IDS)
        and all(
            item["name"] and item["definition"] and item["module"]
            for dataset in DATASETS
            for item in taxonomies[dataset]["categories"]
        ),
        "All 28 observed native IDs resolve to official names, definitions, and groups.",
    )
    add(
        "leakage_scan",
        prepared["leakage"]["passed"]
        and prepared["leakage"]["leakage_count"] == 0,
        f"scanned={prepared['leakage']['rows_scanned']}; leakage={prepared['leakage']['leakage_count']}",
    )
    add(
        "gold_separation",
        all(
            "gold_category" not in row
            and "gold_agent" not in row
            and "gold_step" not in row
            for row in inputs
        ),
        "Prepared model inputs contain no gold category, agent, or step fields.",
    )

    prompt_invariant = True
    output_spaces_fixed = True
    foreign_mapping = True
    delimiters = True
    guidance_present = True
    for record in inputs:
        built = {
            condition: build_prompt(record, condition, validation_taxonomies)
            for condition in CONDITIONS
        }
        reference = built["label_only"]
        prompt_invariant &= all(
            item["system_prompt"] == reference["system_prompt"]
            and item["trajectory"] == reference["trajectory"]
            and item["output_ids"] == reference["output_ids"]
            and item["output_label_block"] == reference["output_label_block"]
            and _without_guidance(item["user_prompt"])
            == _without_guidance(reference["user_prompt"])
            for item in built.values()
        )
        output_spaces_fixed &= all(
            item["output_ids"] == native_ids(record["dataset"], taxonomies)
            for item in built.values()
        )
        expected_foreign = "whowhen" if record["dataset"] == "aegis" else "aegis"
        foreign_mapping &= (
            built["foreign_taxonomy"]["guidance_taxonomy_id"]
            == taxonomies[expected_foreign]["taxonomy_id"]
        )
        delimiters &= all(
            prompt["user_prompt"].count("<TRAJECTORY_DATA>") == 1
            and prompt["user_prompt"].count("</TRAJECTORY_DATA>") == 1
            for prompt in built.values()
        )
        guidance_present &= (
            built["label_only"]["diagnostic_guidance"] == "[NONE]"
            and all(
                built[condition]["diagnostic_guidance"] != "[NONE]"
                for condition in CONDITIONS[1:]
            )
        )
    add(
        "primary_prompt_invariant",
        prompt_invariant,
        "Only DIAGNOSTIC_GUIDANCE differs across the four conditions.",
    )
    add(
        "fixed_native_output_spaces",
        output_spaces_fixed,
        "Every condition uses the same target-native ID/name list and requires ID-only output.",
    )
    add(
        "foreign_guidance_direction",
        foreign_mapping,
        "AEGIS receives Who&When guidance and Who&When receives AEGIS guidance.",
    )
    add(
        "trajectory_delimiters",
        delimiters,
        "Every prompt contains exactly one trusted TRAJECTORY_DATA boundary pair.",
    )
    add(
        "guidance_insertion",
        guidance_present,
        "Label-Only is empty; Own, Foreign, and Merged guidance blocks are populated.",
    )
    add(
        "exact_scoring_contract",
        not ("fm-2.6" in set(AEGIS_IDS))
        and "FM-2.6" in set(AEGIS_IDS)
        and AttributionOutput.model_config.get("extra") == "forbid",
        "Scoring compares raw prediction ID to raw gold ID; no canonicalization layer exists.",
    )
    add(
        "fixed_retry_policy",
        int(config["execution"]["invalid_label_retries"]) == 1
        and _read_prompt("invalid_label_retry.txt").strip()
        == "Return failure_category using exactly one of the permitted OUTPUT LABEL IDs.",
        "Exactly one gold-blind formatting-only retry is configured.",
    )
    passed = sum(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "checks_passed": passed,
        "checks_total": len(checks),
        "merged_taxonomy_available": merged_available,
        "merged_validation_mode": "frozen_artifact" if merged_available else "conservative_preview",
        "checks": checks,
    }
    write_json(resolve_path(config, "prepared") / "pilot_preflight_validation.json", report)
    if report["status"] != "passed":
        failed = [item["name"] for item in checks if not item["passed"]]
        raise ValueError(f"Pilot pre-inference validation failed: {failed}")
    return report


def estimate_budget(
    config: dict[str, Any], prepared: dict[str, Any], *, merge_required: bool
) -> dict[str, Any]:
    taxonomies = dict(prepared["taxonomies"])
    if "merged" not in taxonomies:
        taxonomies["merged"] = _preview_merged_taxonomy(taxonomies)
    model = config["models"]["attribution"]
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    prompt_tokens = []
    context_failures = []
    recovery_output = int(config["execution"]["truncation_retry_max_output_tokens"])
    for record in prepared["inputs"]:
        for condition in CONDITIONS:
            built = build_prompt(record, condition, taxonomies)
            tokens = counter.count(built["system_prompt"] + "\n" + built["user_prompt"])
            prompt_tokens.append(tokens)
            if tokens + recovery_output > int(model["context_window"]):
                context_failures.append(
                    {
                        "dataset": record["dataset"],
                        "trajectory_id": record["trajectory_id"],
                        "condition": condition,
                        "prompt_tokens": tokens,
                    }
                )
    safety = float(config["cost_budget"]["input_token_safety_multiplier"])
    # Reserve every workflow-level attempt and every paid SDK retry. Most transport
    # failures are not billed, but schema/parse failures can be, so count them all.
    workflow_attempts = (
        1
        + int(config["execution"]["invalid_label_retries"])
        + int(config["execution"]["truncation_retries"])
    )
    sdk_attempts = int(config["api"]["max_retries"])
    guarded_input_tokens = math.ceil(
        sum(prompt_tokens) * workflow_attempts * sdk_attempts * safety
    )
    maximum_calls = len(prompt_tokens) * workflow_attempts * sdk_attempts
    prices = model["pricing_usd_per_million"]
    attribution_cost = (
        guarded_input_tokens * float(prices["input"])
        + maximum_calls * recovery_output * float(prices["output"])
    ) / 1_000_000

    merge_cost = 0.0
    if merge_required:
        merge_model = config["models"]["merge"]
        merge_counter = TokenCounter.for_model(
            merge_model["name"], merge_model.get("tokenizer_encoding")
        )
        merge_input = _json_text(
            {dataset: prepared["taxonomies"][dataset] for dataset in DATASETS}
        )
        merge_tokens = merge_counter.count(
            _read_prompt("merge_system.txt")
            + "\n"
            + _read_prompt("merge_user.txt").format(
                source_taxonomies=merge_input, repair_context=""
            )
        )
        merge_attempts = (
            1 + int(config["execution"]["merge_validation_retries"])
        ) * sdk_attempts
        merge_prices = merge_model["pricing_usd_per_million"]
        merge_cost = merge_attempts * (
            math.ceil(merge_tokens * safety) * float(merge_prices["input"])
            + int(merge_model["max_output_tokens"]) * float(merge_prices["output"])
        ) / 1_000_000

    ledger_path = config["_root"] / config["cost_budget"]["ledger_path"]
    incurred = (
        sum(float(row.get("cost_usd", 0.0)) for row in iter_jsonl(ledger_path))
        if ledger_path.exists()
        else 0.0
    )
    projected = incurred + attribution_cost + merge_cost
    fee = max(
        projected * float(config["cost_budget"]["credit_purchase_fee_rate"]),
        float(config["cost_budget"]["minimum_credit_purchase_fee_usd"]),
    )
    report = {
        "status": "ready",
        "prediction_rows": len(prompt_tokens),
        "maximum_reserved_attribution_calls": maximum_calls,
        "workflow_attempts_per_row": workflow_attempts,
        "sdk_attempts_per_workflow_call": sdk_attempts,
        "merge_required": merge_required,
        "raw_single_pass_prompt_tokens": sum(prompt_tokens),
        "maximum_prompt_tokens": max(prompt_tokens),
        "guarded_all_attempt_input_tokens": guarded_input_tokens,
        "projected_new_inference_usd": attribution_cost + merge_cost,
        "incurred_inference_usd": incurred,
        "projected_cumulative_inference_usd": projected,
        "projected_fee_inclusive_total_usd": projected + fee,
        "max_inference_usd": float(config["cost_budget"]["max_inference_usd"]),
        "max_total_charge_usd": float(config["cost_budget"]["max_total_charge_usd"]),
        "within_context": not context_failures,
        "context_failures": context_failures,
        "within_budget": projected
        <= float(config["cost_budget"]["max_inference_usd"])
        and projected + fee
        <= float(config["cost_budget"]["max_total_charge_usd"]),
        "pricing_snapshot_usd_per_million": {
            "attribution": prices,
            "merge": config["models"]["merge"]["pricing_usd_per_million"],
        },
        "estimate_policy": (
            "Conservative: all workflow retries multiplied by all potentially billed "
            "SDK retries, plus all bounded merge provenance-repair attempts; the "
            "runtime ledger independently enforces the hard cap."
        ),
    }
    write_json(resolve_path(config, "prepared") / "pilot_budget_preflight.json", report)
    return report


def _validate_merged_taxonomy(
    artifact: dict[str, Any], source_taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    categories = artifact.get("categories") or []
    if not categories or len(categories) > 28:
        raise ValueError(f"Merged taxonomy category count is invalid: {len(categories)}")
    merged_ids = [str(item.get("id")) for item in categories]
    if len(merged_ids) != len(set(merged_ids)):
        raise ValueError("Merged taxonomy contains duplicate category IDs")
    if not all(re.fullmatch(r"MT-\d{2}", item) for item in merged_ids):
        raise ValueError("Merged taxonomy IDs must use MT-01 style identifiers")

    expected = {
        (taxonomy["taxonomy_id"], str(category["id"]))
        for taxonomy in source_taxonomies.values()
        for category in taxonomy["categories"]
    }
    actual_list = [
        (
            str(ref["source_taxonomy_id"]),
            str(ref["source_category_id"]),
        )
        for category in categories
        for ref in category.get("source_categories", [])
    ]
    actual = set(actual_list)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    duplicates = sorted(item for item, count in Counter(actual_list).items() if count > 1)
    if missing or unknown or duplicates:
        raise ValueError(
            f"Merged provenance invalid; missing={missing}, unknown={unknown}, duplicates={duplicates}"
        )
    if any(
        not str(category.get("name") or "").strip()
        or not str(category.get("definition") or "").strip()
        or not str(category.get("module") or "").strip()
        for category in categories
    ):
        raise ValueError("Merged taxonomy has an empty name, definition, or module")
    expected_hashes = {
        taxonomy["taxonomy_id"]: _sha256_json(taxonomy)
        for taxonomy in source_taxonomies.values()
    }
    recorded_hashes = artifact.get("generated_by", {}).get("source_taxonomy_hashes")
    if recorded_hashes != expected_hashes:
        raise ValueError("Merged taxonomy source hashes do not match frozen inputs")
    return {
        "source_category_count": len(expected),
        "provenance_reference_count": len(actual_list),
        "merged_category_count": len(categories),
        "all_source_categories_exactly_once": True,
        "unknown_provenance": False,
        "source_omissions": False,
        "merged_sha256": _sha256_json(artifact),
    }


def _merge_repair_context(
    *,
    prior_output: dict[str, Any],
    validation_error: str,
    repair_attempt_number: int,
    source_taxonomies: dict[str, dict[str, Any]],
) -> str:
    checklist = [
        {
            "source_taxonomy_id": taxonomy["taxonomy_id"],
            "source_category_id": category["id"],
            "source_category_name": category["name"],
        }
        for dataset in DATASETS
        for taxonomy in [source_taxonomies[dataset]]
        for category in taxonomy["categories"]
    ]
    return (
        "\n<VALIDATION_REPAIR>\n"
        f"Repair attempt number: {repair_attempt_number}.\n"
        "The prior taxonomy failed deterministic provenance validation:\n"
        f"{validation_error}\n\n"
        "Return a complete corrected merged taxonomy. Preserve sound semantic "
        "merges, but ensure every item in REQUIRED_PROVENANCE_CHECKLIST appears "
        "in source_categories exactly once. If a missing source mechanism is "
        "genuinely equivalent, attach its provenance to that merged category; "
        "otherwise create a distinct category. Do not merely copy the prior output "
        "without repairing the reported problem. Count and verify all checklist "
        "entries before returning.\n\n"
        f"REQUIRED_PROVENANCE_CHECKLIST ({len(checklist)} entries):\n"
        f"{_json_text(checklist)}\n\n"
        "PRIOR_OUTPUT:\n"
        f"{_json_text(prior_output)}\n"
        "</VALIDATION_REPAIR>"
    )


def _provenance_issues(
    categories: list[dict[str, Any]],
    source_taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        (taxonomy["taxonomy_id"], str(category["id"]))
        for taxonomy in source_taxonomies.values()
        for category in taxonomy["categories"]
    }
    locations: dict[tuple[str, str], list[str]] = {}
    for category in categories:
        for ref in category.get("source_categories", []):
            key = (
                str(ref["source_taxonomy_id"]),
                str(ref["source_category_id"]),
            )
            locations.setdefault(key, []).append(str(category["id"]))
    actual = set(locations)
    return {
        "missing": sorted(expected - actual),
        "unknown": sorted(actual - expected),
        "duplicates": {
            key: merged_ids
            for key, merged_ids in sorted(locations.items())
            if len(merged_ids) > 1
        },
    }


def _source_category_by_ref(
    source_taxonomies: dict[str, dict[str, Any]], ref: tuple[str, str]
) -> dict[str, Any]:
    taxonomy_id, category_id = ref
    for taxonomy in source_taxonomies.values():
        if taxonomy["taxonomy_id"] != taxonomy_id:
            continue
        for category in taxonomy["categories"]:
            if str(category["id"]) == category_id:
                return {
                    "source_taxonomy_id": taxonomy_id,
                    "source_category_id": category_id,
                    "name": category["name"],
                    "definition": category["definition"],
                    "module": category["module"],
                }
    raise KeyError(f"Unknown source provenance reference: {ref}")


def _artifact_from_categories(
    config: dict[str, Any],
    categories: list[dict[str, Any]],
    source_taxonomies: dict[str, dict[str, Any]],
    *,
    extra_generated_by: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = config["models"]["merge"]
    generated_by = {
        "model": model["name"],
        "temperature": model.get("temperature"),
        "seed": int(config["experiment"]["pilot_seed"]),
        "taxonomy_metadata_only": True,
        "trajectory_input": False,
        "gold_input": False,
        "prediction_or_result_input": False,
        "source_taxonomy_hashes": {
            taxonomy["taxonomy_id"]: _sha256_json(taxonomy)
            for taxonomy in source_taxonomies.values()
        },
    }
    generated_by.update(extra_generated_by or {})
    return {
        "taxonomy_id": "aegis_whowhen_pilot_merged",
        "name": "Pilot Merge of AEGIS and Who&When Pro Text Taxonomies",
        "category_count": len(categories),
        "categories": categories,
        "generated_by": generated_by,
        "freeze_policy": (
            "Create one validated pilot merge, preserve it, and never apply "
            "human semantic edits or result-dependent changes."
        ),
    }


def _adjudicate_duplicate_provenance(
    config: dict[str, Any],
    llm: CachedLLM,
    prior_output: dict[str, Any],
    source_taxonomies: dict[str, dict[str, Any]],
    *,
    base_attempt_number: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    categories = json.loads(json.dumps(prior_output["categories"]))
    issues = _provenance_issues(categories, source_taxonomies)
    if issues["missing"] or issues["unknown"] or not issues["duplicates"]:
        return None, []

    model = config["models"]["merge"]
    decisions = []
    for decision_index, (ref, candidate_ids) in enumerate(
        issues["duplicates"].items(), start=1
    ):
        source_category = _source_category_by_ref(source_taxonomies, ref)
        candidates = [
            {
                "id": category["id"],
                "name": category["name"],
                "definition": category["definition"],
                "module": category["module"],
                "other_source_categories": [
                    item
                    for item in category["source_categories"]
                    if (
                        str(item["source_taxonomy_id"]),
                        str(item["source_category_id"]),
                    )
                    != ref
                ],
            }
            for category in categories
            if str(category["id"]) in candidate_ids
        ]
        user_prompt = _read_prompt("merge_adjudication_user.txt").format(
            source_category=_json_text(source_category),
            candidate_categories=_json_text(candidates),
        )
        decision, call = llm.call(
            model=model["name"],
            system_prompt=_read_prompt("merge_adjudication_system.txt").strip(),
            user_prompt=user_prompt,
            temperature=model.get("temperature"),
            reasoning_effort=model.get("reasoning_effort"),
            seed=int(config["experiment"]["pilot_seed"]) + decision_index,
            provider=model.get("provider"),
            max_output_tokens=500,
            response_model=MergeProvenanceDecision,
        )
        selected = decision.selected_merged_category_id
        if selected not in candidate_ids:
            decisions.append(
                {
                    "source_reference": list(ref),
                    "candidate_ids": candidate_ids,
                    "status": "invalid_candidate_id",
                    "selected_merged_category_id": selected,
                    "reason": decision.reason,
                    "cache_key": call["cache_key"],
                    "cache_hit": call["cache_hit"],
                }
            )
            return None, decisions

        for category in categories:
            category["source_categories"] = [
                item
                for item in category["source_categories"]
                if (
                    str(item["source_taxonomy_id"]),
                    str(item["source_category_id"]),
                )
                != ref
            ]
            if str(category["id"]) == selected:
                category["source_categories"].append(
                    {
                        "source_taxonomy_id": ref[0],
                        "source_category_id": ref[1],
                    }
                )
        categories = [
            category for category in categories if category["source_categories"]
        ]
        record = {
            "source_reference": list(ref),
            "source_category": source_category,
            "candidate_ids": candidate_ids,
            "selected_merged_category_id": selected,
            "reason": decision.reason,
            "status": "applied",
            "cache_key": call["cache_key"],
            "cache_hit": call["cache_hit"],
            "response_metadata": call["response_metadata"],
            "raw_output_text": call["raw_output_text"],
        }
        decisions.append(record)
        write_json(
            resolve_path(config, "artifacts")
            / "provenance_adjudications"
            / f"base_{base_attempt_number:02d}_decision_{decision_index:02d}.json",
            record,
        )

    artifact = _artifact_from_categories(
        config,
        categories,
        source_taxonomies,
        extra_generated_by={
            "base_merge_attempt": base_attempt_number,
            "provenance_adjudication_model": model["name"],
            "provenance_adjudication_count": len(decisions),
            "provenance_repair_method": (
                "LLM chooses among only the duplicated merged-category candidates; "
                "code removes duplicate references and applies that choice exactly."
            ),
        },
    )
    try:
        _validate_merged_taxonomy(artifact, source_taxonomies)
    except ValueError:
        return None, decisions
    return artifact, decisions


def obtain_merged_taxonomy(
    config: dict[str, Any],
    source_taxonomies: dict[str, dict[str, Any]],
    *,
    allow_generation: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    artifact_dir = resolve_path(config, "artifacts")
    destination = artifact_dir / "pilot_merged_taxonomy.json"
    audit_path = artifact_dir / "pilot_merge_audit.json"
    if destination.exists():
        artifact = read_json(destination)
        validation = _validate_merged_taxonomy(artifact, source_taxonomies)
        if audit_path.exists():
            prior_audit = read_json(audit_path)
            if str(prior_audit.get("status", "")).startswith(
                "generated_and_frozen_validated_merge"
            ):
                for key, value in validation.items():
                    if prior_audit.get(key) != value:
                        raise ValueError(
                            f"Frozen merge audit differs from revalidation: {key}"
                        )
                return artifact, prior_audit
        audit = {
            "status": "reused_frozen_validated_merge",
            "human_semantic_edits": False,
            **validation,
        }
        write_json(audit_path, audit)
        return artifact, audit
    if not allow_generation:
        if audit_path.exists():
            existing_audit = read_json(audit_path)
            if existing_audit.get("status") == "failed_validation":
                return None, existing_audit
        audit = {
            "status": "merge_required_no_api_called",
            "human_semantic_edits": False,
        }
        write_json(audit_path, audit)
        return None, audit

    llm = CachedLLM(config, resolve_path(config, "cache") / "merge")
    model = config["models"]["merge"]
    source_payload = {
        dataset: {
            "taxonomy_id": source_taxonomies[dataset]["taxonomy_id"],
            "name": source_taxonomies[dataset]["name"],
            "categories": [
                {
                    "id": category["id"],
                    "name": category["name"],
                    "definition": category["definition"],
                    "module": category["module"],
                }
                for category in source_taxonomies[dataset]["categories"]
            ],
        }
        for dataset in DATASETS
    }
    attempts: list[dict[str, Any]] = []
    repair_context = ""
    previous_output: dict[str, Any] | None = None
    if audit_path.exists():
        prior_audit = read_json(audit_path)
        if prior_audit.get("status") == "failed_validation":
            attempts = list(prior_audit.get("attempts") or [])
            if attempts:
                last_attempt_number = int(attempts[-1]["attempt"])
                last_attempt_path = (
                    artifact_dir
                    / "merge_attempts"
                    / f"attempt_{last_attempt_number:02d}.json"
                )
                if last_attempt_path.exists():
                    last_attempt = read_json(last_attempt_path)
                    previous_output = last_attempt.get("parsed_output")
                    if previous_output:
                        repair_context = _merge_repair_context(
                            prior_output=previous_output,
                            validation_error=str(attempts[-1].get("error") or ""),
                            repair_attempt_number=last_attempt_number + 1,
                            source_taxonomies=source_taxonomies,
                        )

    if previous_output and attempts:
        base_attempt_number = int(attempts[-1]["attempt"])
        adjudicated, decisions = _adjudicate_duplicate_provenance(
            config,
            llm,
            previous_output,
            source_taxonomies,
            base_attempt_number=base_attempt_number,
        )
        if adjudicated is not None:
            validation = _validate_merged_taxonomy(
                adjudicated, source_taxonomies
            )
            _freeze_json(destination, adjudicated)
            audit = {
                "status": (
                    "generated_and_frozen_validated_merge_with_"
                    "llm_provenance_adjudication"
                ),
                "human_semantic_edits": False,
                "attempts": attempts,
                "provenance_adjudications": decisions,
                **validation,
            }
            write_json(audit_path, audit)
            return adjudicated, audit

    first_attempt_number = (
        max(int(item["attempt"]) for item in attempts) + 1 if attempts else 1
    )
    new_attempt_count = 1 + int(config["execution"]["merge_validation_retries"])
    for attempt_index in range(
        first_attempt_number, first_attempt_number + new_attempt_count
    ):
        user_prompt = _read_prompt("merge_user.txt").format(
            source_taxonomies=_json_text(source_payload),
            repair_context=repair_context,
        )
        parsed, call = llm.call(
            model=model["name"],
            system_prompt=_read_prompt("merge_system.txt").strip(),
            user_prompt=user_prompt,
            temperature=model.get("temperature"),
            reasoning_effort=model.get("reasoning_effort"),
            seed=int(config["experiment"]["pilot_seed"]),
            provider=model.get("provider"),
            max_output_tokens=int(model["max_output_tokens"]),
            response_model=MergedTaxonomyOutput,
        )
        previous_output = parsed.model_dump(mode="json")
        artifact = _artifact_from_categories(
            config,
            previous_output["categories"],
            source_taxonomies,
            extra_generated_by={"base_merge_attempt": attempt_index},
        )
        try:
            validation = _validate_merged_taxonomy(artifact, source_taxonomies)
            attempt_record = {
                "attempt": attempt_index,
                "status": "valid",
                "cache_key": call["cache_key"],
                "cache_hit": call["cache_hit"],
                "response_metadata": call["response_metadata"],
                **validation,
            }
            attempts.append(attempt_record)
            write_json(
                artifact_dir / "merge_attempts" / f"attempt_{attempt_index:02d}.json",
                {
                    "audit": attempt_record,
                    "parsed_output": previous_output,
                    "raw_output_text": call["raw_output_text"],
                },
            )
            _freeze_json(destination, artifact)
            audit = {
                "status": "generated_and_frozen_validated_merge",
                "human_semantic_edits": False,
                "attempts": attempts,
                **validation,
            }
            write_json(audit_path, audit)
            return artifact, audit
        except ValueError as exc:
            attempt_record = {
                "attempt": attempt_index,
                "status": "invalid_provenance_or_schema",
                "error": str(exc),
                "cache_key": call["cache_key"],
                "cache_hit": call["cache_hit"],
                "response_metadata": call["response_metadata"],
            }
            attempts.append(attempt_record)
            write_json(
                artifact_dir / "merge_attempts" / f"attempt_{attempt_index:02d}.json",
                {
                    "audit": attempt_record,
                    "parsed_output": previous_output,
                    "raw_output_text": call["raw_output_text"],
                },
            )
            repair_context = (
                _merge_repair_context(
                    prior_output=previous_output,
                    validation_error=str(exc),
                    repair_attempt_number=attempt_index + 1,
                    source_taxonomies=source_taxonomies,
                )
            )
    audit = {
        "status": "failed_validation",
        "human_semantic_edits": False,
        "attempts": attempts,
    }
    write_json(audit_path, audit)
    raise ValueError("Merged taxonomy failed deterministic validation after retries")


def _run_id(
    config: dict[str, Any], prepared: dict[str, Any], merged: dict[str, Any]
) -> str:
    payload = {
        "experiment": config["experiment"],
        "models": config["models"],
        "api_provider_defaults": config["api"]["provider_defaults"],
        "execution": config["execution"],
        "sampling_manifest_sha256": _sha256_json(prepared["manifest"]),
        "source_taxonomy_sha256": {
            dataset: _sha256_json(prepared["taxonomies"][dataset])
            for dataset in DATASETS
        },
        "merged_taxonomy_sha256": _sha256_json(merged),
        "prompts": {
            path.name: _sha256_file(path)
            for path in sorted((EXPERIMENT_DIR / "prompts").glob("*.txt"))
        },
        "attribution_schema_sha256": _sha256_json(
            AttributionOutput.model_json_schema()
        ),
        "merge_schema_sha256": _sha256_json(
            MergedTaxonomyOutput.model_json_schema()
        ),
        "merge_adjudication_schema_sha256": _sha256_json(
            MergeProvenanceDecision.model_json_schema()
        ),
    }
    return "pilot-" + _sha256_json(payload)[:16]


def _prediction_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest = {}
    if path.exists():
        for row in iter_jsonl(path):
            latest[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    return latest


def run_predictions(
    config: dict[str, Any],
    run_id: str,
    run_dir: Path,
    prepared: dict[str, Any],
    merged: dict[str, Any],
) -> None:
    prediction_path = run_dir / "pilot_predictions.jsonl"
    existing = _prediction_index(prediction_path)
    inputs = {
        (row["dataset"], row["trajectory_id"]): row for row in prepared["inputs"]
    }
    taxonomies = {**prepared["taxonomies"], "merged": merged}
    model = config["models"]["attribution"]
    llm = CachedLLM(config, resolve_path(config, "cache") / "attribution")
    counter = TokenCounter.for_model(model["name"], model.get("tokenizer_encoding"))
    keys = planned_keys(prepared["inputs"], int(config["experiment"]["pilot_seed"]))
    completed = sum(row.get("status") == "ok" for row in existing.values())
    for dataset, trajectory_id, condition in keys:
        key = (dataset, trajectory_id, condition)
        if existing.get(key, {}).get("status") == "ok":
            continue
        record = inputs[(dataset, trajectory_id)]
        built = build_prompt(record, condition, taxonomies)
        labels = built["output_ids"]
        label_retry_count = 0
        truncation_retry_count = 0
        initial_valid: bool | None = None
        attempts: list[dict[str, Any]] = []
        final_prompt = built["user_prompt"]
        try:
            while True:
                suffixes = []
                if label_retry_count:
                    suffixes.append(_read_prompt("invalid_label_retry.txt").strip())
                if truncation_retry_count:
                    suffixes.append(_read_prompt("truncation_retry.txt").strip())
                final_prompt = built["user_prompt"]
                if suffixes:
                    final_prompt += "\n\n" + "\n\n".join(suffixes)
                prompt_tokens = counter.count(
                    built["system_prompt"] + "\n" + final_prompt
                )
                output_limit = (
                    int(config["execution"]["truncation_retry_max_output_tokens"])
                    if truncation_retry_count
                    else int(model["max_output_tokens"])
                )
                try:
                    prediction, call = llm.call(
                        model=model["name"],
                        system_prompt=built["system_prompt"],
                        user_prompt=final_prompt,
                        temperature=model.get("temperature"),
                        reasoning_effort=model.get("reasoning_effort"),
                        seed=int(config["experiment"]["pilot_seed"]),
                        provider=model.get("provider"),
                        max_output_tokens=output_limit,
                        response_model=AttributionOutput,
                    )
                except TruncatedResponseError:
                    if initial_valid is None:
                        initial_valid = False
                    attempts.append(
                        {
                            "attempt": len(attempts) + 1,
                            "status": "truncated",
                            "prompt_tokens_estimated": prompt_tokens,
                            "max_output_tokens": output_limit,
                            "label_retry_index": label_retry_count,
                            "truncation_retry_index": truncation_retry_count,
                        }
                    )
                    if truncation_retry_count >= int(
                        config["execution"]["truncation_retries"]
                    ):
                        raise
                    truncation_retry_count += 1
                    continue
                exact_valid = prediction.failure_category in labels
                if initial_valid is None:
                    initial_valid = exact_valid
                usage = call["response_metadata"].get("usage") or {}
                attempts.append(
                    {
                        "attempt": len(attempts) + 1,
                        "status": "valid" if exact_valid else "invalid_output_id",
                        "failure_category": prediction.failure_category,
                        "exact_native_id_valid": exact_valid,
                        "prompt_tokens_estimated": prompt_tokens,
                        "reported_prompt_tokens": usage.get("prompt_tokens"),
                        "reported_response_tokens": usage.get("completion_tokens"),
                        "max_output_tokens": output_limit,
                        "label_retry_index": label_retry_count,
                        "truncation_retry_index": truncation_retry_count,
                        "cache_key": call["cache_key"],
                        "cache_hit": call["cache_hit"],
                        "response_metadata": call["response_metadata"],
                    }
                )
                if exact_valid:
                    break
                if label_retry_count >= int(
                    config["execution"]["invalid_label_retries"]
                ):
                    # Keep the terminal invalid response as an incorrect pilot row so
                    # compliance can be measured and the remaining implementation
                    # checks can finish. It is never normalized or relabeled.
                    break
                label_retry_count += 1

            gold = prepared["gold"][(dataset, trajectory_id)]
            final_valid = prediction.failure_category in labels
            row = {
                "run_id": run_id,
                "dataset": dataset,
                "trajectory_id": trajectory_id,
                "framework": record["framework"],
                "benchmark": record["benchmark"],
                "condition": condition,
                "attribution_model": model["name"],
                "model_parameters": {
                    "temperature": model.get("temperature"),
                    "reasoning_effort": model.get("reasoning_effort"),
                    "seed": int(config["experiment"]["pilot_seed"]),
                    "provider": model.get("provider"),
                    "configured_max_output_tokens": int(model["max_output_tokens"]),
                },
                "output_ids": labels,
                "output_label_block_sha256": _sha256_bytes(
                    built["output_label_block"].encode("utf-8")
                ),
                "guidance_taxonomy_id": built["guidance_taxonomy_id"],
                "diagnostic_guidance_sha256": _sha256_bytes(
                    built["diagnostic_guidance"].encode("utf-8")
                ),
                "system_prompt_sha256": _sha256_bytes(
                    built["system_prompt"].encode("utf-8")
                ),
                "trajectory_payload_sha256": record["trajectory_payload_sha256"],
                "request_prompt_sha256": _sha256_bytes(
                    (built["system_prompt"] + "\n" + final_prompt).encode("utf-8")
                ),
                "raw_model_response": call["raw_output_text"],
                "failure_category": prediction.failure_category,
                "gold_category": gold,
                "correct": final_valid and prediction.failure_category == gold,
                "explanation": prediction.explanation,
                "evidence": prediction.evidence,
                "initial_exact_id_valid": initial_valid,
                "final_exact_id_valid": final_valid,
                "label_retry_count": label_retry_count,
                "truncation_retry_count": truncation_retry_count,
                "retry_count": label_retry_count + truncation_retry_count,
                "attempts": attempts,
                "final_cache_key": call["cache_key"],
                "final_cache_hit": call["cache_hit"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "run_id": run_id,
                "dataset": dataset,
                "trajectory_id": trajectory_id,
                "framework": record["framework"],
                "benchmark": record["benchmark"],
                "condition": condition,
                "initial_exact_id_valid": initial_valid,
                "label_retry_count": label_retry_count,
                "truncation_retry_count": truncation_retry_count,
                "attempts": attempts,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            append_jsonl(prediction_path, row)
            raise
        append_jsonl(prediction_path, row)
        existing[key] = row
        completed += 1
        every = int(config["execution"]["progress_every"])
        if completed % every == 0 or completed == len(keys):
            print(
                f"[aegis-whowhen-pilot] completed={completed}/{len(keys)} "
                f"dataset={dataset} condition={condition}",
                flush=True,
            )


def _latest_complete(
    path: Path, inputs: list[dict[str, Any]], seed: int
) -> dict[tuple[str, str, str], dict[str, Any]]:
    predictions = _prediction_index(path)
    expected = set(planned_keys(inputs, seed))
    missing = expected - set(predictions)
    extras = set(predictions) - expected
    errors = {
        key for key, row in predictions.items() if row.get("status") != "ok"
    }
    if missing or extras or errors:
        raise ValueError(
            f"Incomplete pilot predictions: missing={len(missing)}, "
            f"extras={len(extras)}, errors={len(errors)}"
        )
    return predictions


def _condition_results(
    predictions: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [
                row
                for (d, _, c), row in predictions.items()
                if d == dataset and c == condition
            ]
            correct = sum(bool(row["correct"]) for row in group)
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "correct": correct,
                    "N": len(group),
                    "accuracy": correct / len(group),
                    "interpretation": "descriptive_pilot_only",
                }
            )
    return rows


def _output_compliance(
    predictions: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [
                row
                for (d, _, c), row in predictions.items()
                if d == dataset and c == condition
            ]
            initial = sum(row["initial_exact_id_valid"] is True for row in group)
            final = sum(row["final_exact_id_valid"] is True for row in group)
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "N": len(group),
                    "initial_valid_output": initial,
                    "initial_compliance_rate": initial / len(group),
                    "label_retry_count": sum(row["label_retry_count"] for row in group),
                    "final_valid_output": final,
                    "final_compliance_rate": final / len(group),
                    "terminal_invalid_count": len(group) - final,
                }
            )
    return rows


def _category_frequencies(
    predictions: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [
                row
                for (d, _, c), row in predictions.items()
                if d == dataset and c == condition
            ]
            counts = Counter(row["failure_category"] for row in group)
            for category_id, count in sorted(counts.items()):
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "predicted_category_id": category_id,
                        "count": count,
                        "frequency": count / len(group),
                    }
                )
    return rows


def _guidance_sensitivity(
    predictions: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        trajectory_ids = sorted(
            trajectory_id
            for d, trajectory_id, condition in predictions
            if d == dataset and condition == "label_only"
        )
        for trajectory_id in trajectory_ids:
            labels = {
                condition: predictions[(dataset, trajectory_id, condition)][
                    "failure_category"
                ]
                for condition in CONDITIONS
            }
            baseline = labels["label_only"]
            changed = [
                condition
                for condition in CONDITIONS[1:]
                if labels[condition] != baseline
            ]
            patterns = {
                (): "all_four_same",
                ("own_taxonomy",): "own_changed",
                ("foreign_taxonomy",): "foreign_changed",
                ("merged_taxonomy",): "merged_changed",
            }
            pattern = patterns.get(tuple(changed), "multiple_changed")
            rows.append(
                {
                    "dataset": dataset,
                    "trajectory_id": trajectory_id,
                    **labels,
                    "sensitivity_pattern": pattern,
                }
            )
    return rows


def _transitions(
    predictions: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        trajectory_ids = sorted(
            trajectory_id
            for d, trajectory_id, condition in predictions
            if d == dataset and condition == "label_only"
        )
        for before_condition, after_condition in TRANSITIONS:
            counts: Counter[str] = Counter()
            for trajectory_id in trajectory_ids:
                before = predictions[(dataset, trajectory_id, before_condition)]
                after = predictions[(dataset, trajectory_id, after_condition)]
                if before["failure_category"] == after["failure_category"]:
                    transition = "unchanged"
                elif not before["correct"] and after["correct"]:
                    transition = "incorrect_to_correct"
                elif before["correct"] and not after["correct"]:
                    transition = "correct_to_incorrect"
                else:
                    transition = "incorrect_to_different_incorrect"
                counts[transition] += 1
            for transition in (
                "incorrect_to_correct",
                "correct_to_incorrect",
                "unchanged",
                "incorrect_to_different_incorrect",
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{before_condition} -> {after_condition}",
                        "transition_type": transition,
                        "count": counts[transition],
                        "frequency": counts[transition] / len(trajectory_ids),
                    }
                )
    return rows


def _token_audit(
    predictions: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [
                row
                for (d, _, c), row in predictions.items()
                if d == dataset and c == condition
            ]
            attempts = [attempt for row in group for attempt in row["attempts"]]
            prompt_values = [
                int(
                    attempt.get("reported_prompt_tokens")
                    or attempt.get("prompt_tokens_estimated")
                    or 0
                )
                for attempt in attempts
            ]
            response_values = [
                int(attempt.get("reported_response_tokens") or 0)
                for attempt in attempts
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "prediction_rows": len(group),
                    "attempts": len(attempts),
                    "prompt_tokens": sum(prompt_values),
                    "response_tokens": sum(response_values),
                    "average_prompt_tokens_per_attempt": sum(prompt_values)
                    / len(attempts),
                    "max_prompt_tokens": max(prompt_values),
                    "truncation_retries": sum(
                        row["truncation_retry_count"] for row in group
                    ),
                }
            )
    return rows


def _ledger_summary(config: dict[str, Any]) -> dict[str, Any]:
    path = config["_root"] / config["cost_budget"]["ledger_path"]
    rows = list(iter_jsonl(path)) if path.exists() else []
    return {
        "actual_api_calls": len(rows),
        "attribution_api_calls": sum(
            row.get("model") == config["models"]["attribution"]["name"]
            for row in rows
        ),
        "merge_api_calls": sum(
            row.get("model") == config["models"]["merge"]["name"]
            for row in rows
        ),
        "inference_cost_usd": sum(float(row.get("cost_usd", 0.0)) for row in rows),
    }


def _markdown_table(
    rows: list[dict[str, Any]], fields: list[str], *, decimals: int = 4
) -> str:
    def display(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        if value is None:
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(display(row.get(field)) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _prediction_csv_rows(
    predictions: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(predictions):
        row = predictions[key]
        rows.append(
            {
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "framework": row["framework"],
                "benchmark": row["benchmark"],
                "condition": row["condition"],
                "attribution_model": row["attribution_model"],
                "guidance_taxonomy_id": row.get("guidance_taxonomy_id"),
                "failure_category": row["failure_category"],
                "gold_category": row["gold_category"],
                "correct": row["correct"],
                "explanation": row["explanation"],
                "evidence": row["evidence"],
                "initial_exact_id_valid": row["initial_exact_id_valid"],
                "final_exact_id_valid": row["final_exact_id_valid"],
                "label_retry_count": row["label_retry_count"],
                "truncation_retry_count": row["truncation_retry_count"],
                "retry_count": row["retry_count"],
                "trajectory_payload_sha256": row["trajectory_payload_sha256"],
                "request_prompt_sha256": row["request_prompt_sha256"],
                "raw_model_response": row["raw_model_response"],
                "attempts_json": json.dumps(
                    row["attempts"], ensure_ascii=False, separators=(",", ":")
                ),
                "timestamp_utc": row["timestamp_utc"],
            }
        )
    return rows


def _sensitivity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    patterns = (
        "all_four_same",
        "own_changed",
        "foreign_changed",
        "merged_changed",
        "multiple_changed",
    )
    for dataset in DATASETS:
        group = [row for row in rows if row["dataset"] == dataset]
        counts = Counter(row["sensitivity_pattern"] for row in group)
        for pattern in patterns:
            output.append(
                {
                    "dataset": dataset,
                    "sensitivity_pattern": pattern,
                    "count": counts[pattern],
                    "frequency": counts[pattern] / len(group),
                }
            )
    return output


def _distribution_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [
                row
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            maximum = max(group, key=lambda item: (item["frequency"], item["predicted_category_id"]))
            output.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "most_frequent_id": maximum["predicted_category_id"],
                    "count": maximum["count"],
                    "frequency": maximum["frequency"],
                }
            )
    return output


def _build_report(
    config: dict[str, Any],
    prepared: dict[str, Any],
    merge_audit: dict[str, Any],
    validation: dict[str, Any],
    predictions: dict[tuple[str, str, str], dict[str, Any]],
    results: list[dict[str, Any]],
    compliance: list[dict[str, Any]],
    frequencies: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    token_audit: list[dict[str, Any]],
    ledger: dict[str, Any],
) -> tuple[str, str, list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    if validation["status"] != "passed":
        issues.append("Pre-inference validation did not pass.")
    if prepared["leakage"]["leakage_count"]:
        issues.append(
            f"Gold/metadata leakage was detected in {prepared['leakage']['leakage_count']} serialized trajectories."
        )
    if any(row["terminal_invalid_count"] for row in compliance):
        issues.append("At least one row ended with an invalid native output ID.")
    if any(row["final_compliance_rate"] != 1.0 for row in compliance):
        issues.append("Final native-ID output compliance was below 100%.")

    truncation_threshold = float(
        config["execution"]["systematic_truncation_threshold"]
    )
    for row in token_audit:
        truncation_rate = row["truncation_retries"] / row["prediction_rows"]
        if truncation_rate > truncation_threshold:
            issues.append(
                f"Systematic truncation exceeded the threshold for {row['dataset']}/{row['condition']}: {truncation_rate:.2%}."
            )

    distribution = _distribution_summary(frequencies)
    collapse_warning = float(config["execution"]["collapse_warning_threshold"])
    for row in distribution:
        if row["frequency"] == 1.0:
            issues.append(
                f"Single-category collapse occurred for {row['dataset']}/{row['condition']} ({row['most_frequent_id']})."
            )
        elif row["frequency"] >= collapse_warning:
            warnings.append(
                f"High concentration for {row['dataset']}/{row['condition']}: {row['most_frequent_id']}={row['frequency']:.2%}."
            )

    if not str(merge_audit.get("status", "")).startswith(
        ("generated_and_frozen", "reused_frozen")
    ):
        issues.append("The merged taxonomy was not frozen after provenance validation.")

    sensitivity_summary = _sensitivity_summary(sensitivity)
    total_label_retries = sum(row["label_retry_count"] for row in compliance)
    total_truncations = sum(row["truncation_retries"] for row in token_audit)
    status = "READY_FOR_FULL_EXPERIMENT" if not issues else "IMPLEMENTATION_FIX_REQUIRED"
    issue_lines = "\n".join(f"- {item}" for item in issues) or "- None."
    warning_lines = "\n".join(f"- {item}" for item in warnings) or "- None."

    report = f"""# AEGIS–Who&When Pro Fixed-Guidance Pilot Report

## Purpose

This pilot validates the implementation only: dataset loading, frozen balanced sampling, official taxonomy resolution, leakage prevention, condition isolation, exact native-ID output, retry behavior, and token/truncation safety. It is not a hypothesis test, and accuracy direction is not a readiness criterion.

## Dataset Samples

- AEGIS: 56 trajectories (4 per each of 14 native classes)
- Who&When Pro text: 56 trajectories (4 per each of 14 observed native classes)
- Total: 112 frozen trajectories and 448 prediction rows
- Pilot seed: {config['experiment']['pilot_seed']}
- Future full-experiment seed reserved: {config['experiment']['final_seed_reserved']}; the 112 pilot IDs must be excluded

## Taxonomies

- AEGIS: 14 official MAST-derived failure modes with canonical IDs, names, definitions, and groups
- Who&When Pro text: the 14 modes observed locally, resolved against the official taxonomy artifact
- Merged: {merge_audit.get('merged_category_count', 'unknown')} categories; all 28 source categories recorded exactly once in provenance; no human semantic edits

## Conditions

Label-Only, Own, Foreign, and Merged use the same attribution model, system/task instruction, structured-output schema, trajectory serializer, and target-native `ID -> official name` output list. Only `DIAGNOSTIC_GUIDANCE` changes. Foreign guidance never becomes the output ontology.

## Leakage Validation

- Rows scanned before API inference: {prepared['leakage']['rows_scanned']}
- Leakage findings: {prepared['leakage']['leakage_count']}
- Pre-inference checks: {validation['checks_passed']}/{validation['checks_total']} passed

## Output Compliance

{_markdown_table(compliance, ['dataset', 'condition', 'N', 'initial_compliance_rate', 'label_retry_count', 'final_compliance_rate', 'terminal_invalid_count'])}

Total formatting-only label retries: {total_label_retries}. No relabeler, semantic judge, embedding, synonym matching, or category canonicalization was used.

## Token / Truncation Audit

{_markdown_table(token_audit, ['dataset', 'condition', 'attempts', 'prompt_tokens', 'response_tokens', 'max_prompt_tokens', 'truncation_retries'])}

- Total truncation retries: {total_truncations}
- Actual paid API calls in the isolated ledger: {ledger['actual_api_calls']} (merge={ledger['merge_api_calls']}, attribution={ledger['attribution_api_calls']})
- Recorded inference cost: ${ledger['inference_cost_usd']:.6f}

## Descriptive Accuracy

The following values are descriptive pilot observations only and must not be treated as final comparative performance.

{_markdown_table(results, ['dataset', 'condition', 'correct', 'N', 'accuracy'])}

## Prediction Distribution

The table shows the most frequent predicted native ID in each dataset-condition cell. Complete distributions are in `pilot_category_frequencies.csv`.

{_markdown_table(distribution, ['dataset', 'condition', 'most_frequent_id', 'count', 'frequency'])}

## Guidance Sensitivity

{_markdown_table(sensitivity_summary, ['dataset', 'sensitivity_pattern', 'count', 'frequency'])}

Per-trajectory predictions and sensitivity patterns are in `pilot_guidance_sensitivity.csv`.

## Transition Analysis

{_markdown_table(transitions, ['dataset', 'comparison', 'transition_type', 'count', 'frequency'])}

## Problems Found

{issue_lines}

Warnings that do not independently fail implementation readiness:

{warning_lines}

## Recommendation

`{status}`

Accuracy direction was deliberately excluded from this decision. If ready, retain the frozen taxonomies/prompts/merge and exclude these pilot IDs from the future full sampling pool.

## Final Status

{status}
"""
    return report, status, issues, warnings


def generate_results(
    config: dict[str, Any],
    run_dir: Path,
    prepared: dict[str, Any],
    merge_audit: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Path]:
    predictions = _latest_complete(
        run_dir / "pilot_predictions.jsonl",
        prepared["inputs"],
        int(config["experiment"]["pilot_seed"]),
    )
    prediction_rows = _prediction_csv_rows(predictions)
    results = _condition_results(predictions)
    compliance = _output_compliance(predictions)
    frequencies = _category_frequencies(predictions)
    sensitivity = _guidance_sensitivity(predictions)
    transitions = _transitions(predictions)
    token_audit = _token_audit(predictions)
    ledger = _ledger_summary(config)
    report, status, issues, warnings = _build_report(
        config,
        prepared,
        merge_audit,
        validation,
        predictions,
        results,
        compliance,
        frequencies,
        sensitivity,
        transitions,
        token_audit,
        ledger,
    )

    outputs = {
        "predictions_jsonl": run_dir / "pilot_predictions.jsonl",
        "predictions_csv": run_dir / "pilot_predictions.csv",
        "results_csv": run_dir / "pilot_results.csv",
        "output_compliance_csv": run_dir / "pilot_output_compliance.csv",
        "category_frequencies_csv": run_dir / "pilot_category_frequencies.csv",
        "transitions_csv": run_dir / "pilot_transitions.csv",
        "guidance_sensitivity_csv": run_dir / "pilot_guidance_sensitivity.csv",
        "token_audit_csv": run_dir / "pilot_token_audit.csv",
        "report_markdown": run_dir / "PILOT_REPORT.md",
        "status_json": run_dir / "pilot_status.json",
    }
    prediction_fields = list(prediction_rows[0])
    _write_csv(outputs["predictions_csv"], prediction_rows, prediction_fields)
    _write_csv(outputs["results_csv"], results, list(results[0]))
    _write_csv(outputs["output_compliance_csv"], compliance, list(compliance[0]))
    _write_csv(outputs["category_frequencies_csv"], frequencies, list(frequencies[0]))
    _write_csv(outputs["transitions_csv"], transitions, list(transitions[0]))
    _write_csv(outputs["guidance_sensitivity_csv"], sensitivity, list(sensitivity[0]))
    _write_csv(outputs["token_audit_csv"], token_audit, list(token_audit[0]))
    _write_text(outputs["report_markdown"], report)
    write_json(
        outputs["status_json"],
        {
            "status": status,
            "implementation_issues": issues,
            "warnings": warnings,
            "ledger": ledger,
        },
    )
    return outputs


def _freeze_run_artifacts(
    config: dict[str, Any],
    run_id: str,
    run_dir: Path,
    merged: dict[str, Any],
    merge_audit: dict[str, Any],
    validation: dict[str, Any],
    budget: dict[str, Any],
) -> None:
    prepared_dir = resolve_path(config, "prepared")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "pilot_sampling_manifest.json",
        "pilot_sampling_manifest.csv",
        "aegis_taxonomy.json",
        "whowhen_taxonomy.json",
        "pilot_prompt_templates.json",
        "pilot_leakage_audit.json",
    ):
        _freeze_copy(prepared_dir / name, run_dir / name)
    _freeze_json(run_dir / "pilot_config.json", _sanitized_config(config, run_id))
    _freeze_json(run_dir / "pilot_merged_taxonomy.json", merged)
    _freeze_json(run_dir / "pilot_merge_audit.json", merge_audit)
    _freeze_json(run_dir / "pilot_preflight_validation.json", validation)
    initial_budget_path = run_dir / "pilot_budget_preflight.json"
    if initial_budget_path.exists():
        # Incurred spend necessarily changes after a partial run. Preserve the
        # original preflight and record the latest resume-time check separately.
        write_json(run_dir / "pilot_budget_resume_check.json", budget)
    else:
        _freeze_json(initial_budget_path, budget)


def run_workflow(
    config_path: str | Path = (
        "experiments/aegis_whowhen_pilot_fixed_guidance/experiment.yaml"
    ),
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    prepared = prepare_pilot(config)
    source_taxonomies = {
        dataset: prepared["taxonomies"][dataset] for dataset in DATASETS
    }

    merged, merge_audit = obtain_merged_taxonomy(
        config, source_taxonomies, allow_generation=False
    )
    if merged is not None:
        prepared["taxonomies"]["merged"] = merged
    validation = validate_pre_inference(
        config, prepared, merged_available=merged is not None
    )
    budget = estimate_budget(
        config, prepared, merge_required=merged is None
    )
    if not budget["within_context"]:
        raise ValueError(
            "Pilot prompt context preflight failed; no attribution API calls were made."
        )
    if not budget["within_budget"]:
        raise ValueError(
            "Pilot budget preflight failed; no attribution API calls were made."
        )

    if preflight_only:
        return {
            "status": "PREFLIGHT_READY_NO_API_CALLS",
            "sampled_trajectories": prepared["manifest"][
                "total_sampled_trajectories"
            ],
            "planned_prediction_rows": prepared["manifest"][
                "planned_prediction_rows"
            ],
            "leakage_count": prepared["leakage"]["leakage_count"],
            "merged_taxonomy": (
                str(resolve_path(config, "artifacts") / "pilot_merged_taxonomy.json")
                if merged is not None
                else "will be generated once before inference"
            ),
            "preflight_validation": str(
                resolve_path(config, "prepared")
                / "pilot_preflight_validation.json"
            ),
            "budget_preflight": str(
                resolve_path(config, "prepared") / "pilot_budget_preflight.json"
            ),
        }

    if merged is None:
        merged, merge_audit = obtain_merged_taxonomy(
            config, source_taxonomies, allow_generation=True
        )
        assert merged is not None
        prepared["taxonomies"]["merged"] = merged
        validation = validate_pre_inference(
            config, prepared, merged_available=True
        )
        budget = estimate_budget(config, prepared, merge_required=False)
        if not budget["within_context"] or not budget["within_budget"]:
            raise ValueError(
                "Post-merge prompt/budget validation failed before attribution calls."
            )

    run_id = _run_id(config, prepared, merged)
    run_dir = resolve_path(config, "runs") / run_id
    _freeze_run_artifacts(
        config,
        run_id,
        run_dir,
        merged,
        merge_audit,
        validation,
        budget,
    )
    run_predictions(config, run_id, run_dir, prepared, merged)
    outputs = generate_results(
        config, run_dir, prepared, merge_audit, validation
    )
    return {
        "status": read_json(outputs["status_json"])["status"],
        "run_id": run_id,
        "run_directory": str(run_dir),
        **{name: str(path) for name, path in outputs.items()},
    }
