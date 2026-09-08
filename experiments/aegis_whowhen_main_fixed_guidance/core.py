from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from taxonomy_experiment.config import load_config
from taxonomy_experiment.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from taxonomy_experiment.token_count import TokenCounter

from experiments.aegis_whowhen_pilot_fixed_guidance.core import (
    AEGIS_IDS,
    WHOWHEN_OBSERVED_IDS,
    _load_aegis_candidates,
    _load_whowhen_candidates,
    _validate_merged_taxonomy,
)


HERE = Path(__file__).resolve().parent
DATASETS = ("aegis", "whowhen")
CONDITIONS = (
    "label_only",
    "own_taxonomy",
    "foreign_taxonomy",
    "merged_taxonomy",
)


class ResponseParseError(RuntimeError):
    pass


class ResponseTruncatedError(RuntimeError):
    pass


class RuntimeBudgetExceeded(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def resolve(config: dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key])
    return value if value.is_absolute() else config["_root"] / value


def prompt_text(name: str) -> str:
    return (HERE / "prompts" / name).read_text(encoding="utf-8").strip()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def freeze_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise FileExistsError(f"Frozen artifact differs: {path}")
        return
    write_json(path, value)


def freeze_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Frozen copy differs: {destination}")
        return
    shutil.copyfile(source, destination)


def native_ids(dataset: str, taxonomies: dict[str, dict[str, Any]]) -> list[str]:
    return [str(item["id"]) for item in taxonomies[dataset]["categories"]]


def output_labels(dataset: str, taxonomies: dict[str, dict[str, Any]]) -> str:
    return "\n".join(
        f"- {item['id']} -> {item['name']}"
        for item in taxonomies[dataset]["categories"]
    )


def taxonomy_guidance(taxonomy: dict[str, Any] | None) -> str:
    if taxonomy is None:
        return "[NONE]"
    return json_text(
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


def guidance_for(
    dataset: str, condition: str, taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if condition == "label_only":
        return None
    if condition == "own_taxonomy":
        return taxonomies[dataset]
    if condition == "foreign_taxonomy":
        return taxonomies["whowhen" if dataset == "aegis" else "aegis"]
    if condition == "merged_taxonomy":
        return taxonomies["merged"]
    raise ValueError(condition)


def serialize_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    """Dataset-specific allowlist serializer frozen for the main experiment."""
    source = record["trajectory_payload"]
    if record["dataset"] == "aegis":
        payload = {
            "task_query": source.get("task_query"),
            "conversation_history": source.get("conversation_history"),
        }
        if source.get("final_output") is not None:
            payload["final_output"] = source["final_output"]
        return payload
    if record["dataset"] == "whowhen":
        return {
            "task_query": source.get("task_query"),
            "trajectory": source.get("trajectory"),
        }
    raise ValueError(record["dataset"])


def build_prompt(
    record: dict[str, Any],
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dataset = record["dataset"]
    guidance_taxonomy = guidance_for(dataset, condition, taxonomies)
    trajectory = json_text(serialize_trajectory(record))
    label_block = output_labels(dataset, taxonomies)
    guidance = taxonomy_guidance(guidance_taxonomy)
    return {
        "system_prompt": prompt_text("attribution_system.txt"),
        "user_prompt": prompt_text("attribution_user.txt").format(
            output_labels=label_block,
            diagnostic_guidance=guidance,
            trajectory=trajectory,
        ),
        "trajectory": trajectory,
        "label_block": label_block,
        "guidance": guidance,
        "guidance_taxonomy_id": (
            guidance_taxonomy["taxonomy_id"] if guidance_taxonomy else None
        ),
        "permitted_ids": native_ids(dataset, taxonomies),
    }


def response_schema(permitted_ids: list[str]) -> dict[str, Any]:
    """Create the exact target-dataset enum contract sent to OpenRouter."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "NativeFailureAttribution",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "failure_category": {
                        "type": "string",
                        "enum": list(permitted_ids),
                    },
                    "explanation": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 700,
                    },
                    "evidence": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 700,
                    },
                },
                "required": ["failure_category", "explanation", "evidence"],
            },
        },
    }


def parse_attribution(raw_text: str, permitted_ids: list[str]) -> dict[str, Any]:
    """Parse without fallback; maxLength is advisory to tolerate provider overruns."""
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ResponseParseError(f"invalid_json: {exc}") from exc
    if not isinstance(value, dict):
        raise ResponseParseError("response_is_not_an_object")
    if set(value) != {"failure_category", "explanation", "evidence"}:
        raise ResponseParseError(f"wrong_fields: {sorted(value)}")
    if not all(isinstance(value[key], str) and value[key] for key in value):
        raise ResponseParseError("all_fields_must_be_nonempty_strings")
    value["exact_native_id_valid"] = value["failure_category"] in permitted_ids
    return value


def load_and_freeze_taxonomies(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pilot_run = resolve(config, "pilot_run")
    frozen = resolve(config, "frozen")
    sources = {
        "aegis": pilot_run / "aegis_taxonomy.json",
        "whowhen": pilot_run / "whowhen_taxonomy.json",
        "merged": pilot_run / "pilot_merged_taxonomy.json",
    }
    destinations = {
        "aegis": frozen / "aegis_taxonomy.json",
        "whowhen": frozen / "whowhen_taxonomy.json",
        "merged": frozen / "merged_taxonomy.json",
    }
    for dataset in ("aegis", "whowhen", "merged"):
        freeze_copy(sources[dataset], destinations[dataset])
    freeze_copy(
        pilot_run / "pilot_merge_audit.json", frozen / "merged_taxonomy_audit.json"
    )
    taxonomies = {key: read_json(path) for key, path in destinations.items()}
    if native_ids("aegis", taxonomies) != list(AEGIS_IDS):
        raise ValueError("Frozen AEGIS IDs changed")
    if native_ids("whowhen", taxonomies) != list(WHOWHEN_OBSERVED_IDS):
        raise ValueError("Frozen Who&When IDs changed")
    _validate_merged_taxonomy(
        taxonomies["merged"],
        {dataset: taxonomies[dataset] for dataset in DATASETS},
    )
    return taxonomies


def load_candidates(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    aegis, _ = _load_aegis_candidates(config)
    whowhen, _ = _load_whowhen_candidates(config)
    candidates = {"aegis": aegis, "whowhen": whowhen}
    collision_audit: dict[str, Any] = {
        "policy": (
            "Preserve every eligible row. Keep the source ID as provenance; only "
            "colliding IDs receive a deterministic split-and-payload-hash suffix."
        ),
        "datasets": {},
    }
    for dataset, rows in candidates.items():
        id_counts = Counter(str(row["trajectory_id"]) for row in rows)
        colliding = {key for key, value in id_counts.items() if value > 1}
        details = []
        for row in rows:
            source_id = str(row["trajectory_id"])
            row["source_trajectory_id"] = source_id
            if source_id in colliding:
                suffix = sha256_json(row["trajectory_payload"])[:12]
                row["trajectory_id"] = (
                    f"{source_id}::split={row['split']}::payload={suffix}"
                )
                details.append(
                    {
                        "source_trajectory_id": source_id,
                        "resolved_trajectory_id": row["trajectory_id"],
                        "split": row["split"],
                        "payload_sha256": sha256_json(row["trajectory_payload"]),
                    }
                )
        resolved_ids = [str(row["trajectory_id"]) for row in rows]
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError(f"Unresolved source trajectory ID collision in {dataset}")
        collision_audit["datasets"][dataset] = {
            "eligible_rows": len(rows),
            "source_unique_ids": len(id_counts),
            "colliding_source_id_count": len(colliding),
            "resolved_unique_ids": len(set(resolved_ids)),
            "rows_dropped": 0,
            "details": sorted(details, key=lambda item: item["resolved_trajectory_id"]),
        }
    collision_audit["passed"] = all(
        item["eligible_rows"] == item["resolved_unique_ids"]
        and item["rows_dropped"] == 0
        for item in collision_audit["datasets"].values()
    )
    write_json(resolve(config, "frozen") / "source_id_collision_audit.json", collision_audit)
    return candidates


def exclusion_id(record: dict[str, Any]) -> str:
    return str(record.get("source_trajectory_id") or record["trajectory_id"])


def load_pilot_exclusions(config: dict[str, Any]) -> dict[str, list[str]]:
    manifest = read_json(
        resolve(config, "pilot_prepared") / "pilot_sampling_manifest.json"
    )
    exclusions = {
        dataset: sorted(
            row["trajectory_id"]
            for row in manifest["sampled_rows"]
            if row["dataset"] == dataset
        )
        for dataset in DATASETS
    }
    if {dataset: len(ids) for dataset, ids in exclusions.items()} != {
        "aegis": 56,
        "whowhen": 56,
    }:
        raise ValueError("Pilot exclusion list must contain 56 IDs per dataset")
    freeze_json(
        resolve(config, "frozen") / "pilot_id_exclusions.json",
        {
            "source_pilot_run": str(resolve(config, "pilot_run")),
            "total_excluded": 112,
            "by_dataset": exclusions,
        },
    )
    return exclusions


def deterministic_sample(
    rows: list[dict[str, Any]], *, count: int, seed: int, namespace: str
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["trajectory_id"])
    derived = int.from_bytes(
        hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()[:8], "big"
    )
    random.Random(derived).shuffle(ordered)
    if len(ordered) < count:
        raise ValueError(f"Insufficient pool for {namespace}: {len(ordered)} < {count}")
    return ordered[:count]


def proportional_quotas(
    counts: dict[str, int], *, target: int, seed: int, namespace: str
) -> dict[str, int]:
    """Hamilton allocation with deterministic, seed-derived remainder ties."""
    if target < 0:
        raise ValueError("target must be non-negative")
    if any(value < 0 for value in counts.values()):
        raise ValueError("stratum counts must be non-negative")
    total = sum(counts.values())
    if target > total:
        raise ValueError(f"Requested {target} rows from a pool of {total}")
    if not counts:
        if target:
            raise ValueError("Cannot allocate a non-zero target over no strata")
        return {}
    quotas = {key: (target * value) // total for key, value in counts.items()}
    remaining = target - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            -((target * counts[key]) % total),
            sha256_bytes(f"{seed}:{namespace}:{key}".encode("utf-8")),
            key,
        ),
    )
    for key in ranked:
        if remaining == 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if remaining or sum(quotas.values()) != target:
        raise ValueError("Deterministic proportional quota allocation failed")
    if any(quotas[key] > counts[key] for key in counts):
        raise ValueError("A proportional quota exceeded its eligible stratum")
    return quotas


def proportional_stratified_sample(
    rows: list[dict[str, Any]], *, count: int, seed: int, dataset: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve failure-mode distribution, then framework mix within each mode."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row["gold_category"]), []).append(row)
    category_counts = {
        category: len(items) for category, items in sorted(by_category.items())
    }
    category_quotas = proportional_quotas(
        category_counts,
        target=count,
        seed=seed,
        namespace=f"main:{dataset}:failure-mode",
    )
    selected: list[dict[str, Any]] = []
    category_audit: list[dict[str, Any]] = []
    for category in sorted(by_category):
        category_rows = by_category[category]
        by_framework: dict[str, list[dict[str, Any]]] = {}
        for row in category_rows:
            by_framework.setdefault(str(row["framework"]), []).append(row)
        framework_counts = {
            framework: len(items)
            for framework, items in sorted(by_framework.items())
        }
        category_quota = category_quotas[category]
        framework_quotas = proportional_quotas(
            framework_counts,
            target=category_quota,
            seed=seed,
            namespace=f"main:{dataset}:{category}:framework",
        )
        selected_by_framework: dict[str, int] = {}
        for framework in sorted(by_framework):
            quota = framework_quotas[framework]
            chosen = deterministic_sample(
                by_framework[framework],
                count=quota,
                seed=seed,
                namespace=f"main:{dataset}:{category}:{framework}",
            )
            selected.extend(chosen)
            selected_by_framework[framework] = len(chosen)
        category_audit.append(
            {
                "failure_mode": category,
                "eligible_count": category_counts[category],
                "eligible_proportion": category_counts[category] / len(rows),
                "expected_sample_count": count * category_counts[category] / len(rows),
                "allocated_quota": category_quota,
                "selected_count": sum(selected_by_framework.values()),
                "frameworks": [
                    {
                        "framework": framework,
                        "eligible_count": framework_counts[framework],
                        "eligible_within_mode_proportion": (
                            framework_counts[framework] / category_counts[category]
                        ),
                        "expected_within_mode_sample_count": (
                            category_quota
                            * framework_counts[framework]
                            / category_counts[category]
                        ),
                        "allocated_quota": framework_quotas[framework],
                        "selected_count": selected_by_framework[framework],
                    }
                    for framework in sorted(framework_counts)
                ],
            }
        )
    if len(selected) != count or len({row["trajectory_id"] for row in selected}) != count:
        raise ValueError("Stratified sampling did not produce unique target-sized output")
    category_errors = [
        abs(item["allocated_quota"] - item["expected_sample_count"])
        for item in category_audit
    ]
    framework_errors = [
        abs(item["allocated_quota"] - item["expected_within_mode_sample_count"])
        for category in category_audit
        for item in category["frameworks"]
    ]
    audit = {
        "dataset": dataset,
        "method": "deterministic_hierarchical_proportional_stratified_without_replacement",
        "primary_stratum": "gold_failure_mode",
        "secondary_stratum": "framework",
        "quota_allocation": "hamilton_largest_remainder",
        "eligible_count": len(rows),
        "target_count": count,
        "selected_count": len(selected),
        "rare_class_minimum_quota": 0,
        "rare_class_oversampling_applied": False,
        "all_quotas_within_eligible_counts": all(
            item["allocated_quota"] <= item["eligible_count"]
            and all(
                framework["allocated_quota"] <= framework["eligible_count"]
                for framework in item["frameworks"]
            )
            for item in category_audit
        ),
        "all_selected_counts_match_quotas": all(
            item["selected_count"] == item["allocated_quota"]
            and all(
                framework["selected_count"] == framework["allocated_quota"]
                for framework in item["frameworks"]
            )
            for item in category_audit
        ),
        "maximum_failure_mode_quota_error": max(category_errors, default=0.0),
        "maximum_within_mode_framework_quota_error": max(framework_errors, default=0.0),
        "failure_modes": category_audit,
    }
    audit["passed"] = bool(
        audit["selected_count"] == count
        and audit["all_quotas_within_eligible_counts"]
        and audit["all_selected_counts_match_quotas"]
        and audit["maximum_failure_mode_quota_error"] < 1.0
        and audit["maximum_within_mode_framework_quota_error"] < 1.0
        and not audit["rare_class_oversampling_applied"]
    )
    return selected, audit


def select_smoke_records(
    config: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    pilot_exclusions: dict[str, list[str]],
) -> list[dict[str, Any]]:
    selected = []
    count = int(config["experiment"]["smoke_samples_per_dataset"])
    seed = int(config["experiment"]["smoke_seed"])
    for dataset in DATASETS:
        excluded = set(pilot_exclusions[dataset])
        pool = [row for row in candidates[dataset] if exclusion_id(row) not in excluded]
        selected.extend(
            deterministic_sample(pool, count=count, seed=seed, namespace=f"smoke:{dataset}")
        )
    manifest = {
        "purpose": "non_evaluative_output_contract_smoke",
        "seed": seed,
        "selection_uses_gold_or_pilot_results": False,
        "samples_per_dataset": count,
        "rows": [
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "framework": row["framework"],
                "benchmark": row["benchmark"],
                "payload_sha256": sha256_json(serialize_trajectory(row)),
            }
            for row in selected
        ],
    }
    freeze_json(resolve(config, "smoke") / "smoke_sampling_manifest.json", manifest)
    return selected


def leakage_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden = {
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
        "gold_step",
        "gold_agent",
        "injection_metadata",
    }

    def keys(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                found.add(str(key).casefold())
                found.update(keys(item))
        elif isinstance(value, list):
            for item in value:
                found.update(keys(item))
        return found

    rows = []
    for record in records:
        payload = serialize_trajectory(record)
        allowed = (
            {"task_query", "conversation_history", "final_output"}
            if record["dataset"] == "aegis"
            else {"task_query", "trajectory"}
        )
        hits = sorted(keys(payload) & forbidden)
        unexpected = sorted(set(payload) - allowed)
        rows.append(
            {
                "dataset": record["dataset"],
                "trajectory_id": record["trajectory_id"],
                "forbidden_key_hits": hits,
                "unexpected_top_level_keys": unexpected,
                "passed": not hits and not unexpected,
            }
        )
    return {
        "rows_scanned": len(rows),
        "leakage_count": sum(not row["passed"] for row in rows),
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }


def _without_guidance(value: str) -> str:
    return re.sub(
        r"<DIAGNOSTIC_GUIDANCE>\n.*?\n</DIAGNOSTIC_GUIDANCE>",
        "<DIAGNOSTIC_GUIDANCE>\n[INTENDED_DIFFERENCE]\n</DIAGNOSTIC_GUIDANCE>",
        value,
        flags=re.DOTALL,
    )


def mechanical_audit(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    label_equal = True
    prompt_isolated = True
    label_order_equal = True
    correct_mapping = True
    for record in records:
        prompts = {
            condition: build_prompt(record, condition, taxonomies)
            for condition in CONDITIONS
        }
        reference = prompts["label_only"]
        label_equal &= all(
            item["label_block"].encode("utf-8")
            == reference["label_block"].encode("utf-8")
            for item in prompts.values()
        )
        label_order_equal &= all(
            item["permitted_ids"] == reference["permitted_ids"]
            for item in prompts.values()
        )
        prompt_isolated &= all(
            item["system_prompt"] == reference["system_prompt"]
            and item["trajectory"] == reference["trajectory"]
            and _without_guidance(item["user_prompt"])
            == _without_guidance(reference["user_prompt"])
            for item in prompts.values()
        )
        official = {
            str(item["id"]): str(item["name"])
            for item in taxonomies[record["dataset"]]["categories"]
        }
        rendered = dict(
            line[2:].split(" -> ", 1)
            for line in reference["label_block"].splitlines()
        )
        correct_mapping &= rendered == official
    enum_exact = all(
        response_schema(native_ids(dataset, taxonomies))["json_schema"]["schema"]
        ["properties"]["failure_category"]["enum"]
        == native_ids(dataset, taxonomies)
        for dataset in DATASETS
    )
    invalid_probe = parse_attribution(
        json.dumps(
            {
                "failure_category": "NOT_A_NATIVE_ID",
                "explanation": "probe",
                "evidence": "probe",
            }
        ),
        list(AEGIS_IDS),
    )
    add("output_label_byte_equality", label_equal, "Byte-identical within each dataset across all conditions.")
    add("official_id_name_mapping", correct_mapping, "Rendered ID/name pairs exactly match frozen official taxonomies.")
    add("prompt_isolation", prompt_isolated, "Only DIAGNOSTIC_GUIDANCE differs substantively.")
    add("label_order_stability", label_order_equal, "Native label ordering never changes by condition.")
    add("enum_schema_exact", enum_exact, "failure_category enum equals each target-native ID set exactly.")
    add("no_parser_fallback", invalid_probe["exact_native_id_valid"] is False and invalid_probe["failure_category"] == "NOT_A_NATIVE_ID", "Invalid IDs are retained as invalid; no first-ID/default substitution exists.")
    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "accuracy_or_predictions_used": False,
        "checks": checks,
    }
    write_json(resolve(config, "frozen") / "category_collapse_mechanical_audit.json", report)
    return report


class AuditedOpenRouter:
    def __init__(self, config: dict[str, Any]):
        api_key = next(
            (
                os.getenv(name)
                for name in config["api"]["api_key_envs"]
                if os.getenv(name)
            ),
            None,
        )
        if not api_key:
            raise RuntimeError(
                "Set one of: " + ", ".join(config["api"]["api_key_envs"])
            )
        headers = {"X-OpenRouter-Title": config["api"]["app_title"]}
        referer_name = config["api"].get("http_referer_env")
        if referer_name and os.getenv(referer_name):
            headers["HTTP-Referer"] = os.environ[referer_name]
        self.client = OpenAI(
            api_key=api_key,
            base_url=config["api"]["base_url"],
            timeout=float(config["api"]["timeout_seconds"]),
            max_retries=0,
            default_headers=headers,
        )
        self.config = config
        self.cache_dir = resolve(config, "runtime") / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = resolve(config, "runtime") / "api_call_ledger.jsonl"
        model = config["model"]
        self.counter = TokenCounter.for_model(
            model["name"], model.get("tokenizer_encoding")
        )

    def _spend(self) -> float:
        if not self.ledger.exists():
            return 0.0
        return sum(
            float(row.get("cost_usd") or 0.0)
            for row in iter_jsonl(self.ledger)
            if row.get("actual_request")
        )

    def _reserve(self, system_prompt: str, user_prompt: str, max_tokens: int) -> None:
        model = self.config["model"]
        prices = model["pricing_usd_per_million"]
        input_tokens = self.counter.count(system_prompt + "\n" + user_prompt)
        guarded = int(
            input_tokens
            * float(self.config["budget"]["input_token_safety_multiplier"])
            + 0.999999
        )
        reserve = (
            guarded * float(prices["input"])
            + max_tokens * float(prices["output"])
        ) / 1_000_000
        if self._spend() + reserve > float(self.config["budget"]["max_inference_usd"]):
            raise RuntimeBudgetExceeded("Main experiment inference cap would be exceeded")

    def _append_event(self, context: dict[str, Any], **event: Any) -> dict[str, Any]:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **context,
            **event,
        }
        append_jsonl(self.ledger, row)
        return row

    def call(
        self,
        *,
        context: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        permitted_ids: list[str],
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model = self.config["model"]
        schema = response_schema(permitted_ids)
        fingerprint = sha256_json(
            {
                "model": model,
                "provider": self.config["api"]["provider"],
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": schema,
                "seed": int(self.config["experiment"]["main_seed"]),
                "max_output_tokens": max_output_tokens,
                "workflow_attempt_type": context["workflow_attempt_type"],
                "workflow_attempt_number": context["workflow_attempt_number"],
            }
        )
        cache_path = self.cache_dir / f"{fingerprint}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            parsed = parse_attribution(cached["raw_output_text"], permitted_ids)
            event = self._append_event(
                context,
                accounting_type=context["workflow_attempt_type"],
                sdk_attempt_number=0,
                reason="cache_replay",
                actual_request=False,
                billed=False,
                cost_usd=0.0,
                cache_hit=True,
                cache_key=fingerprint,
                outcome="cache_hit",
            )
            return parsed, {"cache_key": fingerprint, "cache_hit": True, "event": event, "raw_output_text": cached["raw_output_text"], "response_metadata": cached["response_metadata"]}

        retry_kind = context["workflow_attempt_type"]
        retry_reason = "initial_workflow_request"
        maximum = int(self.config["api"]["sdk_attempts"])
        last_error: Exception | None = None
        for sdk_attempt in range(1, maximum + 1):
            self._reserve(system_prompt, user_prompt, max_output_tokens)
            try:
                kwargs: dict[str, Any] = {
                    "model": model["name"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": model["temperature"],
                    "seed": int(self.config["experiment"]["main_seed"]),
                    "max_tokens": max_output_tokens,
                    "response_format": schema,
                    "extra_body": {
                        "provider": {
                            **self.config["api"]["provider"],
                            **model.get("provider", {}),
                        }
                    },
                }
                response = self.client.chat.completions.create(**kwargs)
                raw = response.model_dump(mode="json")
                usage = raw.get("usage") or {}
                cost = usage.get("cost")
                if cost is None:
                    prices = model["pricing_usd_per_million"]
                    cost = (
                        int(usage.get("prompt_tokens") or 0) * float(prices["input"])
                        + int(usage.get("completion_tokens") or 0) * float(prices["output"])
                    ) / 1_000_000
                choice = response.choices[0] if response.choices else None
                raw_text = choice.message.content if choice else None
                finish_reason = choice.finish_reason if choice else "no_choice"
                base_event = {
                    "accounting_type": retry_kind,
                    "sdk_attempt_number": sdk_attempt,
                    "reason": retry_reason,
                    "actual_request": True,
                    "billed": True,
                    "cost_usd": float(cost),
                    "cache_hit": False,
                    "cache_key": fingerprint,
                    "response_id": raw.get("id"),
                    "provider": raw.get("provider"),
                    "finish_reason": finish_reason,
                    "reported_usage": usage,
                }
                if finish_reason == "length":
                    event = self._append_event(context, **base_event, outcome="truncated")
                    raise ResponseTruncatedError(json.dumps(event))
                try:
                    if not isinstance(raw_text, str) or not raw_text.strip():
                        raise ResponseParseError("empty_response_text")
                    parsed = parse_attribution(raw_text, permitted_ids)
                except ResponseParseError as exc:
                    self._append_event(
                        context,
                        **base_event,
                        outcome="schema_parser_error",
                        error=str(exc),
                    )
                    last_error = exc
                    retry_kind = "schema_parser_retry"
                    retry_reason = str(exc)
                    if sdk_attempt < maximum:
                        continue
                    raise
                metadata = {
                    "id": raw.get("id"),
                    "model": raw.get("model"),
                    "provider": raw.get("provider"),
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "cost_usd": float(cost),
                }
                event = self._append_event(
                    context,
                    **base_event,
                    outcome=("valid_native_id" if parsed["exact_native_id_valid"] else "invalid_native_id"),
                )
                write_json(
                    cache_path,
                    {
                        "cache_key": fingerprint,
                        "raw_output_text": raw_text,
                        "parsed_response": parsed,
                        "response_metadata": metadata,
                    },
                )
                return parsed, {"cache_key": fingerprint, "cache_hit": False, "event": event, "raw_output_text": raw_text, "response_metadata": metadata}
            except ResponseTruncatedError:
                raise
            except ResponseParseError:
                raise
            except Exception as exc:
                last_error = exc
                self._append_event(
                    context,
                    accounting_type=retry_kind,
                    sdk_attempt_number=sdk_attempt,
                    reason=retry_reason,
                    actual_request=True,
                    billed=False,
                    cost_usd=0.0,
                    cache_hit=False,
                    cache_key=fingerprint,
                    outcome="provider_transport_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                retry_kind = "provider_transport_retry"
                retry_reason = f"{type(exc).__name__}: {exc}"
                if sdk_attempt < maximum:
                    time.sleep(min(2 ** (sdk_attempt - 1), 4))
                    continue
                raise RuntimeError(f"OpenRouter failed after {maximum} attempts: {last_error}") from exc
        raise RuntimeError(f"OpenRouter failed: {last_error}")


def planned_keys(records: list[dict[str, Any]], seed: int) -> list[tuple[str, str, str]]:
    keys = []
    for record in sorted(records, key=lambda item: (item["dataset"], item["trajectory_id"])):
        conditions = list(CONDITIONS)
        derived = int.from_bytes(
            hashlib.sha256(
                f"{seed}:{record['dataset']}:{record['trajectory_id']}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        random.Random(derived).shuffle(conditions)
        keys.extend((record["dataset"], record["trajectory_id"], item) for item in conditions)
    return keys


def prediction_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = {}
    if path.exists():
        for row in iter_jsonl(path):
            rows[(row["dataset"], row["trajectory_id"], row["condition"])] = row
    return rows


def execute_prediction(
    *,
    config: dict[str, Any],
    client: Any,
    record: dict[str, Any],
    condition: str,
    taxonomies: dict[str, dict[str, Any]],
    purpose: str,
) -> dict[str, Any]:
    built = build_prompt(record, condition, taxonomies)
    formatting_retry = 0
    truncation_retry = 0
    attempts = []
    parsed: dict[str, Any] | None = None
    raw_text: str | None = None
    terminal_reason: str | None = None
    next_attempt_type = "base_attribution"
    while True:
        if next_attempt_type == "formatting_retry":
            attempt_type = "formatting_retry"
            suffix = "\n\n" + prompt_text("formatting_retry.txt")
            workflow_number = formatting_retry
        elif next_attempt_type == "truncation_retry":
            attempt_type = "truncation_retry"
            suffix = ""
            if formatting_retry:
                suffix += "\n\n" + prompt_text("formatting_retry.txt")
            suffix += "\n\n" + prompt_text("truncation_retry.txt")
            workflow_number = truncation_retry
        else:
            attempt_type = "base_attribution"
            suffix = ""
            workflow_number = 1
        context = {
            "purpose": purpose,
            "dataset": record["dataset"],
            "trajectory_id": record["trajectory_id"],
            "condition": condition,
            "workflow_attempt_type": attempt_type,
            "workflow_attempt_number": workflow_number,
        }
        try:
            parsed, call = client.call(
                context=context,
                system_prompt=built["system_prompt"],
                user_prompt=built["user_prompt"] + suffix,
                permitted_ids=built["permitted_ids"],
                max_output_tokens=(
                    int(config["retry_policy"]["truncation_retry_max_output_tokens"])
                    if truncation_retry
                    else int(config["model"]["max_output_tokens"])
                ),
            )
            raw_text = call.get("raw_output_text")
            attempts.append(
                {
                    "workflow_attempt_type": attempt_type,
                    "workflow_attempt_number": workflow_number,
                    "cache_key": call.get("cache_key"),
                    "cache_hit": call.get("cache_hit"),
                    "exact_native_id_valid": parsed["exact_native_id_valid"],
                }
            )
        except ResponseTruncatedError as exc:
            attempts.append({"workflow_attempt_type": attempt_type, "workflow_attempt_number": workflow_number, "outcome": "truncated", "error": str(exc)})
            if truncation_retry >= int(config["retry_policy"]["truncation_retries"]):
                terminal_reason = "terminal_truncation"
                break
            truncation_retry += 1
            next_attempt_type = "truncation_retry"
            continue
        except Exception as exc:
            attempts.append({"workflow_attempt_type": attempt_type, "workflow_attempt_number": workflow_number, "outcome": "terminal_parser_or_transport_error", "error": f"{type(exc).__name__}: {exc}"})
            terminal_reason = "terminal_parser_or_transport_error"
            break

        if parsed["exact_native_id_valid"]:
            break
        if formatting_retry >= int(config["retry_policy"]["formatting_only_retries"]):
            terminal_reason = "terminal_invalid_native_id"
            break
        formatting_retry += 1
        next_attempt_type = "formatting_retry"

    final_valid = bool(parsed and parsed["exact_native_id_valid"])
    return {
        "dataset": record["dataset"],
        "trajectory_id": record["trajectory_id"],
        "framework": record["framework"],
        "benchmark": record["benchmark"],
        "condition": condition,
        "purpose": purpose,
        "failure_category": parsed.get("failure_category") if parsed else None,
        "explanation": parsed.get("explanation") if parsed else None,
        "evidence": parsed.get("evidence") if parsed else None,
        "raw_model_response": raw_text,
        "initial_compliant": bool(attempts and attempts[0].get("exact_native_id_valid")),
        "final_compliant": final_valid,
        "formatting_retry_count": formatting_retry,
        "truncation_retry_count": truncation_retry,
        "terminal_invalid": not final_valid,
        "terminal_reason": terminal_reason,
        "correct": 0 if not final_valid else None,
        "attempts": attempts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }


def synthetic_retry_audit(
    config: dict[str, Any],
    record: dict[str, Any],
    taxonomies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    valid_id = native_ids(record["dataset"], taxonomies)[0]

    def result(label: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return {
            "failure_category": label,
            "explanation": "synthetic contract probe",
            "evidence": "synthetic contract probe",
            "exact_native_id_valid": label
            in native_ids(record["dataset"], taxonomies),
        }, {
            "cache_key": "synthetic",
            "cache_hit": False,
            "raw_output_text": "{}",
        }

    class SequenceClient:
        def __init__(self, values: list[Any]):
            self.values = values
            self.calls: list[str] = []

        def call(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            self.calls.append(kwargs["context"]["workflow_attempt_type"])
            value = self.values.pop(0)
            if isinstance(value, Exception):
                raise value
            return result(value)

    two_retry_client = SequenceClient(["INVALID", "INVALID", valid_id])
    two_retry = execute_prediction(
        config=config,
        client=two_retry_client,
        record=record,
        condition="label_only",
        taxonomies=taxonomies,
        purpose="offline_contract_probe",
    )
    terminal_client = SequenceClient(["INVALID", "INVALID", "INVALID"])
    terminal = execute_prediction(
        config=config,
        client=terminal_client,
        record=record,
        condition="foreign_taxonomy",
        taxonomies=taxonomies,
        purpose="offline_contract_probe",
    )
    truncation_client = SequenceClient(
        [ResponseTruncatedError("synthetic"), valid_id]
    )
    truncation = execute_prediction(
        config=config,
        client=truncation_client,
        record=record,
        condition="merged_taxonomy",
        taxonomies=taxonomies,
        purpose="offline_contract_probe",
    )
    checks = {
        "two_formatting_retries_then_valid": (
            two_retry_client.calls
            == ["base_attribution", "formatting_retry", "formatting_retry"]
            and two_retry["formatting_retry_count"] == 2
            and two_retry["final_compliant"]
        ),
        "terminal_invalid_retained_and_scored_zero": (
            terminal["terminal_invalid"]
            and terminal["correct"] == 0
            and terminal["status"] == "complete"
            and terminal["formatting_retry_count"] == 2
        ),
        "truncation_retry_once": (
            truncation_client.calls == ["base_attribution", "truncation_retry"]
            and truncation["truncation_retry_count"] == 1
            and truncation["final_compliant"]
        ),
    }
    audit = {
        "status": "passed" if all(checks.values()) else "failed",
        "uses_api": False,
        "uses_gold": False,
        "checks": checks,
    }
    write_json(resolve(config, "frozen") / "synthetic_retry_audit.json", audit)
    return audit


def run_smoke(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    taxonomies: dict[str, dict[str, Any]],
    implementation_run_id: str,
) -> dict[str, Any]:
    smoke_dir = resolve(config, "smoke")
    path = smoke_dir / "smoke_predictions.jsonl"
    existing = prediction_index(path)
    record_map = {(row["dataset"], row["trajectory_id"]): row for row in records}
    client = AuditedOpenRouter(config)
    keys = planned_keys(records, int(config["experiment"]["smoke_seed"]))
    for index, (dataset, trajectory_id, condition) in enumerate(keys, start=1):
        key = (dataset, trajectory_id, condition)
        if (
            existing.get(key, {}).get("status") == "complete"
            and existing[key].get("terminal_reason")
            != "terminal_parser_or_transport_error"
        ):
            continue
        row = execute_prediction(
            config=config,
            client=client,
            record=record_map[(dataset, trajectory_id)],
            condition=condition,
            taxonomies=taxonomies,
            purpose="non_evaluative_smoke",
        )
        append_jsonl(path, row)
        existing[key] = row
        print(f"[main-contract-smoke] completed={index}/{len(keys)} dataset={dataset} condition={condition}", flush=True)
    latest = prediction_index(path)
    if set(latest) != set(keys):
        raise ValueError("Smoke rows are incomplete")
    rows = [latest[key] for key in keys]
    events_path = resolve(config, "runtime") / "api_call_ledger.jsonl"
    all_events = list(iter_jsonl(events_path)) if events_path.exists() else []
    key_set = set(keys)
    events = [
        event
        for event in all_events
        if event.get("purpose") == "non_evaluative_smoke"
        and (
            event.get("dataset"),
            event.get("trajectory_id"),
            event.get("condition"),
        )
        in key_set
    ]
    actual_events = [event for event in events if event.get("actual_request")]
    billed_events = [event for event in actual_events if event.get("billed")]
    parsed_events = [
        event
        for event in actual_events
        if event.get("outcome") in {"valid_native_id", "invalid_native_id"}
    ]
    report = {
        "purpose": "non_evaluative_output_contract_smoke",
        "accuracy_computed": False,
        "implementation_run_id": implementation_run_id,
        "logical_prediction_rows": len(keys),
        "attempted_rows": sum(key in latest for key in keys),
        "prediction_rows": len(rows),
        "responses_actually_received": len(billed_events),
        "api_network_failures": sum(
            bool(event.get("actual_request")) and not bool(event.get("billed"))
            for event in events
        ),
        "provider_transport_failures": sum(
            event.get("outcome") == "provider_transport_error" for event in events
        ),
        "parsed_responses": len(parsed_events),
        "exact_id_compliant_responses": sum(
            event.get("outcome") == "valid_native_id" for event in actual_events
        ),
        "initial_compliant": sum(row["initial_compliant"] for row in rows),
        "formatting_retries": sum(row["formatting_retry_count"] for row in rows),
        "truncation_retries": sum(row["truncation_retry_count"] for row in rows),
        "final_compliant": sum(row["final_compliant"] for row in rows),
        "terminal_invalid": sum(row["terminal_invalid"] for row in rows),
        "billed_responses": len(billed_events),
        "actual_network_calls": len(actual_events),
        "total_cost_usd": sum(float(event.get("cost_usd") or 0.0) for event in actual_events),
        "all_rows_preserved": len(rows) == len(keys),
        "source_of_truth": [
            str(path),
            str(events_path),
        ],
    }
    write_json(smoke_dir / "smoke_summary.json", report)
    return report


def ledger_reconciliation(
    config: dict[str, Any], implementation_run_id: str | None = None
) -> dict[str, Any]:
    path = resolve(config, "runtime") / "api_call_ledger.jsonl"
    events = list(iter_jsonl(path)) if path.exists() else []
    actual = [row for row in events if row["actual_request"]]
    types = (
        "base_attribution",
        "formatting_retry",
        "truncation_retry",
        "schema_parser_retry",
        "provider_transport_retry",
        "merge_adjudication",
    )
    counts = {item: sum(row["accounting_type"] == item for row in actual) for item in types}
    equation_sum = sum(counts.values())
    report = {
        "implementation_run_id": implementation_run_id,
        "logical_prediction_rows": (
            int(config["experiment"]["smoke_samples_per_dataset"])
            * len(DATASETS)
            * len(CONDITIONS)
        ),
        "equation": (
            "base attribution + formatting retries + truncation retries + "
            "schema/parser retries + provider/transport retries + "
            "merge/adjudication calls = total actual API calls"
        ),
        "counts": counts,
        "total_actual_api_calls": len(actual),
        "equation_sum": equation_sum,
        "reconciled": equation_sum == len(actual),
        "billed_response_count": sum(bool(row["billed"]) for row in actual),
        "successful_provider_responses": sum(bool(row["billed"]) for row in actual),
        "non_billed_network_attempts": sum(not bool(row["billed"]) for row in actual),
        "failed_network_attempts": sum(not bool(row["billed"]) for row in actual),
        "cache_events": sum(not row["actual_request"] for row in events),
        "total_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in actual),
        "required_context_fields_present": all(
            all(
                key in row
                for key in (
                    "dataset",
                    "trajectory_id",
                    "condition",
                    "workflow_attempt_type",
                    "workflow_attempt_number",
                    "reason",
                    "actual_request",
                    "billed",
                )
            )
            for row in events
        ),
    }
    write_json(resolve(config, "smoke") / "api_ledger_reconciliation.json", report)
    return report


def implementation_manifest(
    config: dict[str, Any], taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    code_files = [
        HERE / "core.py",
        HERE / "run.py",
        config["_root"]
        / "experiments"
        / "aegis_whowhen_pilot_fixed_guidance"
        / "core.py",
        config["_root"] / "src" / "taxonomy_experiment" / "config.py",
        config["_root"] / "src" / "taxonomy_experiment" / "io.py",
        config["_root"] / "src" / "taxonomy_experiment" / "token_count.py",
    ]
    prompt_files = sorted((HERE / "prompts").glob("*.txt"))
    schemas = {
        dataset: response_schema(native_ids(dataset, taxonomies))
        for dataset in DATASETS
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config["_root"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unavailable-no-git-metadata"
    model_config = {
        "experiment": config["experiment"],
        "sampling": config["sampling"],
        "model": config["model"],
        "api": {
            "base_url": config["api"]["base_url"],
            "sdk_attempts": config["api"]["sdk_attempts"],
        },
        "api_provider": config["api"]["provider"],
        "retry_policy": config["retry_policy"],
        "budget": config["budget"],
    }
    content = {
        "code_commit": commit,
        "code_files": {
            str(path.relative_to(config["_root"])): sha256_file(path)
            for path in code_files
        },
        "code_hash": sha256_json(
            {
                str(path.relative_to(config["_root"])): sha256_file(path)
                for path in code_files
            }
        ),
        "parser_version": "strict-shape-tolerant-max-length-no-fallback-v1",
        "parser_hash": sha256_bytes(inspect.getsource(parse_attribution).encode("utf-8")),
        "serializer_version": "dataset-allowlist-v1",
        "serializer_hash": sha256_bytes(inspect.getsource(serialize_trajectory).encode("utf-8")),
        "prompt_hashes": {path.name: sha256_file(path) for path in prompt_files},
        "schema_hashes": {dataset: sha256_json(schema) for dataset, schema in schemas.items()},
        "schemas": schemas,
        "output_label_set_hashes": {
            dataset: sha256_json(native_ids(dataset, taxonomies)) for dataset in DATASETS
        },
        "taxonomy_hashes": {
            dataset: sha256_json(taxonomies[dataset])
            for dataset in ("aegis", "whowhen", "merged")
        },
        "own_foreign_mapping": {"aegis_foreign": "whowhen", "whowhen_foreign": "aegis"},
        "model_config_hash": sha256_json(model_config),
        "experiment_config_file_hash": sha256_file(config["_config_path"]),
        "model_config": model_config,
        "scoring": "exact_native_id_only_no_relabeler",
    }
    run_id = "main-" + sha256_json(content)[:16]
    manifest = {"implementation_run_id": run_id, **content}
    path = resolve(config, "frozen") / "implementation_manifest.json"
    ready_path = resolve(config, "frozen") / "READY_FOR_MAIN_EXPERIMENT.json"
    if ready_path.exists():
        freeze_json(path, manifest)
    else:
        # Preflight development may still uncover an implementation bug. The
        # manifest becomes immutable as soon as the READY gate is written.
        write_json(path, manifest)
    return manifest


def freeze_main_sample(
    config: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    pilot_exclusions: dict[str, list[str]],
    smoke_records: list[dict[str, Any]],
    implementation_run_id: str,
) -> dict[str, Any]:
    count = int(config["experiment"]["main_samples_per_dataset"])
    seed = int(config["experiment"]["main_seed"])
    smoke_ids = {
        dataset: {
            row["trajectory_id"] for row in smoke_records if row["dataset"] == dataset
        }
        for dataset in DATASETS
    }
    selected = []
    sampling_audits: dict[str, dict[str, Any]] = {}
    eligible_counts: dict[str, int] = {}
    for dataset in DATASETS:
        pilot_ids = set(pilot_exclusions[dataset])
        pool = [
            row
            for row in candidates[dataset]
            if exclusion_id(row) not in pilot_ids
            and row["trajectory_id"] not in smoke_ids[dataset]
        ]
        eligible_counts[dataset] = len(pool)
        picked, sampling_audit = proportional_stratified_sample(
            pool,
            count=count,
            seed=seed,
            dataset=dataset,
        )
        sampling_audits[dataset] = sampling_audit
        selected.extend(picked)
    inputs = []
    gold = []
    for row in selected:
        payload = serialize_trajectory(row)
        inputs.append(
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "source_trajectory_id": exclusion_id(row),
                "framework": row["framework"],
                "benchmark": row["benchmark"],
                "trajectory_payload": payload,
                "trajectory_payload_sha256": sha256_json(payload),
            }
        )
        gold.append(
            {
                "dataset": row["dataset"],
                "trajectory_id": row["trajectory_id"],
                "source_trajectory_id": exclusion_id(row),
                "gold_category": row["gold_category"],
            }
        )
    inputs.sort(key=lambda row: (row["dataset"], row["trajectory_id"]))
    gold.sort(key=lambda row: (row["dataset"], row["trajectory_id"]))
    prepared_dir = resolve(config, "frozen") / "main_sample"
    if (prepared_dir / "main_inputs.jsonl").exists():
        if list(iter_jsonl(prepared_dir / "main_inputs.jsonl")) != inputs:
            raise FileExistsError("Frozen main inputs differ")
    else:
        write_jsonl(prepared_dir / "main_inputs.jsonl", inputs)
    if (prepared_dir / "main_gold.jsonl").exists():
        if list(iter_jsonl(prepared_dir / "main_gold.jsonl")) != gold:
            raise FileExistsError("Frozen main gold differs")
    else:
        write_jsonl(prepared_dir / "main_gold.jsonl", gold)
    manifest = {
        "implementation_run_id": implementation_run_id,
        "seed": seed,
        "sampling": "deterministic_hierarchical_proportional_stratified_without_replacement",
        "primary_stratum": "gold_failure_mode",
        "secondary_stratum": "framework",
        "quota_allocation": "hamilton_largest_remainder",
        "rare_class_minimum_quota": 0,
        "rare_class_oversampling_applied": False,
        "sampling_design_deviation": False,
        "uses_pilot_results": False,
        "samples_per_dataset": count,
        "eligible_after_exclusions": eligible_counts,
        "dataset_sample_counts": dict(
            sorted(Counter(row["dataset"] for row in selected).items())
        ),
        "total_trajectories": len(inputs),
        "planned_prediction_rows": len(inputs) * len(CONDITIONS),
        "pilot_exclusion_count": sum(len(ids) for ids in pilot_exclusions.values()),
        "smoke_exclusion_count": sum(len(ids) for ids in smoke_ids.values()),
        "pilot_overlap": sum(
            exclusion_id(row) in set(pilot_exclusions[row["dataset"]])
            for row in inputs
        ),
        "smoke_overlap": sum(
            row["trajectory_id"] in smoke_ids[row["dataset"]] for row in inputs
        ),
        "input_hash": sha256_json(inputs),
        "gold_hash": sha256_json(gold),
        "stratification_audits": sampling_audits,
        "stratification_passed": all(
            audit["passed"] for audit in sampling_audits.values()
        ),
    }
    main_leakage = leakage_audit(selected)
    collision_audit = read_json(
        resolve(config, "frozen") / "source_id_collision_audit.json"
    )
    manifest["source_id_collision_audit_passed"] = collision_audit["passed"]
    manifest["source_id_collisions_resolved"] = sum(
        item["colliding_source_id_count"]
        for item in collision_audit["datasets"].values()
    )
    manifest["source_id_collision_rows_dropped"] = sum(
        item["rows_dropped"] for item in collision_audit["datasets"].values()
    )
    manifest["leakage_rows_scanned"] = main_leakage["rows_scanned"]
    manifest["leakage_count"] = main_leakage["leakage_count"]
    manifest["leakage_passed"] = main_leakage["passed"]
    write_json(prepared_dir / "main_leakage_audit.json", main_leakage)
    freeze_json(prepared_dir / "main_sampling_manifest.json", manifest)
    return manifest


def readiness_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    leakage: dict[str, Any],
    mechanical: dict[str, Any],
    smoke: dict[str, Any],
    ledger: dict[str, Any],
    main_sample: dict[str, Any],
    retry_audit: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "leakage_audit_passed": leakage["passed"] and main_sample["leakage_passed"],
        "taxonomy_id_name_mapping_passed": next(item for item in mechanical["checks"] if item["name"] == "official_id_name_mapping")["passed"],
        "output_label_equality_passed": next(item for item in mechanical["checks"] if item["name"] == "output_label_byte_equality")["passed"],
        "prompt_isolation_passed": next(item for item in mechanical["checks"] if item["name"] == "prompt_isolation")["passed"],
        "enum_schema_validation_passed": next(item for item in mechanical["checks"] if item["name"] == "enum_schema_exact")["passed"],
        "retry_handling_passed": retry_audit["status"] == "passed",
        "terminal_invalid_handling_defined": retry_audit["checks"]["terminal_invalid_retained_and_scored_zero"],
        "parser_frozen": bool(manifest["parser_hash"]),
        "schema_frozen": bool(manifest["schema_hashes"]),
        "serializer_frozen": bool(manifest["serializer_hash"]),
        "truncation_policy_frozen": int(config["retry_policy"]["truncation_retries"]) == 1,
        "api_ledger_reconciled": ledger["reconciled"] and ledger["required_context_fields_present"],
        "code_config_hashes_recorded": bool(manifest["implementation_run_id"]),
        "pilot_ids_excluded": main_sample["pilot_overlap"] == 0 and main_sample["pilot_exclusion_count"] == 112,
        "smoke_ids_excluded": main_sample["smoke_overlap"] == 0,
        "proportional_stratified_sampling_restored": (
            main_sample["sampling"]
            == "deterministic_hierarchical_proportional_stratified_without_replacement"
            and main_sample["stratification_passed"]
            and not main_sample["rare_class_oversampling_applied"]
            and not main_sample["sampling_design_deviation"]
        ),
        "main_dataset_counts_exact": main_sample["dataset_sample_counts"]
        == {"aegis": 1000, "whowhen": 1000},
        "source_id_collisions_resolved_without_drops": (
            main_sample["source_id_collision_audit_passed"]
            and main_sample["source_id_collision_rows_dropped"] == 0
        ),
        "smoke_rows_complete": (
            smoke["attempted_rows"] == smoke["logical_prediction_rows"]
            and smoke["final_compliant"] == smoke["logical_prediction_rows"]
            and smoke["terminal_invalid"] == 0
        ),
        "live_enum_parser_smoke_passed": (
            smoke["responses_actually_received"] >= smoke["logical_prediction_rows"]
            and smoke["parsed_responses"] >= smoke["logical_prediction_rows"]
            and smoke["exact_id_compliant_responses"]
            >= smoke["logical_prediction_rows"]
            and smoke["final_compliant"] == smoke["logical_prediction_rows"]
            and smoke["terminal_invalid"] == 0
            and ledger["billed_response_count"] >= smoke["logical_prediction_rows"]
        ),
    }
    status = "READY_FOR_MAIN_EXPERIMENT" if all(gates.values()) else "IMPLEMENTATION_FIX_REQUIRED"
    result = {
        "status": status,
        "accuracy_used_for_readiness": False,
        "implementation_run_id": manifest["implementation_run_id"],
        "gates": gates,
        "smoke_summary": smoke,
        "ledger_reconciliation": ledger,
        "main_sample_manifest": main_sample,
    }
    write_json(resolve(config, "frozen") / "READY_FOR_MAIN_EXPERIMENT.json", result)
    lines = [
        "# Main Experiment Readiness Report",
        "",
        f"Status: `{status}`",
        "",
        "Accuracy was not computed for the smoke test and was not used for readiness.",
        "",
        "## Implementation fixes",
        "",
        "- `failure_category` is constrained by a dataset-specific exact native-ID enum in the API JSON Schema.",
        "- Local exact-ID validation remains active; no fallback, normalization, relabeling, or semantic judging exists.",
        "- At most two formatting-only retries use a gold-blind contract-correction message.",
        "- Terminal invalid rows are retained with `terminal_invalid=true` and `correct=0`; no drop or resampling.",
        "- Truncation uses one deterministic, condition-invariant retry with a frozen 1,200-token recovery limit.",
        "- Every actual, billed, retry, transport, parser, and cache event is context-labeled in the isolated ledger.",
        "",
        "## Smoke test",
        "",
        f"- Rows: {smoke['prediction_rows']}",
        f"- Attempted rows: {smoke['attempted_rows']}",
        f"- Responses actually received: {smoke['responses_actually_received']}",
        f"- API/network failures: {smoke['api_network_failures']}",
        f"- Provider transport failures: {smoke['provider_transport_failures']}",
        f"- Parsed responses: {smoke['parsed_responses']}",
        f"- Exact-ID compliant responses: {smoke['exact_id_compliant_responses']}",
        f"- Initial compliant: {smoke['initial_compliant']}",
        f"- Formatting retries: {smoke['formatting_retries']}",
        f"- Truncation retries: {smoke['truncation_retries']}",
        f"- Final compliant: {smoke['final_compliant']}",
        f"- Terminal invalid: {smoke['terminal_invalid']}",
        f"- Billed responses: {smoke['billed_responses']}",
        f"- Actual network calls: {smoke['actual_network_calls']}",
        f"- Total cost: ${smoke['total_cost_usd']:.8f}",
        "- Accuracy: not computed",
        "",
        "## Ledger reconciliation",
        "",
        f"- Counts: `{json.dumps(ledger['counts'], sort_keys=True)}`",
        f"- Equation sum: {ledger['equation_sum']}",
        f"- Total actual API calls: {ledger['total_actual_api_calls']}",
        f"- Reconciled: {ledger['reconciled']}",
        f"- Cost: ${ledger['total_cost_usd']:.8f}",
        f"- Offline retry/terminal/truncation audit: {retry_audit['status']}",
        "",
        "## Mechanical category-collapse audit",
        "",
        f"- Checks: {mechanical['checks_passed']}/{mechanical['checks_total']}",
        "- No output-label mismatch, ordering drift, unintended prompt difference, parser fallback, or default-category selection was found.",
        "- Any category concentration is therefore treated as model behavior, not a tuning signal.",
        "",
        "## Freeze and exclusions",
        "",
        f"- Implementation run ID: `{manifest['implementation_run_id']}`",
        "- Parser, schema, serializer, prompts, model configuration, label sets, and all three taxonomies are content-hashed.",
        f"- Pilot exclusions: {main_sample['pilot_exclusion_count']} IDs; overlap with main sample: {main_sample['pilot_overlap']}",
        f"- Smoke overlap with main sample: {main_sample['smoke_overlap']}",
        f"- Sampling: `{main_sample['sampling']}`",
        f"- Sampling seed: {main_sample['seed']}",
        f"- Dataset counts: AEGIS={main_sample['dataset_sample_counts'].get('aegis', 0)}, Who&When={main_sample['dataset_sample_counts'].get('whowhen', 0)}",
        f"- Stratification audit: {'passed' if main_sample['stratification_passed'] else 'failed'}; rare-class oversampling applied={main_sample['rare_class_oversampling_applied']}",
        f"- Source ID collisions resolved: {main_sample['source_id_collisions_resolved']}; rows dropped={main_sample['source_id_collision_rows_dropped']}",
        f"- Main-sample leakage count: {main_sample['leakage_count']}",
        f"- Frozen main sample: {main_sample['total_trajectories']} trajectories / {main_sample['planned_prediction_rows']} planned rows",
        "",
        "## Final status",
        "",
        status,
        "",
    ]
    (resolve(config, "frozen") / "MAIN_READINESS_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return result


def prepare_and_run_gate(
    config_path: str | Path = "experiments/aegis_whowhen_main_fixed_guidance/experiment.yaml",
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    taxonomies = load_and_freeze_taxonomies(config)
    exclusions = load_pilot_exclusions(config)
    candidates = load_candidates(config)
    smoke_records = select_smoke_records(config, candidates, exclusions)
    leak = leakage_audit(smoke_records)
    write_json(resolve(config, "frozen") / "smoke_leakage_audit.json", leak)
    mechanical = mechanical_audit(config, smoke_records, taxonomies)
    retry_audit = synthetic_retry_audit(config, smoke_records[0], taxonomies)
    manifest = implementation_manifest(config, taxonomies)
    if (
        not leak["passed"]
        or mechanical["status"] != "passed"
        or retry_audit["status"] != "passed"
    ):
        raise ValueError("Pre-smoke implementation gate failed")
    if preflight_only:
        return {
            "status": "PREFLIGHT_READY_NO_API_CALLS",
            "implementation_run_id": manifest["implementation_run_id"],
            "smoke_rows_planned": len(smoke_records) * len(CONDITIONS),
            "leakage_count": leak["leakage_count"],
            "mechanical_checks": f"{mechanical['checks_passed']}/{mechanical['checks_total']}",
        }
    implementation_run_id = manifest["implementation_run_id"]
    before = ledger_reconciliation(config, implementation_run_id)["total_actual_api_calls"]
    smoke = run_smoke(config, smoke_records, taxonomies, implementation_run_id)
    after_first = ledger_reconciliation(config, implementation_run_id)["total_actual_api_calls"]
    smoke_again = run_smoke(config, smoke_records, taxonomies, implementation_run_id)
    after_resume = ledger_reconciliation(config, implementation_run_id)["total_actual_api_calls"]
    if smoke_again != smoke or after_resume != after_first:
        raise ValueError("Smoke cache/resume check caused new calls or changed results")
    ledger = ledger_reconciliation(config, implementation_run_id)
    ledger["actual_calls_added_by_this_gate"] = after_first - before
    ledger["resume_added_actual_calls"] = after_resume - after_first
    write_json(resolve(config, "smoke") / "api_ledger_reconciliation.json", ledger)
    main_sample = freeze_main_sample(
        config,
        candidates,
        exclusions,
        smoke_records,
        implementation_run_id,
    )
    readiness = readiness_report(
        config,
        manifest,
        leak,
        mechanical,
        smoke,
        ledger,
        main_sample,
        retry_audit,
    )
    return {
        "status": readiness["status"],
        "implementation_run_id": manifest["implementation_run_id"],
        "report": str(resolve(config, "frozen") / "MAIN_READINESS_REPORT.md"),
        "readiness_json": str(resolve(config, "frozen") / "READY_FOR_MAIN_EXPERIMENT.json"),
        "smoke_summary": str(resolve(config, "smoke") / "smoke_summary.json"),
        "ledger_reconciliation": str(resolve(config, "smoke") / "api_ledger_reconciliation.json"),
        "main_sampling_manifest": str(resolve(config, "frozen") / "main_sample" / "main_sampling_manifest.json"),
    }
