from __future__ import annotations

from pathlib import Path
from typing import Any

from taxonomy_experiment.io import iter_jsonl


CATEGORY_NORMALIZATION = {
    "instruction adherence failure": "Instruction/Plan Adherence Failure",
    "instruction/plan adherence failure": "Instruction/Plan Adherence Failure",
    "invention of new information": "Invention of New Information",
    "invalid invocation": "Invalid Invocation",
    "misinterpretation of tool output": "Misinterpretation of Tool Output / Handoff Failure",
    "misinterpretation of tool output / handoff failure": "Misinterpretation of Tool Output / Handoff Failure",
    "intent plan misalignment": "Intent-Plan Misalignment",
    "intent-plan misalignment": "Intent-Plan Misalignment",
    "underspecified user intent": "Underspecified User Intent",
    "intent not supported": "Intent Not Supported",
    "guardrails triggered": "Guardrails Triggered",
    "system failure": "System Failure",
    "inconclusive (use sparingly)": "Inconclusive (USE SPARINGLY)",
}


def canonicalize_category(value: Any) -> str:
    raw = str(value or "").strip()
    return CATEGORY_NORMALIZATION.get(raw.lower(), raw)


def _normalize_input(raw: dict[str, Any], domain: str, source_file: Path) -> dict[str, Any]:
    return {
        "trajectory_id": str(raw["trajectory_id"]),
        "dataset": "agentrx",
        "domain": domain,
        "instruction": str(raw.get("instruction") or ""),
        "context": {},
        "steps": raw.get("steps") or [],
        "metadata": {"source_file": source_file.as_posix()},
    }


def _normalize_gold(raw: dict[str, Any], normalized_id: str, domain: str) -> dict[str, Any]:
    root_id = str(raw.get("root_cause_failure_id") or (raw.get("root_cause") or {}).get("failure_id") or "")
    root_failure = next(
        (failure for failure in raw.get("failures", []) if str(failure.get("failure_id")) == root_id),
        None,
    )
    if root_failure is None:
        raise ValueError(f"AgentRx annotation {raw.get('trajectory_id')} has no root failure {root_id}")
    category_raw = str(root_failure.get("failure_category") or "").strip()
    return {
        "trajectory_id": normalized_id,
        "dataset": "agentrx",
        "domain": domain,
        "eligible_for_attribution": bool(category_raw),
        "failure_type_raw": category_raw,
        "failure_type": canonicalize_category(category_raw),
        "failure_types_raw": [category_raw],
        "failure_types": [canonicalize_category(category_raw)],
        "failure_module_raw": "",
        "failure_module": "",
        "critical_failure_step": root_failure.get("step_number"),
        "reference_reasoning": str(
            raw.get("root_cause_reason")
            or (raw.get("root_cause") or {}).get("reason_for_root_cause")
            or root_failure.get("category_reason")
            or root_failure.get("step_reason")
            or ""
        ).strip(),
        "failure_summary": str(raw.get("failure_summary") or "").strip(),
        "root_failure": root_failure,
        "raw_label": raw,
    }


def load_agentrx(
    tau_trajectories: Path,
    tau_annotations: Path,
    magentic_trajectories: Path,
    magentic_annotations: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tau_inputs_all = {str(row["trajectory_id"]): row for row in iter_jsonl(tau_trajectories)}
    mag_inputs_all = {str(row["trajectory_id"]): row for row in iter_jsonl(magentic_trajectories)}
    tau_gold_raw = list(iter_jsonl(tau_annotations))
    mag_gold_raw = list(iter_jsonl(magentic_annotations))

    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    matched_tau: set[str] = set()
    matched_mag: set[str] = set()
    unmatched_annotation_ids: list[str] = []

    for annotation in tau_gold_raw:
        normalized_id = f"tau_retail_{annotation['trajectory_id']}"
        trajectory = tau_inputs_all.get(normalized_id)
        if trajectory is None:
            unmatched_annotation_ids.append(f"tau_retail:{annotation['trajectory_id']}")
            continue
        matched_tau.add(normalized_id)
        inputs.append(
            _normalize_input(
                trajectory,
                "tau_retail",
                tau_trajectories.relative_to(project_root),
            )
        )
        gold.append(_normalize_gold(annotation, normalized_id, "tau_retail"))

    for annotation in mag_gold_raw:
        normalized_id = str(annotation["trajectory_id"])
        trajectory = mag_inputs_all.get(normalized_id)
        if trajectory is None:
            unmatched_annotation_ids.append(f"magentic_one:{normalized_id}")
            continue
        matched_mag.add(normalized_id)
        inputs.append(
            _normalize_input(
                trajectory,
                "magentic_one",
                magentic_trajectories.relative_to(project_root),
            )
        )
        gold.append(_normalize_gold(annotation, normalized_id, "magentic_one"))

    inputs.sort(key=lambda item: (item["domain"], item["trajectory_id"]))
    gold.sort(key=lambda item: (item["domain"], item["trajectory_id"]))
    stats = {
        "total_trajectories": len(tau_inputs_all) + len(mag_inputs_all),
        "total_official_annotations": len(tau_gold_raw) + len(mag_gold_raw),
        "matched_pairs": len(inputs),
        "eligible_attribution_pairs": sum(item["eligible_for_attribution"] for item in gold),
        "missing_gold_failure_types": sum(not item["failure_type_raw"] for item in gold),
        "missing_gold_reasoning": sum(not item["reference_reasoning"] for item in gold),
        "domain_trajectory_counts": {
            "tau_retail": len(tau_inputs_all),
            "magentic_one": len(mag_inputs_all),
        },
        "domain_annotation_counts": {
            "tau_retail": len(tau_gold_raw),
            "magentic_one": len(mag_gold_raw),
        },
        "unmatched_annotation_ids": sorted(unmatched_annotation_ids),
        "unmatched_trajectory_ids": sorted(
            [f"tau_retail:{item}" for item in set(tau_inputs_all) - matched_tau]
            + [f"magentic_one:{item}" for item in set(mag_inputs_all) - matched_mag]
        ),
    }
    return inputs, gold, stats
