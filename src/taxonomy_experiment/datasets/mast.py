from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MAST_CATEGORY_NAMES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Reasoning-Action Mismatch",
    "3.1": "Premature Termination",
    "3.2": "No or Incomplete Verification",
    "3.3": "Incorrect Verification",
}

MAST_MODULES = {
    "1": "system_design_issues",
    "2": "inter_agent_misalignment",
    "3": "task_verification",
}

_SCENARIO_MARKER = "SCENARIO.PY STARTING !#!#"
_EVENT_HEADER = re.compile(r"(?m)^---------- (.+?) ----------\r?$")
_ID_COMPONENT = re.compile(r"[^a-z0-9]+")


def canonicalize_mast_category(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.upper().startswith("FM-"):
        raw = raw[3:]
    return MAST_CATEGORY_NAMES.get(raw, raw)


def _trace_identity(row: dict[str, Any]) -> str:
    """Build a stable ID from the release's run key plus its run-local trace ID."""
    trace = row.get("trace") or {}
    run_key = str(trace.get("key") or "unknown_run").casefold()
    run_key = _ID_COMPONENT.sub("_", run_key).strip("_")
    return f"mast_{run_key}_{row['trace_id']}"


def _scenario_trace(raw: str) -> str:
    marker = raw.find(_SCENARIO_MARKER)
    return raw[marker:] if marker >= 0 else raw


def _parse_events(raw: str) -> tuple[str, list[dict[str, Any]]]:
    trace = _scenario_trace(raw)
    matches = list(_EVENT_HEADER.finditer(trace))
    if not matches:
        return "", [{"index": 1, "actor": "trace", "content": trace.strip()}]

    events: list[dict[str, Any]] = []
    instruction = ""
    for index, match in enumerate(matches, start=1):
        end = matches[index].start() if index < len(matches) else len(trace)
        actor = match.group(1).strip()
        content = trace[match.end() : end].strip()
        if actor.casefold() == "user" and not instruction:
            instruction = content
        events.append({"index": index, "actor": actor, "content": content})
    return instruction, events


def load_mast(
    dataset_path: Path,
    project_root: Path,
    *,
    mas_name: str = "Magentic",
    benchmark_name: str = "GAIA",
    only_annotated_failures: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"MAST dataset must be a JSON array: {dataset_path}")

    subset = [
        row
        for row in rows
        if str(row.get("mas_name")) == mas_name
        and str(row.get("benchmark_name")) == benchmark_name
    ]
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    skipped_without_failure = 0
    unknown_codes: set[str] = set()

    for row in subset:
        annotation = row.get("mast_annotation") or {}
        positive_codes = sorted(code for code, value in annotation.items() if value == 1)
        unknown_codes.update(code for code in annotation if code not in MAST_CATEGORY_NAMES)
        if only_annotated_failures and not positive_codes:
            skipped_without_failure += 1
            continue
        # trace_id is only run-local in the MAST release.  The same GAIA IDs occur
        # under both MagenticOne_GAIA_GPT4o and Magentic_One_GPT4o.
        prompt_trace_id = (
            f"mast_{mas_name.casefold()}_{benchmark_name.casefold()}_{row['trace_id']}"
        )
        trace_id = _trace_identity(row)
        raw_trace = str((row.get("trace") or {}).get("trajectory") or "")
        instruction, steps = _parse_events(raw_trace)
        positive_names = [MAST_CATEGORY_NAMES[code] for code in positive_codes]
        positive_modules = sorted({MAST_MODULES[code.split(".", 1)[0]] for code in positive_codes})
        inputs.append(
            {
                "trajectory_id": trace_id,
                "prompt_trajectory_id": prompt_trace_id,
                "dataset": "mast",
                "domain": f"{mas_name.casefold()}_{benchmark_name.casefold()}",
                "instruction": instruction,
                "context": {
                    "multi_agent_system": mas_name,
                    "benchmark": benchmark_name,
                    "backbone_model": row.get("llm_name"),
                },
                "steps": steps,
                "metadata": {
                    "source_file": dataset_path.relative_to(project_root).as_posix(),
                    "source_trace_id": row.get("trace_id"),
                    "source_trace_key": (row.get("trace") or {}).get("key"),
                    "setup_preamble_removed": _SCENARIO_MARKER in raw_trace,
                },
            }
        )
        gold.append(
            {
                "trajectory_id": trace_id,
                "dataset": "mast",
                "domain": f"{mas_name.casefold()}_{benchmark_name.casefold()}",
                "eligible_for_attribution": bool(positive_names),
                "failure_type_raw": " | ".join(positive_codes),
                "failure_type": " | ".join(positive_names),
                "failure_types_raw": positive_codes,
                "failure_types": positive_names,
                "failure_module_raw": " | ".join(positive_modules),
                "failure_module": " | ".join(positive_modules),
                "critical_failure_step": None,
                "reference_reasoning": "",
                "failure_summary": "Official MAST multi-label annotations mark these failure modes as present: "
                + "; ".join(positive_names),
                "root_failure": {},
                "raw_label": annotation,
                "annotation_kind": "llm_judge_multilabel",
            }
        )

    inputs.sort(key=lambda item: item["trajectory_id"])
    gold.sort(key=lambda item: item["trajectory_id"])
    stats = {
        "total_trajectories": len(rows),
        "total_official_annotations": len(rows),
        "matched_pairs": len(inputs),
        "eligible_attribution_pairs": sum(item["eligible_for_attribution"] for item in gold),
        "missing_gold_failure_types": sum(not item["failure_types"] for item in gold),
        "missing_gold_reasoning": sum(not item["reference_reasoning"] for item in gold),
        "domain_trajectory_counts": {f"{mas_name.casefold()}_{benchmark_name.casefold()}": len(subset)},
        "domain_annotation_counts": {f"{mas_name.casefold()}_{benchmark_name.casefold()}": len(subset)},
        "selected_mas": mas_name,
        "selected_benchmark": benchmark_name,
        "selected_before_failure_filter": len(subset),
        "skipped_without_positive_failure_label": skipped_without_failure,
        "annotation_kind": "official_release_llm_judge_multilabel",
        "unknown_annotation_codes": sorted(unknown_codes),
        "unmatched_annotation_ids": [],
        "unmatched_trajectory_ids": [],
    }
    return inputs, gold, stats
