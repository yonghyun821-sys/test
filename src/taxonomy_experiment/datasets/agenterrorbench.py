from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from taxonomy_experiment.io import read_json


FAILURE_TYPE_NORMALIZATION = {
    "parameter_error": "parameter_error",
    "plan_inefficient": "inefficient_plan",
}

MODULE_NORMALIZATION = {"plan": "planning"}


def canonicalize_failure_type(value: Any) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower()
    return FAILURE_TYPE_NORMALIZATION.get(lowered, lowered)


def canonicalize_module(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return MODULE_NORMALIZATION.get(raw, raw)


def _first_marker(text: str, markers: list[str], start: int) -> int:
    positions = [
        position
        for marker in markers
        if (position := text.lower().find(marker.lower(), start)) >= 0
    ]
    return min(positions) if positions else len(text)


def _extract_after_marker(text: str, markers: list[str], end_markers: list[str]) -> str:
    lowered = text.lower()
    starts: list[tuple[int, int]] = []
    for marker in markers:
        position = lowered.find(marker.lower())
        if position >= 0:
            starts.append((position, len(marker)))
    if not starts:
        return ""
    position, marker_length = min(starts)
    start = position + marker_length
    end = _first_marker(text, end_markers, start)
    return text[start:end].strip()


def _extract_instruction(first_user_message: str, domain: str) -> str:
    if domain.lower() == "gaia":
        return _extract_after_marker(
            first_user_message,
            ["\nTask:", "Task:"],
            ["\n\nAvailable Tools:", "\nAvailable Tools:", "\n\nCurrent Observation:"],
        )
    return _extract_after_marker(
        first_user_message,
        ["Your task is to:", "Your task is:"],
        ["\n", "\r"],
    )


def _extract_observation(user_message: str) -> str:
    return _extract_after_marker(
        user_message,
        [
            "and your current observation is:",
            "Current Observation:",
            "Your current observation is:",
        ],
        [
            "\nYour admissible actions",
            "\n\nAvailable Tools:",
            "\nAvailable Tools:",
            "\n\nInstructions:",
            "\nInstructions:",
            "\n\nYou should first",
            "\nYou should first",
        ],
    )


def _extract_available_actions(user_message: str) -> str:
    block = _extract_after_marker(
        user_message,
        ["Your admissible actions of the current situation are:"],
        ["\n\nNow it's your turn", "\nNow it's your turn"],
    )
    return block


def _extract_constant_context(first_user_message: str, domain: str) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if domain.lower() == "gaia":
        tools = _extract_after_marker(
            first_user_message,
            ["Available Tools:"],
            ["\n\nCurrent Observation:", "\nCurrent Observation:"],
        )
        if tools:
            context["available_tools"] = tools
    return context


def _strip_redundant_memory(agent_output: str) -> tuple[str, int]:
    """Remove cumulative memory wrappers whose source steps remain in the trajectory."""
    pattern = re.compile(r"<memory>.*?</memory>\s*", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(agent_output)
    stripped = pattern.sub("", agent_output).strip()
    if matches and not stripped:
        return agent_output, 0
    return stripped, len(matches)


def _compact_unreadable_pdf_payload(observation: str) -> tuple[str, int]:
    """Compact raw PDF bytes while retaining a visible header and explicit audit marker."""
    marker = "%PDF-"
    start = observation.find(marker)
    if start < 0:
        return observation, 0
    keep_through = min(len(observation), start + 240)
    omitted = len(observation) - keep_through
    if omitted <= 0:
        return observation, 0
    compacted = (
        observation[:keep_through]
        + f"\n[... {omitted} characters of unreadable raw PDF binary omitted by "
        "deterministic normalization ...]"
    )
    return compacted, omitted


def normalize_trajectory(raw: dict[str, Any], trajectory_id: str, domain: str, source: Path) -> dict[str, Any]:
    messages = raw.get("messages", raw.get("chat_history", []))
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"No messages in AgentErrorBench trajectory: {source}")
    first_user = str(messages[0].get("content", ""))
    steps: list[dict[str, Any]] = []
    redundant_memory_blocks_removed = 0
    unreadable_pdf_characters_omitted = 0
    step_index = 1
    for position, message in enumerate(messages):
        if str(message.get("role", "")).lower() != "user":
            continue
        user_content = str(message.get("content", ""))
        assistant_content = ""
        if position + 1 < len(messages):
            next_message = messages[position + 1]
            if str(next_message.get("role", "")).lower() == "assistant":
                assistant_content = str(next_message.get("content", ""))
        assistant_content, removed = _strip_redundant_memory(assistant_content)
        redundant_memory_blocks_removed += removed
        explicit = re.search(r"you are now at step\s+(\d+)", user_content, re.IGNORECASE)
        current_index = int(explicit.group(1)) if explicit else step_index
        observation, omitted = _compact_unreadable_pdf_payload(
            _extract_observation(user_content)
        )
        unreadable_pdf_characters_omitted += omitted
        steps.append(
            {
                "index": current_index,
                "observation": observation,
                "available_actions": _extract_available_actions(user_content),
                "agent_output": assistant_content,
            }
        )
        step_index = current_index + 1
    metadata = raw.get("metadata", {})
    return {
        "trajectory_id": trajectory_id,
        "dataset": "agenterrorbench",
        "domain": domain.lower(),
        "instruction": _extract_instruction(first_user, domain),
        "context": _extract_constant_context(first_user, domain),
        "steps": steps,
        "metadata": {
            "source_file": str(source.as_posix()),
            "source_message_count": len(messages),
            "model": metadata.get("model"),
            "environment": metadata.get("environment"),
            "won": metadata.get("won"),
            "normalization_version": "v4",
            "redundant_memory_blocks_removed": redundant_memory_blocks_removed,
            "unreadable_pdf_characters_omitted": unreadable_pdf_characters_omitted,
        },
    }


def _critical_annotation(label: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    critical_step = label.get("critical_failure_step")
    annotations = label.get("step_annotations") or []
    annotation = next((item for item in annotations if item.get("step") == critical_step), None)
    if annotation is None and annotations:
        annotation = annotations[0]
    if not annotation:
        return None, None
    raw_module = str(label.get("critical_failure_module") or "")
    candidates = [raw_module, canonicalize_module(raw_module)]
    if raw_module == "planning":
        candidates.append("plan")
    if raw_module == "plan":
        candidates.append("planning")
    for key in candidates:
        detail = annotation.get(key)
        if isinstance(detail, dict):
            return annotation, detail
    for key, detail in annotation.items():
        if key != "step" and isinstance(detail, dict) and "failure_type" in detail:
            return annotation, detail
    return annotation, None


def load_agenterrorbench(
    trajectory_root: Path, label_root: Path, project_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trajectory_paths: dict[str, tuple[str, Path]] = {}
    domain_trajectory_counts: dict[str, int] = {}
    for domain_dir in sorted(path for path in trajectory_root.iterdir() if path.is_dir()):
        files = sorted(domain_dir.glob("*.json"))
        domain_trajectory_counts[domain_dir.name.lower()] = len(files)
        for path in files:
            if path.stem in trajectory_paths:
                raise ValueError(f"Duplicate AgentErrorBench trajectory ID: {path.stem}")
            trajectory_paths[path.stem] = (domain_dir.name, path)

    labels: list[dict[str, Any]] = []
    for path in sorted(label_root.glob("*.json")):
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected label list in {path}")
        labels.extend(payload)

    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    unmatched_label_ids: list[str] = []
    for label in labels:
        trajectory_id = str(label["trajectory_id"])
        match = trajectory_paths.get(trajectory_id)
        if match is None:
            unmatched_label_ids.append(trajectory_id)
            continue
        domain, trajectory_path = match
        matched_ids.add(trajectory_id)
        raw_trajectory = read_json(trajectory_path)
        relative_source = trajectory_path.relative_to(project_root)
        inputs.append(normalize_trajectory(raw_trajectory, trajectory_id, domain, relative_source))
        annotation, detail = _critical_annotation(label)
        raw_type = str((detail or {}).get("failure_type") or "").strip()
        raw_reasoning = str((detail or {}).get("reasoning") or "").strip()
        gold.append(
            {
                "trajectory_id": trajectory_id,
                "dataset": "agenterrorbench",
                "domain": domain.lower(),
                "eligible_for_attribution": bool(raw_type),
                "failure_type_raw": raw_type,
                "failure_type": canonicalize_failure_type(raw_type),
                "failure_module_raw": str(label.get("critical_failure_module") or "").strip(),
                "failure_module": canonicalize_module(label.get("critical_failure_module")),
                "critical_failure_step": label.get("critical_failure_step"),
                "reference_reasoning": raw_reasoning,
                "critical_annotation": annotation,
                "raw_label": label,
            }
        )

    unmatched_trajectory_ids = sorted(set(trajectory_paths) - matched_ids)
    inputs.sort(key=lambda item: (item["domain"], item["trajectory_id"]))
    gold.sort(key=lambda item: (item["domain"], item["trajectory_id"]))
    stats = {
        "total_trajectories": len(trajectory_paths),
        "total_official_annotations": len(labels),
        "matched_pairs": len(inputs),
        "eligible_attribution_pairs": sum(item["eligible_for_attribution"] for item in gold),
        "missing_gold_failure_types": sum(not item["failure_type_raw"] for item in gold),
        "missing_gold_reasoning": sum(not item["reference_reasoning"] for item in gold),
        "domain_trajectory_counts": domain_trajectory_counts,
        "unmatched_annotation_ids": sorted(unmatched_label_ids),
        "unmatched_trajectory_ids": unmatched_trajectory_ids,
    }
    return inputs, gold, stats
