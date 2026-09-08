from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.io import write_json


PAPER_AGENTERROR_CATEGORIES = {
    "memory": ["over_simplification", "memory_retrieval_failure", "hallucination"],
    "reflection": ["progress_misjudge", "outcome_misinterpretation", "causal_misattribution"],
    "planning": ["constraint_ignorance", "impossible_action", "inefficient_plan"],
    "action": ["misalignment", "invalid_action", "format_error", "parameter_error"],
    "system": ["step_limit", "tool_execution_error", "llm_limit", "environment_error"],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agenterror_source(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    spec = importlib.util.spec_from_file_location("canonical_agenterror_definitions", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ErrorDefinitionsLoader().definitions


def _agentrx_source(path: Path) -> dict[int, dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TAXONOMY_DATA"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            return value
    raise ValueError(f"TAXONOMY_DATA literal not found in {path}")


def extract_taxonomies(config_path: str | Path = "config/experiment.yaml") -> tuple[Path, Path]:
    config = load_config(config_path)
    root = config["_root"]
    agenterror_source_path = root / "repos" / "AgentDebug" / "detector" / "error_definitions.py"
    agentrx_source_path = root / "repos" / "AgentRx" / "agentrx" / "judge" / "judge.py"
    agenterror_source = _agenterror_source(agenterror_source_path)
    agentrx_source = _agentrx_source(agentrx_source_path)

    agenterror_categories = []
    for module, names in PAPER_AGENTERROR_CATEGORIES.items():
        for name in names:
            details = agenterror_source[module][name]
            agenterror_categories.append(
                {
                    "id": f"{module}.{name}",
                    "module": module,
                    "name": name,
                    "definition": details["definition"],
                }
            )
    agenterror = {
        "taxonomy_id": "agenterror_taxonomy",
        "name": "AgentErrorTaxonomy",
        "category_count": len(agenterror_categories),
        "source": {
            "repository": "repos/AgentDebug",
            "implementation": "repos/AgentDebug/detector/error_definitions.py",
            "implementation_sha256": _sha256(agenterror_source_path),
            "paper": "papers/agenterrorbench.pdf",
            "paper_summary": "repos/AgentDebug/README.md lines 124-134",
        },
        "representation_policy": "Canonical paper categories with module, category name, and source definition only; implementation examples are omitted for information-content parity with AgentRx.",
        "categories": agenterror_categories,
        "excluded_implementation_entries": [
            {
                "id": "reflection.hallucination",
                "reason": "Present in error_definitions.py but absent from the repository's canonical 17-type README table; the paper taxonomy counts hallucination under memory.",
            },
            {
                "id": "others.others",
                "reason": "Implementation fallback category, not one of the 17 paper taxonomy types.",
            },
        ],
    }
    agentrx_categories = [
        {
            "id": str(category_id),
            "module": None,
            "name": details["name"],
            "definition": details["desc_standard"],
        }
        for category_id, details in sorted(agentrx_source.items())
    ]
    agentrx = {
        "taxonomy_id": "agentrx_taxonomy",
        "name": "AgentRx Taxonomy",
        "category_count": len(agentrx_categories),
        "source": {
            "repository": "repos/AgentRx",
            "implementation": "repos/AgentRx/agentrx/judge/judge.py:TAXONOMY_DATA",
            "implementation_sha256": _sha256(agentrx_source_path),
            "paper": "papers/agentrx.pdf",
        },
        "representation_policy": "Canonical TAXONOMY_DATA names and desc_standard definitions only; checklists and examples are omitted for information-content parity with AgentErrorTaxonomy.",
        "categories": agentrx_categories,
    }
    output_a = resolve_path(config, "taxonomy_a")
    output_b = resolve_path(config, "taxonomy_b")
    write_json(output_a, agenterror)
    write_json(output_b, agentrx)
    return output_a, output_b

