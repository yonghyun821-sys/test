from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from taxonomy_experiment.io import read_json


class Condition(StrEnum):
    NO_TAXONOMY = "no_taxonomy"
    OWN_TAXONOMY = "own_taxonomy"
    FOREIGN_TAXONOMY = "foreign_taxonomy"
    MERGED_TAXONOMY = "merged_taxonomy"


CONDITION_ORDER = [
    Condition.NO_TAXONOMY,
    Condition.OWN_TAXONOMY,
    Condition.FOREIGN_TAXONOMY,
    Condition.MERGED_TAXONOMY,
]


DATASET_TAXONOMY_MAPPING = {
    "agentrx": {
        Condition.OWN_TAXONOMY: "agentrx_taxonomy",
        Condition.FOREIGN_TAXONOMY: "mast_taxonomy",
        Condition.MERGED_TAXONOMY: "llm_merged_agentrx_mast",
    },
    "mast": {
        Condition.OWN_TAXONOMY: "mast_taxonomy",
        Condition.FOREIGN_TAXONOMY: "agentrx_taxonomy",
        Condition.MERGED_TAXONOMY: "llm_merged_agentrx_mast",
    },
}


def taxonomy_digest(taxonomy: dict[str, Any]) -> str:
    payload = json.dumps(taxonomy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_taxonomy_payload(taxonomy: dict[str, Any]) -> dict[str, Any]:
    return {
        "taxonomy_id": taxonomy["taxonomy_id"],
        "name": taxonomy["name"],
        "categories": [
            {
                "id": category["id"],
                "module": category.get("module"),
                "name": category["name"],
                "definition": category["definition"],
                **(
                    {"provenance": category["provenance"]}
                    if "provenance" in category
                    else {}
                ),
            }
            for category in taxonomy["categories"]
        ],
    }


def format_taxonomy(taxonomy: dict[str, Any]) -> str:
    return json.dumps(canonical_taxonomy_payload(taxonomy), ensure_ascii=False, indent=2)


def attribution_taxonomy_payload(taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Use the same information fields for original, foreign, and merged conditions."""
    return {
        "name": taxonomy["name"],
        "categories": [
            {
                "module": category.get("module"),
                "name": category["name"],
                "definition": category["definition"],
            }
            for category in taxonomy["categories"]
        ],
    }


def taxonomy_block(taxonomy: dict[str, Any] | None) -> str:
    if taxonomy is None:
        return ""
    payload = attribution_taxonomy_payload(taxonomy)
    return f"<TAXONOMY>\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n</TAXONOMY>\n"


def load_taxonomy_set(taxonomy_a: Path, taxonomy_b: Path, merged: Path) -> dict[str, dict[str, Any]]:
    first = read_json(taxonomy_a)
    second = read_json(taxonomy_b)
    items = {first["taxonomy_id"]: first, second["taxonomy_id"]: second}
    if merged.exists():
        merged_item = read_json(merged)
        items[merged_item["taxonomy_id"]] = merged_item
    return items


def taxonomy_for_condition(
    dataset: str, condition: Condition, taxonomies: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if condition == Condition.NO_TAXONOMY:
        return None
    taxonomy_id = DATASET_TAXONOMY_MAPPING[dataset][condition]
    if taxonomy_id not in taxonomies:
        raise FileNotFoundError(
            f"Required taxonomy {taxonomy_id!r} is not available for {dataset}/{condition.value}"
        )
    return taxonomies[taxonomy_id]
