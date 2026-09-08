from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taxonomy_experiment.config import load_config, resolve_path
from taxonomy_experiment.io import read_json, write_json
from taxonomy_experiment.llm import CachedLLM
from taxonomy_experiment.models import MergedTaxonomyResult
from taxonomy_experiment.prompts import format_template, read_prompt
from taxonomy_experiment.taxonomy import format_taxonomy, taxonomy_digest


def _validate_provenance(
    result: MergedTaxonomyResult,
    taxonomy_a: dict[str, Any],
    taxonomy_b: dict[str, Any],
) -> None:
    allowed = {
        taxonomy_a["taxonomy_id"]: {item["id"] for item in taxonomy_a["categories"]},
        taxonomy_b["taxonomy_id"]: {item["id"] for item in taxonomy_b["categories"]},
    }
    if not result.categories:
        raise ValueError("Merged taxonomy has no categories")
    ids = [category.id for category in result.categories]
    if len(ids) != len(set(ids)):
        raise ValueError("Merged taxonomy contains duplicate category IDs")
    for category in result.categories:
        if not category.provenance:
            raise ValueError(f"Merged category {category.id!r} has no provenance")
        for provenance in category.provenance:
            if provenance.taxonomy_id not in allowed:
                raise ValueError(
                    f"Merged category {category.id!r} uses unknown taxonomy provenance {provenance.taxonomy_id!r}"
                )
            if provenance.original_category_id not in allowed[provenance.taxonomy_id]:
                raise ValueError(
                    f"Merged category {category.id!r} uses unknown source category "
                    f"{provenance.taxonomy_id}/{provenance.original_category_id}"
                )


def create_merged_taxonomy(config_path: str | Path = "config/experiment.yaml") -> Path:
    config = load_config(config_path)
    output_path = resolve_path(config, "merged_taxonomy")
    raw_path = resolve_path(config, "merged_taxonomy_raw")
    if output_path.exists() or raw_path.exists():
        raise FileExistsError(
            "Merged taxonomy is frozen once created. Refusing to overwrite existing merged taxonomy artifacts."
        )
    from taxonomy_experiment.budget import estimate_api_budget

    estimate_api_budget(config_path, mode="full")
    taxonomy_a = read_json(resolve_path(config, "taxonomy_a"))
    taxonomy_b = read_json(resolve_path(config, "taxonomy_b"))
    template = read_prompt(config["_root"], "taxonomy_merge.txt")
    user_prompt = format_template(
        template,
        taxonomy_a=format_taxonomy(taxonomy_a),
        taxonomy_b=format_taxonomy(taxonomy_b),
    )
    model_config = config["models"]["merge"]
    llm = CachedLLM(config, resolve_path(config, "cache") / "merge")
    result, call = llm.call(
        model=model_config["name"],
        system_prompt="Merge only the two supplied taxonomy definitions. Do not use dataset knowledge.",
        user_prompt=user_prompt,
        temperature=model_config.get("temperature"),
        reasoning_effort=model_config.get("reasoning_effort"),
        seed=int(config["experiment"]["seed"]),
        provider=model_config.get("provider"),
        max_output_tokens=int(model_config["max_output_tokens"]),
        response_model=MergedTaxonomyResult,
    )
    _validate_provenance(result, taxonomy_a, taxonomy_b)
    merged_id = "llm_merged_" + "_".join(
        taxonomy["taxonomy_id"].removesuffix("_taxonomy")
        for taxonomy in (taxonomy_a, taxonomy_b)
    )
    prompt_sha = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
    artifact = {
        "taxonomy_id": merged_id,
        "name": result.name,
        "description": result.description,
        "category_count": len(result.categories),
        "generated_by": {
            "model": model_config["name"],
            "temperature": model_config.get("temperature"),
            "reasoning_effort": model_config.get("reasoning_effort"),
            "seed": int(config["experiment"]["seed"]),
            "api": "openrouter_chat_completions",
            "provider_routing": {
                **config["api"].get("provider_defaults", {}),
                **model_config.get("provider", {}),
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": prompt_sha,
            "cache_key": call["cache_key"],
        },
        "source_taxonomies": [
            {"taxonomy_id": taxonomy_a["taxonomy_id"], "sha256": taxonomy_digest(taxonomy_a)},
            {"taxonomy_id": taxonomy_b["taxonomy_id"], "sha256": taxonomy_digest(taxonomy_b)},
        ],
        "freeze_policy": "Generated once before attribution evaluation; no human semantic edits permitted.",
        "categories": [item.model_dump(mode="json") for item in result.categories],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(call["raw_output_text"] + "\n", encoding="utf-8", newline="\n")
    write_json(output_path, artifact)
    return output_path
