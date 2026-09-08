from __future__ import annotations

import copy
from pathlib import Path

from taxonomy_experiment.config import load_config

from experiments.aegis_whowhen_gpt5_merge_gemini25flash_main.core import (
    derive_implementation_id,
    ledger_summary,
    offline_preflight,
)
from experiments.aegis_whowhen_main_fixed_guidance import core as prediction_impl


CONFIG = Path(
    "experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/experiment.yaml"
)


def test_offline_preflight_has_zero_api_calls() -> None:
    config = load_config(CONFIG)
    prepared = offline_preflight(config)
    assert prepared["audit"]["api_calls"] == 0
    assert prepared["audit"]["planned_logical_rows"] == 2388
    assert prepared["audit"]["models"] == {
        "taxonomy_merge": "openai/gpt-5",
        "attribution": "google/gemini-2.5-flash",
    }


def test_implementation_id_depends_on_generated_merge() -> None:
    config = load_config(CONFIG)
    frozen_root = Path(config["paths"]["frozen_implementation_root"])
    if not frozen_root.is_absolute():
        frozen_root = config["_root"] / frozen_root
    merged = prediction_impl.read_json(frozen_root / "merged_taxonomy.json")
    merged["generated_by"]["model"] = "openai/gpt-5"
    first, _ = derive_implementation_id(config, merged)
    changed = copy.deepcopy(merged)
    changed["categories"][0]["definition"] += " changed"
    second, _ = derive_implementation_id(config, changed)
    assert first.startswith("main-")
    assert first != second


def test_ledger_summary_excludes_cache_events(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"actual_request":true,"cost_usd":0.25,"reported_usage":{"prompt_tokens":10,"completion_tokens":2}}\n'
        '{"actual_request":false,"cost_usd":0.0}\n',
        encoding="utf-8",
    )
    summary = ledger_summary(ledger)
    assert summary["actual_api_calls"] == 1
    assert summary["cost_usd"] == 0.25
    assert summary["prompt_tokens"] == 10
    assert summary["response_tokens"] == 2
