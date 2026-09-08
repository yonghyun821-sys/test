from __future__ import annotations

from pathlib import Path


def read_prompt(root: Path, name: str) -> str:
    return (root / "prompts" / name).read_text(encoding="utf-8").strip()


def format_template(template: str, **values: str) -> str:
    return template.format(**values)

