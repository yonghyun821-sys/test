from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.aegis_whowhen_gpt5_merge_gemini25flash_main.core import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated GPT-5 merge + Gemini 2.5 Flash main experiment."
    )
    parser.add_argument(
        "--config",
        default="experiments/aegis_whowhen_gpt5_merge_gemini25flash_main/experiment.yaml",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    result = run_workflow(
        args.config,
        preflight_only=args.preflight_only,
        analyze_only=args.analyze_only,
    )
    for key, value in result.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()

