from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.aegis_whowhen_main_fixed_guidance.core import prepare_and_run_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and validate the AEGIS–Who&When main implementation."
    )
    parser.add_argument(
        "--config",
        default="experiments/aegis_whowhen_main_fixed_guidance/experiment.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run all local checks without an OpenRouter call.",
    )
    args = parser.parse_args()
    outputs = prepare_and_run_gate(args.config, preflight_only=args.preflight_only)
    for key, value in outputs.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
