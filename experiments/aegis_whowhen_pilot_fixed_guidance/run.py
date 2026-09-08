from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.aegis_whowhen_pilot_fixed_guidance.core import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated AEGIS–Who&When implementation pilot."
    )
    parser.add_argument(
        "--config",
        default="experiments/aegis_whowhen_pilot_fixed_guidance/experiment.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Freeze samples and validate locally without any API calls.",
    )
    args = parser.parse_args()
    outputs = run_workflow(args.config, preflight_only=args.preflight_only)
    for name, value in outputs.items():
        print(f"{name}: {value}", flush=True)


if __name__ == "__main__":
    main()
