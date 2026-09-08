from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.aegis_whowhen_final_main_attribution.core import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen task-independent AEGIS/Who&When main experiment."
    )
    parser.add_argument(
        "--config",
        default="experiments/aegis_whowhen_final_main_attribution/config.yaml",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify every frozen identity and estimate cost without any API call.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Regenerate result artifacts from a complete prediction ledger without API calls.",
    )
    args = parser.parse_args()
    outputs = run_workflow(
        args.config,
        preflight_only=args.preflight_only,
        analyze_only=args.analyze_only,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
