# Main Experiment Readiness Report

Status: `READY_FOR_MAIN_EXPERIMENT`

Accuracy was not computed for the smoke test and was not used for readiness.

## Implementation fixes

- `failure_category` is constrained by a dataset-specific exact native-ID enum in the API JSON Schema.
- Local exact-ID validation remains active; no fallback, normalization, relabeling, or semantic judging exists.
- At most two formatting-only retries use a gold-blind contract-correction message.
- Terminal invalid rows are retained with `terminal_invalid=true` and `correct=0`; no drop or resampling.
- Truncation uses one deterministic, condition-invariant retry with a frozen 1,200-token recovery limit.
- Every actual, billed, retry, transport, parser, and cache event is context-labeled in the isolated ledger.

## Smoke test

- Rows: 16
- Attempted rows: 16
- Responses actually received: 16
- API/network failures: 0
- Provider transport failures: 0
- Parsed responses: 16
- Exact-ID compliant responses: 16
- Initial compliant: 16
- Formatting retries: 0
- Truncation retries: 0
- Final compliant: 16
- Terminal invalid: 0
- Billed responses: 16
- Actual network calls: 16
- Total cost: $0.01662600
- Accuracy: not computed

## Ledger reconciliation

- Counts: `{"base_attribution": 16, "formatting_retry": 0, "merge_adjudication": 0, "provider_transport_retry": 0, "schema_parser_retry": 0, "truncation_retry": 0}`
- Equation sum: 16
- Total actual API calls: 16
- Reconciled: True
- Cost: $0.01662600
- Offline retry/terminal/truncation audit: passed

## Mechanical category-collapse audit

- Checks: 6/6
- No output-label mismatch, ordering drift, unintended prompt difference, parser fallback, or default-category selection was found.
- Any category concentration is therefore treated as model behavior, not a tuning signal.

## Freeze and exclusions

- Implementation run ID: `main-95e2d9ee2ba7c1fd`
- Parser, schema, serializer, prompts, model configuration, label sets, and all three taxonomies are content-hashed.
- Pilot exclusions: 112 IDs; overlap with main sample: 0
- Smoke overlap with main sample: 0
- Sampling: `deterministic_hierarchical_proportional_stratified_without_replacement`
- Sampling seed: 20260907
- Dataset counts: AEGIS=1000, Who&When=1000
- Stratification audit: passed; rare-class oversampling applied=False
- Source ID collisions resolved: 1; rows dropped=0
- Main-sample leakage count: 0
- Frozen main sample: 2000 trajectories / 8000 planned rows

## Final status

READY_FOR_MAIN_EXPERIMENT
