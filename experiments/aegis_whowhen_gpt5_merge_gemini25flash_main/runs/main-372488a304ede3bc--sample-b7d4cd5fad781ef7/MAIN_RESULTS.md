# Final Main Attribution Experiment

Status: `FINAL_MAIN_EXPERIMENT_COMPLETE`

## Experiment identity

- Research question: **How does taxonomy guidance affect the accuracy of agent failure attribution when the native output label space is held constant?**
- Prediction implementation ID: `main-372488a304ede3bc`
- Main sample ID: `sample-b7d4cd5fad781ef7`
- Namespace: `main-372488a304ede3bc--sample-b7d4cd5fad781ef7`
- Merge model: `openai/gpt-5`; reasoning effort=medium
- Attribution model: `google/gemini-2.5-flash`; temperature=0
- Conditions: Label-Only, Own, Foreign, Merged

## Sample

- AEGIS N = 97
- Who&When N = 500
- **One trajectory per unique original task was used in the primary analysis.**
- All 14 native classes are represented in each dataset; very small supports are interpreted descriptively.

### Native-category support

| Dataset | Native category | Support |
|---|---|---:|
| aegis | FM-1.1 | 13 |
| aegis | FM-1.2 | 7 |
| aegis | FM-1.3 | 4 |
| aegis | FM-1.4 | 2 |
| aegis | FM-1.5 | 9 |
| aegis | FM-2.1 | 8 |
| aegis | FM-2.2 | 6 |
| aegis | FM-2.3 | 8 |
| aegis | FM-2.4 | 3 |
| aegis | FM-2.5 | 5 |
| aegis | FM-2.6 | 10 |
| aegis | FM-3.1 | 8 |
| aegis | FM-3.2 | 6 |
| aegis | FM-3.3 | 8 |
| whowhen | A.1 | 2 |
| whowhen | A.2 | 23 |
| whowhen | A.3 | 141 |
| whowhen | A.4 | 3 |
| whowhen | C.1 | 1 |
| whowhen | C.2 | 1 |
| whowhen | C.3 | 45 |
| whowhen | PL.1 | 3 |
| whowhen | R.1 | 43 |
| whowhen | R.2 | 71 |
| whowhen | R.3 | 49 |
| whowhen | R.4 | 79 |
| whowhen | V.1 | 6 |
| whowhen | V.2 | 33 |

## Output compliance

| Dataset | Condition | Initial valid | Formatting retries | Truncation retries | Final valid | Terminal invalid |
|---|---|---:|---:|---:|---:|---:|
| aegis | label_only | 98.97% | 0 | 1 | 100.00% | 0 |
| aegis | own_taxonomy | 100.00% | 0 | 0 | 100.00% | 0 |
| aegis | foreign_taxonomy | 100.00% | 0 | 0 | 100.00% | 0 |
| aegis | merged_taxonomy | 100.00% | 0 | 0 | 100.00% | 0 |
| whowhen | label_only | 100.00% | 0 | 0 | 100.00% | 0 |
| whowhen | own_taxonomy | 99.40% | 0 | 3 | 100.00% | 0 |
| whowhen | foreign_taxonomy | 99.80% | 0 | 1 | 100.00% | 0 |
| whowhen | merged_taxonomy | 100.00% | 0 | 0 | 100.00% | 0 |

## Primary condition accuracy

| Dataset | Label-Only | Own | Foreign | Merged |
|---|---:|---:|---:|---:|
| aegis | 19.59% | 20.62% | 16.49% | 22.68% |
| whowhen | 22.20% | 23.00% | 20.20% | 21.20% |

## Primary paired contrasts

| Dataset | Contrast | Difference (pp) | Improved | Regressed | Exact McNemar p | Holm p | Paired 95% CI (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| aegis | own_taxonomy − label_only | 1.03 | 5 | 4 | 1 | 1 | [-5.15, 7.22] |
| aegis | foreign_taxonomy − label_only | -3.09 | 1 | 4 | 0.375 | 1 | [-8.25, 1.03] |
| aegis | merged_taxonomy − label_only | 3.09 | 5 | 2 | 0.453125 | 1 | [-2.06, 8.25] |
| aegis | merged_taxonomy − foreign_taxonomy | 6.19 | 7 | 1 | 0.0703125 | 0.5625 | [1.03, 12.37] |
| whowhen | own_taxonomy − label_only | 0.80 | 32 | 28 | 0.698883 | 1 | [-2.20, 3.80] |
| whowhen | foreign_taxonomy − label_only | -2.00 | 13 | 23 | 0.132498 | 0.927487 | [-4.40, 0.40] |
| whowhen | merged_taxonomy − label_only | -1.00 | 18 | 23 | 0.532709 | 1 | [-3.40, 1.40] |
| whowhen | merged_taxonomy − foreign_taxonomy | 1.00 | 21 | 16 | 0.511376 | 1 | [-1.40, 3.40] |

## Secondary metrics

| Dataset | Label-Only Macro-F1 | Own | Foreign | Merged |
|---|---:|---:|---:|---:|
| aegis | 0.1068 | 0.1284 | 0.0847 | 0.1217 |
| whowhen | 0.1383 | 0.1523 | 0.1205 | 0.1409 |

### Major confusion patterns

| Dataset | Condition | Gold | Predicted | Count |
|---|---|---|---|---:|
| aegis | label_only | FM-2.6 | FM-1.1 | 5 |
| aegis | label_only | FM-2.2 | FM-1.1 | 4 |
| aegis | label_only | FM-1.3 | FM-2.1 | 3 |
| aegis | label_only | FM-1.5 | FM-1.1 | 3 |
| aegis | label_only | FM-1.5 | FM-2.1 | 3 |
| aegis | own_taxonomy | FM-2.1 | FM-1.3 | 4 |
| aegis | own_taxonomy | FM-2.2 | FM-1.1 | 4 |
| aegis | own_taxonomy | FM-3.3 | FM-1.1 | 4 |
| aegis | own_taxonomy | FM-1.1 | FM-1.3 | 3 |
| aegis | own_taxonomy | FM-1.5 | FM-1.1 | 3 |
| aegis | foreign_taxonomy | FM-3.3 | FM-2.6 | 5 |
| aegis | foreign_taxonomy | FM-1.1 | FM-2.3 | 4 |
| aegis | foreign_taxonomy | FM-3.2 | FM-2.1 | 4 |
| aegis | foreign_taxonomy | FM-1.2 | FM-2.1 | 3 |
| aegis | foreign_taxonomy | FM-1.3 | FM-2.1 | 3 |
| aegis | merged_taxonomy | FM-1.1 | FM-2.3 | 4 |
| aegis | merged_taxonomy | FM-2.2 | FM-1.1 | 4 |
| aegis | merged_taxonomy | FM-2.6 | FM-1.1 | 4 |
| aegis | merged_taxonomy | FM-3.3 | FM-2.6 | 4 |
| aegis | merged_taxonomy | FM-1.2 | FM-2.1 | 3 |
| whowhen | label_only | A.3 | R.2 | 93 |
| whowhen | label_only | R.4 | R.2 | 46 |
| whowhen | label_only | A.3 | R.4 | 34 |
| whowhen | label_only | R.3 | R.2 | 31 |
| whowhen | label_only | C.3 | R.2 | 27 |
| whowhen | own_taxonomy | A.3 | R.4 | 66 |
| whowhen | own_taxonomy | A.3 | R.2 | 62 |
| whowhen | own_taxonomy | C.3 | R.2 | 29 |
| whowhen | own_taxonomy | R.4 | R.2 | 26 |
| whowhen | own_taxonomy | R.3 | R.2 | 24 |
| whowhen | foreign_taxonomy | A.3 | R.2 | 69 |
| whowhen | foreign_taxonomy | A.3 | R.4 | 56 |
| whowhen | foreign_taxonomy | R.4 | R.2 | 46 |
| whowhen | foreign_taxonomy | R.3 | R.2 | 29 |
| whowhen | foreign_taxonomy | C.3 | R.2 | 25 |
| whowhen | merged_taxonomy | A.3 | R.2 | 100 |
| whowhen | merged_taxonomy | R.4 | R.2 | 45 |
| whowhen | merged_taxonomy | C.3 | R.2 | 35 |
| whowhen | merged_taxonomy | R.3 | R.2 | 28 |
| whowhen | merged_taxonomy | A.3 | R.4 | 27 |

Full per-class precision, recall, F1, and support are reported in `main_per_class_metrics.csv`; full confusion matrices are in `main_confusion_matrices.json`.

## Guidance sensitivity

- aegis label_only_to_own_taxonomy: improved=5, regressed=4, prediction-change=37.11%.
- aegis label_only_to_foreign_taxonomy: improved=1, regressed=4, prediction-change=30.93%.
- aegis label_only_to_merged_taxonomy: improved=5, regressed=2, prediction-change=22.68%.
- aegis foreign_taxonomy_to_merged_taxonomy: improved=7, regressed=1, prediction-change=29.90%.
- whowhen label_only_to_own_taxonomy: improved=32, regressed=28, prediction-change=30.60%.
- whowhen label_only_to_foreign_taxonomy: improved=13, regressed=23, prediction-change=22.00%.
- whowhen label_only_to_merged_taxonomy: improved=18, regressed=23, prediction-change=20.20%.
- whowhen foreign_taxonomy_to_merged_taxonomy: improved=21, regressed=16, prediction-change=23.00%.

### Four-condition prediction patterns

| Dataset | Pattern | Count | Percent |
|---|---|---:|---:|
| aegis | all_four_same | 40 | 41.24% |
| aegis | only_own_changed | 19 | 19.59% |
| aegis | only_foreign_changed | 11 | 11.34% |
| aegis | only_merged_changed | 4 | 4.12% |
| aegis | multiple_conditions_changed | 23 | 23.71% |
| whowhen | all_four_same | 278 | 55.60% |
| whowhen | only_own_changed | 66 | 13.20% |
| whowhen | only_foreign_changed | 28 | 5.60% |
| whowhen | only_merged_changed | 25 | 5.00% |
| whowhen | multiple_conditions_changed | 103 | 20.60% |

## Framework analysis

Framework results are descriptive only; no confirmatory framework-level tests were performed.

| Dataset | Framework | N | Label-Only | Own | Foreign | Merged |
|---|---|---:|---:|---:|---:|---:|
| aegis | agentverse | 1 | 0.00% | 100.00% | 0.00% | 100.00% |
| aegis | dylan | 26 | 19.23% | 26.92% | 19.23% | 23.08% |
| aegis | llm_debate | 22 | 18.18% | 9.09% | 18.18% | 27.27% |
| aegis | magentic_one | 23 | 26.09% | 30.43% | 21.74% | 17.39% |
| aegis | smoagents | 25 | 16.00% | 12.00% | 8.00% | 20.00% |
| whowhen | alfagent | 21 | 33.33% | 23.81% | 38.10% | 28.57% |
| whowhen | debate | 46 | 15.22% | 10.87% | 13.04% | 13.04% |
| whowhen | dylan | 11 | 27.27% | 36.36% | 27.27% | 27.27% |
| whowhen | macnet | 34 | 8.82% | 8.82% | 11.76% | 8.82% |
| whowhen | magentic-one | 49 | 40.82% | 40.82% | 36.73% | 46.94% |
| whowhen | mathchat | 59 | 44.07% | 45.76% | 45.76% | 44.07% |
| whowhen | metagpt | 6 | 16.67% | 16.67% | 16.67% | 16.67% |
| whowhen | smolagents | 274 | 16.06% | 18.25% | 12.41% | 13.87% |

## Execution

- Planned logical rows: 2388
- Completed logical rows: 2388
- Actual API calls, combined: 2401
- GPT-5 merge API calls: 1
- Gemini 2.5 Flash attribution API calls: 2400
- Formatting retries: 0
- Truncation retries: 5
- Schema/parser retries: 7
- Provider/transport retries: 0
- Prompt tokens: 14820553
- Response tokens: 409827
- GPT-5 merge cost: $0.06947875
- Gemini 2.5 Flash attribution cost: $5.29819071
- Total recorded pipeline cost: $5.36766946
- API ledger reconciled: True

## Interpretation

- aegis: relative to Label-Only, Own changed accuracy by 1.03 pp, Foreign by -3.09 pp, and Merged by 3.09 pp. 0/4 pre-specified paired contrasts remained significant after Holm adjustment.
- whowhen: relative to Label-Only, Own changed accuracy by 0.80 pp, Foreign by -2.00 pp, and Merged by -1.00 pp. 0/4 pre-specified paired contrasts remained significant after Holm adjustment.

These results answer only how diagnostic taxonomy guidance changed exact native-label attribution accuracy while the native output label space was held constant. Mixed or null effects are retained as experimental results. AEGIS estimates have limited power for small effects; categories with support 1–3 are not used for strong class-specific claims.
