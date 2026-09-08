from __future__ import annotations

from experiments.aegis_whowhen_final_main_attribution.core import (
    class_metrics,
    exact_mcnemar_p,
    holm_adjust,
    paired_bootstrap_ci,
)


def test_exact_two_sided_mcnemar() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(0, 5) == 0.0625
    assert exact_mcnemar_p(3, 3) == 1.0


def test_holm_adjustment_is_order_preserving_and_monotone() -> None:
    raw = [0.01, 0.04, 0.03, 0.5]
    adjusted = holm_adjust(raw)
    assert adjusted == [0.04, 0.09, 0.09, 0.5]


def test_paired_bootstrap_is_deterministic_and_keeps_pair_differences() -> None:
    differences = [1, 1, 0, -1, 0]
    first = paired_bootstrap_ci(differences, resamples=2_000, seed=1234)
    second = paired_bootstrap_ci(differences, resamples=2_000, seed=1234)
    assert first == second
    assert -1.0 <= first[0] <= first[1] <= 1.0


def test_class_metrics_include_unpredicted_native_class() -> None:
    rows = [
        {"gold_native_category": "a", "predicted_native_category": "a"},
        {"gold_native_category": "b", "predicted_native_category": "a"},
    ]
    metrics, macro_f1, matrix = class_metrics(rows, ["a", "b"])
    by_class = {item["category"]: item for item in metrics}
    assert by_class["a"]["precision"] == 0.5
    assert by_class["a"]["recall"] == 1.0
    assert by_class["b"]["f1"] == 0.0
    assert macro_f1 == by_class["a"]["f1"] / 2
    assert matrix == {"a": {"a": 1}, "b": {"a": 1}}
