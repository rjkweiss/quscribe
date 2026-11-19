"""Tests for evaluation metrics."""

import pytest

from quscribe.core.models import TrustLabel
from quscribe.evaluation.metrics import (
    BinaryConfusion,
    ItemEvaluation,
    summarize_configuration,
)


def make_evaluation(
    is_correct: bool,
    label: TrustLabel,
    cost_usd: float | None = 0.01
) -> ItemEvaluation:
    """
    Build an item evaluation with the fields metrics depend on.

    Args:
        is_correct: Whether the circuit matched the reference.
        label: The assigned trust label.
        cost_usd: The item cost, or None without pricing.

    Returns:
        ItemEvaluation: The evaluation.
    """
    return ItemEvaluation(
        configuration_name="test",
        item_id="item",
        fidelity=1.0 if is_correct else 0.2,
        is_correct=is_correct,
        label=label,
        repair_iterations=0,
        model_calls=4,
        input_tokens=100,
        output_tokens=50,
        latency_seconds=2.0,
        cost_usd=cost_usd,
        error_message=None,
    )


def test_confusion_precision_recall_and_inversion() -> None:
    """
    Precision and recall follow their definitions, and inversion swaps classes.

    Args:
        None.

    Returns:
        None.
    """
    confusion = BinaryConfusion.from_outcomes(
        [True, True, False, False],
        [True, False, False, True]
    )

    assert confusion.precision == pytest.approx(0.5)
    assert confusion.recall == pytest.approx(0.5)
    inverted = confusion.inverted()
    assert inverted.true_positives == confusion.true_negatives
    assert inverted.false_negatives == confusion.false_positives


def test_summary_reports_verifier_metrics_and_cost() -> None:
    """
    A trustworthy verifier yields perfect verified precision and
    flagged recall.

    Args:
        None.

    Returns:
        None.
    """
    evaluations = [
        make_evaluation(True, TrustLabel.VERIFIED),
        make_evaluation(True, TrustLabel.FLAGGED),
        make_evaluation(False, TrustLabel.FLAGGED),
    ]

    summary = summarize_configuration(
        "test",
        evaluations,
        include_verifier_metrics=True
    )

    assert summary.accuracy == pytest.approx(2 / 3)
    assert summary.verified_precision == pytest.approx(1.0)
    assert summary.verified_recall == pytest.approx(0.5)
    assert summary.flagged_recall == pytest.approx(1.0)
    assert summary.mean_cost_usd == pytest.approx(0.01)


def test_oracle_summary_omits_verifier_metrics_and_partial_cost() -> None:
    """
    Oracle runs report no verifier metrics, and partial pricing reports
    no cost.

    Args:
        None.

    Returns:
        None.
    """
    evaluations = [make_evaluation(True, TrustLabel.VERIFIED, cost_usd=None)]
    summary = summarize_configuration(
        "oracle",
        evaluations,
        include_verifier_metrics=False
    )

    assert summary.verified_precision is None
    assert summary.flagged_recall is None
    assert summary.mean_cost_usd is None
