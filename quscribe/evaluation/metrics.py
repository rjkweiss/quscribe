"""
Evaluation metrics: generation accuracy, verifier reliability,
cost and latency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import fmean

from quscribe.core.models import TrustLabel


@dataclass(frozen=True)
class ItemEvaluation:
    """
    Scored outcome of one benchmark item under one configuration.

    Attributes:
        configuration_name: The ablation configuration.
        item_id: The benchmark item.
        fidelity: Unitary fidelity of the final circuit to the reference.
        is_correct: Whether the fidelity meets the threshold (the ground truth)
        label: The trust label QuScribe assigned.
        repair_iterations: Repair iterations performed.
        model_calls: Number of model calls.
        input_tokens: Prompt tokens used.
        output_tokens: Completion tokens used.
        latency_seconds: Wall-clock time for the item.
        cost_usd: Estimated cost, or None when no pricing was supplied.
        error_message: Why the run stopped early, if it did.
        check_statuses: Each verification check's name mapped to its final
                        status (passed, failed, skipped or error), showing
                        which check caught what.
    """

    configuration_name: str
    item_id: str
    fidelity: float
    is_correct: bool
    label: TrustLabel
    repair_iterations: int
    model_calls: int
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float | None
    error_message: str | None
    check_statuses: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BinaryConfusion:
    """
    Confusion counts for a binary classifier.

    Attributes:
        true_positives: Predicted positive, actually positive.
        false_positives: Predicted positive, actually negative.
        true_negatives: Predicted negative, actually negative.
        false_negatives: Predicted negative, actually positive.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @classmethod
    def from_outcomes(
        cls,
        predicted: Sequence[bool],
        actual: Sequence[bool]
    ) -> BinaryConfusion:
        """
        Count outcomes from paired predictions and ground truth.

        Args:
            predicted: Predicted positives.
            actual: Actual positives, aligned with ``predicted``.

        Returns:
            BinaryConfusion: The confusion counts.

        Raises:
            ValueError: If the sequences differ in length.
        """
        pairs = list(zip(predicted, actual, strict=True))
        return cls(
            true_positives=sum(guess and truth for guess, truth in pairs),
            false_positives=sum(guess and not truth for guess, truth in pairs),
            true_negatives=sum(
                not guess and not truth for guess, truth in pairs),
            false_negatives=sum(not guess and truth for guess, truth in pairs),
        )

    @property
    def precision(self) -> float | None:
        """
        Fraction of predicted positives that are actually positive.

        Args:
            None.

        Returns:
            float | None: The precision, or None with no predicted positives.
        """
        return _safe_ratio(
            self.true_positives,
            self.true_positives + self.false_positives,
        )

    @property
    def recall(self) -> float | None:
        """
        Fraction of actual positives that are predicted positive.

        Args:
            None.

        Returns:
            float | None: The recall, or None with no actual positives.
        """
        return _safe_ratio(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )

    def inverted(self) -> BinaryConfusion:
        """
        Swap the positive and negative classes.

        Args:
            None.

        Returns:
            BinaryConfusion: Counts with the classes swapped.
        """
        return BinaryConfusion(
            true_positives=self.true_negatives,
            false_positives=self.false_negatives,
            true_negatives=self.true_positives,
            false_negatives=self.false_positives,
        )


@dataclass(frozen=True)
class ConfigurationSummary:
    """
    Aggregate metrics for one ablation configuration.

    "Verified" metrics measure trust: precision is P(correct | verified).
    "Flagged" metrics measure error detection: recall is P(flagged | incorrect)

    Attributes:
        configuration_name: The ablation configuration.
        item_count: Number of items evaluated.
        accuracy: Fraction of items whose final circuit is correct.
        mean_fidelity: Mean unitary fidelity to the reference.
        verified_precision: P(correct | verified), or None if not applicable.
        verified_recall: P(verified | correct), or None if not applicable.
        flagged_precision: P(incorrect | flagged), or None if not applicable.
        flagged_recall: P(flagged | incorrect), or None if not applicable.
        mean_repair_iterations: Mean repair iterations per item.
        mean_input_tokens: Mean prompt tokens per item.
        mean_output_tokens: Mean completion tokens per item.
        mean_latency_seconds: Mean wall-clock time per item.
        mean_cost_usd: Mean cost per item, or None without pricing.
        error_count: Items whose run stopped early.
    """

    configuration_name: str
    item_count: int
    accuracy: float | None
    mean_fidelity: float | None
    verified_precision: float | None
    verified_recall: float | None
    flagged_precision: float | None
    flagged_recall: float | None
    mean_repair_iterations: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None
    mean_latency_seconds: float | None
    mean_cost_usd: float | None
    error_count: int


def summarize_configuration(
    configuration_name: str,
    evaluations: Iterable[ItemEvaluation],
    include_verifier_metrics: bool
) -> ConfigurationSummary:
    """
    Aggregate item evaluations into configuration-level metrics.

    Args:
        configuration_name: The ablation configuration.
        evaluations: The configuration's item evaluations.
        include_verifier_metrics: False for the oracle, whose labels come from
                                    the reference and therefore say nothing
                                    about the reference-free verifier.

    Returns:
        ConfigurationSummary: The aggregated metrics.
    """
    evaluation_list = list(evaluations)
    verified_confusion = (
        BinaryConfusion.from_outcomes(
            [
                evaluation.label is TrustLabel.VERIFIED
                for evaluation in evaluation_list
            ],
            [evaluation.is_correct for evaluation in evaluation_list],
        )
        if include_verifier_metrics
        else None
    )

    verified_precision, verified_recall = _precision_and_recall(
        verified_confusion
    )

    flagged_precision, flagged_recall = _precision_and_recall(
        verified_confusion.inverted() if verified_confusion else None
    )

    costs = [
        evaluation.cost_usd
        for evaluation in evaluation_list if evaluation.cost_usd is not None
    ]

    return ConfigurationSummary(
        configuration_name=configuration_name,
        item_count=len(evaluation_list),
        accuracy=_mean(
            [float(evaluation.is_correct) for evaluation in evaluation_list]
        ),
        mean_fidelity=_mean(
            [evaluation.fidelity for evaluation in evaluation_list]
        ),
        verified_precision=verified_precision,
        verified_recall=verified_recall,
        flagged_precision=flagged_precision,
        flagged_recall=flagged_recall,
        mean_repair_iterations=_mean(
            [
                float(evaluation.repair_iterations)
                for evaluation in evaluation_list
            ]
        ),
        mean_input_tokens=_mean(
            [float(evaluation.input_tokens) for evaluation in evaluation_list]
        ),
        mean_output_tokens=_mean(
            [float(evaluation.output_tokens) for evaluation in evaluation_list]
        ),
        mean_latency_seconds=_mean(
            [evaluation.latency_seconds for evaluation in evaluation_list]
        ),
        mean_cost_usd=(
            _mean(costs) if len(costs) == len(evaluation_list) else None
        ),
        error_count=sum(
            evaluation.error_message is not None
            for evaluation in evaluation_list
        ),
    )


def _precision_and_recall(
    confusion: BinaryConfusion | None
) -> tuple[float | None, float | None]:
    """
    Unpack precision and recall, tolerating a missing confusion matrix.

    Args:
        confusion: The confusion counts, or None when not applicable.

    Returns:
        tuple[float | None, float | None]: Precision and recall, or
                                            (None, None).
    """
    if confusion is None:
        return None, None
    return confusion.precision, confusion.recall


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """
    Divide, returning None instead of failing on a zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        float | None: The ratio, or None when the denominator is 0.
    """
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float]) -> float | None:
    """
    Average values, returning None for an empty sequence.

    Args:
        values: The values to average.

    Returns:
        float | None: The mean, or None when there are no values.
    """
    return fmean(values) if values else None
