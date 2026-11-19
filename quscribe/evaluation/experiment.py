"""
Ablation experiments: run the pipeline over a benchmark under
several configurations.

The standard ablations answer the proposal's evaluation questions:
- image only vs image plus paper text (does text context help reading?);
- no repair vs reference-free repair (does the verifier-guided loop help?);
- oracle repair, guided by the true reference (how much of the possible gain
    the reference-free approach captures).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from qiskit import QuantumCircuit

from quscribe.components.verification.verifiers import OracleVerifier
from quscribe.core.circuit_analysis import unitary_fidelity
from quscribe.core.models import PipelineResult
from quscribe.core.usage import ModelPricing
from quscribe.errors import CircuitAnalysisError
from quscribe.evaluation.dataset import BenchmarkItem
from quscribe.evaluation.metrics import (
    ConfigurationSummary,
    ItemEvaluation,
    summarize_configuration,
)
from quscribe.pipeline.builder import PipelineBuilder
from quscribe.pipeline.orchestrator import QuScribePipeline
from quscribe.pipeline.settings import PipelineSettings

logger = logging.getLogger(__name__)

# (column title, ConfigurationSummary attribute) pairs for the markdown report.
SUMMARY_COLUMNS = (
    ("Configuration", "configuration_name"),
    ("Items", "item_count"),
    ("Accuracy", "accuracy"),
    ("Mean fidelity", "mean_fidelity"),
    ("Verified precision", "verified_precision"),
    ("Verified recall", "verified_recall"),
    ("Flagged precision", "flagged_precision"),
    ("Flagged recall", "flagged_recall"),
    ("Mean repairs", "mean_repair_iterations"),
    ("Mean cost (USD)", "mean_cost_usd"),
    ("Mean latency (s)", "mean_latency_seconds"),
    ("Errors", "error_count"),
)
MISSING_VALUE_TEXT = "n/a"


@dataclass(frozen=True)
class AblationConfiguration:
    """
    One experimental condition.

    Attributes:
        name: Identifier used in reports.
        use_text_context: Whether to send the caption and paper text.
        max_repair_iterations: Repair budget (0 disables repair).
        use_oracle_repair: Whether repair is guided by the true
                            reference circuit.
    """

    name: str
    use_text_context: bool
    max_repair_iterations: int
    use_oracle_repair: bool = False

    @property
    def reports_verifier_reliability(self) -> bool:
        """
        Whether this configuration's labels measure the
        reference-free verifier.

        Args:
            None.

        Returns:
            bool: False for oracle repair, whose labels come from
                    the reference.
        """
        return not self.use_oracle_repair


def standard_ablations(
    max_repair_iterations: int
) -> tuple[AblationConfiguration, ...]:
    """
    Return the four configurations from the project proposal.

    Args:
        max_repair_iterations: Repair budget for the repair configurations.

    Returns:
        tuple[AblationConfiguration, ...]: The standard ablation set.
    """
    return (
        AblationConfiguration(
            "image_only",
            use_text_context=False,
            max_repair_iterations=0
        ),
        AblationConfiguration(
            "image_and_text",
            use_text_context=True,
            max_repair_iterations=0
        ),
        AblationConfiguration(
            "reference_free_repair",
            use_text_context=True,
            max_repair_iterations=max_repair_iterations
        ),
        AblationConfiguration(
            "oracle_repair",
            use_text_context=True,
            max_repair_iterations=max_repair_iterations,
            use_oracle_repair=True,
        ),
    )


@dataclass(frozen=True)
class ExperimentReport:
    """
    Results of an experiment across configurations.

    Attributes:
        summaries: One summary per configuration.
        item_evaluations: Every scored item, for per-item and
            per-style analysis.
        model_roles: Which model played each role, so results can be
            reproduced.
    """

    summaries: tuple[ConfigurationSummary, ...]
    item_evaluations: tuple[ItemEvaluation, ...]
    model_roles: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Format the summaries as a markdown table.

        Args:
            None.

        Returns:
            str: A markdown document with one row per configuration.
        """
        header = "| " + " | ".join(
            column_title for column_title, attribute in SUMMARY_COLUMNS) + " |"

        separator = "|" + "---|" * len(SUMMARY_COLUMNS)
        rows = [
            "| " + " | ".join(
                _format_cell(getattr(summary, attribute))
                for column_title, attribute in SUMMARY_COLUMNS) + " |"
            for summary in self.summaries
        ]

        model_lines = [
            f"- {role}: {model}" for role, model in self.model_roles.items()
        ]
        return "\n".join([
            "# QuScribe evaluation", "", *model_lines,
            "", header, separator, *rows
        ]) + "\n"


class ExperimentRunner:
    """
    Runs ablation configurations over benchmark items and scores the results.
    """

    def __init__(
        self,
        base_settings: PipelineSettings,
        pricing: ModelPricing | None = None
    ) -> None:
        """
        Create a runner.

        Args:
            base_settings: Settings shared by all configurations
                            (models, thresholds).
            pricing: Token prices for cost estimates; None reports tokens only.

        Returns:
            None.
        """
        self._base_settings = base_settings
        self._pricing = pricing

    def run(
        self,
        items: Sequence[BenchmarkItem],
        configurations: Sequence[AblationConfiguration]
    ) -> ExperimentReport:
        """
        Evaluate every configuration on every item.

        Args:
            items: The benchmark items.
            configurations: The ablation configurations to compare.

        Returns:
            ExperimentReport: Summaries and per-item evaluations.

        Raises:
            ConfigurationError: If model settings are invalid.
            InvalidInputError: If an item's image cannot be read.
        """
        all_evaluations: list[ItemEvaluation] = []
        summaries: list[ConfigurationSummary] = []
        for configuration in configurations:
            evaluations = self._evaluate_configuration(configuration, items)
            all_evaluations.extend(evaluations)
            summaries.append(summarize_configuration(
                configuration.name,
                evaluations,
                configuration.reports_verifier_reliability
            ))

        return ExperimentReport(
            summaries=tuple(summaries),
            item_evaluations=tuple(all_evaluations),
            model_roles=self._base_settings.model_roles(),
        )

    def _evaluate_configuration(
        self,
        configuration: AblationConfiguration,
        items: Sequence[BenchmarkItem]
    ) -> list[ItemEvaluation]:
        """
        Evaluate one configuration on every item.

        Args:
            configuration: The configuration to evaluate.
            items: The benchmark items.

        Returns:
            list[ItemEvaluation]: One evaluation per item.
        """
        settings = replace(
            self._base_settings,
            use_text_context=configuration.use_text_context,
            max_repair_iterations=configuration.max_repair_iterations,
        )
        builder = PipelineBuilder.from_settings(settings)
        # The reference-free pipeline is shared; the oracle needs each
        # item's reference.
        shared_pipeline = (
            None if configuration.use_oracle_repair else
            builder.build()
        )

        evaluations: list[ItemEvaluation] = []
        for item_number, item in enumerate(items, start=1):
            reference_circuit = item.reference_circuit()
            pipeline = (
                shared_pipeline if shared_pipeline is not None else
                builder.build(
                    verifier=OracleVerifier(
                        reference_circuit,
                        settings.fidelity_threshold
                    )
                )
            )

            evaluation = self._evaluate_item(
                configuration,
                item,
                reference_circuit,
                pipeline,
                settings
            )

            logger.info(
                "[%s] %d/%d %s: %s, fidelity %.3f",
                configuration.name, item_number, len(items), item.item_id,
                evaluation.label.value, evaluation.fidelity,
            )
            evaluations.append(evaluation)

        return evaluations

    def _evaluate_item(
        self,
        configuration: AblationConfiguration,
        item: BenchmarkItem,
        reference_circuit: QuantumCircuit,
        pipeline: QuScribePipeline,
        settings: PipelineSettings,
    ) -> ItemEvaluation:
        """
        Run the pipeline on one item and score it against the reference.

        Args:
            configuration: The configuration being evaluated.
            item: The benchmark item.
            reference_circuit: The item's ground-truth circuit.
            pipeline: The pipeline to run.
            settings: The configuration's settings (for the fidelity threshold)

        Returns:
            ItemEvaluation: The scored outcome.
        """
        result = pipeline.run(item.to_circuit_input())
        fidelity = _final_fidelity(result, reference_circuit)

        return ItemEvaluation(
            configuration_name=configuration.name,
            item_id=item.item_id,
            fidelity=fidelity,
            is_correct=fidelity >= settings.fidelity_threshold,
            label=result.label,
            repair_iterations=len(result.repair_steps),
            model_calls=result.usage.call_count,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            latency_seconds=result.wall_clock_seconds,
            cost_usd=(
                self._pricing.cost_of(result.usage) if self._pricing else None
            ),
            error_message=result.error_message,
            check_statuses=(
                {
                    check.check_name: check.status.value
                    for check in result.report.check_results
                } if result.report else {}
            ),
        )


def _final_fidelity(
    result: PipelineResult,
    reference_circuit: QuantumCircuit
) -> float:
    """
    Score a pipeline result against the reference circuit.

    Args:
        result: The pipeline result.
        reference_circuit: The ground-truth circuit.

    Returns:
        float: Unitary fidelity, or 0.0 if no circuit was produced or
                it cannot be analyzed.
    """
    if result.generated is None:
        return 0.0

    try:
        return unitary_fidelity(result.generated.circuit, reference_circuit)
    except CircuitAnalysisError as error:
        logger.warning("Scoring failed: %s", error)
        return 0.0


def _format_cell(value: object) -> str:
    """
    Format a summary value for the markdown table.

    Args:
        value: A number, string, or None.

    Returns:
        str: The formatted cell.
    """
    if value is None:
        return MISSING_VALUE_TEXT

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)
