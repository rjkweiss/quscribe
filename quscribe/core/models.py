"""
Immutable data objects passed between QuScribe components.

Every pipeline stage communicates through these types; a stage only needs to
know the shape of its input and output, never the internals of its neighbors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from quscribe.core.images import ImagePayload
from quscribe.core.usage import UsageSummary

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class TrustLabel(str, Enum):
    """
    Final trust decision attached to every generated circuit.

    Attributes:
        VERIFIED: Every check that ran agreed with the figure.
        FLAGGED: A check failed or errored, or nothing could
                confirm the circuit.
    """

    VERIFIED = "verified"
    FLAGGED = "flagged"


class CheckStatus(str, Enum):
    """
    Outcome of a single verification check.

    Attributes:
        PASSED: The check found the candidate consistent.
        FAILED: The check found at least one discrepancy.
        SKIPPED: The check had nothing to test (no claims in the paper text).
        ERROR: The check could not run.
    """

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# Statuses that prevent a circuit from being verified.
PROBLEM_STATUSES = frozenset({CheckStatus.FAILED, CheckStatus.ERROR})


@dataclass(frozen=True)
class CircuitInput:
    """
    Everything known about one circuit figure.

    Attributes:
        figure: The circuit diagram image.
        caption: The figure caption, if available.
        paper_text: Relevant passages from the paper, if available.
    """

    figure: ImagePayload
    caption: str = ""
    paper_text: str = ""

    @property
    def has_text_context(self) -> bool:
        """
        Report whether a caption or paper text accompanies the figure.

        Args:
            None.

        Returns:
            bool: True when either text field is non-blank.
        """
        return bool(self.caption.strip() or self.paper_text.strip())


@dataclass(frozen=True)
class ParsedCircuit:
    """
    The intermediate representation produced by reading a figure.

    Attributes:
        qasm: The OpenQASM 3 program.
        circuit: The circuit loaded from that program.
    """

    qasm: str
    circuit: QuantumCircuit


@dataclass(frozen=True)
class GeneratedCode:
    """
    Executable Qiskit code and the circuit it builds.

    Attributes:
        source_code: The generated Python module.
        circuit: The circuit returned by running that module.
    """

    source_code: str
    circuit: QuantumCircuit


@dataclass(frozen=True)
class CheckResult:
    """
    Outcome of one verification check.

    Attributes:
        check_name: Name of the check that produced the result.
        status: The check outcome.
        summary: One-sentence explanation of the outcome.
        discrepancies: Specific problems found, phrased for the repair agent.
    """

    check_name: str
    status: CheckStatus
    summary: str
    discrepancies: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationReport:
    """
    Aggregated verification outcome for one candidate circuit.

    Attributes:
        label: The trust label derived from the check results.
        check_results: The individual check outcomes.
    """

    label: TrustLabel
    check_results: tuple[CheckResult, ...]

    @classmethod
    def from_check_results(
        cls, check_results: Sequence[CheckResult]
    ) -> VerificationReport:
        """
        Derive the trust label from individual check results.

        A circuit is verified only when no check failed or errored and
        at least one check passed; skipped checks neither confirm
        nor refute it.

        Args:
            check_results: The outcomes of every check that was run.

        Returns:
            VerificationReport: The report with its derived label.
        """
        has_problem = any(
            result.status in PROBLEM_STATUSES for result in check_results)
        has_pass = any(
            result.status is CheckStatus.PASSED for result in check_results)
        label = (
            TrustLabel.VERIFIED
            if has_pass and not has_problem
            else TrustLabel.FLAGGED
        )

        return cls(label=label, check_results=tuple(check_results))

    @property
    def discrepancies(self) -> tuple[str, ...]:
        """
        Collect the problems from failed or errored checks, tagged by
        check name.

        Args:
            None.

        Returns:
            tuple[str, ...]: One entry per problem,
                            e.g.``"[check] missing cx"``.
        """
        collected: list[str] = []
        for result in self.check_results:
            if result.status not in PROBLEM_STATUSES:
                continue

            # A check that errored may have no itemized discrepancies;
            # use its summary.
            details = result.discrepancies or (result.summary,)
            collected.extend(
                f"[{result.check_name}] {detail}" for detail in details
            )

        return tuple(collected)


@dataclass(frozen=True)
class RepairStep:
    """
    Record of one repair iteration, shown in the web demo and reports.

    Attributes:
        iteration: 1-based repair iteration number.
        addressed_discrepancies: The problems sent to the repair agent.
        revised_qasm: The corrected OpenQASM 3 program.
        label_after_repair: The trust label after re-verification.
    """

    iteration: int
    addressed_discrepancies: tuple[str, ...]
    revised_qasm: str
    label_after_repair: TrustLabel


@dataclass(frozen=True)
class PipelineResult:
    """
    Everything the pipeline produced for one figure.

    Attributes:
        label: Final trust label; always FLAGGED when an error occurred.
        parsed: Final transcription, or None if parsing failed.
        generated: Final generated code, or None if generation failed.
        report: Final verification report, or None if verification never ran.
        repair_steps: The repair iterations performed, in order.
        usage: Model usage for this run.
        wall_clock_seconds: Total time for this run.
        error_message: Why the run stopped early, or None if it completed.
    """

    label: TrustLabel
    parsed: ParsedCircuit | None
    generated: GeneratedCode | None
    report: VerificationReport | None
    repair_steps: tuple[RepairStep, ...]
    usage: UsageSummary
    wall_clock_seconds: float
    error_message: str | None = None
