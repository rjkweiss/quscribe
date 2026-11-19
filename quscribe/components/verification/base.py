"""
Verification abstractions shared by the reference-free and oracle verifiers.

A ``Verifier`` builds a ``VerificationSuite`` once per figure. Checks in the
suite may cache work that depends only on the figure (an independent reading,
extracted paper claims), so repair iterations re-verify new code without
repeating that work.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from qiskit import QuantumCircuit

from quscribe.core.circuit_analysis import compare_circuits
from quscribe.core.models import (
    CheckResult,
    CheckStatus,
    CircuitInput,
    GeneratedCode,
    VerificationReport,
)
from quscribe.errors import QuScribeError


class VerificationCheck(ABC):
    """
    One independent test of whether generated code matches the figure.

    Subclasses define a ``name`` class attribute and implement ``run``.
    """

    name: str

    @abstractmethod
    def run(self, candidate: GeneratedCode) -> CheckResult:
        """Test a candidate.

        Args:
            candidate: The generated code and its circuit.

        Returns:
            CheckResult: The outcome of the check.

        Raises:
            QuScribeError: If the check cannot run; the suite records it
                            as an error.
        """


class VerificationSuite:
    """
    Runs a fixed set of checks against candidates for one figure."""

    def __init__(self, checks: Sequence[VerificationCheck]) -> None:
        """
        Create a suite.

        Args:
            checks: The checks to run, in order.
        """
        self._checks = tuple(checks)

    def verify(self, candidate: GeneratedCode) -> VerificationReport:
        """
        Run every check and derive the trust label.

        Args:
            candidate: The generated code and its circuit.

        Returns:
            VerificationReport: All check outcomes and the resulting label.
        """
        results = [
            self._run_safely(check, candidate)
            for check in self._checks
        ]

        return VerificationReport.from_check_results(results)

    @staticmethod
    def _run_safely(
        check: VerificationCheck,
        candidate: GeneratedCode
    ) -> CheckResult:
        """
        Run one check, turning a failure to run into an ERROR result.

        Args:
            check: The check to run.
            candidate: The generated code and its circuit.

        Returns:
            CheckResult: The check outcome, or ERROR result if it couldn't run.
        """
        try:
            return check.run(candidate)
        except QuScribeError as error:
            return CheckResult(
                check.name,
                CheckStatus.ERROR,
                f"The check could not run: {error}"
            )


class Verifier(ABC):
    """
    Builds the verification suite for a figure.
    """

    @abstractmethod
    def create_suite(self, circuit_input: CircuitInput) -> VerificationSuite:
        """
        Create the checks for one figure.

        Args:
            circuit_input: The figure and its accompanying text.

        Returns:
            VerificationSuite: The suite to run on every candidate for
                this figure.
        """


class CircuitComparisonCheck(VerificationCheck):
    """
    Passes when the candidate's unitary matches a comparison circuit.

    Subclasses decide where the comparison circuit comes from: an independent
    reading of the figure (reference-free) or the true reference circuit
    (oracle upper bound).
    """

    comparison_description: str

    def __init__(self, fidelity_threshold: float) -> None:
        """
        Create the check.

        Args:
            fidelity_threshold: Minimum unitary fidelity to pass.
        """
        self._fidelity_threshold = fidelity_threshold

    @abstractmethod
    def _comparison_circuit(self) -> QuantumCircuit:
        """
        Return the circuit to compare candidates against.

        Returns:
            QuantumCircuit: The comparison circuit.

        Raises:
            QuScribeError: If the comparison circuit cannot be obtained.
        """

    def run(self, candidate: GeneratedCode) -> CheckResult:
        """
        Compare the candidate with the comparison circuit.

        Args:
            candidate: The generated code and its circuit.

        Returns:
            CheckResult: PASSED if equivalent, otherwise FAILED with
                        gate-level differences.

        Raises:
            CircuitAnalysisError: If the unitaries cannot be computed.
        """
        comparison = compare_circuits(
            candidate.circuit,
            self._comparison_circuit(),
            self._fidelity_threshold,
            self.comparison_description,
        )
        fidelity_note = f"(fidelity {comparison.fidelity:.4f})"
        if comparison.is_equivalent:
            return CheckResult(
                self.name,
                CheckStatus.PASSED,
                f"Matches {self.comparison_description} {fidelity_note}."
            )

        return CheckResult(
            self.name,
            CheckStatus.FAILED,
            f"Differs from {self.comparison_description} {fidelity_note}.",
            comparison.differences,
        )
