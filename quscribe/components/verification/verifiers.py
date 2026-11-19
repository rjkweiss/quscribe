"""
The two verifiers: reference-free (the contribution) and oracle
(the upper bound).

Both implement the same ``Verifier`` interface, so the pipeline and repair
loop are identical in every ablation; only the source of discrepancies changes.
"""

from qiskit import QuantumCircuit

from quscribe.components.parser import DiagramParser
from quscribe.components.verification.base import (
    CircuitComparisonCheck,
    VerificationSuite,
    Verifier,
)
from quscribe.components.verification.behavioral import (
    PaperClaimExtractor,
    PaperClaimsCheck,
)
from quscribe.components.verification.structural import (
    IndependentReadingCheck,
    RedrawComparisonCheck,
)
from quscribe.core.models import CircuitInput
from quscribe.llm.base import VisionLanguageModel


class ReferenceFreeVerifier(Verifier):
    """
    Verifies candidates using only the figure and the paper text.
    """

    def __init__(
        self,
        independent_parser: DiagramParser,
        judge_model: VisionLanguageModel,
        claim_extractor: PaperClaimExtractor,
        use_text_context: bool,
        fidelity_threshold: float,
        max_attempts: int,
    ) -> None:
        """
        Create the verifier.

        Args:
            independent_parser: Produces the second, independent reading of
                                the figure.
            judge_model: Compares redrawn candidates with the figure.
            claim_extractor: Extracts testable claims from the paper text.
            use_text_context: Whether checks may use the caption and paper
                                text.
            fidelity_threshold: Minimum unitary fidelity for structural
                                agreement.
            max_attempts: Model calls allowed per check request.
        """
        self._independent_parser = independent_parser
        self._judge_model = judge_model
        self._claim_extractor = claim_extractor
        self._use_text_context = use_text_context
        self._fidelity_threshold = fidelity_threshold
        self._max_attempts = max_attempts

    def create_suite(self, circuit_input: CircuitInput) -> VerificationSuite:
        """
        Create the structural and behavioral checks for one figure.

        Args:
            circuit_input: The figure and its accompanying text.

        Returns:
            VerificationSuite: Independent reading, redraw comparison
                                and paper claims.
        """
        return VerificationSuite((
            IndependentReadingCheck(
                self._independent_parser,
                circuit_input,
                self._fidelity_threshold
            ),
            RedrawComparisonCheck(
                self._judge_model,
                circuit_input,
                self._use_text_context,
                self._max_attempts
            ),
            PaperClaimsCheck(
                self._claim_extractor,
                circuit_input,
                self._use_text_context
            ),
        ))


class ReferenceCircuitCheck(CircuitComparisonCheck):
    """
    Compares the candidate with the true reference circuit (evaluation only).
    """

    name = "oracle: reference circuit"
    comparison_description = "the reference circuit"

    def __init__(
        self,
        reference_circuit: QuantumCircuit,
        fidelity_threshold: float
    ) -> None:
        """
        Create the check.

        Args:
            reference_circuit: The ground-truth circuit.
            fidelity_threshold: Minimum unitary fidelity to pass.
        """
        super().__init__(fidelity_threshold)
        self._reference_circuit = reference_circuit

    def _comparison_circuit(self) -> QuantumCircuit:
        """
        Return the ground-truth circuit.

        Args:
            None.

        Returns:
            QuantumCircuit: The reference circuit.
        """
        return self._reference_circuit


class OracleVerifier(Verifier):
    """
    Verifies against the true reference, giving the upper bound for repair.
    """

    def __init__(
        self,
        reference_circuit: QuantumCircuit,
        fidelity_threshold: float
    ) -> None:
        """
        Create the verifier for one benchmark item.

        Args:
            reference_circuit: The ground-truth circuit for the item.
            fidelity_threshold: Minimum unitary fidelity to pass.
        """
        self._reference_circuit = reference_circuit
        self._fidelity_threshold = fidelity_threshold

    def create_suite(self, circuit_input: CircuitInput) -> VerificationSuite:
        """
        Create the single reference-comparison check.

        Args:
            circuit_input: Unused; the oracle already knows the answer.

        Returns:
            VerificationSuite: A suite with one reference-comparison check.
        """
        return VerificationSuite(
            (ReferenceCircuitCheck(
                self._reference_circuit,
                self._fidelity_threshold
            ),)
        )
