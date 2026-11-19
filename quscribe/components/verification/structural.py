"""
Structural consistency: does the generated circuit look like the figure,
gate by gate?

Two complementary checks:
- an independent second reading of a figure, compared with candidate's unitary;
- a redrawing of the candidate, compared visually with the original figure
    by a model.
"""

from dataclasses import dataclass
from functools import cached_property

from qiskit import QuantumCircuit

from quscribe.components.figure_context import build_figure_message
from quscribe.components.parser import DiagramParser
from quscribe.components.verification.base import (
    CircuitComparisonCheck,
    VerificationCheck,
)
from quscribe.core.conventions import QUBIT_ORDER_CONVENTION
from quscribe.core.models import (
    CheckResult,
    CheckStatus,
    CircuitInput,
    GeneratedCode,
    ParsedCircuit,
)
from quscribe.core.rendering import render_circuit
from quscribe.core.text_extraction import extract_json_object
from quscribe.errors import ResponseValidationError
from quscribe.llm.base import VisionLanguageModel
from quscribe.llm.validated_generation import generate_validated

REDRAW_SYSTEM_PROMPT = """You are a meticulous reviewer who compares
quantum circuit diagrams gate by gate.
"""

REDRAW_INSTRUCTION = f"""The first image is a circuit figure from a research
paper. The second image is a Qiskit drawing of a candidate transcription of
that figure, with q[0] on the top wire. {QUBIT_ORDER_CONVENTION}

Compare the two gate by gate: gate types, target and control qubits, order,
and rotation angles. Ignore differences in visual style, colours, wire labels,
barriers, layout and line wrapping. Qiskit adds a double line at the bottom
for classical bits (labelled c, with a slash and the bit count) and an arrow
from each measurement down to it, numbered by bit. These only show where
results are stored: they are not extra wires, controls or conditions, and
figures often omit them, so never count them as differences. If the figure is
a circuit identity (two circuits joined by "="), compare the candidate with
the right-hand side only. A labelled box in the second image stands for the
figure's box or group of gates with the same label: check its label, position
and wires, because its contents are verified by another check.

Reply with a single ```json block of the form
{{"consistent": true, "discrepancies": []}}. When they differ, set "consistent"
to false and list each concrete mismatch as one short sentence.
"""


@dataclass(frozen=True)
class RedrawVerdict:
    """
    A model's judgement of whether a redrawing matches the original figure.

    Attributes:
        consistent: Whether the two diagrams show the same circuit.
        discrepancies: Concrete mismatches; empty when consistent.
    """

    consistent: bool
    discrepancies: tuple[str, ...]


def parse_redraw_verdict(response_text: str) -> RedrawVerdict:
    """
    Validate and parse the redraw-comparison reply.

    Args:
        response_text: The raw model response containing a JSON verdict.

    Returns:
        RedrawVerdict: The parsed verdict.

    Raises:
        ResponseValidationError: If the verdict is malformed or
                                self-contradictory.
    """
    payload = extract_json_object(response_text)
    consistent = payload.get("consistent")
    if not isinstance(consistent, bool):
        raise ResponseValidationError('"consistent" must be true or false.')
    raw_discrepancies = payload.get("discrepancies", [])

    if (
        not isinstance(raw_discrepancies, list)
        or not all(isinstance(item, str) for item in raw_discrepancies)
    ):
        raise ResponseValidationError(
            '"discrepancies" must be a list of strings.'
        )

    discrepancies = tuple(
        item.strip()
        for item in raw_discrepancies if item.strip()
    )

    if consistent and discrepancies:
        raise ResponseValidationError(
            'Set "consistent" to false when you list discrepancies.'
        )

    if not consistent and not discrepancies:
        raise ResponseValidationError(
            "List at least one discrepancy when the circuits differ."
        )

    return RedrawVerdict(consistent=consistent, discrepancies=discrepancies)


class IndependentReadingCheck(CircuitComparisonCheck):
    """
    Compares the candidate with a second, independently produced reading of
    the figure.
    """

    name = "structural: independent reading"
    comparison_description = "an independent reading of the figure"

    def __init__(
        self,
        independent_parser: DiagramParser,
        circuit_input: CircuitInput,
        fidelity_threshold: float,
    ) -> None:
        """
        Create the check.

        Args:
            independent_parser: A parser using a different reading strategy
                    (and optionally a different model) from the primary parser.
            circuit_input: The figure to re-read.
            fidelity_threshold: Minimum unitary fidelity to pass.
        """
        super().__init__(fidelity_threshold)
        self._independent_parser = independent_parser
        self._circuit_input = circuit_input

    @cached_property
    def independent_reading(self) -> ParsedCircuit:
        """
        Read the figure once; later repair iterations reuse the reading.

        Args:
            None.

        Returns:
            ParsedCircuit: The independent transcription.

        Raises:
            GenerationFailedError: If the reading fails
                                    (retried on the next run).
        """
        return self._independent_parser.parse(self._circuit_input)

    def _comparison_circuit(self) -> QuantumCircuit:
        """
        Return the independent reading's circuit.

        Args:
            None.

        Returns:
            QuantumCircuit: The circuit from the independent reading.
        """
        return self.independent_reading.circuit


class RedrawComparisonCheck(VerificationCheck):
    """
    Redraws the candidate and asks a model to compare it
    with the original figure.
    """

    name = "structural: redraw comparison"

    def __init__(
        self,
        judge_model: VisionLanguageModel,
        circuit_input: CircuitInput,
        use_text_context: bool,
        max_attempts: int,
    ) -> None:
        """
        Create the check.

        Args:
            judge_model: The model that compares the two diagrams.
            circuit_input: The original figure and its text.
            use_text_context: Whether to send the caption and paper text.
            max_attempts: Model calls allowed per comparison.
        """
        self._judge_model = judge_model
        self._circuit_input = circuit_input
        self._use_text_context = use_text_context
        self._max_attempts = max_attempts

    def run(self, candidate: GeneratedCode) -> CheckResult:
        """
        Render the candidate and compare it with the figure.

        Args:
            candidate: The generated code and its circuit.

        Returns:
            CheckResult: PASSED if the model finds the diagrams consistent,
                        else FAILED.

        Raises:
            RenderingError: If the candidate cannot be drawn.
            GenerationFailedError: If the model gives no valid verdict.
        """
        redrawn_circuit = render_circuit(candidate.circuit)

        message = build_figure_message(
            self._circuit_input,
            self._use_text_context,
            REDRAW_INSTRUCTION,
            extra_images=(redrawn_circuit,)
        )

        verdict = generate_validated(
            self._judge_model,
            REDRAW_SYSTEM_PROMPT,
            message,
            parse_redraw_verdict,
            self._max_attempts
        )

        if verdict.consistent:
            return CheckResult(
                self.name,
                CheckStatus.PASSED,
                "The redrawn circuit matches the figure."
            )

        return CheckResult(
            self.name,
            CheckStatus.FAILED,
            "The redrawn circuit differs from the figure.",
            verdict.discrepancies
        )
