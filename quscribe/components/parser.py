"""
Diagram Parser: reads a circuit figure and transcribes it into OpenQASM 3.
"""

from quscribe.components.figure_context import build_figure_message
from quscribe.core.conventions import QASM_FORMAT_RULES
from quscribe.core.models import CircuitInput, ParsedCircuit
from quscribe.core.qasm import parse_qasm_response
from quscribe.llm.base import VisionLanguageModel
from quscribe.llm.validated_generation import generate_validated

PARSER_SYSTEM_PROMPT = (
    "You are an expert in quantum computing who transcribes quantum circuit "
    f"diagrams from research papers into OpenQASM 3.\n\n{QASM_FORMAT_RULES}"
)

PRIMARY_READING_INSTRUCTION = (
    "Transcribe the quantum circuit shown in the figure. Scan it from left to "
    "right one time step at a time, and for each step record every gate, its "
    "target and control  qubits, and its parameters."
)

# A different reading strategy makes the verifier's second
# reading genuinely independent.
INDEPENDENT_READING_INSTRUCTION = (
    "Transcribe the quantum circuit shown in the figure. Read it wire by wire "
    "from the top: first list every element that touches each wire, then "
    "assemble the  gates in time order, checking which dots are controls and "
    "which symbols are targets."
)


class DiagramParser:
    """
    Transcribes figures into OpenQASM 3, retrying with feedback
    on invalid programs.
    """

    def __init__(
        self,
        model: VisionLanguageModel,
        reading_instruction: str = PRIMARY_READING_INSTRUCTION,
        use_text_context: bool = True,
        max_attempts: int = 3,
    ) -> None:
        """
        Create a parser.

        Args:
            model: The vision-language model that reads the figure.
            reading_instruction: How to read the figure - the verifier uses
                                a different one.
            use_text_context: Whether to send the caption and paper text.
            max_attempts: Model calls allowed per transcription.
        """
        self._model = model
        self._reading_instruction = reading_instruction
        self._use_text_context = use_text_context
        self._max_attempts = max_attempts

    def parse(self, circuit_input: CircuitInput) -> ParsedCircuit:
        """
        Transcribe a figure using this parser's reading instruction.

        Args:
            circuit_input: The figure and its accompanying text.

        Returns:
            ParsedCircuit: The validated OpenQASM 3 transcription.

        Raises:
            GenerationFailedError: If no valid program is produced.
        """
        return self.transcribe(circuit_input, self._reading_instruction)

    def transcribe(
        self,
        circuit_input: CircuitInput,
        instruction: str
    ) -> ParsedCircuit:
        """
        Transcribe a figure following an arbitrary instruction.

        Repair Agent reuses the same prompt framing and validation.

        Args:
            circuit_input: The figure and its accompanying text.
            instruction: The task description for this transcription.

        Returns:
            ParsedCircuit: The validated OpenQASM 3 transcription.

        Raises:
            GenerationFailedError: If no valid program is produced.
        """
        message = build_figure_message(
            circuit_input,
            self._use_text_context,
            instruction
        )

        return generate_validated(
            self._model,
            PARSER_SYSTEM_PROMPT,
            message,
            parse_qasm_response,
            self._max_attempts
        )
