"""
Code Generator: translates the OpenQASM 3 transcription
into runnable Qiskit code.

Because the input is already a structured gate list,
translation faithfulness can be checked exactly: the generated code is
executed and its circuit must match the transcription's unitary.
Mismatches and runtime errors go back to the model as feedback.
"""

from functools import partial

from quscribe.core.circuit_analysis import compare_circuits
from quscribe.core.execution.circuit_runner import ENTRY_POINT_NAME
from quscribe.core.execution.executor import CircuitCodeExecutor
from quscribe.core.models import GeneratedCode, ParsedCircuit
from quscribe.core.text_extraction import extract_fenced_block
from quscribe.errors import (
    CircuitAnalysisError,
    CodeExecutionError,
    ResponseValidationError,
)
from quscribe.llm.base import ChatMessage, MessageRole, VisionLanguageModel
from quscribe.llm.validated_generation import generate_validated

PYTHON_BLOCK_LANGUAGES = ("python", "py")
IR_DESCRIPTION = "the OpenQASM input"

GENERATOR_SYSTEM_PROMPT = f"""You are an expert Qiskit engineer. You translate
OpenQASM 3 programs into Python code for the current Qiskit release.

Rules:
- Write one self-contained Python module that defines
    `def {ENTRY_POINT_NAME}() -> QuantumCircuit:`, taking no arguments and
    returning the circuit.
- Import only from qiskit, math and numpy.
- Reproduce every gate exactly: the same qubits, order, angles
    and measurements.
- Build any custom gate declared in the program as a QuantumCircuit
    named after it (e.g. `QuantumCircuit(4, name="U_f")`) and convert it
    with `.to_gate()`, so drawings show the same labelled box.
- Do not simulate, transpile, draw, print, read files or access the network,
    and write no code outside imports, constants and function definitions.
- Use descriptive variable names such as `qubits`, `classical_bits` and
    `circuit`; never single letters.
- Give the function a short docstring and add brief comments that describe
    what the gates do. Do not name or guess the algorithm: the program does
    not say which one it is.
- Return the module in a single fenced block that starts with ```python.
"""

GENERATION_INSTRUCTION_TEMPLATE = """Translate this OpenQASM 3 program into
Qiskit code:\n```qasm\n{qasm}\n```
"""


class CodeGenerator:
    """
    Generates Qiskit code and accepts it only if it reproduces the
    transcription.
    """

    def __init__(
        self,
        model: VisionLanguageModel,
        executor: CircuitCodeExecutor,
        fidelity_threshold: float,
        max_attempts: int = 3,
    ) -> None:
        """
        Create a generator.

        Args:
            model: The model that writes the code.
            executor: Runs generated code in a separate process.
            fidelity_threshold: Minimum unitary fidelity to the transcription.
            max_attempts: Model calls allowed per generation.
        """
        self._model = model
        self._executor = executor
        self._fidelity_threshold = fidelity_threshold
        self._max_attempts = max_attempts

    def generate(self, parsed: ParsedCircuit) -> GeneratedCode:
        """
        Generate verified-faithful Qiskit code for a transcription.

        Args:
            parsed: The OpenQASM 3 transcription to translate.

        Returns:
            GeneratedCode: Code that runs and reproduces the transcription.

        Raises:
            GenerationFailedError: If no faithful code is produced.
        """
        message = ChatMessage(
            role=MessageRole.USER,
            text=GENERATION_INSTRUCTION_TEMPLATE.format(qasm=parsed.qasm),
        )

        return generate_validated(
            self._model,
            GENERATOR_SYSTEM_PROMPT,
            message,
            partial(self._validate_response, target=parsed),
            self._max_attempts,
        )

    def _validate_response(
        self,
        response_text: str,
        target: ParsedCircuit
    ) -> GeneratedCode:
        """
        Run the generated code and check it against the transcription.

        Args:
            response_text: The raw model response containing a ```python block.
            target: The transcription the code must reproduce.

        Returns:
            GeneratedCode: The accepted code and its circuit.

        Raises:
            ResponseValidationError: If the code fails to run or
                                    builds a different circuit.
        """
        source_code = extract_fenced_block(
            response_text,
            PYTHON_BLOCK_LANGUAGES
        )

        try:
            circuit = self._executor.execute(source_code)
            comparison = compare_circuits(
                circuit,
                target.circuit,
                self._fidelity_threshold,
                IR_DESCRIPTION
            )
        except CodeExecutionError as error:
            raise ResponseValidationError(
                f"The code failed to run:\n{error}") from error
        except CircuitAnalysisError as error:
            raise ResponseValidationError(
                f"The circuit could not be analyzed: {error}") from error
        if not comparison.is_equivalent:
            raise ResponseValidationError(
                "The code builds a different circuit "
                f"(fidelity {comparison.fidelity:.4f}). "
                + " ".join(comparison.differences)
            )

        return GeneratedCode(source_code=source_code, circuit=circuit)
