"""
Loading and exporting the OpenQASM 3 intermediate representation.
"""
import re

from qiskit import QuantumCircuit, qasm3

from quscribe.core.conventions import UNDERSPECIFIED_MARKER
from quscribe.core.models import ParsedCircuit
from quscribe.core.text_extraction import extract_fenced_block
from quscribe.errors import (
    QasmParseError,
    ResponseValidationError,
    UnderspecifiedCircuitError,
)

QASM_BLOCK_LANGUAGES = ("qasm", "openqasm", "qasm3")
# The marker must start a line, so a passing mention inside prose does
# not trigger it.
UNDERSPECIFIED_PATTERN = re.compile(
    rf"^\s*{re.escape(UNDERSPECIFIED_MARKER)}\s*(.*)$", re.MULTILINE
)


def load_qasm(qasm_text: str) -> QuantumCircuit:
    """
    Load an OpenQASM 3 program into a Qiskit circuit.

    Args:
        qasm_text: The OpenQASM 3 source.

    Returns:
        QuantumCircuit: The loaded circuit.

    Raises:
        QasmParseError: If the program is invalid or unsupported.
        ImportError: If the optional ``qiskit-qasm3-import`` package is missing
    """
    try:
        return qasm3.loads(qasm_text)
    except ImportError:
        raise  # missing import, not correctness issue
    except Exception as error:  # importer and parser raise different errors.
        raise QasmParseError(f"Invalid OpenQASM 3: {error}") from error


def dump_qasm(circuit: QuantumCircuit) -> str:
    """Export a circuit as OpenQASM 3.

    Args:
        circuit: The circuit to export.

    Returns:
        str: The OpenQASM 3 source.
    """
    return qasm3.dumps(circuit)


def parse_qasm_response(response_text: str) -> ParsedCircuit:
    """
    Extract and load the OpenQASM 3 program from a model response.

    Used as the validator for every stage that produces a transcription, so a
    malformed program is fed back to the model instead
    of propagating downstream.

    Args:
        response_text: The raw model response containing a ```qasm block.

    Returns:
        ParsedCircuit: The program and its loaded circuit.

    Raises:
        UnderspecifiedCircuitError: If the model reports that the figure does
            not determine a unique circuit. This is not retried: asking again
            would only pressure the model into inventing the missing part.

        ResponseValidationError: If the block is missing, invalid, or
            has no qubits.
    """
    underspecified_match = UNDERSPECIFIED_PATTERN.search(response_text)
    if underspecified_match:
        missing_part = (
            underspecified_match.group(1).strip() or "no details given"
        )

        raise UnderspecifiedCircuitError(
            f"The figure does not fully specify the circuit: {missing_part}"
        )

    # get fenced qasm block from raw model response
    qasm_text = extract_fenced_block(response_text, QASM_BLOCK_LANGUAGES)

    try:
        circuit = load_qasm(qasm_text)
    except QasmParseError as error:
        raise ResponseValidationError(
            f"{error}. Return a corrected program that follows "
            f"the formatting rules."
        ) from error

    # circuit must have at least 1 qubit
    if circuit.num_qubits == 0:
        raise ResponseValidationError("The program declares no qubits.")

    return ParsedCircuit(qasm=qasm_text, circuit=circuit)
