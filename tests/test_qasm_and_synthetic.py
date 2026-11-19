"""
Tests for QASM parsing and the synthetic dataset generator (requires Qiskit).
"""

from pathlib import Path

import pytest

from quscribe.core.conventions import UNDERSPECIFIED_MARKER
from quscribe.core.qasm import parse_qasm_response
from quscribe.core.rendering import DiagramStyle
from quscribe.errors import ResponseValidationError, UnderspecifiedCircuitError
from quscribe.evaluation.dataset import MANIFEST_FILENAME, load_manifest
from quscribe.evaluation.synthetic import (
    SyntheticDatasetConfig,
    SyntheticDatasetGenerator,
)

BELL_QASM = (
    'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
    'h q[0];\ncx q[0], q[1];'
)


def test_valid_qasm_response_is_parsed() -> None:
    """
    A fenced OpenQASM 3 program loads into a circuit.

    Args:
        None.

    Returns:
        None.
    """
    parsed = parse_qasm_response(f"```qasm\n{BELL_QASM}\n```")
    assert parsed.circuit.num_qubits == 2


def test_invalid_qasm_response_raises_feedback_error() -> None:
    """
    An unknown gate produces feedback for the model.

    Args:
        None.

    Returns:
        None.
    """
    broken_program = BELL_QASM.replace("h q[0];", "notagate q[0];")
    with pytest.raises(ResponseValidationError, match="Invalid OpenQASM 3"):
        parse_qasm_response(f"```qasm\n{broken_program}\n```")


def test_underspecified_reply_raises_with_reason() -> None:
    """
    A reply starting a line with the marker reports what the figure leaves out.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(UnderspecifiedCircuitError, match="oracle contents"):
        parse_qasm_response(
            f"I cannot transcribe this.\n{UNDERSPECIFIED_MARKER} "
            "oracle contents not shown"
        )


def test_synthetic_dataset_round_trips_through_manifest(
    tmp_path: Path
) -> None:
    """
    Generated items are written as images plus a manifest that loads back.

    Args:
        tmp_path: Pytest's temporary directory.

    Returns:
        None.
    """
    styles = (
        DiagramStyle.IQP,
        DiagramStyle.TEXT,
        DiagramStyle.BLACK_AND_WHITE
    )

    config = SyntheticDatasetConfig(
        circuit_count=3,
        max_qubits=3,
        max_gates=6,
        styles=styles,
        seed=7,
    )

    generated_items = SyntheticDatasetGenerator(config).generate(tmp_path)
    loaded_items = load_manifest(tmp_path / MANIFEST_FILENAME)
    assert [
        item.item_id for item in loaded_items
    ] == [item.item_id for item in generated_items]

    assert [
        item.style for item in loaded_items
    ] == [style.value for style in styles]

    for item in loaded_items:
        assert item.image_path.is_file()
        assert 2 <= item.reference_circuit().num_qubits <= 3
