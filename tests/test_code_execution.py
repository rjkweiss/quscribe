"""
Tests for running generated code in a subprocess (requires Qiskit).
"""

import pytest

from quscribe.core.execution.executor import CircuitCodeExecutor
from quscribe.errors import CodeExecutionError

VALID_CODE = '''
from qiskit import QuantumCircuit


def build_circuit() -> QuantumCircuit:
    """Build a Bell circuit."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit
'''


def test_valid_code_returns_its_circuit() -> None:
    """
    The circuit built by the generated code is returned intact.

    Returns:
        None.
    """
    circuit = CircuitCodeExecutor().execute(VALID_CODE)
    assert circuit.num_qubits == 2
    assert [
        instruction.operation.name for instruction in circuit.data
    ] == ["h", "cx"]


def test_missing_entry_point_is_reported() -> None:
    """
    Code without build_circuit() fails with a message naming it.

    Returns:
        None.
    """
    with pytest.raises(CodeExecutionError, match="build_circuit"):
        CircuitCodeExecutor().execute("value = 1\n")


def test_runtime_error_is_reported_with_traceback() -> None:
    """
    An exception in the generated code is surfaced to the caller.

    Returns:
        None.
    """
    with pytest.raises(CodeExecutionError, match="ZeroDivisionError"):
        CircuitCodeExecutor().execute(
            "def build_circuit():\n    return 1 / 0\n")


def test_slow_code_times_out() -> None:
    """
    Code that runs too long is stopped.

    Returns:
        None.
    """
    slow_code = "import time\n\n\ndef build_circuit():\n    time.sleep(60)\n"
    with pytest.raises(CodeExecutionError, match="did not finish"):
        CircuitCodeExecutor(timeout_seconds=5).execute(slow_code)
