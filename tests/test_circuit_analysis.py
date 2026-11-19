"""
Tests for unitary comparison and gate-level diffs (requires Qiskit).
"""

import pytest
from qiskit import QuantumCircuit

from quscribe.core.circuit_analysis import compare_circuits, unitary_fidelity

THRESHOLD = 0.99


def bell_circuit() -> QuantumCircuit:
    """
    Build the Bell-state preparation circuit.

    Returns:
        QuantumCircuit: H on q[0] followed by CX from q[0] to q[1].
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def test_different_decompositions_are_equivalent() -> None:
    """
    CZ equals H-CX-H on the target, despite different gate lists.

    Returns:
        None.
    """
    controlled_z = QuantumCircuit(2)
    controlled_z.cz(0, 1)
    decomposed = QuantumCircuit(2)
    decomposed.h(1)
    decomposed.cx(0, 1)
    decomposed.h(1)
    assert compare_circuits(decomposed, controlled_z, THRESHOLD).is_equivalent


def test_missing_gate_is_reported_by_name_and_qubits() -> None:
    """
    A dropped CX is named, with its qubits, in the differences.

    Returns:
        None.
    """
    missing_cx = QuantumCircuit(2)
    missing_cx.h(0)
    comparison = compare_circuits(
        missing_cx, bell_circuit(), THRESHOLD, "the figure"
    )
    assert not comparison.is_equivalent
    assert any(
        "cx on q[0], q[1]" in difference and "Missing" in difference
        for difference in comparison.differences
    )


def test_reversed_control_is_detected() -> None:
    """
    Swapping control and target changes the unitary.

    Returns:
        None.
    """
    reversed_control = QuantumCircuit(2)
    reversed_control.h(0)
    reversed_control.cx(1, 0)
    assert not compare_circuits(
        reversed_control, bell_circuit(), THRESHOLD).is_equivalent


def test_measurements_are_ignored_and_qubit_mismatch_scores_zero() -> None:
    """
    Final measurements do not affect fidelity; differing widths score 0.

    Returns:
        None.
    """
    measured = bell_circuit()
    measured.measure_all()
    assert unitary_fidelity(measured, bell_circuit()) == pytest.approx(1.0)
    assert unitary_fidelity(QuantumCircuit(3), bell_circuit()) == 0.0
