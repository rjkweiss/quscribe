"""
Tests for drawing named gates with their contents (requires Qiskit).
"""

from qiskit import QuantumCircuit

from quscribe.core.circuit_analysis import unitary_fidelity
from quscribe.core.rendering import expand_named_gates, render_circuit


def circuit_with_oracle() -> QuantumCircuit:
    """
    Build H gates around a two-CNOT oracle named U_f, as in a
    Bernstein-Vazirani figure.

    Returns:
        QuantumCircuit: A 4-qubit circuit containing one named gate.
    """
    oracle = QuantumCircuit(4, name="U_f")
    oracle.cx(0, 3)
    oracle.cx(2, 3)
    circuit = QuantumCircuit(4)
    circuit.h([0, 1, 2])
    circuit.append(oracle.to_gate(), [0, 1, 2, 3])
    circuit.h([0, 1, 2])
    return circuit


def test_named_gate_is_expanded_between_labelled_barriers() -> None:
    """
    The oracle's CNOTs appear between barriers, the first labelled U_f.
    """
    expanded = expand_named_gates(circuit_with_oracle())
    names = [instruction.operation.name for instruction in expanded.data]
    assert names == [
        "h", "h", "h", "barrier", "cx", "cx", "barrier", "h", "h", "h"
    ]
    assert expanded.data[3].operation.label == "U_f"
    cnot_wires = [
        tuple(expanded.find_bit(qubit).index for qubit in instruction.qubits)
        for instruction in expanded.data
        if instruction.operation.name == "cx"
    ]
    assert cnot_wires == [(0, 3), (2, 3)]


def test_expansion_keeps_the_circuit_equivalent_and_drawable() -> None:
    """
    Expanding for display never changes what the circuit does.

    Args:
        None.

    Returns:
        None.
    """
    original = circuit_with_oracle()
    assert unitary_fidelity(expand_named_gates(original), original) > 0.999999
    assert render_circuit(original, show_named_gate_contents=True).png_bytes
