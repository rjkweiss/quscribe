"""
Circuit simulation and comparison shared by the generator,
verifier and evaluation.
"""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import NamedTuple

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, Statevector, process_fidelity

from quscribe.errors import CircuitAnalysisError

# operations that carry no unitary action;
# static circuits are compared without them.
NON_UNITARY_OPERATION_NAMES = frozenset({"barrier", "measure", "delay"})
PARAMETER_SIGNIFICANT_DIGITS = 6
DEFAULT_REFERENCE_DESCRIPTION = "the reference circuit"


class GateSignature(NamedTuple):
    """
    Hashable description of one gate application, used for gate-level diffs.

    Attributes:
        name: The gate name, e.g. ``"cx"``.
        qubits: Indices of the qubits the gate acts on, in argument order.
        parameters: Gate parameters formatted as strings.
    """

    name: str
    qubits: tuple[int, ...]
    parameters: tuple[str, ...]

    def describe(self) -> str:
        """
        Format the signature for humans and for repair prompts.

        Returns:
            str: For example ``"rz(0.785398) on q[1]"``.
        """
        parameter_text = (
            f"({', '.join(self.parameters)})" if self.parameters else ""
        )
        qubit_text = ", ".join(
            f"q[{qubit_index}]" for qubit_index in self.qubits
        )

        return f"{self.name}{parameter_text} on {qubit_text}"


@dataclass(frozen=True)
class CircuitComparison:
    """
    Result of comparing a candidate circuit with a reference circuit.

    Attributes:
        fidelity: Process fidelity between the two unitaries (0.0 to 1.0).
        is_equivalent: Whether the fidelity meets the threshold.
        differences: Gate-level differences; empty when equivalent.
    """

    fidelity: float
    is_equivalent: bool
    differences: tuple[str, ...] = ()


@contextmanager
def _analysis_errors(action_description: str) -> Iterator[None]:
    """
    Translate Qiskit simulation failures into ``CircuitAnalysisError``.

    Args:
        action_description: action being attempted, used in the error message.

    Returns:
        Iterator[None]: A context manager that yields once.

    Raises:
        CircuitAnalysisError: If the wrapped block raises a simulation error.
    """
    try:
        yield
    except (QiskitError, TypeError, ValueError) as error:
        raise CircuitAnalysisError(
            f"Could not {action_description}: {error}") from error


def strip_non_unitary_operations(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Copy a circuit without measurements, barriers, delays or classical bits.

    Args:
        circuit: The circuit to copy.

    Returns:
        QuantumCircuit: A purely unitary circuit on the same number of qubits.
    """
    unitary_circuit = QuantumCircuit(
        circuit.num_qubits,
        global_phase=circuit.global_phase
    )

    for instruction in circuit.data:
        if instruction.operation.name in NON_UNITARY_OPERATION_NAMES:
            continue

        qubit_indices = [
            circuit.find_bit(qubit).index for qubit in instruction.qubits
        ]
        unitary_circuit.append(instruction.operation, qubit_indices)

    return unitary_circuit


def simulate_statevector(circuit: QuantumCircuit) -> Statevector:
    """
    Simulate the ideal output state of circuit started in the all-zeros state.

    Args:
        circuit: The circuit to simulate; final measurements are ignored.

    Returns:
        Statevector: The output state.

    Raises:
        CircuitAnalysisError: If the circuit cannot be simulated.
    """
    with _analysis_errors("simulate the circuit"):
        return Statevector(strip_non_unitary_operations(circuit))


def unitary_fidelity(
    candidate: QuantumCircuit,
    reference: QuantumCircuit
) -> float:
    """
    Compute the process fidelity between two circuits' unitaries.

    Process fidelity ignores global phase, so equivalent circuits
    score 1.0 even when they use different gate decompositions.

    Args:
        candidate: The circuit under test.
        reference: The circuit to compare against.

    Returns:
        float: Fidelity in [0, 1]; 0.0 when the qubit counts differ.

    Raises:
        CircuitAnalysisError: If either unitary cannot be computed.
    """
    if candidate.num_qubits != reference.num_qubits:
        return 0.0

    with _analysis_errors("compute the circuit unitaries"):
        fidelity = process_fidelity(
            Operator(strip_non_unitary_operations(candidate)),
            Operator(strip_non_unitary_operations(reference)),
        )

    return min(1.0, float(fidelity))  # Clamp floating-point overshoot.


def compare_circuits(
    candidate: QuantumCircuit,
    reference: QuantumCircuit,
    fidelity_threshold: float,
    reference_description: str = DEFAULT_REFERENCE_DESCRIPTION,
) -> CircuitComparison:
    """
    Compare two circuits by unitary fidelity and explain any mismatch.

    Args:
        candidate: The circuit under test.
        reference: The circuit to compare against.
        fidelity_threshold: Minimum fidelity to count as equivalent.
        reference_description: How to name the reference in difference messages

    Returns:
        CircuitComparison: Fidelity, equivalence and gate-level differences.

    Raises:
        CircuitAnalysisError: If either unitary cannot be computed.
    """
    fidelity = unitary_fidelity(candidate, reference)
    is_equivalent = fidelity >= fidelity_threshold
    differences = () if is_equivalent else describe_gate_differences(
        candidate, reference, reference_description
    )

    return CircuitComparison(
        fidelity=fidelity,
        is_equivalent=is_equivalent,
        differences=differences
    )


def describe_gate_differences(
    candidate: QuantumCircuit,
    reference: QuantumCircuit,
    reference_description: str = DEFAULT_REFERENCE_DESCRIPTION,
) -> tuple[str, ...]:
    """
    List the gates that one circuit has and the other lacks.

    Args:
        candidate: The circuit under test.
        reference: The circuit to compare against.
        reference_description: How to name the reference in the messages.

    Returns:
        tuple[str, ...]: Human-readable differences, at least one entry.
    """
    differences: list[str] = []
    if candidate.num_qubits != reference.num_qubits:
        differences.append(
            f"The candidate uses {candidate.num_qubits} qubits but "
            f"{reference_description} "
            f"uses {reference.num_qubits}."
        )

    candidate_counts = Counter(gate_signatures(candidate))
    reference_counts = Counter(gate_signatures(reference))
    differences.extend(_format_count_differences(
        reference_counts - candidate_counts,
        f"Missing from the candidate but present in {reference_description}",
    ))

    differences.extend(_format_count_differences(
        candidate_counts - reference_counts,
        f"Present in the candidate but not in {reference_description}",
    ))

    if not differences:
        differences.append(
            f"The candidate has the same gates as {reference_description}, "
            "but in a different order."
        )
    return tuple(differences)


def gate_signatures(circuit: QuantumCircuit) -> list[GateSignature]:
    """
    List the signatures of every unitary gate in a circuit, in order.

    Args:
        circuit: The circuit to describe.

    Returns:
        list[GateSignature]: One signature per gate application.
    """
    unitary_circuit = strip_non_unitary_operations(circuit)
    return [
        GateSignature(
            name=instruction.operation.name,
            qubits=tuple(
                unitary_circuit.find_bit(qubit).index
                for qubit in instruction.qubits
            ),
            parameters=tuple(
                _format_parameter(parameter)
                for parameter in instruction.operation.params
            ),
        )

        for instruction in unitary_circuit.data
    ]


def _format_count_differences(
    signature_counts: Counter,
    prefix: str
) -> list[str]:
    """
    Format a multiset of gate signatures as difference messages.

    Args:
        signature_counts: Signatures mapped to how many times they differ.
        prefix: Sentence prefix describing the direction of the difference.

    Returns:
        list[str]: One message per distinct signature, sorted for stable output
    """
    return [
        f"{prefix}: {signature.describe()} (x{count})."
        for signature, count in sorted(signature_counts.items())
    ]


def _format_parameter(parameter: object) -> str:
    """
    Format a gate parameter so equal angles produce equal strings.

    Args:
        parameter: A numeric value or a Qiskit parameter expression.

    Returns:
        str: The rounded number, or the symbolic expression when it is unbound.
    """
    try:
        return f"{float(parameter):.{PARAMETER_SIGNIFICANT_DIGITS}g}"
    except (TypeError, ValueError):
        return str(parameter)  # Unbound symbolic parameters keep their name.
