"""
Tests for paper-claim parsing and evaluation (requires Qiskit).
"""

import json

import pytest
from qiskit import QuantumCircuit

from quscribe.components.verification.behavioral import (
    ClaimTolerances,
    parse_claims,
)
from quscribe.core.circuit_analysis import simulate_statevector
from quscribe.errors import ResponseValidationError

TOLERANCES = ClaimTolerances()


def claims_response(*claims: dict) -> str:
    """
    Wrap claim dictionaries in a fenced JSON reply.

    Args:
        *claims: The claim entries.

    Returns:
        str: A model-style reply.
    """
    return "```json\n" + json.dumps({"claims": list(claims)}) + "\n```"


def evaluate_single_claim(circuit: QuantumCircuit, claim: dict) -> bool:
    """
    Parse one claim and evaluate it on a circuit.

    Args:
        circuit: The circuit to simulate.
        claim: The claim entry.

    Returns:
        bool: Whether the claim holds.
    """
    (parsed_claim,) = parse_claims(claims_response(claim), TOLERANCES)
    return parsed_claim.evaluate(simulate_statevector(circuit)).passed


def bell_circuit() -> QuantumCircuit:
    """
    Build the Bell-state preparation circuit.

    Args:
        None.

    Returns:
        QuantumCircuit: H on q[0] followed by CX from q[0] to q[1].
    """
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def test_bell_distribution_and_state_claims_hold() -> None:
    """
    Correct distribution and output-state claims pass on a Bell circuit.

    Args:
        None.

    Returns:
        None.
    """
    assert evaluate_single_claim(bell_circuit(), {
        "kind": "probability_distribution",
        "description": "Bell",
        "probabilities": {"00": 0.5, "11": 0.5},
    })

    assert evaluate_single_claim(bell_circuit(), {
        "kind": "output_state",
        "description": "Bell",
        "amplitudes": {"00": [0.7071, 0], "11": [0.7071, 0]},
    })


def test_wrong_relative_phase_fails_state_claim() -> None:
    """
    A minus sign on |11> makes an orthogonal state, so the claim fails.

    Args:
        None.

    Returns:
        None.
    """
    assert not evaluate_single_claim(bell_circuit(), {
        "kind": "output_state",
        "description": "Bell",
        "amplitudes": {"00": 0.7071, "11": -0.7071},
    })


def test_bitstrings_put_qubit_zero_leftmost() -> None:
    """
    X on q[0] gives "10" in the conventional order, not Qiskit's "01".

    Args:
        None.

    Returns:
        None.
    """
    flip_first_qubit = QuantumCircuit(2)
    flip_first_qubit.x(0)
    assert evaluate_single_claim(
        flip_first_qubit,
        {"kind": "probability_distribution", "probabilities": {"10": 1.0}}
    )

    assert not evaluate_single_claim(
        flip_first_qubit,
        {"kind": "probability_distribution", "probabilities": {"01": 1.0}}
    )


def test_marginal_distribution_on_listed_qubits() -> None:
    """
    A claim about a subset of qubits uses the marginal distribution.

    Args:
        None.

    Returns:
        None.
    """
    flip_last_qubit = QuantumCircuit(3)
    flip_last_qubit.x(2)
    assert evaluate_single_claim(flip_last_qubit, {
        "kind": "probability_distribution",
        "qubits": [2, 0],
        "probabilities": {"10": 1.0},
    })


def test_qubit_count_claim_and_unknown_kinds() -> None:
    """
    Qubit counts are compared; unsupported kinds are skipped.

    Args:
        None.

    Returns:
        None.
    """
    claims = parse_claims(claims_response(
        {"kind": "qubit_count", "value": 3}, {"kind": "gate_count", "value": 7}
    ), TOLERANCES)
    assert len(claims) == 1
    assert not claims[0].evaluate(simulate_statevector(bell_circuit())).passed


def test_malformed_claim_raises_feedback_error() -> None:
    """
    A non-binary bitstring is rejected so the model can fix it.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(ResponseValidationError, match="0s and 1s"):
        parse_claims(
            claims_response(
                {
                    "kind": "probability_distribution",
                    "probabilities": {"0a": 1.0}
                }
            ),
            TOLERANCES
        )
