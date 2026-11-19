"""
Behavioral consistency: test claims stated in the paper by
simulating the circuit.

A model extracts claims (qubit count, ideal measurement distribution,
output state) from the caption and paper text once per figure; each candidate
is then simulated from the all-zeros state and every claim is evaluated
against the simulation.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property, partial
from operator import itemgetter
from typing import Any, TypeVar

import numpy
from qiskit.quantum_info import Statevector, state_fidelity

from quscribe.components.figure_context import describe_text_context
from quscribe.components.verification.base import VerificationCheck
from quscribe.core.circuit_analysis import simulate_statevector
from quscribe.core.conventions import (
    BITSTRING_CONVENTION,
    QUBIT_ORDER_CONVENTION,
    swap_bitstring_endianness,
)
from quscribe.core.models import (
    CheckResult,
    CheckStatus,
    CircuitInput,
    GeneratedCode,
)
from quscribe.core.text_extraction import extract_json_object
from quscribe.errors import ResponseValidationError
from quscribe.llm.base import ChatMessage, MessageRole, VisionLanguageModel
from quscribe.llm.validated_generation import generate_validated

logger = logging.getLogger(__name__)

ValueType = TypeVar("ValueType")

BINARY_DIGITS = frozenset("01")
MAX_DISPLAYED_OUTCOMES = 8
NEGLIGIBLE_PROBABILITY = 1e-9


CLAIMS_SYSTEM_PROMPT = f"""
You extract testable claims about a quantum circuit from research paper: from
the caption and text you are given, and from any caption or annotation shown
inside the figure image.

Extract only claims about the ideal, noiseless behavior of the circuit in
the figure when it starts in the all-zeros state. Ignore experimental
or hardware results, claims that need a different input state, and
statements about complexity or resources other than the number of qubits.

{BITSTRING_CONVENTION} {QUBIT_ORDER_CONVENTION}

Reply with a single ```json block of the form:
{{"claims": [
  {{"kind": "qubit_count", "description": "...", "value": 3}},
    {{"kind": "probability_distribution", "description": "...",
        "qubits": [0, 1], "probabilities": {{"00": 0.5, "11": 0.5}}}},
    {{"kind": "output_state", "description": "...",
        "amplitudes": {{"000": [0.7071, 0.0], "111": [0.7071, 0.0]}}}}
]}}

Omit "qubits" when a distribution covers every qubit. Give amplitudes over all
qubits as [real, imaginary] pairs. Return {{"claims": []}} when neither the
figure nor the text states such a claim.
"""

CLAIMS_INSTRUCTION = (
    "Extract the testable claims about this circuit from the figure, including"
    " any caption or annotation drawn inside it, and from the text below."
)

NO_CLAIMS_WITH_TEXT_SUMMARY = (
    "No testable claims were found in the caption, "
    "the paper text or the figure."
)

NO_CLAIMS_WITHOUT_TEXT_SUMMARY = (
    "No caption or paper text was provided, and the figure "
    "itself states no testable claims."
)


@dataclass(frozen=True)
class ClaimTolerances:
    """How closely the simulation must match each kind of claim.

    Attributes:
        probability_tolerance: Maximum total variation distance for
                                distributions.
        state_fidelity_threshold: Minimum state fidelity for output-state
                                claims.
    """

    probability_tolerance: float = 0.05
    state_fidelity_threshold: float = 0.99


@dataclass(frozen=True)
class ClaimOutcome:
    """
    Result of testing one claim.

    Attributes:
        passed: Whether the simulation agrees with the claim.
        message: Explanation, phrased for the repair agent.
    """

    passed: bool
    message: str


@dataclass(frozen=True)
class PaperClaim(ABC):
    """
    A testable statement from the paper about the circuit's ideal behavior.

    Attributes:
        description: The claim as described in the paper.
    """

    description: str

    @classmethod
    @abstractmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        tolerances: ClaimTolerances,
    ) -> "PaperClaim":
        """
        Build a claim from its JSON representation.

        Args:
            payload: One entry of the extracted ``claims`` list.
            tolerances: Matching tolerances to attach to the claim.

        Returns:
            PaperClaim: The validated claim.

        Raises:
            ResponseValidationError: If the payload is malformed.
        """

    @abstractmethod
    def evaluate(self, statevector: Statevector) -> ClaimOutcome:
        """
        Test the claim against a simulated output state.

        Args:
            statevector: The candidate's ideal output state.

        Returns:
            ClaimOutcome: Whether the claim holds, with an explanation.
        """


@dataclass(frozen=True)
class QubitCountClaim(PaperClaim):
    """
    The paper states how many qubits the circuit uses.

    Attributes:
        expected_qubit_count: The stated number of qubits.
    """

    expected_qubit_count: int

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        tolerances: ClaimTolerances
    ) -> "QubitCountClaim":
        """
        Build the claim from JSON.

        Args:
            payload: The claim entry, with an integer ``value``.
            tolerances: Unused; accepted for a uniform construction interface.

        Returns:
            QubitCountClaim: The validated claim.

        Raises:
            ResponseValidationError: If ``value`` is not a positive integer.
        """
        value = payload.get("value")
        if not _is_integer(value) or value < 1:
            raise ResponseValidationError(
                'A qubit_count claim needs a positive integer "value".'
            )
        return cls(
            description=_description_of(payload),
            expected_qubit_count=value
        )

    def evaluate(self, statevector: Statevector) -> ClaimOutcome:
        """
        Compare the stated and actual qubit counts.

        Args:
            statevector: The candidate's ideal output state.

        Returns:
            ClaimOutcome: Whether the counts agree.
        """
        actual_qubit_count = statevector.num_qubits
        return ClaimOutcome(
            passed=actual_qubit_count == self.expected_qubit_count,
            message=(
                f"{self.description}: the paper states "
                f"{self.expected_qubit_count} qubits "
                f"and the circuit has {actual_qubit_count}."
            ),
        )


@dataclass(frozen=True)
class ProbabilityDistributionClaim(PaperClaim):
    """
    The paper states the ideal measurement distribution over some or
    all qubits.

    Attributes:
        probabilities: Bitstrings (first listed qubit leftmost) mapped to
                        probabilities.
        qubit_indices: The measured qubits, or None for all qubits in order.
        tolerance: Maximum total variation distance to pass.
    """

    probabilities: dict[str, float]
    qubit_indices: tuple[int, ...] | None
    tolerance: float

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], tolerances: ClaimTolerances
    ) -> "ProbabilityDistributionClaim":
        """
        Build the claim from JSON.

        Args:
            payload: The claim entry, with ``probabilities`` and optional
                    ``qubits``.
            tolerances: Supplies the distribution tolerance.

        Returns:
            ProbabilityDistributionClaim: The validated claim.

        Raises:
            ResponseValidationError: If the payload is malformed.
        """
        probabilities = _parse_bitstring_mapping(
            payload.get("probabilities"),
            _parse_probability,
            "probabilities"
        )

        qubit_indices = _parse_qubit_indices(payload.get("qubits"))
        if (
            qubit_indices is not None
            and len(qubit_indices) != _common_bitstring_length(probabilities)
        ):
            raise ResponseValidationError(
                'Each bitstring needs one character per qubit listed in '
                '"qubits".'
            )
        return cls(
            description=_description_of(payload),
            probabilities=probabilities,
            qubit_indices=qubit_indices,
            tolerance=tolerances.probability_tolerance,
        )

    def evaluate(self, statevector: Statevector) -> ClaimOutcome:
        """
        Compare the stated distribution with the simulated one.

        Args:
            statevector: The candidate's ideal output state.

        Returns:
            ClaimOutcome: Whether the total variation distance is
                        within tolerance.
        """
        qubit_count = statevector.num_qubits
        qubit_indices = (
            self.qubit_indices
            if self.qubit_indices is not None else tuple(range(qubit_count))
        )

        if any(qubit_index >= qubit_count for qubit_index in qubit_indices):
            return ClaimOutcome(False, (
                f"{self.description}: the claim refers to qubits "
                f"{list(qubit_indices)} but the circuit has only "
                f"{qubit_count} qubits."
            ))

        outcome_length = _common_bitstring_length(self.probabilities)

        if outcome_length != len(qubit_indices):
            return ClaimOutcome(False, (
                f"{self.description}: the claim describes "
                f"{outcome_length}-qubit outcomes but the circuit "
                f"has {qubit_count} qubits."
            ))

        simulated = _simulated_distribution(statevector, qubit_indices)
        outcomes = set(simulated) | set(self.probabilities)
        distance = 0.5 * sum(
            abs(
                simulated.get(outcome, 0.0)
                - self.probabilities.get(outcome, 0.0)
            )
            for outcome in outcomes
        )
        return ClaimOutcome(
            passed=distance <= self.tolerance,
            message=(
                f"{self.description}: the paper gives "
                f"{_format_distribution(self.probabilities)} and "
                f"simulation gives {_format_distribution(simulated)} "
                f"(total variation distance {distance:.3f}, "
                f"tolerance {self.tolerance:g})."
            ),
        )


@dataclass(frozen=True)
class OutputStateClaim(PaperClaim):
    """
    The paper states the ideal output state of the circuit.

    Attributes:
        amplitudes: Bitstrings (q[0] leftmost) mapped to amplitudes;
                    normalized on use.
        fidelity_threshold: Minimum state fidelity to pass.
    """

    amplitudes: dict[str, complex]
    fidelity_threshold: float

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        tolerances: ClaimTolerances
    ) -> "OutputStateClaim":
        """
        Build the claim from JSON.

        Args:
            payload: The claim entry, with ``amplitudes`` over all qubits.
            tolerances: Supplies the state fidelity threshold.

        Returns:
            OutputStateClaim: The validated claim.

        Raises:
            ResponseValidationError: If the payload is malformed or all
                                    amplitudes are zero.
        """
        amplitudes = _parse_bitstring_mapping(
            payload.get("amplitudes"),
            _parse_amplitude,
            "amplitudes"
        )
        _common_bitstring_length(amplitudes)
        if all(amplitude == 0 for amplitude in amplitudes.values()):
            raise ResponseValidationError(
                "An output_state claim needs at least one non-zero amplitude.")

        return cls(
            description=_description_of(payload),
            amplitudes=amplitudes,
            fidelity_threshold=tolerances.state_fidelity_threshold,
        )

    def evaluate(self, statevector: Statevector) -> ClaimOutcome:
        """
        Compare the stated state with the simulated state
        (global phase ignored).

        Args:
            statevector: The candidate's ideal output state.

        Returns:
            ClaimOutcome: Whether the state fidelity meets the threshold.
        """
        qubit_count = statevector.num_qubits
        state_length = _common_bitstring_length(self.amplitudes)
        if state_length != qubit_count:
            return ClaimOutcome(False, (
                f"{self.description}: the stated state covers {state_length} "
                f"qubits but the circuit has {qubit_count}."
            ))

        claimed_vector = numpy.zeros(2**qubit_count, dtype=complex)
        for bitstring, amplitude in self.amplitudes.items():
            # Statevector indices follow Qiskit's order,
            # where q[0] is the rightmost bit.
            index = int(swap_bitstring_endianness(bitstring), 2)
            claimed_vector[index] = amplitude

        claimed_vector /= numpy.linalg.norm(claimed_vector)
        fidelity = state_fidelity(Statevector(claimed_vector), statevector)
        return ClaimOutcome(
            passed=fidelity >= self.fidelity_threshold,
            message=(
                f"{self.description}: the stated and simulated output states "
                f"have fidelity "
                f"{fidelity:.4f} (threshold {self.fidelity_threshold:g})."
            ),
        )


CLAIM_TYPES: dict[str, type[PaperClaim]] = {
    "qubit_count": QubitCountClaim,
    "probability_distribution": ProbabilityDistributionClaim,
    "output_state": OutputStateClaim,
}


def parse_claims(
    response_text: str,
    tolerances: ClaimTolerances
) -> tuple[PaperClaim, ...]:
    """
    Validate and parse the claim-extraction reply.

    Args:
        response_text: The raw model response containing a JSON ``claims`` list
        tolerances: Matching tolerances to attach to every claim.

    Returns:
        tuple[PaperClaim, ...]: The supported claims; unknown kinds are skipped

    Raises:
        ResponseValidationError: If the reply or a supported claim is malformed
    """
    raw_claims = extract_json_object(response_text).get("claims")
    if not isinstance(raw_claims, list):
        raise ResponseValidationError('The JSON must contain a "claims" list.')

    claims: list[PaperClaim] = []

    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            raise ResponseValidationError("Each claim must be a JSON object.")

        claim_type = CLAIM_TYPES.get(raw_claim.get("kind"))
        if claim_type is None:
            logger.info(
                "Ignoring unsupported claim kind %r", raw_claim.get("kind")
            )
            continue

        claims.append(claim_type.from_payload(raw_claim, tolerances))

    return tuple(claims)


class PaperClaimExtractor:
    """
    Extracts testable claims from a figure's caption and paper text.
    """

    def __init__(
        self,
        model: VisionLanguageModel,
        tolerances: ClaimTolerances,
        max_attempts: int
    ) -> None:
        """
        Create an extractor.

        Args:
            model: The model that reads the text.
            tolerances: Matching tolerances attached to extracted claims.
            max_attempts: Model calls allowed per extraction.
        """
        self._model = model
        self._tolerances = tolerances
        self._max_attempts = max_attempts

    def extract(self, circuit_input: CircuitInput) -> tuple[PaperClaim, ...]:
        """
        Extract claims from the figure image, its caption and the paper text.

        The figure is always sent, because screenshots from papers often
        include the caption or annotations such as the expected output state.

        Args:
            circuit_input: The figure and its accompanying text.

        Returns:
            tuple[PaperClaim, ...]: The extracted claims, possibly empty.

        Raises:
            GenerationFailedError: If the model gives no valid claim list.
        """
        if not circuit_input.has_text_context:
            return ()
        text_context = describe_text_context(
            circuit_input,
            include_text_context=True,
        )
        message = ChatMessage(
            role=MessageRole.USER,
            text=(f"{CLAIMS_INSTRUCTION}\n\n{text_context}"),
        )
        return generate_validated(
            self._model,
            CLAIMS_SYSTEM_PROMPT,
            message,
            partial(parse_claims, tolerances=self._tolerances),
            self._max_attempts,
        )


class PaperClaimsCheck(VerificationCheck):
    """
    Simulates the candidate and tests every claim extracted from the paper.
    """

    name = "behavioral: paper claims"

    def __init__(
        self,
        claim_extractor: PaperClaimExtractor,
        circuit_input: CircuitInput,
        use_text_context: bool,
    ) -> None:
        """
        Create the check.

        Args:
            claim_extractor: Extracts claims from the figure and its text.
            circuit_input: The figure and its accompanying text.
            use_text_context: False disables the check (image-only ablation).
        """
        self._claim_extractor = claim_extractor
        self._circuit_input = circuit_input
        self._use_text_context = use_text_context

    @cached_property
    def claims(self) -> tuple[PaperClaim, ...]:
        """
        Extract the claims once; later repair iterations reuse them.

        Args:
            None.

        Returns:
            tuple[PaperClaim, ...]: The extracted claims.

        Raises:
            GenerationFailedError: If extraction fails (retried on the
                next run).
        """
        return self._claim_extractor.extract(self._circuit_input)

    def run(self, candidate: GeneratedCode) -> CheckResult:
        """
        Test every claim against the candidate's simulated output.

        Args:
            candidate: The generated code and its circuit.

        Returns:
            CheckResult: SKIPPED without claims, otherwise PASSED or FAILED.

        Raises:
            CircuitAnalysisError: If the candidate cannot be simulated.
        """
        if not self._use_text_context:
            return CheckResult(
                self.name,
                CheckStatus.SKIPPED,
                "Text context is disabled, so no claims were tested.",
            )
        if not self.claims:
            summary = (
                NO_CLAIMS_WITH_TEXT_SUMMARY if
                self._circuit_input.has_text_context else
                NO_CLAIMS_WITHOUT_TEXT_SUMMARY
            )
            return CheckResult(
                self.name,
                CheckStatus.SKIPPED,
                summary
            )

        statevector = simulate_statevector(candidate.circuit)
        outcomes = [claim.evaluate(statevector) for claim in self.claims]
        failures = tuple(
            outcome.message for outcome in outcomes if not outcome.passed
        )

        if failures:
            return CheckResult(
                self.name,
                CheckStatus.FAILED,
                f"{len(failures)} of {len(outcomes)} paper claims failed.",
                failures,
            )

        return CheckResult(
            self.name,
            CheckStatus.PASSED,
            f"All {len(outcomes)} paper claims hold in simulation."
        )


def _is_number(value: object) -> bool:
    """
    Report whether a JSON value is a real number (booleans excluded).

    Args:
        value: The value to test.

    Returns:
        bool: True for ints and floats that are not booleans.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: object) -> bool:
    """
    Report whether a JSON value is an integer (booleans excluded).

    Args:
        value: The value to test.

    Returns:
        bool: True for ints that are not booleans.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _description_of(payload: Mapping[str, Any]) -> str:
    """
    Return a claim's description, falling back to its kind.

    Args:
        payload: The claim entry.

    Returns:
        str: A non-empty description.
    """
    description = payload.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()

    return f"Paper claim ({payload.get('kind')})"


def _parse_bitstring_mapping(
    raw_mapping: object,
    parse_value: Callable[[object], ValueType],
    field_name: str
) -> dict[str, ValueType]:
    """
    Validate a JSON object keyed by bitstrings and parse its values.

    Args:
        raw_mapping: The JSON value to validate.
        parse_value: Parses and validates one value.
        field_name: The JSON field name, used in error messages.

    Returns:
        dict[str, ValueType]: The validated mapping.

    Raises:
        ResponseValidationError: If the mapping, a key or a value is invalid.
    """
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ResponseValidationError(
            f'"{field_name}" must be a non-empty object keyed by bitstrings.'
        )

    parsed: dict[str, ValueType] = {}

    for bitstring, raw_value in raw_mapping.items():
        if not bitstring or not set(bitstring) <= BINARY_DIGITS:
            raise ResponseValidationError(
                f'"{bitstring}" in "{field_name}" is not a string of 0s and 1s'
            )
        parsed[bitstring] = parse_value(raw_value)

    return parsed


def _parse_probability(raw_value: object) -> float:
    """
    Validate a probability.

    Args:
        raw_value: The JSON value to validate.

    Returns:
        float: The probability.

    Raises:
        ResponseValidationError: If the value is not a number in [0, 1].
    """
    if not _is_number(raw_value) or not 0.0 <= raw_value <= 1.0:
        raise ResponseValidationError(
            f"Probability {raw_value!r} must be a number between 0 and 1."
        )

    return float(raw_value)


def _parse_amplitude(raw_value: object) -> complex:
    """
    Validate an amplitude given as a number or a [real, imaginary] pair.

    Args:
        raw_value: The JSON value to validate.

    Returns:
        complex: The amplitude.

    Raises:
        ResponseValidationError: If the value has neither accepted form.
    """
    if _is_number(raw_value):
        return complex(raw_value)

    if (
        isinstance(raw_value, list) and
        len(raw_value) == 2 and
        all(_is_number(part) for part in raw_value)
    ):
        return complex(raw_value[0], raw_value[1])

    raise ResponseValidationError(
        f"Amplitude {raw_value!r} must be a number or a [real, imaginary] pair"
    )


def _parse_qubit_indices(raw_indices: object) -> tuple[int, ...] | None:
    """
    Validate an optional list of distinct, non-negative qubit indices.

    Args:
        raw_indices: The JSON value to validate, or None.

    Returns:
        tuple[int, ...] | None: The indices, or None when absent.

    Raises:
        ResponseValidationError: If the list is empty, repeats or holds
                                invalid indices.
    """
    if raw_indices is None:
        return None
    is_valid = (
        isinstance(raw_indices, list) and bool(raw_indices) and
        all(
            _is_integer(qubit_index) and qubit_index >= 0
            for qubit_index in raw_indices
        ) and len(set(raw_indices)) == len(raw_indices)
    )

    if not is_valid:
        raise ResponseValidationError(
            '"qubits" must be a list of distinct, non-negative qubit indices.'
        )

    return tuple(raw_indices)


def _common_bitstring_length(mapping: Mapping[str, object]) -> int:
    """
    Return the shared length of a mapping's bitstring keys.

    Args:
        mapping: A mapping keyed by bitstrings.

    Returns:
        int: The common key length.

    Raises:
        ResponseValidationError: If the keys have different lengths.
    """
    lengths = {len(bitstring) for bitstring in mapping}
    if len(lengths) != 1:
        raise ResponseValidationError(
            "All bitstrings in a claim must have the same length."
        )
    return lengths.pop()


def _simulated_distribution(
    statevector: Statevector,
    qubit_indices: tuple[int, ...]
) -> dict[str, float]:
    """
    Compute the ideal measurement distribution in the conventional bit order.

    Args:
        statevector: The simulated output state.
        qubit_indices: The measured qubits; the first is leftmost in
                        each bitstring.

    Returns:
        dict[str, float]: Bitstrings mapped to non-negligible probabilities.
    """
    # Qiskit puts qargs[0] in the rightmost position, so reverse every key.
    qiskit_distribution = statevector.probabilities_dict(
        qargs=list(qubit_indices)
    )

    return {
        swap_bitstring_endianness(outcome): float(probability)
        for outcome, probability in qiskit_distribution.items()
        if probability > NEGLIGIBLE_PROBABILITY
    }


def _format_distribution(distribution: Mapping[str, float]) -> str:
    """
    Format the most likely outcomes of a distribution compactly.

    Args:
        distribution: Bitstrings mapped to probabilities.

    Returns:
        str: For example ``"{00: 0.500, 11: 0.500}"``.
    """
    top_outcomes = sorted(
        distribution.items(),
        key=itemgetter(1),
        reverse=True
    )[:MAX_DISPLAYED_OUTCOMES]

    return (
        "{"
        + ", ".join(
            f"{outcome}: {probability:.3f}"
            for outcome, probability in top_outcomes
        )
        + "}"
    )
