"""Validated configuration for a QuScribe pipeline."""

from dataclasses import dataclass, field

from quscribe.errors import ConfigurationError
from quscribe.llm.factory import ModelSpec


@dataclass(frozen=True)
class PipelineSettings:
    """All tunable pipeline parameters in one place.

    Attributes:
        primary_model: Model for parsing, code generation and repair.
        verifier_model: Model for independent reading, redraw comparison, and
            claim extraction; None reuses the primary model. A different
            model makes the verifier more independent.
        generator_model: Model that writes the Qiskit code; None reuses the
            primary model.
        use_text_context: Whether to send the caption and paper text
            (ablation switch).
        max_repair_iterations: Repair attempts after a failed verification
            (0 disables).
        max_attempts_per_call: Model calls allowed per stage when output is
            invalid.
        fidelity_threshold: Minimum unitary fidelity for two circuits to
            count as equal.
        code_timeout_seconds: Maximum run time for generated code.
        max_output_tokens: Reply token limit for providers that require one.
    """

    primary_model: ModelSpec = field(default_factory=ModelSpec)
    verifier_model: ModelSpec | None = None
    generator_model: ModelSpec | None = None
    use_text_context: bool = True
    max_repair_iterations: int = 3
    max_attempts_per_call: int = 3
    fidelity_threshold: float = 0.99
    code_timeout_seconds: float = 60.0
    max_output_tokens: int = 4096

    def __post_init__(self) -> None:
        """
        Reject out-of-range values early, before any model call is made.

        Args:
            None.

        Returns:
            None.

        Raises:
            ConfigurationError: If any value is out of range.
        """
        _require(
            self.max_repair_iterations >= 0,
            "max_repair_iterations must be 0 or more.",
        )
        _require(
            self.max_attempts_per_call >= 1,
            "max_attempts_per_call must be at least 1.",
        )
        _require(
            0.0 < self.fidelity_threshold <= 1.0,
            "fidelity_threshold must be in (0, 1].",
        )
        _require(
            self.code_timeout_seconds > 0,
            "code_timeout_seconds must be positive.",
        )
        _require(
            self.max_output_tokens > 0,
            "max_output_tokens must be positive.",
        )

    def model_roles(self) -> dict[str, str]:
        """
        Describe which model plays each role, for the web app, the CLI
        and reports.

        Args:
        None.

        Returns:
            dict[str, str]: Role name mapped to "model (provider)".

        Raises:
            ConfigurationError: If a role's model name cannot be resolved.
        """
        return {
            "Transcription and repair": _describe_model(self.primary_model),
            "Code generation": _describe_model(
                self.generator_model or self.primary_model
            ),
            "Verification": _describe_model(
                self.verifier_model or self.primary_model
            ),
        }


def _describe_model(spec: ModelSpec) -> str:
    """
    Format a model specification for people to read.

    Args:
        spec: The model specification.

    Returns:
        str: For example "claude-opus-5-5 (anthropic)".

    Raises:
        ConfigurationError: If the model name cannot be resolved.
    """
    return f"{spec.resolved_model_name()} ({spec.provider.value})"


def _require(condition: bool, message: str) -> None:
    """
    Raise a configuration error when a condition does not hold.

    Args:
        condition: The condition that must be true.
        message: The error message if it is false.

    Returns:
        None.

    Raises:
        ConfigurationError: If ``condition`` is false.
    """
    if not condition:
        raise ConfigurationError(message)
