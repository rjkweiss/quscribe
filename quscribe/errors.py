"""
Exception hierarchy shared by every QuScribe component.

Components raise these instead of library-specific exceptions, so callers (
Pipeline, cli, web) can handle failures uniformly.
"""


class QuScribeError(Exception):
    """
    Base class for every error QuScribe raises.

    Args:
        message: Description of the problem.
    """


class ConfigurationError(QuScribeError):
    """
    Raised when settings are invalid, a dependency or credential is missing.

    Args:
        message: Description of the problem.
    """


class InvalidInputError(QuScribeError):
    """
    Raised when user-supplied input, such as an image cannot be read.

    Args:
        message: Description of the problem.
    """


class ResponseValidationError(QuScribeError):
    """
    Raised when a model response is unusable;
    the message is sent back as feedback.

    Args:
        message: Feedback telling the model what to correct.
    """


class GenerationFailedError(QuScribeError):
    """
    Raised when a model produces no valid response within its attempt budget.

    Args:
        message: Description of the last rejection.
    """


class ModelCallError(QuScribeError):
    """
    Raised when a request to a model provider fails (
    network, authentication, quota).

    Args:
        message: Description of the provider failure.
    """


class QasmParseError(QuScribeError):
    """
    Raised when OpenQASM 3 text cannot be loaded into a circuit.

    Args:
        message: Description of the parse failure.
    """


class CodeExecutionError(QuScribeError):
    """
    Raised when generated Qiskit code fails to run or
    does not return a circuit.

    Args:
        message: The error output of the failed run.
    """


class CircuitAnalysisError(QuScribeError):
    """
    Raised when a circuit cannot be simulated or compared (
    e.g. unbound parameters).

    Args:
        message: Description of the analysis failure.
    """


class RenderingError(QuScribeError):
    """
    Raised when a circuit cannot be drawn as an image.

    Args:
        message: Description of the drawing failure.
    """


class UnderspecifiedCircuitError(QuScribeError):
    """
    Raised when a figure does not determine a unique circuit, e.g. an
    undefined oracle box.

    Args:
        message: What the figure leaves unspecified.
    """
