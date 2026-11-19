"""
Creates model clients from provider-independent specifications.
"""

from dataclasses import dataclass
from enum import Enum

from quscribe.core.usage import UsageTracker
from quscribe.errors import ConfigurationError
from quscribe.llm.base import VisionLanguageModel


class ModelProvider(str, Enum):
    """
    Supported model providers.

    Attributes:
        ANTHROPIC: Claude models via the Anthropic API.
        OPENAI: OpenAI models, or any OpenAI-compatible server given a base
                URL (e.g. vLLM).
        OLLAMA: Open-weight models served locally by Ollama.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


# Only providers with a stable, known default get one;
# others need an explicit model.
DEFAULT_MODEL_NAMES = {
    ModelProvider.ANTHROPIC: "claude-opus-5-5",
}

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_PLACEHOLDER_API_KEY = "ollama"
# A local call can generate for many minutes. The SDK default
# (600 s, retried twice) cut such calls off, and each retry restarted
# the whole generation from scratch.
LOCAL_REQUEST_TIMEOUT_SECONDS = 3600.0
LOCAL_MAX_RETRIES = 0
# Ollama turns a model's thinking off when its OpenAI-compatible endpoint
# receives reasoning_effort "none"; thinking multiplied output tokens
# several-fold in local trials.
OLLAMA_REQUEST_BODY = {"reasoning_effort": "none"}


@dataclass(frozen=True)
class ModelSpec:
    """
    Which model to use and where to reach it.

    Attributes:
        provider: The model provider.
        model_name: The model identifier; None uses provider default, if any.
        base_url: Optional server URL override (OpenAI-compatible providers).
    """

    provider: ModelProvider = ModelProvider.ANTHROPIC
    model_name: str | None = None
    base_url: str | None = None

    def resolved_model_name(self) -> str:
        """
        Return the explicit model name or the provider default.

        Args:
            None.

        Returns:
            str: The model identifier to request.

        Raises:
            ConfigurationError: If no name was given and provider has
                                no default.
        """
        model_name = (
            (self.model_name or "").strip()
            or DEFAULT_MODEL_NAMES.get(self.provider)
        )
        if not model_name:
            raise ConfigurationError(
                "Choose a model name for the "
                f"{self.provider.value} provider."
            )

        return model_name


def create_vision_model(
    spec: ModelSpec, usage_tracker: UsageTracker, max_output_tokens: int
) -> VisionLanguageModel:
    """
    Create a model client for a specification.

    Provider modules are imported on demand so an unused
    provider SDK never has to load.

    Args:
        spec: Which model to create.
        usage_tracker: Where the client records usage.
        max_output_tokens: Reply token limit, sent to Anthropic (which requires
            one) and to Ollama (where it stops a runaway generation from
            filling the context window).

    Returns:
        VisionLanguageModel: A ready-to-use client.

    Raises:
        ConfigurationError: If the specification or credentials are invalid.
    """
    model_name = spec.resolved_model_name()
    if spec.provider is ModelProvider.ANTHROPIC:
        from quscribe.llm.anthropic_model import AnthropicVisionModel

        return AnthropicVisionModel(
            model_name,
            usage_tracker,
            max_output_tokens
        )

    from quscribe.llm.openai_model import OpenAICompatibleVisionModel

    if spec.provider is ModelProvider.OLLAMA:
        return OpenAICompatibleVisionModel(
            model_name,
            usage_tracker,
            base_url=spec.base_url or OLLAMA_DEFAULT_BASE_URL,
            api_key=OLLAMA_PLACEHOLDER_API_KEY,
            request_timeout_seconds=LOCAL_REQUEST_TIMEOUT_SECONDS,
            max_retries=LOCAL_MAX_RETRIES,
            extra_body={
                **OLLAMA_REQUEST_BODY,
                "max_tokens": max_output_tokens,
            },
        )

    return OpenAICompatibleVisionModel(
        model_name,
        usage_tracker,
        base_url=spec.base_url,
    )
