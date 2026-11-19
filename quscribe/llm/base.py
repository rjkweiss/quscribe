"""
Provider-independent interface for vision-language models.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from quscribe.core.images import ImagePayload
from quscribe.core.usage import UsageRecord, UsageTracker


class MessageRole(str, Enum):
    """
    Author of a chat message.

    Attributes:
        USER: Messages QuScribe sends to the model.
        ASSISTANT: Earlier model replies, replayed when asking
                    for a correction.
    """

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    """
    One message in a conversation with a model.

    Attributes:
        role: Who wrote the message.
        text: The message text.
        images: Images attached to the message, sent before the text.
    """

    role: MessageRole
    text: str
    images: tuple[ImagePayload, ...] = ()


@dataclass(frozen=True)
class ModelResponse:
    """
    A model reply and its token usage.

    Attributes:
        text: The reply text.
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class VisionLanguageModel(ABC):
    """
    Base class for model clients; records usage for every call.

    Subclasses implement only ``_send``; ``complete`` wraps it with timing and
    usage recording so cost and latency accounting is identical
    for every provider.
    """

    def __init__(self, model_name: str, usage_tracker: UsageTracker) -> None:
        """
        Initialize the shared client state.

        Args:
            model_name: The provider's model identifier.
            usage_tracker: Where to record the usage of every call.
        """
        self._model_name = model_name
        self._usage_tracker = usage_tracker

    @property
    def model_name(self) -> str:
        """
        Return the provider's model identifier.

        Returns:
            str: The model identifier.
        """
        return self._model_name

    def complete(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessage]
    ) -> ModelResponse:
        """
        Send a conversation to the model, recording tokens and latency.

        Args:
            system_prompt: Instructions that frame the whole conversation.
            messages: The conversation so far, ending with a user message.

        Returns:
            ModelResponse: The model reply.

        Raises:
            ModelCallError: If the provider request fails.
        """
        started_at = time.perf_counter()
        response = self._send(system_prompt, messages)
        self._usage_tracker.record(UsageRecord(
            model_name=self._model_name,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_seconds=time.perf_counter() - started_at,
        ))

        return response

    @abstractmethod
    def _send(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessage]
    ) -> ModelResponse:
        """
        Perform the provider-specific request.

        Args:
            system_prompt: Instructions that frame the whole conversation.
            messages: The conversation so far, ending with a user message.

        Returns:
            ModelResponse: The model reply.

        Raises:
            ModelCallError: If the provider request fails.
        """
