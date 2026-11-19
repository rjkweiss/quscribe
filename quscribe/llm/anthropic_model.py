"""
Anthropic Claude implementation of the vision-language model interface.
"""

from collections.abc import Sequence
from typing import Any

import anthropic

from quscribe.core.usage import UsageTracker
from quscribe.errors import ConfigurationError, ModelCallError
from quscribe.llm.base import ChatMessage, ModelResponse, VisionLanguageModel


class AnthropicVisionModel(VisionLanguageModel):
    """
    Claude models through the Anthropic Messages API
    (reads ``ANTHROPIC_API_KEY``)
    """

    def __init__(
        self,
        model_name: str,
        usage_tracker: UsageTracker,
        max_output_tokens: int
    ) -> None:
        """
        Create the Anthropic client.

        Args:
            model_name: The Claude model identifier.
            usage_tracker: Where to record the usage of every call.
            max_output_tokens: Upper bound on tokens per reply; required by API

        Raises:
            ConfigurationError: If client cannot be created (e.g. no API key).
        """
        super().__init__(model_name, usage_tracker)
        try:
            self._client = anthropic.Anthropic()
        except anthropic.AnthropicError as error:
            raise ConfigurationError(
                f"Cannot create the Anthropic client: {error}") from error

        self._max_output_tokens = max_output_tokens

    def _send(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessage]
    ) -> ModelResponse:
        """
        Call the Messages API.

        Args:
            system_prompt: Instructions that frame the whole conversation.
            messages: The conversation so far, ending with a user message.

        Returns:
            ModelResponse: The reply text and token usage.

        Raises:
            ModelCallError: If the request fails.
        """
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=self._max_output_tokens,
                system=system_prompt,
                messages=[
                    _to_anthropic_message(message) for message in messages
                ],
            )
        except anthropic.AnthropicError as error:
            raise ModelCallError(
                f"The Anthropic request failed: {error}") from error

        response_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return ModelResponse(
            text=response_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def _to_anthropic_message(message: ChatMessage) -> dict[str, Any]:
    """
    Convert a chat message to the Anthropic content-block format.

    Args:
        message: The message to convert.

    Returns:
        dict[str, Any]: The message with image blocks first, then text block.
    """
    image_blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": image.to_base64()
            },
        }
        for image in message.images
    ]

    return {
        "role": message.role.value,
        "content": [
            *image_blocks,
            {"type": "text", "text": message.text}
        ]
    }
