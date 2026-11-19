"""
OpenAI-compatible implementation, covering OpenAI, Ollama,
vLLM and similar servers.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import openai

from quscribe.core.usage import UsageTracker
from quscribe.errors import ConfigurationError, ModelCallError
from quscribe.llm.base import ChatMessage, ModelResponse, VisionLanguageModel


class OpenAICompatibleVisionModel(VisionLanguageModel):
    """
    Any vision model served through the Chat Completions API.

    No token limit or temperature is sent: OpenAI reasoning models reject those
    parameters, and local servers apply sensible defaults.
    """

    def __init__(
        self,
        model_name: str,
        usage_tracker: UsageTracker,
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout_seconds: float | None = None,
        max_retries: int | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Create the OpenAI client.

        Args:
            model_name: The model identifier on the target server.
            usage_tracker: Where to record the usage of every call.
            base_url: Server URL; None targets OpenAI itself.
            api_key: API key; None reads ``OPENAI_API_KEY`` from the
                environment.
            request_timeout_seconds: Per-request timeout; None keeps the SDK
                default.
            max_retries: Automatic retries after a failed request; None keeps
                the SDK default.
            extra_body: Extra fields added to every request body
                (server-specific options).

        Returns:
            None.

        Raises:
            ConfigurationError: If the client cannot be created
                (e.g. no API key).
        """
        super().__init__(model_name, usage_tracker)
        client_options: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key
        }

        if request_timeout_seconds is not None:
            client_options["timeout"] = request_timeout_seconds
        if max_retries is not None:
            client_options["max_retries"] = max_retries
        try:
            self._client = openai.OpenAI(**client_options)
        except openai.OpenAIError as error:
            raise ConfigurationError(
                f"Cannot create the OpenAI-compatible client: {error}"
            ) from error

        self._extra_body = dict(extra_body) if extra_body else None

    def _send(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessage]
    ) -> ModelResponse:
        """
        Call the Chat Completions API.

        Args:
            system_prompt: Instructions that frame the whole conversation.
            messages: The conversation so far, ending with a user message.

        Returns:
            ModelResponse: The reply text and token usage.

        Raises:
            ModelCallError: If the request fails.
        """
        payload = [{"role": "system", "content": system_prompt}]
        payload.extend(_to_openai_message(message) for message in messages)

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                extra_body=self._extra_body
            )
        except openai.OpenAIError as error:
            raise ModelCallError(
                f"The OpenAI-compatible request failed: {error}") from error

        usage = response.usage

        return ModelResponse(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


def _to_openai_message(message: ChatMessage) -> dict[str, Any]:
    """
    Convert a chat message to the Chat Completions format.

    Text-only messages use plain string content, which every compatible
    server accepts.

    Args:
        message: The message to convert.

    Returns:
        dict[str, Any]: The converted message.
    """
    if not message.images:
        return {"role": message.role.value, "content": message.text}
    image_parts = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image.media_type};base64,{image.to_base64()}"
            }
        }
        for image in message.images
    ]

    return {
        "role": message.role.value,
        "content": [
            {"type": "text", "text": message.text},
            *image_parts
        ]
    }
