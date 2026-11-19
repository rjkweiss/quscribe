"""
Test doubles shared by the QuScribe test suite.
"""

from collections.abc import Callable, Sequence

from PIL import Image

from quscribe.core.images import ImagePayload
from quscribe.core.usage import UsageTracker
from quscribe.llm.base import ChatMessage, ModelResponse, VisionLanguageModel

Responder = Callable[[str, Sequence[ChatMessage]], str]

FAKE_INPUT_TOKENS = 10
FAKE_OUTPUT_TOKENS = 5


class CallbackVisionModel(VisionLanguageModel):
    """
    A model whose replies come from a Python callback, for offline tests.
    """

    def __init__(
        self,
        responder: Responder,
        usage_tracker: UsageTracker | None = None
    ) -> None:
        """
        Create the fake model.

        Args:
            responder: Maps (system prompt, conversation) to reply text.
            usage_tracker: Tracker to record into; a new one if None.
        """
        super().__init__(
            "fake-model",
            usage_tracker if usage_tracker is not None else UsageTracker()
        )
        self._responder = responder

    def _send(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessage]
    ) -> ModelResponse:
        """
        Reply using the callback, with fixed token counts.

        Args:
            system_prompt: The system prompt.
            messages: The conversation so far.

        Returns:
            ModelResponse: The callback's reply.
        """
        return ModelResponse(
            text=self._responder(system_prompt, messages),
            input_tokens=FAKE_INPUT_TOKENS,
            output_tokens=FAKE_OUTPUT_TOKENS,
        )


def fenced(language: str, body: str) -> str:
    """
    Wrap text in a fenced code block, as a model would.

    Args:
        language: The block's language tag.
        body: The block content.

    Returns:
        str: The fenced block.
    """
    return f"```{language}\n{body}\n```"


def blank_figure() -> ImagePayload:
    """
    Create a small blank image to stand in for a circuit figure.

    Args:
        None.

    Returns:
        ImagePayload: A white 16x16 image.
    """
    return ImagePayload.from_pil_image(Image.new("RGB", (16, 16), "white"))
