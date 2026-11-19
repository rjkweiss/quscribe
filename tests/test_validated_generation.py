"""Tests for the shared generate-validate-retry loop."""

from collections.abc import Sequence

import pytest
from fakes import CallbackVisionModel

from quscribe.core.usage import UsageTracker
from quscribe.errors import GenerationFailedError, ResponseValidationError
from quscribe.llm.base import ChatMessage, MessageRole
from quscribe.llm.validated_generation import generate_validated

GOOD_ANSWER = "good"


def require_good_answer(response_text: str) -> str:
    """
    Accept only the good answer.

    Args:
        response_text: The model reply.

    Returns:
        str: The reply, if accepted.

    Raises:
        ResponseValidationError: If the reply is not the good answer.
    """
    if response_text != GOOD_ANSWER:
        raise ResponseValidationError("say good")
    return response_text


def answer_good_after_feedback(
    system_prompt: str,
    messages: Sequence[ChatMessage]
) -> str:
    """
    Reply badly first, then correctly once feedback has been given.

    Args:
        system_prompt: Unused.
        messages: The conversation so far.

    Returns:
        str: "bad" on the first turn, the good answer afterwards.
    """
    return GOOD_ANSWER if len(messages) > 1 else "bad"


def always_answer_badly(
    system_prompt: str,
    messages: Sequence[ChatMessage]
) -> str:
    """
    Always reply with an unacceptable answer.

    Args:
        system_prompt: Unused.
        messages: Unused.

    Returns:
        str: Always "bad".
    """
    return "bad"


def test_feedback_leads_to_accepted_answer_and_usage_is_recorded() -> None:
    """
    A rejected answer is replayed with feedback and the retry is accepted.
    """
    tracker = UsageTracker()
    model = CallbackVisionModel(answer_good_after_feedback, tracker)
    result = generate_validated(
        model,
        "system",
        ChatMessage(MessageRole.USER, "go"),
        require_good_answer,
        max_attempts=3
    )
    assert result == GOOD_ANSWER
    assert tracker.summarize().call_count == 2


def test_exhausted_attempts_raise_generation_failed() -> None:
    """
    When every answer is rejected, the last problem is reported.
    """
    model = CallbackVisionModel(always_answer_badly)
    with pytest.raises(GenerationFailedError, match="say good"):
        generate_validated(
            model,
            "system",
            ChatMessage(MessageRole.USER, "go"),
            require_good_answer, max_attempts=2
        )
