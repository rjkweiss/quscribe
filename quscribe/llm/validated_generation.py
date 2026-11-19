"""
Generate-validate-retry loop shared by every stage that asks
a model for output.

Each stage supplies a validator that turns response text into a typed result or
raises ``ResponseValidationError``; the error message is sent back to
the model as feedback. This one loop replaces per-stage retry logic.
"""

import logging
from collections.abc import Callable
from typing import TypeVar

from quscribe.errors import GenerationFailedError, ResponseValidationError
from quscribe.llm.base import ChatMessage, MessageRole, VisionLanguageModel

logger = logging.getLogger(__name__)

ResultType = TypeVar("ResultType")

FEEDBACK_TEMPLATE = (
    "Your previous answer was rejected: {problem}\n"
    "Return a corrected, complete answer in the same format."
)


def generate_validated(
    model: VisionLanguageModel,
    system_prompt: str,
    initial_message: ChatMessage,
    validate_response: Callable[[str], ResultType],
    max_attempts: int,
) -> ResultType:
    """
    Ask a model for output until a validator accepts it or attempts run out.

    Args:
        model: The model to query.
        system_prompt: Instructions that frame the conversation.
        initial_message: The first user message (may include images).
        validate_response: Converts response text to a result, or raises
            ``ResponseValidationError`` describing what to fix.
        max_attempts: Maximum number of model calls, at least 1.

    Returns:
        ResultType: The first accepted result.

    Raises:
        ValueError: If ``max_attempts`` is below 1.
        GenerationFailedError: If no response is accepted within the budget.
        ModelCallError: If a provider request fails.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    conversation = [initial_message]
    last_problem = ""

    # try until validator accepts (within given number of attempts)
    for attempt_number in range(1, max_attempts + 1):
        response = model.complete(system_prompt, conversation)
        try:
            return validate_response(response.text)
        except ResponseValidationError as error:
            last_problem = str(error)
            logger.info(
                "Attempt %d/%d rejected: %s",
                attempt_number,
                max_attempts,
                last_problem
            )

            # Replay the rejected answer with feedback so the model
            # can correct itself.
            conversation = [
                *conversation,
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    text=response.text
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    text=FEEDBACK_TEMPLATE.format(problem=last_problem)
                ),
            ]

    # ran out of attempts budget
    raise GenerationFailedError(
        f"No valid answer after {max_attempts} attempt(s): {last_problem}"
    )
