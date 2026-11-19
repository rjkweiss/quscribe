"""
Builds the figure-plus-text messages that several components send to models.
"""

from collections.abc import Sequence

from quscribe.core.images import ImagePayload
from quscribe.core.models import CircuitInput
from quscribe.llm.base import ChatMessage, MessageRole

# Long papers are truncated to keep requests within context and cost budgets.
MAX_PAPER_TEXT_CHARACTERS = 12_000
NO_TEXT_CONTEXT_NOTE = (
    "No caption or paper text is provided; rely on the figure alone."
)


def describe_text_context(
    circuit_input: CircuitInput,
    include_text_context: bool,
) -> str:
    """
    Format the caption and paper passages for a prompt.

    Args:
        circuit_input: The figure and its accompanying text.
        include_text_context: False for image-only runs (the ablation baseline)

    Returns:
        str: The formatted text context, or a note that none is available.
    """
    if not include_text_context or not circuit_input.has_text_context:
        return NO_TEXT_CONTEXT_NOTE

    sections = []
    if circuit_input.caption.strip():
        sections.append(f"Figure caption:\n{circuit_input.caption.strip()}")

    if circuit_input.paper_text.strip():
        passages = circuit_input.paper_text.strip()[:MAX_PAPER_TEXT_CHARACTERS]
        sections.append(f"Relevant passages from the paper:\n{passages}")

    return "\n\n".join(sections)


def build_figure_message(
    circuit_input: CircuitInput,
    include_text_context: bool,
    instruction: str,
    extra_images: Sequence[ImagePayload] = (),
) -> ChatMessage:
    """
    Build user message with the figure, optional extra images and text context.

    Args:
        circuit_input: The figure and its accompanying text.
        include_text_context: Whether to include the caption and paper text.
        instruction: The task description for the model.
        extra_images: Images sent after the figure (e.g. a redrawn circuit).

    Returns:
        ChatMessage: The assembled user message.
    """
    message_text = (
        f"{instruction}\n\n"
        f"{describe_text_context(circuit_input, include_text_context)}"
    )
    return ChatMessage(
        role=MessageRole.USER,
        text=message_text,
        images=(circuit_input.figure, *extra_images),
    )
