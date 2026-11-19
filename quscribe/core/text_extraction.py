"""
Helpers that pull structured content out of raw model responses.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from quscribe.errors import ResponseValidationError

# Matches ```language ... ``` blocks; the language tag may be empty.
FENCED_BLOCK_PATTERN = re.compile(
    r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL
)
JSON_LANGUAGES = ("json",)


def extract_fenced_block(
    response_text: str,
    accepted_languages: Sequence[str]
) -> str:
    """
    Return the content of the fenced code block.

    When several matching blocks exist, the last one wins, because models that
    revise an answer put the final version last. A single unlabeled block is
    accepted too.

    Args:
        response_text: The raw model response.
        accepted_languages: Language tags to accept, preferred tag first.

    Returns:
        str: The block content without the fences.

    Raises:
        ResponseValidationError: If no suitable block exists.
    """
    # e.g. ```qasm, ```python, etc
    accepted = {language.lower() for language in accepted_languages}

    # find matching fenced code blocks from raw response
    blocks = [
        (match.group(1).lower(), match.group(2).strip())
        for match in FENCED_BLOCK_PATTERN.finditer(response_text)
    ]

    # get all matching blocks for all the languages
    matching_blocks = [
        content for language, content in blocks
        if language in accepted
    ]

    # return the last code block
    if matching_blocks:
        return matching_blocks[-1]

    # unlabeled blocks
    unlabeled_blocks = [
        content for language, content in blocks
        if not language
    ]

    if len(unlabeled_blocks) == 1:
        return unlabeled_blocks[0]

    # there is fenced blocks, so an error is raised
    raise ResponseValidationError(
        f"Put your answer in one fenced code block that "
        f"starts with ```{accepted_languages[0]}."
    )


def extract_json_object(response_text: str) -> dict[str, Any]:
    """
    Parse the JSON object in a model response, preferring a ```json block.

    Args:
        response_text: The raw model response.

    Returns:
        dict[str, Any]: The parsed JSON object.

    Raises:
        ResponseValidationError: If the response holds no valid JSON object.
    """
    # extract any fenced json code from response if it exists
    try:
        candidate_text = extract_fenced_block(response_text, JSON_LANGUAGES)
    except ResponseValidationError:
        candidate_text = response_text  # scan the whole response.

    # special chars defining start/end of JSON block
    object_start = candidate_text.find("{")
    object_end = candidate_text.rfind("}")

    if object_start == -1 or object_end < object_start:
        raise ResponseValidationError(
            "Reply with a JSON object inside a ```json block."
        )

    # parse the JSON if valid
    try:
        parsed = json.loads(candidate_text[object_start: object_end + 1])
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            f"The JSON is invalid: {error}.") from error

    # correctly parsed JSON should be an object (dictionary in python)
    if not isinstance(parsed, dict):
        raise ResponseValidationError(
            "The JSON must be an object, not a list or value.")

    return parsed
