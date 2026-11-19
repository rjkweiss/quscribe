"""
Tests for extracting code blocks and JSON from model responses.
"""

import pytest

from quscribe.core.text_extraction import (
    extract_fenced_block,
    extract_json_object,
)
from quscribe.errors import ResponseValidationError


def test_extracts_last_block_in_accepted_language() -> None:
    """
    The final matching block wins when a model revises its answer.
    """
    response = "Draft:\n```qasm\nold\n```\nFinal:\n```qasm\nnew\n```"
    assert extract_fenced_block(response, ("qasm",)) == "new"


def test_accepts_single_unlabeled_block() -> None:
    """
    A lone block without a language tag is accepted.
    """
    assert extract_fenced_block("```\nbody\n```", ("python",)) == "body"


def test_missing_block_raises_feedback_error() -> None:
    """
    A response without a block yields feedback naming the expected tag.
    """
    with pytest.raises(ResponseValidationError, match="```qasm"):
        extract_fenced_block("no code here", ("qasm",))


def test_extracts_json_from_fenced_or_bare_text() -> None:
    """
    JSON is found inside a json block or embedded in prose.
    """
    assert extract_json_object('```json\n{"consistent": true}\n```') == {
        "consistent": True
    }
    assert extract_json_object(
        'Verdict: {"consistent": false} done'
    ) == {"consistent": False}


def test_invalid_json_raises_feedback_error() -> None:
    """
    Malformed JSON yields a feedback error rather than a crash.
    """
    with pytest.raises(ResponseValidationError, match="invalid"):
        extract_json_object('```json\n{"consistent": tru}\n```')
