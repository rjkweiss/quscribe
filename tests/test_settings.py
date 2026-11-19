"""
Tests for pipeline settings: model roles and validation.
"""

import pytest

from quscribe.errors import ConfigurationError
from quscribe.llm.factory import ModelProvider, ModelSpec
from quscribe.pipeline.settings import PipelineSettings


def test_roles_fall_back_to_the_transcription_model() -> None:
    """
    Unset roles reuse the transcription model; set roles show their own model.
    """
    settings = PipelineSettings(
        primary_model=ModelSpec(ModelProvider.OLLAMA, "qwen3.5:9b"),
        verifier_model=ModelSpec(ModelProvider.OPENAI, "gpt-6-astra"),
    )
    assert settings.model_roles() == {
        "Transcription and repair": "qwen3.5:9b (ollama)",
        "Code generation": "qwen3.5:9b (ollama)",
        "Verification": "gpt-6-astra (openai)",
    }


def test_out_of_range_settings_are_rejected() -> None:
    """
    Invalid values fail before any model call is made.
    """
    with pytest.raises(ConfigurationError, match="fidelity_threshold"):
        PipelineSettings(fidelity_threshold=1.5)
