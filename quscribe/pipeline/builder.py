"""
Assembles pipelines from settings: the single place where components are wired.
"""

from quscribe.components.generator import CodeGenerator
from quscribe.components.parser import (
    INDEPENDENT_READING_INSTRUCTION,
    PRIMARY_READING_INSTRUCTION,
    DiagramParser,
)
from quscribe.components.repair import RepairAgent
from quscribe.components.verification.base import Verifier
from quscribe.components.verification.behavioral import (
    ClaimTolerances,
    PaperClaimExtractor,
)
from quscribe.components.verification.verifiers import ReferenceFreeVerifier
from quscribe.core.execution.executor import CircuitCodeExecutor
from quscribe.core.usage import UsageTracker
from quscribe.llm.base import VisionLanguageModel
from quscribe.llm.factory import ModelSpec, create_vision_model
from quscribe.pipeline.orchestrator import QuScribePipeline
from quscribe.pipeline.settings import PipelineSettings


class PipelineBuilder:
    """
    Builds pipelines that share one pair of model clients and one
    usage tracker.

    Models are injected, so tests can pass fakes; ``from_settings``
    creates real clients.
    """

    def __init__(
        self,
        settings: PipelineSettings,
        primary_model: VisionLanguageModel,
        verifier_model: VisionLanguageModel,
        usage_tracker: UsageTracker,
        generator_model: VisionLanguageModel | None = None
    ) -> None:
        """
        Create a builder from ready-made models.

        Args:
            settings: The pipeline settings.
            primary_model: Model for parsing and repair, and for code
                generation by default.
            verifier_model: Model for verification (may be the same object).
            usage_tracker: The tracker both models record into.
            generator_model: Model for code generation; None uses the
                primary model.

        Returns:
            None.
        """
        self._settings = settings
        self._primary_model = primary_model
        self._verifier_model = verifier_model
        self._usage_tracker = usage_tracker
        self._generator_model = (
            generator_model if generator_model is not None else primary_model
        )

    @classmethod
    def from_settings(cls, settings: PipelineSettings) -> "PipelineBuilder":
        """
        Create model clients from the settings and return a builder for them.

        Args:
            settings: The pipeline settings.

        Returns:
            PipelineBuilder: A builder with real model clients.

        Raises:
            ConfigurationError: If a model specification or credential
                                is invalid.
        """
        usage_tracker = UsageTracker()
        primary_model = create_vision_model(
            settings.primary_model,
            usage_tracker,
            settings.max_output_tokens,
        )
        verifier_model = cls._model_or_fallback(
            settings.verifier_model,
            primary_model,
            usage_tracker,
            settings,
        )
        generator_model = cls._model_or_fallback(
            settings.generator_model,
            primary_model,
            usage_tracker,
            settings,
        )
        return cls(
            settings,
            primary_model,
            verifier_model,
            usage_tracker,
            generator_model=generator_model,
        )

    @staticmethod
    def _model_or_fallback(
        spec: ModelSpec | None,
        fallback: VisionLanguageModel,
        usage_tracker: UsageTracker,
        settings: PipelineSettings,
    ) -> VisionLanguageModel:
        """
        Create a client for an optional role, or reuse the fallback client.

        Args:
            spec: The role's model specification, or None to reuse the
                fallback.
            fallback: The client to reuse.
            usage_tracker: The tracker a new client records into.
            settings: Supplies the reply token limit.

        Returns:
            VisionLanguageModel: The client for the role.

        Raises:
            ConfigurationError: If the specification or credentials are
                                invalid.
        """
        if spec is None:
            return fallback
        return create_vision_model(
            spec,
            usage_tracker,
            settings.max_output_tokens,
        )

    def build(self, verifier: Verifier | None = None) -> QuScribePipeline:
        """
        Build a pipeline.

        Args:
            verifier: Verifier override (e.g. the oracle for ablations); None
                        uses the reference-free verifier.

        Returns:
            QuScribePipeline: The assembled pipeline.
        """
        settings = self._settings
        parser = DiagramParser(
            self._primary_model,
            PRIMARY_READING_INSTRUCTION,
            settings.use_text_context,
            settings.max_attempts_per_call,
        )

        generator = CodeGenerator(
            self._generator_model,
            CircuitCodeExecutor(settings.code_timeout_seconds),
            settings.fidelity_threshold,
            settings.max_attempts_per_call,
        )

        return QuScribePipeline(
            parser=parser,
            generator=generator,
            verifier=(
                verifier if verifier is not None else
                self.build_reference_free_verifier()
            ),
            repair_agent=RepairAgent(parser),
            max_repair_iterations=settings.max_repair_iterations,
            usage_tracker=self._usage_tracker,
        )

    def build_reference_free_verifier(self) -> ReferenceFreeVerifier:
        """
        Build the reference-free verifier on the verifier model.

        Args:
            None.

        Returns:
            ReferenceFreeVerifier: The verifier.
        """
        settings = self._settings
        independent_parser = DiagramParser(
            self._verifier_model,
            INDEPENDENT_READING_INSTRUCTION,
            settings.use_text_context,
            settings.max_attempts_per_call,
        )

        claim_extractor = PaperClaimExtractor(
            self._verifier_model,
            ClaimTolerances(),
            settings.max_attempts_per_call
        )

        return ReferenceFreeVerifier(
            independent_parser=independent_parser,
            judge_model=self._verifier_model,
            claim_extractor=claim_extractor,
            use_text_context=settings.use_text_context,
            fidelity_threshold=settings.fidelity_threshold,
            max_attempts=settings.max_attempts_per_call,
        )
