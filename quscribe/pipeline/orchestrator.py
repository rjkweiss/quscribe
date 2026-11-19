"""
The QuScribe pipeline: parse, generate, verify, and repair until verified or
out of budget.
"""

import logging
import time
from dataclasses import dataclass

from quscribe.components.generator import CodeGenerator
from quscribe.components.parser import DiagramParser
from quscribe.components.repair import RepairAgent
from quscribe.components.verification.base import VerificationSuite, Verifier
from quscribe.core.models import (
    CircuitInput,
    GeneratedCode,
    ParsedCircuit,
    PipelineResult,
    RepairStep,
    TrustLabel,
    VerificationReport,
)
from quscribe.core.usage import UsageTracker
from quscribe.errors import QuScribeError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Candidate:
    """
    A transcription, the code generated from it, and that code's
    verification report.

    Keeping the three together guarantees a result never mixes a new
    transcription with code or a report from an older one.

    Attributes:
        parsed: The transcription.
        generated: The code generated from the transcription.
        report: The verification report for the code.
    """

    parsed: ParsedCircuit
    generated: GeneratedCode
    report: VerificationReport


class QuScribePipeline:
    """
    Orchestrates the four components for one figure at a time.
    """

    def __init__(
        self,
        parser: DiagramParser,
        generator: CodeGenerator,
        verifier: Verifier,
        repair_agent: RepairAgent,
        max_repair_iterations: int,
        usage_tracker: UsageTracker,
    ) -> None:
        """
        Create a pipeline from its components.

        Args:
            parser: Transcribes figures into OpenQASM 3.
            generator: Translates transcriptions into verified-faithful Qiskit
                        code.
            verifier: Builds the verification checks for each figure.
            repair_agent: Corrects transcriptions that fail verification.
            max_repair_iterations: Repair attempts allowed per figure.
            usage_tracker: The tracker shared by this pipeline's models.

        Returns:
            None.
        """
        self._parser = parser
        self._generator = generator
        self._verifier = verifier
        self._repair_agent = repair_agent
        self._max_repair_iterations = max_repair_iterations
        self._usage_tracker = usage_tracker

    def run(self, circuit_input: CircuitInput) -> PipelineResult:
        """
        Convert one figure into code with a trust label.

        Any failure yields a FLAGGED result with an error message instead of an
        exception, so system doesn't silently returns code it couldn't confirm.

        Args:
            circuit_input: The figure and its accompanying text.

        Returns:
            PipelineResult: The final code, verification report, repair
                            history and usage.
        """
        usage_mark = self._usage_tracker.mark()
        started_at = time.perf_counter()
        candidate: _Candidate | None = None
        repair_steps: list[RepairStep] = []
        error_message: str | None = None
        try:
            suite = self._verifier.create_suite(circuit_input)
            candidate = self._build_candidate(
                self._parser.parse(circuit_input),
                suite
            )
            for iteration in range(1, self._max_repair_iterations + 1):
                if candidate.report.label is TrustLabel.VERIFIED:
                    break
                repaired_parse = self._repair_agent.repair(
                    circuit_input,
                    candidate.parsed,
                    candidate.report
                )
                repaired = self._build_candidate(repaired_parse, suite)
                repair_steps.append(RepairStep(
                    iteration=iteration,
                    addressed_discrepancies=candidate.report.discrepancies,
                    revised_qasm=repaired.parsed.qasm,
                    label_after_repair=repaired.report.label,
                ))
                candidate = repaired
        except QuScribeError as error:
            logger.warning("Pipeline stopped early: %s", error)
            error_message = str(error)

        completed_cleanly = candidate is not None and error_message is None

        return PipelineResult(
            label=(
                candidate.report.label
                if completed_cleanly else TrustLabel.FLAGGED
            ),
            parsed=candidate.parsed if candidate else None,
            generated=candidate.generated if candidate else None,
            report=candidate.report if candidate else None,
            repair_steps=tuple(repair_steps),
            usage=self._usage_tracker.summarize(usage_mark),
            wall_clock_seconds=time.perf_counter() - started_at,
            error_message=error_message,
        )

    def _build_candidate(
        self,
        parsed: ParsedCircuit,
        suite: VerificationSuite
    ) -> _Candidate:
        """
        Generate code for a transcription and verify it.

        Args:
            parsed: The transcription.
            suite: The verification checks for the current figure.

        Returns:
            _Candidate: The transcription, its code, and the verification
                        report.

        Raises:
            GenerationFailedError: If no faithful code can be generated.
        """
        generated = self._generator.generate(parsed)
        return _Candidate(
            parsed=parsed,
            generated=generated,
            report=suite.verify(generated)
        )
