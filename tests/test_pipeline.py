"""
End-to-end pipeline tests with a scripted model
(requires Qiskit and matplotlib).
"""

from collections.abc import Callable, Sequence

from fakes import CallbackVisionModel, blank_figure, fenced

from quscribe.components.generator import GENERATOR_SYSTEM_PROMPT
from quscribe.components.parser import INDEPENDENT_READING_INSTRUCTION
from quscribe.components.repair import REPAIR_INSTRUCTION_TEMPLATE
from quscribe.components.verification.behavioral import CLAIMS_SYSTEM_PROMPT
from quscribe.components.verification.structural import REDRAW_SYSTEM_PROMPT
from quscribe.core.conventions import UNDERSPECIFIED_MARKER
from quscribe.core.models import CheckStatus, CircuitInput, TrustLabel
from quscribe.core.usage import UsageTracker
from quscribe.llm.base import ChatMessage
from quscribe.pipeline.builder import PipelineBuilder
from quscribe.pipeline.orchestrator import QuScribePipeline
from quscribe.pipeline.settings import PipelineSettings

QASM_HEADER = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
BELL_QASM = QASM_HEADER + "h q[0];\ncx q[0], q[1];"
MISREAD_QASM = QASM_HEADER + "h q[0];"  # The CX was missed.
REPAIR_MARKER = REPAIR_INSTRUCTION_TEMPLATE.strip().splitlines()[0]

CODE_TEMPLATE = '''from qiskit import QuantumCircuit


def build_circuit() -> QuantumCircuit:
    """Build the transcribed circuit."""
    circuit = QuantumCircuit(2)
{body}
    return circuit
'''
BELL_CODE = CODE_TEMPLATE.format(body="    circuit.h(0)\n    circuit.cx(0, 1)")
MISREAD_CODE = CODE_TEMPLATE.format(body="    circuit.h(0)")
BELL_CLAIMS = (
    '{"claims": [{"kind": "probability_distribution", '
    '"description": "Bell state", '
    '"probabilities": {"00": 0.5, "11": 0.5}}]}'
)
CONSISTENT_VERDICT = '{"consistent": true, "discrepancies": []}'

Responder = Callable[[str, Sequence[ChatMessage]], str]


def make_responder(first_reading_qasm: str) -> Responder:
    """
    Script a model that reads the figure as given, but re-reads and repairs
    correctly.

    Args:
        first_reading_qasm: What the primary parser reads on its first attempt.

    Returns:
        Responder: The scripted reply function.
    """

    def respond(system_prompt: str, messages: Sequence[ChatMessage]) -> str:
        """
        Reply according to which stage is calling.

        Args:
            system_prompt: Identifies the stage.
            messages: The conversation; the first message holds the task.

        Returns:
            str: A model-style reply for that stage.
        """
        task_text = messages[0].text
        if system_prompt == GENERATOR_SYSTEM_PROMPT:
            return fenced(
                "python",
                BELL_CODE if "cx" in task_text else MISREAD_CODE
            )

        if system_prompt == CLAIMS_SYSTEM_PROMPT:
            return fenced("json", BELL_CLAIMS)

        if system_prompt == REDRAW_SYSTEM_PROMPT:
            return fenced("json", CONSISTENT_VERDICT)

        if (
            INDEPENDENT_READING_INSTRUCTION in task_text or
            REPAIR_MARKER in task_text
        ):
            return fenced("qasm", BELL_QASM)

        return fenced("qasm", first_reading_qasm)

    return respond


def build_pipeline(
    responder: Responder,
    max_repair_iterations: int
) -> QuScribePipeline:
    """
    Assemble a pipeline around the scripted model.

    Args:
        responder: The scripted reply function.
        max_repair_iterations: Repair budget.

    Returns:
        QuScribePipeline: The pipeline under test.
    """
    tracker = UsageTracker()
    model = CallbackVisionModel(responder, tracker)
    settings = PipelineSettings(max_repair_iterations=max_repair_iterations)
    return PipelineBuilder(settings, model, model, tracker).build()


def bell_input() -> CircuitInput:
    """
    Build a figure input whose caption states the Bell distribution.

    Returns:
        CircuitInput: The test input.
    """
    return CircuitInput(
        figure=blank_figure(),
        caption="Bell pair: outcomes 00 and 11, each with probability 1/2."
    )


def test_correct_reading_is_verified_without_repair() -> None:
    """
    A correct first reading passes every check.
    """
    result = build_pipeline(
        make_responder(BELL_QASM),
        max_repair_iterations=2
    ).run(bell_input())

    assert result.label is TrustLabel.VERIFIED
    assert result.repair_steps == ()
    assert all(
        check.status is CheckStatus.PASSED
        for check in result.report.check_results
    )

    # Parse, generate, independent read, redraw, claims.
    assert result.usage.call_count == 5


def test_misreading_is_caught_and_repaired() -> None:
    """
    A missed gate is caught by the verifier and fixed by one repair.
    """
    result = build_pipeline(
        make_responder(MISREAD_QASM),
        max_repair_iterations=2
    ).run(bell_input())

    assert result.label is TrustLabel.VERIFIED
    assert len(result.repair_steps) == 1
    addressed = " ".join(result.repair_steps[0].addressed_discrepancies)
    assert "cx" in addressed and "paper claims" in addressed


def test_misreading_without_repair_budget_is_flagged() -> None:
    """
    With repair disabled, the misreading is flagged rather than returned
    as correct.
    """
    result = build_pipeline(
        make_responder(MISREAD_QASM),
        max_repair_iterations=0
    ).run(bell_input())

    assert result.label is TrustLabel.FLAGGED
    failed_checks = {
        check.check_name for check in result.report.check_results
        if check.status is CheckStatus.FAILED
    }

    assert failed_checks == {
        "structural: independent reading", "behavioral: paper claims"
    }


def test_unusable_model_output_yields_flagged_result_not_exception() -> None:
    """
    When the model never produces valid output, the result is flagged
    with a reason.
    """

    def respond_with_prose(
        system_prompt: str,
        messages: Sequence[ChatMessage]
    ) -> str:
        """
        Reply without any code block.

        Args:
            system_prompt: Unused.
            messages: Unused.

        Returns:
            str: Prose with no code.
        """
        return "I think it is a Bell circuit."

    result = build_pipeline(
        respond_with_prose,
        max_repair_iterations=1
    ).run(bell_input())

    assert result.label is TrustLabel.FLAGGED
    assert result.generated is None
    assert "No valid answer" in result.error_message


def test_underspecified_figure_is_flagged_without_retrying() -> None:
    """
    An undefined oracle box stops the run with the reason, after a single
    model call.
    """

    def report_missing_oracle(
        system_prompt: str,
        messages: Sequence[ChatMessage],
    ) -> str:
        """
        Reply that the oracle's contents are not shown.

        Args:
            system_prompt: Unused.
            messages: Unused.

        Returns:
            str: An underspecified-figure reply.
        """
        return (
            f"{UNDERSPECIFIED_MARKER} the gates inside the U_f box "
            "are not shown or defined."
        )

    result = build_pipeline(
        report_missing_oracle,
        max_repair_iterations=2,
    ).run(bell_input())
    assert result.label is TrustLabel.FLAGGED
    assert "does not fully specify" in result.error_message
    assert "U_f" in result.error_message
    # Retrying would only pressure the model into inventing gates.
    assert result.usage.call_count == 1


def test_code_generation_can_use_its_own_model() -> None:
    """
    With a separate generator model, only that model receives code-generation
    requests.
    """
    responder = make_responder(BELL_QASM)
    prompts_by_model: dict[str, list[str]] = {"primary": [], "generator": []}

    def recording_responder(model_label: str) -> Responder:
        """
        Wrap the scripted responder so it records which model was asked what.

        Args:
            model_label: The key under which to record system prompts.

        Returns:
            Responder: The recording reply function.
        """

        def respond(
            system_prompt: str,
            messages: Sequence[ChatMessage]
        ) -> str:
            """
            Record the system prompt, then reply as scripted.

            Args:
                system_prompt: Identifies the stage.
                messages: The conversation so far.

            Returns:
                str: The scripted reply.
            """
            prompts_by_model[model_label].append(system_prompt)
            return responder(system_prompt, messages)

        return respond

    tracker = UsageTracker()
    primary_model = CallbackVisionModel(
        recording_responder("primary"), tracker
    )
    generator_model = CallbackVisionModel(
        recording_responder("generator"), tracker
    )
    pipeline = PipelineBuilder(
        PipelineSettings(max_repair_iterations=0),
        primary_model,
        primary_model,
        tracker,
        generator_model=generator_model,
    ).build()
    result = pipeline.run(bell_input())
    assert result.label is TrustLabel.VERIFIED
    assert prompts_by_model["generator"] == [GENERATOR_SYSTEM_PROMPT]
    assert GENERATOR_SYSTEM_PROMPT not in prompts_by_model["primary"]
