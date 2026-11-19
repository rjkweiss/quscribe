"""
Turns pipeline results into the text, HTML and images the web demo displays.
"""

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

from PIL import Image

from quscribe.core.models import (
    PipelineResult,
    RepairStep,
    TrustLabel,
    VerificationReport,
)
from quscribe.core.rendering import render_circuit
from quscribe.core.usage import UsageSummary
from quscribe.errors import RenderingError

# (text colour, background colour, message) per label; both colours are set
# so the badge stays legible in light and dark themes.
LABEL_PRESENTATION = {
    TrustLabel.VERIFIED: (
        "#14532d",
        "#dcfce7",
        "Verified: every check that ran agreed with the figure."
    ),
    TrustLabel.FLAGGED: (
        "#7c2d12",
        "#ffedd5",
        "Flagged: review this code before relying on it."
    ),
}

NOT_PRODUCED_TEXT = "Not produced because an earlier step failed."


@dataclass(frozen=True)
class ResultView:
    """
    Display-ready pipeline result; field order matches the web demo's outputs.

    Attributes:
        label_html: Trust-label badge.
        qasm: The final OpenQASM 3 transcription.
        circuit_image: The transcription redrawn by Qiskit, if drawable.
        source_code: The generated Qiskit code.
        report_markdown: The verification checks and discrepancies.
        repair_markdown: The repair history.
        usage_markdown: Model calls, tokens and timing.
    """

    label_html: str
    qasm: str
    circuit_image: Image.Image | None
    source_code: str
    report_markdown: str
    repair_markdown: str
    usage_markdown: str

    def as_outputs(self) -> tuple:
        """
        Return the fields in declaration order, as Gradio expects.

        Args:
            None.

        Returns:
            tuple: One value per output component.
        """
        return tuple(
            getattr(self, field_info.name) for field_info in fields(self)
        )


def build_result_view(
    result: PipelineResult,
    model_roles: Mapping[str, str] | None = None,
) -> ResultView:
    """
    Format a pipeline result for display.

    Args:
        result: The pipeline result.
        model_roles: Which model played each role, shown with cost and time.

    Returns:
        ResultView: Display-ready values for every output panel.
    """
    return ResultView(
        label_html=format_label_html(result.label, result.error_message),
        qasm=result.parsed.qasm if result.parsed else "",
        circuit_image=_render_preview(result),
        source_code=(
            result.generated.source_code
            if result.generated else f"# {NOT_PRODUCED_TEXT}"
        ),
        report_markdown=format_report_markdown(result.report),
        repair_markdown=format_repair_markdown(result.repair_steps),
        usage_markdown=format_usage_markdown(
            result.usage,
            result.wall_clock_seconds,
            model_roles
        ),
    )


def format_label_html(label: TrustLabel, error_message: str | None) -> str:
    """
    Render the trust label as a coloured badge.

    Args:
        label: The trust label.
        error_message: Why the run stopped early, if it did.

    Returns:
        str: An HTML snippet.
    """
    text_colour, background_colour, message = LABEL_PRESENTATION[label]
    detail = (
        f"<div style='margin-top:6px;font-weight:400'>Stopped early: "
        f"{html.escape(error_message)}</div>"
        if error_message else ""
    )
    return (
        f"<div style='padding:14px 18px;border-radius:8px;"
        f"background:{background_colour};color:{text_colour};font-weight:600;"
        f"font-size:1.05rem'>{html.escape(message)}{detail}</div>"
    )


def format_report_markdown(report: VerificationReport | None) -> str:
    """
    Render the verification checks as a table followed by any discrepancies.

    Args:
        report: The verification report, or None if verification never ran.

    Returns:
        str: Markdown text.
    """
    if report is None:
        return NOT_PRODUCED_TEXT
    lines = ["| Check | Result | Summary |", "|---|---|---|"]
    lines.extend(
        f"| {result.check_name} | {result.status.value} | "
        f"{_escape_table_cell(result.summary)} |"
        for result in report.check_results
    )
    if report.discrepancies:
        lines.extend([
            "",
            "**Discrepancies**",
            _bullet_list(report.discrepancies)
        ])

    return "\n".join(lines)


def format_repair_markdown(repair_steps: Sequence[RepairStep]) -> str:
    """
    Render the repair history, one section per iteration.

    Args:
        repair_steps: The repair iterations, in order.

    Returns:
        str: Markdown text.
    """
    if not repair_steps:
        return "No repairs were made."
    sections = [
        f"### Repair {step.iteration}: {step.label_after_repair.value}\n\n"
        f"Problems addressed:\n\n{_bullet_list(step.addressed_discrepancies)}"
        f"\n\n```qasm\n{step.revised_qasm}\n```"
        for step in repair_steps
    ]

    return "\n\n".join(sections)


def format_usage_markdown(
    usage: UsageSummary,
    wall_clock_seconds: float,
    model_roles: Mapping[str, str] | None = None
) -> str:
    """
    Render the models used, then model usage and timing.

    Args:
        usage: Model usage for the run.
        wall_clock_seconds: Total run time.
        model_roles: Which model played each role; omitted when None.

    Returns:
        str: Markdown text.
    """
    model_lines = [
        f"- {role}: {model}"
        for role, model in (model_roles or {}).items()
    ]
    models_section = (
        ["**Models used**", *model_lines, "", "**Cost and time**"]
        if model_lines
        else []
    )
    return "\n".join([
        *models_section,
        f"- Model calls: {usage.call_count}",
        f"- Model calls: {usage.call_count}",
        f"- Input tokens: {usage.input_tokens:,}",
        f"- Output tokens: {usage.output_tokens:,}",
        f"- Time waiting for models: {usage.model_latency_seconds:.1f} s",
        f"- Total time: {wall_clock_seconds:.1f} s",
    ])


def _render_preview(result: PipelineResult) -> Image.Image | None:
    """
    Redraw the final transcription for display.

    Args:
        result: The pipeline result.

    Returns:
        Image.Image | None: The drawing, or None if there is nothing drawable.
    """
    if result.parsed is None:
        return None
    try:
        return render_circuit(
            result.parsed.circuit,
            show_named_gate_contents=True
        ).to_pil_image()
    except RenderingError:
        return None


def _bullet_list(items: Sequence[str]) -> str:
    """
    Format items as a markdown bullet list.

    Args:
        items: The items to list.

    Returns:
        str: The bullet list, or a placeholder when empty.
    """
    return (
        "\n".join(f"- {item}" for item in items) or
        "- No details were given."
    )


def _escape_table_cell(text: str) -> str:
    """
    Escape characters that would break a markdown table cell.

    Args:
        text: The cell text.

    Returns:
        str: Text safe to place inside a table cell.
    """
    return text.replace("|", "\\|").replace("\n", " ")
