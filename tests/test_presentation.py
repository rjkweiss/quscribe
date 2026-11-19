"""
Tests for formatting pipeline results for the web demo.
"""

from dataclasses import fields

from quscribe.app.presentation import (
    ResultView,
    build_result_view,
    format_report_markdown,
)
from quscribe.core.models import (
    CheckResult,
    CheckStatus,
    PipelineResult,
    TrustLabel,
    VerificationReport,
)
from quscribe.core.usage import UsageSummary

# Must match the order of the `outputs` list in quscribe/app/web_app.py.
EXPECTED_OUTPUT_ORDER = (
    "label_html", "qasm", "circuit_image", "source_code",
    "report_markdown", "repair_markdown", "usage_markdown",
)


def make_result(
    report: VerificationReport | None,
    error_message: str | None = None,
) -> PipelineResult:
    """
    Build a pipeline result with no transcription or code,
    as after an early failure.

    Args:
        report: The verification report, or None.
        error_message: Why the run stopped, if it did.

    Returns:
        PipelineResult: The result.
    """
    return PipelineResult(
        label=report.label if report else TrustLabel.FLAGGED,
        parsed=None,
        generated=None,
        report=report,
        repair_steps=(),
        usage=UsageSummary(
            call_count=2,
            input_tokens=1500,
            output_tokens=300,
            model_latency_seconds=4.2,
        ),
        wall_clock_seconds=5.0,
        error_message=error_message,
    )


def test_output_order_matches_web_app_components() -> None:
    """
    Reordering ResultView fields must match web_app.py's outputs list.

    Args:
        None.

    Returns:
        None.
    """
    assert (
        tuple(field_info.name for field_info in fields(ResultView))
        == EXPECTED_OUTPUT_ORDER
    )


def test_pipe_in_summary_does_not_break_the_table() -> None:
    """
    A "|" in a summary is escaped; discrepancies appear below the table.

    Args:
        None.

    Returns:
        None.
    """
    report = VerificationReport.from_check_results(
            [
                CheckResult(
                    "check",
                    CheckStatus.FAILED,
                    "differs | see below",
                    ("missing cx",),
                )
            ]
    )
    markdown = format_report_markdown(report)
    assert "differs \\| see below" in markdown
    assert "- [check] missing cx" in markdown


def test_failed_run_shows_escaped_reason_and_placeholders() -> None:
    """
    An early failure shows its reason safely and placeholders instead of empty
    panels.

    Args:
        None.

    Returns:
        None.
    """
    view = build_result_view(
        make_result(report=None, error_message="model <unavailable>")
    )
    assert "Stopped early: model &lt;unavailable&gt;" in view.label_html
    assert view.qasm == "" and view.circuit_image is None
    assert view.report_markdown.startswith("Not produced")
    assert view.repair_markdown == "No repairs were made."
    assert "- Model calls: 2" in view.usage_markdown


def test_verified_run_shows_verified_badge() -> None:
    """
    A verified report produces the verified badge.

    Args:
        None.

    Returns:
        None.
    """
    report = VerificationReport.from_check_results(
        [CheckResult("check", CheckStatus.PASSED, "Matches.")]
    )
    assert "Verified" in build_result_view(make_result(report)).label_html


def test_usage_panel_names_the_model_behind_each_role() -> None:
    """
    The cost-and-time panel starts with the models used, then the
    usage figures.

    Args:
        None.

    Returns:
        None.
    """
    roles = {
        "Transcription and repair": "qwen3.5:9b (ollama)",
        "Verification": "claude-opus-5-5 (anthropic)",
    }
    view = build_result_view(make_result(report=None), roles)
    assert view.usage_markdown.startswith(
        "**Models used**\n"
        "- Transcription and repair: qwen3.5:9b (ollama)"
    )
    assert "- Verification: claude-opus-5-5 (anthropic)" in view.usage_markdown
    assert "- Model calls: 2" in view.usage_markdown
