"""
Command-line interface:
``quscribe convert | synthesize | import-figures | evaluate | app``.

Exit status: 0 when the work succeeded (for ``convert``: the code was
verified), 1 when ``convert`` produced flagged code, and 2 on errors.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

from quscribe.core.images import ImagePayload
from quscribe.core.models import CircuitInput, PipelineResult, TrustLabel
from quscribe.core.rendering import DiagramStyle
from quscribe.core.serialization import to_jsonable
from quscribe.core.usage import ModelPricing
from quscribe.errors import InvalidInputError, QuScribeError
from quscribe.evaluation.dataset import (
    FIGURE_SET_IMAGE_VARIANTS,
    import_figure_set,
    load_manifest,
    write_manifest,
)
from quscribe.evaluation.experiment import ExperimentRunner, standard_ablations
from quscribe.evaluation.synthetic import (
    SyntheticDatasetConfig,
    SyntheticDatasetGenerator,
)
from quscribe.llm.factory import ModelProvider, ModelSpec
from quscribe.pipeline.builder import PipelineBuilder
from quscribe.pipeline.settings import PipelineSettings

logger = logging.getLogger("quscribe")

EXIT_SUCCESS = 0
EXIT_FLAGGED = 1
EXIT_ERROR = 2
PROVIDER_CHOICES = [provider.value for provider in ModelProvider]
STYLE_CHOICES = [style.value for style in DiagramStyle]
ABLATION_CHOICES = (
    "image_only",
    "image_and_text",
    "reference_free_repair",
    "oracle_repair",
)
DEFAULTS = PipelineSettings()


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and run the chosen subcommand.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        int: The process exit status.
    """
    # The entry point chooses a headless backend; GUI backends break
    # in server threads.
    matplotlib.use("Agg")
    arguments = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=arguments.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return arguments.handler(arguments)
    except QuScribeError as error:
        logger.error("%s", error)
        return EXIT_ERROR


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Define the command-line interface.

    Args:
        None.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="quscribe",
        description="Verified Qiskit code from circuit figures.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert one figure into verified Qiskit code.",
    )
    convert_parser.add_argument(
        "figure",
        type=Path,
        help="Path to the circuit figure image.",
    )
    convert_parser.add_argument(
        "--caption",
        default="",
        help="The figure caption.",
    )
    convert_parser.add_argument(
        "--paper-text-file",
        type=Path,
        help="Text file with relevant paper passages.",
    )
    convert_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quscribe_output"),
    )
    _add_pipeline_arguments(convert_parser)
    convert_parser.set_defaults(handler=_run_convert)

    synthesize_parser = subparsers.add_parser(
        "synthesize",
        help="Generate the synthetic multi-style dataset.",
    )
    synthesize_parser.add_argument("--output-dir", type=Path, required=True)
    synthesize_parser.add_argument("--count", type=int, default=100)
    synthesize_parser.add_argument("--min-qubits", type=int, default=2)
    synthesize_parser.add_argument("--max-qubits", type=int, default=6)
    synthesize_parser.add_argument("--min-gates", type=int, default=3)
    synthesize_parser.add_argument("--max-gates", type=int, default=20)
    synthesize_parser.add_argument(
        "--styles",
        nargs="+",
        choices=STYLE_CHOICES,
        default=STYLE_CHOICES,
    )
    synthesize_parser.add_argument("--seed", type=int, default=0)
    synthesize_parser.set_defaults(handler=_run_synthesize)

    import_parser = subparsers.add_parser(
        "import-figures",
        help=(
            "Write manifests for a figure set "
            "(metadata/, qasm/, clean/, with_caption/)."
        ),
    )
    import_parser.add_argument(
        "source",
        type=Path,
        help="Root folder of the figure set.",
    )
    import_parser.add_argument("--output-dir", type=Path, required=True)
    import_parser.set_defaults(handler=_run_import_figures)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run ablations over a benchmark manifest.",
    )
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quscribe_evaluation"),
    )
    evaluate_parser.add_argument(
        "--max-items",
        type=int,
        help="Evaluate only the first N items.",
    )
    evaluate_parser.add_argument(
        "--configurations",
        nargs="+",
        choices=ABLATION_CHOICES,
        default=list(ABLATION_CHOICES),
    )
    evaluate_parser.add_argument(
        "--input-price",
        type=float,
        help="USD per million input tokens.",
    )
    evaluate_parser.add_argument(
        "--output-price",
        type=float,
        help="USD per million output tokens.",
    )
    _add_pipeline_arguments(evaluate_parser)
    evaluate_parser.set_defaults(handler=_run_evaluate)

    app_parser = subparsers.add_parser(
        "app",
        help="Launch the Gradio web demo.",
    )
    app_parser.add_argument("--host", default="127.0.0.1")
    app_parser.add_argument("--port", type=int)
    app_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public link.",
    )
    _add_pipeline_arguments(app_parser)
    app_parser.set_defaults(handler=_run_app)
    return parser


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add the model and pipeline options shared by several subcommands.

    Args:
        parser: The subcommand parser to extend.

    Returns:
        None.
    """
    parser.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=DEFAULTS.primary_model.provider.value,
    )
    parser.add_argument(
        "--model",
        help="Model name; required for openai and ollama.",
    )
    parser.add_argument(
        "--base-url",
        help="Server URL for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--verifier-provider",
        choices=PROVIDER_CHOICES,
        help="Separate provider for verification.",
    )
    parser.add_argument(
        "--verifier-model",
        help="Separate model for verification.",
    )
    parser.add_argument(
        "--verifier-base-url",
        help="Server URL for the verifier model.",
    )
    parser.add_argument(
        "--generator-provider",
        choices=PROVIDER_CHOICES,
        help="Separate provider for code generation."
    )
    parser.add_argument(
        "--generator-model",
        help="Separate model for code generation."
    )
    parser.add_argument(
        "--generator-base-url",
        help="Server URL for the code-generation model."
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=DEFAULTS.max_repair_iterations,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULTS.max_attempts_per_call,
    )
    parser.add_argument(
        "--fidelity-threshold",
        type=float,
        default=DEFAULTS.fidelity_threshold,
    )
    parser.add_argument(
        "--code-timeout",
        type=float,
        default=DEFAULTS.code_timeout_seconds,
    )
    parser.add_argument(
        "--no-text-context",
        action="store_true",
        help="Ignore the caption and paper text.",
    )


def _settings_from_arguments(
    arguments: argparse.Namespace,
) -> PipelineSettings:
    """
    Build pipeline settings from parsed arguments.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        PipelineSettings: The validated settings.

    Raises:
        ConfigurationError: If a value is out of range.
    """
    return PipelineSettings(
        primary_model=ModelSpec(
            ModelProvider(arguments.provider),
            arguments.model,
            arguments.base_url,
        ),
        verifier_model=_optional_model_spec(
            arguments.verifier_provider,
            arguments.verifier_model,
            arguments.verifier_base_url,
            arguments.provider
        ),
        generator_model=_optional_model_spec(
            arguments.generator_provider,
            arguments.generator_model,
            arguments.generator_base_url,
            arguments.provider
        ),
        use_text_context=not arguments.no_text_context,
        max_repair_iterations=arguments.max_repairs,
        max_attempts_per_call=arguments.max_attempts,
        fidelity_threshold=arguments.fidelity_threshold,
        code_timeout_seconds=arguments.code_timeout,
    )


def _optional_model_spec(
    provider_value: str | None,
    model_name: str | None,
    base_url: str | None,
    default_provider: str
) -> ModelSpec | None:
    """
    Build the model specification for an optional role, if any of its flags
    were given.

    Args:
        provider_value: The role's provider flag, or None.
        model_name: The role's model flag, or None.
        base_url: The role's server URL flag, or None.
        default_provider: The primary provider, used when only a model name
            is given.

    Returns:
        ModelSpec | None: The specification, or None when the role reuses the
            primary model.
    """
    if not provider_value and not model_name:
        return None
    return ModelSpec(
        ModelProvider(provider_value or default_provider),
        model_name,
        base_url
    )


def _run_convert(arguments: argparse.Namespace) -> int:
    """
    Convert one figure and write the code, transcription and full result.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        int: EXIT_SUCCESS if verified, EXIT_FLAGGED otherwise.
    """
    paper_text = (
        _read_text_file(arguments.paper_text_file)
        if arguments.paper_text_file
        else ""
    )
    circuit_input = CircuitInput(
        ImagePayload.from_file(arguments.figure),
        arguments.caption,
        paper_text,
    )
    settings = _settings_from_arguments(arguments)
    result = PipelineBuilder.from_settings(settings).build().run(circuit_input)
    _write_convert_outputs(result, arguments.output_dir)
    for role, model in settings.model_roles().items():
        print(f"{role}: {model}")
    print(f"Trust label: {result.label.value}")
    if result.error_message:
        print(f"Stopped early: {result.error_message}")
    for discrepancy in result.report.discrepancies if result.report else ():
        print(f"  - {discrepancy}")
    print(f"Outputs written to {arguments.output_dir}")
    return (
        EXIT_SUCCESS
        if result.label is TrustLabel.VERIFIED
        else EXIT_FLAGGED
    )


def _write_convert_outputs(
    result: PipelineResult,
    output_directory: Path,
) -> None:
    """
    Write the generated code, transcription and full JSON result.

    Args:
        result: The pipeline result.
        output_directory: Destination directory (created if needed).

    Returns:
        None.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    if result.generated:
        (
            output_directory / "circuit.py"
        ).write_text(result.generated.source_code, encoding="utf-8")
    if result.parsed:
        (
            output_directory / "circuit.qasm"
        ).write_text(result.parsed.qasm, encoding="utf-8")
    _write_json(to_jsonable(result), output_directory / "result.json")


def _run_synthesize(arguments: argparse.Namespace) -> int:
    """
    Generate the synthetic multi-style dataset.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        int: EXIT_SUCCESS.
    """
    config = SyntheticDatasetConfig(
        circuit_count=arguments.count,
        min_qubits=arguments.min_qubits,
        max_qubits=arguments.max_qubits,
        min_gates=arguments.min_gates,
        max_gates=arguments.max_gates,
        styles=tuple(DiagramStyle(style) for style in arguments.styles),
        seed=arguments.seed,
    )
    items = SyntheticDatasetGenerator(config).generate(arguments.output_dir)
    print(f"Wrote {len(items)} circuits to {arguments.output_dir}")
    return EXIT_SUCCESS


def _run_import_figures(arguments: argparse.Namespace) -> int:
    """
    Write ``manifest_clean.jsonl`` and ``manifest_with_caption.jsonl`` for a
    figure set.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        int: EXIT_SUCCESS.
    """
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for image_variant in FIGURE_SET_IMAGE_VARIANTS:
        items = import_figure_set(arguments.source, image_variant)
        manifest_path = (
            arguments.output_dir / f"manifest_{image_variant}.jsonl"
        )
        write_manifest(items, manifest_path)
        print(f"Wrote {len(items)} figures to {manifest_path}")

    return EXIT_SUCCESS

# def _run_import_figures(arguments: argparse.Namespace) -> int:
#     """Import a figure set and write one manifest for each image variant."""
#     figure_sets = import_figure_set(arguments.source)
#     for variant in FIGURE_SET_IMAGE_VARIANTS:
#         output_directory = arguments.output_dir / variant
#         output_directory.mkdir(parents=True, exist_ok=True)
#         write_manifest(
#             figure_sets[variant],
#             output_directory / "manifest.jsonl",
#         )
#     print(f"Wrote figure manifests to {arguments.output_dir}")
#     return EXIT_SUCCESS


def _run_evaluate(arguments: argparse.Namespace) -> int:
    """
    Run the selected ablations and write JSON and markdown reports.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        int: EXIT_SUCCESS.

    Raises:
        InvalidInputError: If only one of the two prices is given.
    """
    settings = _settings_from_arguments(arguments)
    items = load_manifest(arguments.manifest)[: arguments.max_items]
    selected_names = set(arguments.configurations)
    configurations = [
        configuration
        for configuration in standard_ablations(settings.max_repair_iterations)
        if configuration.name in selected_names
    ]
    report = ExperimentRunner(
        settings,
        _pricing_from_arguments(arguments),
    ).run(items, configurations)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        to_jsonable(report),
        arguments.output_dir / "evaluation_report.json"
    )
    markdown_report = report.to_markdown()
    (arguments.output_dir / "evaluation_report.md").write_text(
        markdown_report,
        encoding="utf-8",
    )
    print(markdown_report)
    return EXIT_SUCCESS


def _run_app(arguments: argparse.Namespace) -> int:
    """
    Launch the Gradio web demo (blocks until stopped).

    Gradio is imported here so the other subcommands start without loading it.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        int: EXIT_SUCCESS once the server stops.
    """
    from quscribe.app.gradio_app import launch_app

    launch_app(
        _settings_from_arguments(arguments),
        arguments.host,
        arguments.port,
        arguments.share,
    )
    return EXIT_SUCCESS


def _pricing_from_arguments(
    arguments: argparse.Namespace,
) -> ModelPricing | None:
    """
    Build token pricing when both prices are supplied.

    Args:
        arguments: Parsed command-line arguments.

    Returns:
        ModelPricing | None: The pricing, or None when neither price is given.

    Raises:
        InvalidInputError: If only one of the two prices is given.
    """
    prices = (arguments.input_price, arguments.output_price)
    if all(price is None for price in prices):
        return None
    if any(price is None for price in prices):
        raise InvalidInputError(
            "Give both --input-price and --output-price, or neither."
        )
    return ModelPricing(arguments.input_price, arguments.output_price)


def _read_text_file(text_path: Path) -> str:
    """
    Read a UTF-8 text file.

    Args:
        text_path: The file to read.

    Returns:
        str: The file contents.

    Raises:
        InvalidInputError: If the file cannot be read.
    """
    try:
        return text_path.read_text(encoding="utf-8")
    except OSError as error:
        raise InvalidInputError(f"Cannot read {text_path}: {error}") from error


def _write_json(payload: object, output_path: Path) -> None:
    """
    Write a JSON-compatible payload with indentation.

    Args:
        payload: The data to write.
        output_path: Destination file.

    Returns:
        None.
    """
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
