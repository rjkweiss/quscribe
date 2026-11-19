"""
Gradio web demo: upload a figure, get Qiskit code, a verification report and
a trust label.
"""

from dataclasses import replace

import gradio
import matplotlib

from quscribe.app.presentation import build_result_view
from quscribe.core.images import ImagePayload
from quscribe.core.models import CircuitInput
from quscribe.errors import ConfigurationError, InvalidInputError
from quscribe.llm.factory import ModelProvider, ModelSpec
from quscribe.pipeline.builder import PipelineBuilder
from quscribe.pipeline.settings import PipelineSettings

PROVIDER_CHOICES = [provider.value for provider in ModelProvider]
# Shortcuts for the models in the evaluation; any other model can still be
# typed in.
MODEL_PRESETS = {
    "Claude Opus 5.5": ModelSpec(ModelProvider.ANTHROPIC, "claude-opus-5-5"),
    "Claude Sonnet 5.5": ModelSpec(
        ModelProvider.ANTHROPIC,
        "claude-sonnet-5-5"
    ),
    "Qwen3.5 9B (local, Ollama)": ModelSpec(
        ModelProvider.OLLAMA,
        "qwen3.5:9b"
    ),
    "GPT Astra 6.2": ModelSpec(
        ModelProvider.OPENAI,
        "gpt-6.2-astra"
    )
}
SAME_AS_TRANSCRIPTION_MODEL = "Same as the transcription model"
VERIFIER_CHOICES = [SAME_AS_TRANSCRIPTION_MODEL, *MODEL_PRESETS]

MAX_REPAIR_ITERATIONS_IN_UI = 5
APP_INTRODUCTION = (
    "# QuScribe\n"
    "Turn a quantum circuit figure from a paper into Qiskit code, and see "
    "whether the code can be trusted. Adding the caption and nearby paper "
    "text lets QuScribe test the paper's own claims against a simulation of "
    "the code."
)


class QuScribeWebApp:
    """
    Builds the Gradio interface and runs the pipeline for each request."""

    def __init__(self, base_settings: PipelineSettings) -> None:
        """
        Create the app.

        Args:
            base_settings: Defaults for the controls and settings the
                UI does not expose.
        """
        self._base_settings = base_settings

    def convert(
        self,
        figure_path: str | None,
        caption: str,
        paper_text: str,
        provider_value: str,
        model_name: str,
        use_text_context: bool,
        max_repair_iterations: float,
        verifier_choice: str,
    ) -> tuple:
        """
        Run the pipeline for one request; Gradio calls this on button click.

        A fresh pipeline is built per request, so concurrent users never
        share state.

        Args:
            figure_path: Path of the uploaded figure, or None if
                nothing was uploaded.
            caption: The figure caption.
            paper_text: Relevant paper passages.
            provider_value: The selected provider's value.
            model_name: The model name; blank uses the provider default.
            use_text_context: Whether to send the caption and paper text.
            max_repair_iterations: Repair budget from the slider.
            verifier_choice: A preset name for the verifier model, or the
                same-model option. It replaces any verifier given on the
                command line.

        Returns:
            tuple: Values for the output components, in ``ResultView``
                    field order.

        Raises:
            gradio.Error: If the input or settings are invalid.
        """
        if not figure_path:
            raise gradio.Error("Upload a circuit figure to convert.")

        settings = replace(
            self._base_settings,
            primary_model=self._model_spec_for(
                ModelProvider(provider_value),
                model_name
            ),
            use_text_context=use_text_context,
            max_repair_iterations=int(max_repair_iterations),
        )

        try:
            pipeline = PipelineBuilder.from_settings(settings).build()
            circuit_input = CircuitInput(
                ImagePayload.from_file(figure_path),
                caption,
                paper_text
            )
        except (ConfigurationError, InvalidInputError) as error:
            raise gradio.Error(str(error)) from error

        return build_result_view(pipeline.run(circuit_input)).as_outputs()

    def build_interface(self) -> gradio.Blocks:
        """
        Lay out the interface.

        Returns:
            gradio.Blocks: The interface, ready to launch.
        """
        defaults = self._base_settings
        with gradio.Blocks(title="QuScribe") as interface:
            with gradio.Accordion("Model settings", open=True):
                preset_input = gradio.Dropdown(
                    choices=list(MODEL_PRESETS),
                    value=None,
                    label="Quick choice of transcription model"
                )
            gradio.Markdown(APP_INTRODUCTION)
            with gradio.Row():
                with gradio.Column(scale=1):
                    figure_input = gradio.Image(
                        type="filepath",
                        label="Circuit figure"
                    )
                    caption_input = gradio.Textbox(
                        label="Figure caption (optional)",
                        lines=2
                    )
                    paper_text_input = gradio.Textbox(
                        label="Paper text about the circuit (optional)",
                        lines=6,
                    )
                    with gradio.Accordion("Model settings", open=False):
                        provider_input = gradio.Dropdown(
                            choices=PROVIDER_CHOICES,
                            value=defaults.primary_model.provider.value,
                            label="Provider",
                        )
                        model_name_input = gradio.Textbox(
                            value=defaults.primary_model.model_name or "",
                            label="Model name",
                            placeholder=(
                                "Blank uses the Anthropic default; "
                                "OpenAI and Ollama need a name"
                            ),
                        )
                        use_text_input = gradio.Checkbox(
                            value=defaults.use_text_context,
                            label="Use the caption and paper text"
                        )
                        repair_input = gradio.Slider(
                            minimum=0,
                            maximum=MAX_REPAIR_ITERATIONS_IN_UI,
                            step=1,
                            value=defaults.max_repair_iterations,
                            label="Repair attempts",
                        )
                        verifier_input = gradio.Dropdown(
                            choices=VERIFIER_CHOICES,
                            value=SAME_AS_TRANSCRIPTION_MODEL,
                            label="Verifier model"
                        )
                    preset_input.change(
                        fn=self.apply_preset,
                        inputs=[preset_input],
                        outputs=[provider_input, model_name_input]
                    )
                    convert_button = gradio.Button(
                        "Convert figure",
                        variant="primary"
                    )
                with gradio.Column(scale=2):
                    # Output components are created in ResultView field order.
                    label_output = gradio.HTML()
                    with gradio.Tabs():
                        with gradio.Tab("Code"):
                            code_output = gradio.Code(
                                language="python",
                                label="Qiskit code",
                                interactive=False
                            )
                        with gradio.Tab("Transcription"):
                            circuit_image_output = gradio.Image(
                                type="pil",
                                label="Transcription, redrawn by Qiskit"
                            )
                            qasm_output = gradio.Code(
                                label="OpenQASM 3",
                                interactive=False
                            )
                        with gradio.Tab("Verification"):
                            report_output = gradio.Markdown()
                        with gradio.Tab("Repairs"):
                            repair_output = gradio.Markdown()
                        with gradio.Tab("Cost and time"):
                            usage_output = gradio.Markdown()
            convert_button.click(
                fn=self.convert,
                inputs=[
                    figure_input,
                    caption_input,
                    paper_text_input,
                    provider_input,
                    model_name_input,
                    use_text_input,
                    repair_input,
                    verifier_input
                ],
                outputs=[
                    label_output,
                    qasm_output,
                    circuit_image_output,
                    code_output,
                    report_output,
                    repair_output,
                    usage_output,
                ],
            )

        return interface

    @staticmethod
    def apply_preset(preset_name: str | None) -> tuple:
        """
        Fill in the provider and model name for a quick-choice preset.

        Args:
            preset_name: The chosen preset, or None when the choice is cleared.

        Returns:
            tuple: The provider value and model name, or two no-op updates
            when cleared.
        """
        spec = MODEL_PRESETS.get(preset_name)
        if spec is None:
            return gradio.update(), gradio.update()
        return spec.provider.value, spec.model_name

    def _model_spec_for(
        self,
        provider: ModelProvider,
        model_name: str
    ) -> ModelSpec:
        """
        Build the primary model specification from the UI controls.

        The configured base URL is kept only when the provider is unchanged,
        since a URL for one provider is meaningless for another.

        Args:
            provider: The selected provider.
            model_name: The model name from the text box.

        Returns:
            ModelSpec: The model specification for this request.
        """
        default_spec = self._base_settings.primary_model
        base_url = (
            default_spec.base_url
            if provider is default_spec.provider
            else None
        )
        return ModelSpec(
            provider=provider,
            model_name=model_name.strip() or None,
            base_url=base_url
        )


def launch_app(
    settings: PipelineSettings,
    host: str,
    port: int | None,
    share: bool
) -> None:
    """
    Build and serve the web demo.

    Args:
        settings: Default pipeline settings.
        host: Interface to bind, e.g. ``"127.0.0.1"`` or ``"0.0.0.0"``.
        port: Port to bind; None lets Gradio choose.
        share: Whether to create a temporary public Gradio link.
    """
    matplotlib.use("Agg")

    QuScribeWebApp(settings).build_interface().launch(
        server_name=host,
        server_port=port,
        share=share
    )
