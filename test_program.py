import matplotlib

# see note below
from quscribe.app.gradio_app import launch_app
from quscribe.pipeline.settings import PipelineSettings

matplotlib.use("Agg")

launch_app(PipelineSettings(), host="127.0.0.1", port=7860, share=False)
