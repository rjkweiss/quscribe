"""
Rendering circuits as images in several drawing styles.

Used by the redraw-comparison check, the synthetic multi-style dataset
generator and the web demo, so all three draw circuits the same way.
"""

import io
from enum import Enum

from matplotlib import pyplot
from matplotlib.figure import Figure
from qiskit import QuantumCircuit
from qiskit.circuit import Operation
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError

from quscribe.core.images import ImagePayload
from quscribe.errors import RenderingError

RENDER_DPI = 150
TEXT_FONT_SIZE_POINTS = 10
# Approximate glyph advance and line height of a 10 pt monospace font (inches).
TEXT_CHARACTER_WIDTH_INCHES = 0.085
TEXT_LINE_HEIGHT_INCHES = 0.17
MIN_TEXT_FIGURE_WIDTH_INCHES = 2.0
MIN_TEXT_FIGURE_HEIGHT_INCHES = 1.0
STANDARD_GATE_NAMES = frozenset(get_standard_gate_name_mapping())


class DiagramStyle(str, Enum):
    """
    Drawing styles available without external tools such as LaTeX.

    Attributes:
        IQP: Qiskit's default colourful matplotlib style.
        CLIFFORD: Qiskit's Clifford matplotlib style.
        TEXTBOOK: Qiskit's textbook matplotlib style.
        BLACK_AND_WHITE: Qiskit's monochrome matplotlib style.
        TEXT: Qiskit's ASCII-art drawing rendered as an image.
    """

    IQP = "iqp"
    CLIFFORD = "clifford"
    TEXTBOOK = "textbook"
    BLACK_AND_WHITE = "bw"
    TEXT = "text"


def render_circuit(
    circuit: QuantumCircuit,
    style: DiagramStyle = DiagramStyle.IQP,
    show_named_gate_contents: bool = False,
) -> ImagePayload:
    """
    Draw a circuit as a PNG image, with q[0] on the top wire.

    Args:
        circuit: The circuit to draw.
        style: The drawing style.
        show_named_gate_contents: Draw each named gate (such as U_f) as its
            gates between labelled barriers instead of a solid block, as
            figures in papers do.

    Returns:
        ImagePayload: The rendered image.

    Raises:
        RenderingError: If drawing fails (e.g. ``pylatexenc`` is not installed)
    """
    try:
        drawable_circuit = (
            expand_named_gates(circuit)
            if show_named_gate_contents else circuit
        )
        if style is DiagramStyle.TEXT:
            figure = _draw_text_figure(drawable_circuit)
        else:
            figure = drawable_circuit.draw(output="mpl", style=style.value)
    except (ImportError, QiskitError) as error:
        raise RenderingError(
            f"Could not draw the circuit in the {style.value!r} style: {error}"
        ) from error

    buffer = io.BytesIO()

    try:
        figure.savefig(
            buffer,
            format="png",
            dpi=RENDER_DPI,
            bbox_inches="tight",
            facecolor="white"
        )
    finally:
        pyplot.close(figure)

    return ImagePayload.from_bytes(buffer.getvalue())


def expand_named_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Replace each named gate with its own gates, framed by barriers,
    for display.

    The first barrier carries the gate's name, so the drawing shows a
    labeled dashed region like a paper's oracle box, and the barriers keep
    the columns around it aligned. Gates nested inside a named gate stay as
    blocks; only one level is expanded.

    Args:
        circuit: The circuit to expand.

    Returns:
        QuantumCircuit: An equivalent circuit for drawing, with the same
            registers.
    """
    expanded_circuit = circuit.copy_empty_like()
    for instruction in circuit.data:
        operation = instruction.operation
        if not _is_named_gate(operation):
            expanded_circuit.append(
                operation,
                instruction.qubits,
                instruction.clbits
            )
            continue
        definition = operation.definition
        expanded_circuit.barrier(*instruction.qubits, label=operation.name)
        for inner_instruction in definition.data:
            # Map the definition's own qubits onto the wires the
            # named gate acts on.
            outer_qubits = [
                instruction.qubits[definition.find_bit(qubit).index]
                for qubit in inner_instruction.qubits
            ]
            expanded_circuit.append(inner_instruction.operation, outer_qubits)
        expanded_circuit.barrier(*instruction.qubits)
    return expanded_circuit


def _is_named_gate(operation: Operation) -> bool:
    """
    Report whether an operation is a user-defined gate with drawable contents.

    Args:
        operation: The operation to classify.

    Returns:
        bool: True for gates outside Qiskit's standard set that have a
            definition.
    """
    return (
        operation.name not in STANDARD_GATE_NAMES and
        getattr(operation, "definition", None) is not None
    )


def _draw_text_figure(circuit: QuantumCircuit) -> Figure:
    """
    Render Qiskit's ASCII-art drawing of a circuit into a matplotlib figure.

    Args:
        circuit: The circuit to draw.

    Returns:
        Figure: A figure containing the monospace drawing.
    """
    diagram_text = str(circuit.draw(output="text", fold=-1))
    diagram_lines = diagram_text.splitlines() or [""]
    widest_line_length = max(len(line) for line in diagram_lines)
    figure = Figure(figsize=(
        max(
            MIN_TEXT_FIGURE_WIDTH_INCHES,
            widest_line_length * TEXT_CHARACTER_WIDTH_INCHES
        ),
        max(
            MIN_TEXT_FIGURE_HEIGHT_INCHES,
            len(diagram_lines) * TEXT_LINE_HEIGHT_INCHES
        ),
    ))

    figure.text(
        0.02,
        0.98,
        diagram_text,
        family="monospace",
        fontsize=TEXT_FONT_SIZE_POINTS,
        verticalalignment="top",
        linespacing=1.0,
    )

    return figure
