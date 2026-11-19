"""
Benchmark items and the JSONL manifest format shared by every dataset.

The real-figure benchmark and the synthetic multi-style dataset use the
same format: one JSON object per line, with image paths relative to the
manifest file.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from qiskit import QuantumCircuit

from quscribe.core.images import ImagePayload
from quscribe.core.models import CircuitInput
from quscribe.core.qasm import load_qasm
from quscribe.errors import InvalidInputError

MANIFEST_FILENAME = "manifest.jsonl"

# Image folders of a figure set: plain figures, and figures with title
# and caption drawn in.
FIGURE_SET_IMAGE_VARIANTS = ("clean", "with_caption")
METADATA_DIRECTORY_NAME = "metadata"


@dataclass(frozen=True)
class BenchmarkItem:
    """
    One figure with its ground-truth circuit.

    Attributes:
        item_id: Unique identifier.
        image_path: Path to the figure image.
        reference_qasm: The manually verified (or generated) reference circuit.
        caption: The figure caption, if any.
        paper_text: Relevant paper passages, if any.
        style: Drawing style label, for per-style analysis.
        source: Provenance, e.g. an arXiv identifier or ``"synthetic"``.
    """

    item_id: str
    image_path: Path
    reference_qasm: str
    caption: str = ""
    paper_text: str = ""
    style: str = ""
    source: str = ""

    def to_circuit_input(self) -> CircuitInput:
        """
        Build the pipeline input for this item (the reference is
        never included).

        Args:
            None.

        Returns:
            CircuitInput: The figure, caption and paper text.

        Raises:
            InvalidInputError: If the image cannot be read.
        """
        return CircuitInput(
            figure=ImagePayload.from_file(self.image_path),
            caption=self.caption,
            paper_text=self.paper_text,
        )

    def reference_circuit(self) -> QuantumCircuit:
        """
        Load the ground-truth circuit.

        Args:
            None.

        Returns:
            QuantumCircuit: The reference circuit.

        Raises:
            QasmParseError: If the reference program is invalid.
        """
        return load_qasm(self.reference_qasm)


def write_manifest(items: list[BenchmarkItem], manifest_path: Path) -> None:
    """
    Write items to a JSONL manifest with image paths relative to it.

    Args:
        items: The items to write.
        manifest_path: Destination file.

    Returns:
        None.
    """
    manifest_directory = manifest_path.parent
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for item in items:
            record = {
                **asdict(item),
                "image_path": os.path.relpath(
                    item.image_path,
                    manifest_directory
                )
            }

            manifest_file.write(json.dumps(record) + "\n")


def load_manifest(manifest_path: Path) -> list[BenchmarkItem]:
    """
    Read a JSONL manifest.

    Args:
        manifest_path: The manifest file.

    Returns:
        list[BenchmarkItem]: The items, with image paths resolved.

    Raises:
        InvalidInputError: If the file is missing or an entry is malformed.
    """
    items: list[BenchmarkItem] = []
    try:
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InvalidInputError(
            f"Cannot read manifest {manifest_path}: {error}") from error

    for line_number, line in enumerate(manifest_lines, start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
            record["image_path"] = manifest_path.parent / record["image_path"]
            items.append(BenchmarkItem(**record))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise InvalidInputError(
                f"{manifest_path}:{line_number}: invalid entry ({error})."
            ) from error

    return items


def import_figure_set(
    source_directory: Path, image_variant: str
) -> list[BenchmarkItem]:
    """
    Build benchmark items from a figure set with one JSON metadata
    file per figure.

    Expected layout: ``metadata/<id>.json`` (with ``figure_id``, ``caption``
    and ``reference_qasm``, a path relative to the set), the reference QASM
    files, and one image folder per variant holding ``<id>.png``.

    Args:
        source_directory: Root folder of the figure set.
        image_variant: Which image folder to use, one of
                        ``FIGURE_SET_IMAGE_VARIANTS``.

    Returns:
        list[BenchmarkItem]: One item per metadata file, sorted by file name.

    Raises:
        InvalidInputError: If the variant is unknown, the set is empty, or a
                        figure's metadata, reference QASM or image is missing.
    """
    if image_variant not in FIGURE_SET_IMAGE_VARIANTS:
        raise InvalidInputError(
            f"Unknown image variant {image_variant!r}; use one of "
            f"{FIGURE_SET_IMAGE_VARIANTS}."
        )
    metadata_paths = sorted(
        (source_directory / METADATA_DIRECTORY_NAME).glob("*.json")
    )

    if not metadata_paths:
        raise InvalidInputError(
            f"No {METADATA_DIRECTORY_NAME}/*.json files found in "
            f"{source_directory}."
        )

    return [
        _figure_set_item(metadata_path, source_directory, image_variant)
        for metadata_path in metadata_paths
    ]


def _figure_set_item(
    metadata_path: Path, source_directory: Path, image_variant: str
) -> BenchmarkItem:
    """Build one benchmark item from a figure's metadata file.

    Args:
        metadata_path: The figure's JSON metadata file.
        source_directory: Root folder of the figure set.
        image_variant: Which image folder to use.

    Returns:
        BenchmarkItem: The item, with the caption as text context.

    Raises:
        InvalidInputError: If the metadata, reference QASM or image is missing
            or unreadable.
    """
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        figure_id = metadata["figure_id"]
        reference_qasm_path = source_directory / metadata["reference_qasm"]
        reference_qasm = reference_qasm_path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise InvalidInputError(
            f"{metadata_path}: cannot import this figure ({error})."
        ) from error
    image_path = source_directory / image_variant / f"{figure_id}.png"
    if not image_path.is_file():
        raise InvalidInputError(
            f"{metadata_path}: image {image_path} does not exist."
        )
    return BenchmarkItem(
        item_id=figure_id,
        image_path=image_path,
        reference_qasm=reference_qasm,
        caption=metadata.get("caption", ""),
        style=metadata.get("drawing_style", ""),
        source=metadata.get("category", ""),
    )
