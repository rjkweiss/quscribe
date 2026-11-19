"""
Tests for importing a figure set laid out as metadata/, qasm/ and
image folders.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from quscribe.errors import InvalidInputError
from quscribe.evaluation.dataset import (
    import_figure_set,
    load_manifest,
    write_manifest,
)

BELL_QASM = (
    'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\n'
    'h q[0];\ncx q[0], q[1];\n'
)


def make_figure_set(root: Path, include_captioned_image: bool = True) -> None:
    """
    Create a one-figure set in the demo-set layout.

    Args:
        root: Folder to create the set in.
        include_captioned_image: False leaves the with_caption image out.

    Returns:
        None.
    """
    for folder_name in ("metadata", "qasm", "clean", "with_caption"):
        (root / folder_name).mkdir(parents=True)
    (root / "qasm" / "bell_state.qasm").write_text(BELL_QASM)
    metadata = {
        "figure_id": "bell_state",
        "caption": (
            "Bell pair: measuring gives 00 or 11 with equal probability."
        ),
        "reference_qasm": "qasm/bell_state.qasm",
        "drawing_style": "paper-like",
        "category": "dev_demo",
    }
    (root / "metadata" / "bell_state.json").write_text(json.dumps(metadata))
    Image.new("RGB", (8, 8), "white").save(root / "clean" / "bell_state.png")
    if include_captioned_image:
        Image.new("RGB", (8, 8), "white").save(
            root / "with_caption" / "bell_state.png"
        )


def test_import_uses_metadata_caption_and_reference(
    tmp_path: Path,
) -> None:
    """
    Items take their caption, reference program and provenance from
    the metadata.

    Args:
        tmp_path: Pytest's temporary directory.

    Returns:
        None.
    """
    make_figure_set(tmp_path / "set")
    (item,) = import_figure_set(tmp_path / "set", "clean")
    assert item.item_id == "bell_state"
    assert item.caption.startswith("Bell pair")
    assert item.image_path == tmp_path / "set" / "clean" / "bell_state.png"
    assert "cx q[0], q[1];" in item.reference_qasm
    assert item.source == "dev_demo"


def test_imported_items_round_trip_through_a_manifest_elsewhere(
    tmp_path: Path,
) -> None:
    """
    A manifest written outside the set still points at the set's images.

    Args:
        tmp_path: Pytest's temporary directory.

    Returns:
        None.
    """
    make_figure_set(tmp_path / "set")
    manifest_path = tmp_path / "manifests" / "manifest_with_caption.jsonl"
    manifest_path.parent.mkdir()
    write_manifest(
        import_figure_set(tmp_path / "set", "with_caption"), manifest_path
    )
    (loaded_item,) = load_manifest(manifest_path)
    expected_image = tmp_path / "set" / "with_caption" / "bell_state.png"
    assert loaded_item.image_path.resolve() == expected_image.resolve()
    assert loaded_item.caption.startswith("Bell pair")


def test_missing_image_and_unknown_variant_are_reported(
    tmp_path: Path,
) -> None:
    """
    Report set problems clearly at import time, not mid-evaluation.

    Args:
        tmp_path: Pytest's temporary directory.

    Returns:
        None.
    """
    make_figure_set(tmp_path / "set", include_captioned_image=False)
    with pytest.raises(InvalidInputError, match="does not exist"):
        import_figure_set(tmp_path / "set", "with_caption")
    with pytest.raises(InvalidInputError, match="Unknown image variant"):
        import_figure_set(tmp_path / "set", "sketch")
