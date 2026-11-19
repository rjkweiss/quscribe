"""
Conversion of QuScribe result objects into JSON-compatible structures.
"""

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError

from quscribe.core.qasm import dump_qasm


def to_jsonable(value: Any) -> Any:
    """
    Recursively convert dataclasses, enums, paths and circuits into JSON types.

    Args:
        value: Any QuScribe result object or plain value.

    Returns:
        Any: A structure made only of dicts, lists, strings, numbers,
            booleans and None.
    """
    # handle enums
    if isinstance(value, Enum):
        return value.value

    # handle dataclasses
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: to_jsonable(getattr(value, field_info.name))
            for field_info in fields(value)
        }

    # handle circuits
    if isinstance(value, QuantumCircuit):
        return _circuit_to_text(value)

    # handle paths
    if isinstance(value, Path):
        return str(value)

    # handle objects (dictionary)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    # handle tuples
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    # already JSON -> just return
    return value


def _circuit_to_text(circuit: QuantumCircuit) -> str:
    """
    Represent a circuit as OpenQASM 3, falling back to a text drawing.

    Args:
        circuit: The circuit to represent.

    Returns:
        str: OpenQASM 3 source, or an ASCII drawing if export is unsupported.
    """
    try:
        return dump_qasm(circuit)
    except QiskitError:  # Some custom instructions have no OpenQASM 3 export.
        return str(circuit.draw(output="text"))
