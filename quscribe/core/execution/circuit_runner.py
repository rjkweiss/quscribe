"""
Subprocess entry point that executes generated circuit code in isolation.

Usage: ``python circuit_runner.py <generated_code.py> <output.qpy>``

It imports nothing from QuScribe so it runs as a plain script in a fresh
interpreter. It loads the generated module, calls its entry-point function and
serializes the returned circuit with QPY for the parent process to read.
"""

import importlib.util
import sys
from types import ModuleType

from qiskit import QuantumCircuit, qpy

ENTRY_POINT_NAME = "build_circuit"

# Exit constants - tells us how the program executed
EXIT_SUCCESS = 0
EXIT_CONTRACT_VIOLATION = 2
EXPECTED_ARGUMENT_COUNT = 3  # Script path, code path, output path.


def load_generated_module(code_path: str) -> ModuleType:
    """
    Import a Python file as a module without running its ``__main__`` block.

    Args:
        code_path: Path to the generated Python file.

    Returns:
        ModuleType: The executed module.

    Raises:
        ImportError: If the file cannot be loaded as a module.
    """
    specification = importlib.util.spec_from_file_location(
        "generated_circuit", code_path)

    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load generated code from {code_path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


def main(arguments: list[str]) -> int:
    """
    Run the generated code and write its circuit to a QPY file.

    Exceptions raised by the generated code propagate, so the interpreter
    prints the traceback to stderr and exits with a non-zero status for the
    parent to report.

    Args:
        arguments: ``sys.argv``: script path, generated code path,
        QPY output path.

    Returns:
        int: Process exit status.
    """
    if len(arguments) != EXPECTED_ARGUMENT_COUNT:
        print(
            "usage: circuit_runner.py <generated_code.py> <output.qpy>",
            file=sys.stderr
        )
        return EXIT_CONTRACT_VIOLATION

    code_path, output_path = arguments[1], arguments[2]
    module = load_generated_module(code_path)
    circuit_builder = getattr(module, ENTRY_POINT_NAME, None)

    if not callable(circuit_builder):
        print(
            f"The code must define a function named {ENTRY_POINT_NAME}().",
            file=sys.stderr
        )
        return EXIT_CONTRACT_VIOLATION

    circuit = circuit_builder()
    if not isinstance(circuit, QuantumCircuit):
        print(
            f"{ENTRY_POINT_NAME}() returned {type(circuit).__name__}, "
            f"expected QuantumCircuit.",
            file=sys.stderr,
        )
        return EXIT_CONTRACT_VIOLATION

    with open(output_path, "wb") as output_file:
        qpy.dump(circuit, output_file)

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main(sys.argv))
