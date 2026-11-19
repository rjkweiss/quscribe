"""
Runs generated Qiskit code in a separate process and returns
the circuit it builds.

A subprocess with a timeout keeps crashes, infinite loops and global
side effects of model-written code out of the QuScribe process.
It is isolation and not a security sandbox: only run QuScribe on machines
where executing model-written code is acceptable.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from qiskit import QuantumCircuit, qpy

from quscribe.core.execution import circuit_runner
from quscribe.errors import CodeExecutionError

RUNNER_SCRIPT_PATH = Path(circuit_runner.__file__).resolve()
GENERATED_MODULE_FILENAME = "generated_circuit.py"
OUTPUT_FILENAME = "circuit.qpy"
ERROR_TAIL_CHARACTERS = 2000  # Keep feedback to the model short and focused.
DEFAULT_TIMEOUT_SECONDS = 60.0


class CircuitCodeExecutor:
    """
    Executes generated code in a fresh interpreter and loads
    the resulting circuit.
    """

    def __init__(
        self,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ) -> None:
        """
        Create an executor.

        Args:
            timeout_seconds: Maximum run time for one piece of generated code.
        """
        self._timeout_seconds = timeout_seconds

    def execute(self, source_code: str) -> QuantumCircuit:
        """
        Run generated code and return the circuit its entry point builds.

        Args:
            source_code: A Python module defining ``build_circuit()``.

        Returns:
            QuantumCircuit: The circuit returned by ``build_circuit()``.

        Raises:
            CodeExecutionError: If the code fails, times out,
                                or breaks the contract.
        """
        with tempfile.TemporaryDirectory(
            prefix="quscribe_"
        ) as working_directory:
            # get path to working directory and output results
            working_path = Path(working_directory)
            code_path = working_path / GENERATED_MODULE_FILENAME
            output_path = working_path / OUTPUT_FILENAME

            # write source code to the code file
            code_path.write_text(source_code, encoding="utf-8")

            # execute the code and check resulting returned codes.
            completed_process = self._run_in_subprocess(
                code_path, output_path, working_path
            )

            if completed_process.returncode != 0:
                error_output = completed_process.stderr.strip()[
                    -ERROR_TAIL_CHARACTERS:
                ]
                raise CodeExecutionError(
                    error_output or
                    f"exit status {completed_process.returncode}"
                )

            #  execution was successful -> load resulting qpy file
            with output_path.open("rb") as output_file:
                circuits = qpy.load(output_file)

        # return the circuit created
        return circuits[0]

    def _run_in_subprocess(
        self,
        code_path: Path,
        output_path: Path,
        working_path: Path
    ) -> subprocess.CompletedProcess:
        """
        Launch the runner script on the generated code.

        Args:
            code_path: Path of the generated module.
            output_path: Where the runner writes the QPY file.
            working_path: Working directory for the subprocess.

        Returns:
            subprocess.CompletedProcess: The finished process,
                                        with captured output.

        Raises:
            CodeExecutionError: If the process exceeds the timeout.
        """
        try:
            return subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_SCRIPT_PATH),
                    str(code_path),
                    str(output_path)
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                cwd=working_path,
                env={**os.environ, "MPLBACKEND": "Agg"},  # don't open GUI
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise CodeExecutionError(
                f"The code did not finish within {self._timeout_seconds:g} "
                f"seconds."
            ) from error
