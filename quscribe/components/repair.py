"""
Repair Agent: corrects a transcription using the discrepancies
the verifier found.

Repairs happen at the OpenQASM 3 level, because figure-reading errors
originate there; the generator then regenerates code from the corrected
transcription.
"""

from quscribe.components.parser import DiagramParser
from quscribe.core.models import (
    CircuitInput,
    ParsedCircuit,
    VerificationReport,
)

NO_DETAILS_NOTE = "- The checks failed without giving specific details."

REPAIR_INSTRUCTION_TEMPLATE = """A previous OpenQASM 3 transcription of this
figure failed automated checks.

Previous transcription:
```qasm
{current_qasm}
```

Problems reported by the checks:
{discrepancy_list}

Re-examine the figure carefully. The checks are imperfect, because an
independent reading or a paper claim can itself be wrong, so change the
transcription only where the figure supports the change.
Return the complete corrected program.
"""


class RepairAgent:
    """
    Asks the model for a corrected transcription, given specific discrepancies.
    """

    def __init__(self, transcriber: DiagramParser) -> None:
        """
        Create a repair agent.

        Args:
            transcriber: Parser whose prompt framing and validation are reused

        Returns:
            None.
        """
        self._transcriber = transcriber

    def repair(
        self,
        circuit_input: CircuitInput,
        current: ParsedCircuit,
        report: VerificationReport,
    ) -> ParsedCircuit:
        """
        Produce a corrected transcription.

        Args:
            circuit_input: The figure and its accompanying text.
            current: The transcription that failed verification.
            report: The failed verification report.

        Returns:
            ParsedCircuit: The revised, validated transcription.

        Raises:
            GenerationFailedError: If no valid program is produced.
        """
        discrepancy_list = "\n".join(
            f"- {discrepancy}" for discrepancy in report.discrepancies
        )
        instruction = REPAIR_INSTRUCTION_TEMPLATE.format(
            current_qasm=current.qasm,
            discrepancy_list=discrepancy_list or NO_DETAILS_NOTE,
        )
        return self._transcriber.transcribe(circuit_input, instruction)
