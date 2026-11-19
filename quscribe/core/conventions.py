"""
Conventions shared by prompts & code, so both agree on qubit and bit ordering.

Keeping the prompt wording and the matching conversion function in one module
ensures the model and the verifier never drift apart.
"""

QUBIT_ORDER_CONVENTION = (
    "Number qubits from the top wire of the figure downwards: the top wire is "
    "q[0], the wire below it is q[1], and so on."
)

BITSTRING_CONVENTION = (
    "Write every bitstring with the bit of the first listed qubit as the "
    "leftmost character; when no qubits are listed, the leftmost "
    "character is q[0]."
)

# A model replies with this marker instead of a program when the figure
# is underspecified.
UNDERSPECIFIED_MARKER = "UNDERSPECIFIED:"

QASM_FORMAT_RULES = f"""Write the circuit as an OpenQASM 3 program that
follows these rules:
- Start with `OPENQASM 3.0;` followed by `include "stdgates.inc";`.
- Declare a single qubit register named q, e.g. `qubit[3] q;`.
    {QUBIT_ORDER_CONVENTION}
- Use gates from stdgates.inc: h, x, y, z, s, sdg, t, tdg, sx, rx, ry, rz, p,
    cx, cy, cz, ch, cp, crx, cry, crz, swap, ccx, cswap.
- Write controlled phase gates with cp: a controlled-S is `cp(pi/2)` and a
    controlled-T is `cp(pi/4)`.
- Use numeric angles, writing multiples of pi with the `pi` constant, e.g.
    `rz(pi/4) q[0];`.
- When the figure groups gates in a labelled box (such as U_f, Oracle or QFT†),
    declare that group as a `gate` named after the label, using letters, digits
    and underscores only (e.g. `U_f`, `QFT_dagger`), and apply it where the
    box appears, so the program keeps the figure's structure.
- A dashed outline only groups gates: when wires run through it, the gates
    drawn inside are its contents, even if there is only one. A box is opaque
    only when it is a solid block with no gates visible inside. Build an
    opaque box's gate from the caption or text when they define it precisely
    (e.g. "f(x) = x0 ⊕ x2"). If an opaque box is not defined there either,
    never invent its contents: reply with the single line
    `{UNDERSPECIFIED_MARKER} <what is missing>` instead of a program.
- If the figure is a circuit identity (two circuits joined by "="), transcribe
    only the right-hand side.
- Include measurements only when the figure shows them: declare `bit[n] c;`
    and write `c[i] = measure q[j];`.
- Ignore barriers, slice labels and unlabelled dashed boxes.
- Return the program in a single fenced block that starts with ```qasm.
"""


def swap_bitstring_endianness(bitstring: str) -> str:
    """
    Convert between the conventional (q[0] leftmost) and Qiskit (q[0]
    rightmost) orders.

    Reversing the string is its own inverse, so the same function converts in
    both directions.

    Args:
        bitstring: A bitstring in either ordering.

    Returns:
        str: The bitstring in the other ordering.
    """
    return bitstring[::-1]
