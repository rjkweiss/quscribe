# QScribe Dev/Demo Circuit Set (16 circuits)

Textbook circuits for the progress demo. **Not benchmark data**: these are famous circuits that models have likely memorized.

## Folders
- `with_caption/`: PNG with title + caption (use for the "image + text" setting).
- `clean/`: the same circuit with no title or caption (use for the "image only" setting).
- `qasm/`: reference OpenQASM 3 for each circuit. For equation figures (SWAP, CZ, Toffoli) the reference is the right-hand side.
- `metadata/`: JSON with caption, wire mapping, checkable claims, and expected outcome or target unitary.

## Conventions
- q[0] is the top wire. Bitstrings are read top to bottom (q0 first).
- Qiskit prints counts with the highest-numbered bit on the LEFT, so reverse Qiskit's strings before comparing with `expected_outcome`.
- Every reference was checked by simulation: outcome distributions match `expected_outcome`; equation figures match the target unitary up to global phase.

## Circuits
| File | Oracle / variant | Expected result |
|---|---|---|
| deutsch_constant | f(x) = 1 | q0 = 0 |
| deutsch_balanced | f(x) = x | q0 = 1 |
| dj_constant | f(x) = 1 (n = 3) | 000 |
| dj_balanced | f(x) = x0 ⊕ x1 ⊕ x2 ⊕ 1 | 111 |
| bv_secret_101 | s = 101 (balanced) | 101 |
| bv_constant | s = 000, f(x) = 1 (constant) | 000 |
| simon_s11 | two-to-one, s = 11 | 00 or 11 (½ each) |
| simon_s00 | one-to-one, s = 00 | uniform (¼ each) |
| qft_3qubit | 3-qubit QFT | unitary = DFT |
| qpe_t_gate | phase estimation of T | 001 |
| bell_state | | 00 or 11 (½ each) |
| ghz_state | | 000 or 111 (½ each) |
| swap_decomposition | SWAP = 3 CNOTs | unitary = SWAP |
| cz_decomposition | CZ = H·CNOT·H | unitary = CZ |
| toffoli_decomposition | 6-CNOT Toffoli | unitary = CCX |
| teleportation_deferred | deferred measurement | q2 = 0 |
