OPENQASM 3.0;
include "stdgates.inc";
// Deutsch's Algorithm (Balanced Oracle)
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[2] q;
bit[1] c;
x q[1];
h q[0];
h q[1];
// --- begin oracle ---
cx q[0], q[1];
// --- end ---
h q[0];
c[0] = measure q[0];
