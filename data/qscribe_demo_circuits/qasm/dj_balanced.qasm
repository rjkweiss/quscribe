OPENQASM 3.0;
include "stdgates.inc";
// Deutsch–Jozsa Algorithm, n = 3 (Balanced Oracle)
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[4] q;
bit[3] c;
x q[3];
h q[0];
h q[1];
h q[2];
h q[3];
// --- begin oracle ---
x q[1];
cx q[0], q[3];
cx q[1], q[3];
cx q[2], q[3];
x q[1];
// --- end ---
h q[0];
h q[1];
h q[2];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
