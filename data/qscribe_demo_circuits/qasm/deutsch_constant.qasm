OPENQASM 3.0;
include "stdgates.inc";
// Deutsch's Algorithm (Constant Oracle)
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[2] q;
bit[1] c;
x q[1];
h q[0];
h q[1];
// --- begin oracle ---
x q[1];
// --- end ---
h q[0];
c[0] = measure q[0];
