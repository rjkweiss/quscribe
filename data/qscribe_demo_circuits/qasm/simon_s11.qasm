OPENQASM 3.0;
include "stdgates.inc";
// Simon's Algorithm, n = 2 (Two-to-One, s = 11)
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[4] q;
bit[2] c;
h q[0];
h q[1];
// --- begin oracle ---
cx q[0], q[2];
cx q[1], q[3];
cx q[0], q[2];
cx q[0], q[3];
// --- end ---
h q[0];
h q[1];
c[0] = measure q[0];
c[1] = measure q[1];
