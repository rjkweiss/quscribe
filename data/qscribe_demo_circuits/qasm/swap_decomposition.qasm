OPENQASM 3.0;
include "stdgates.inc";
// SWAP Gate Decomposition
// Reference = right-hand side of the figure
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[2] q;
cx q[0], q[1];
cx q[1], q[0];
cx q[0], q[1];
