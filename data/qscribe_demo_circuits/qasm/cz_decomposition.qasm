OPENQASM 3.0;
include "stdgates.inc";
// Controlled-Z from a CNOT
// Reference = right-hand side of the figure
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[2] q;
h q[1];
cx q[0], q[1];
h q[1];
