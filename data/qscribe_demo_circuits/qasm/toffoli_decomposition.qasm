OPENQASM 3.0;
include "stdgates.inc";
// Toffoli Gate Decomposition
// Reference = right-hand side of the figure
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[3] q;
h q[2];
cx q[1], q[2];
tdg q[2];
cx q[0], q[2];
t q[2];
cx q[1], q[2];
tdg q[2];
cx q[0], q[2];
t q[1];
t q[2];
h q[2];
cx q[0], q[1];
t q[0];
tdg q[1];
cx q[0], q[1];
