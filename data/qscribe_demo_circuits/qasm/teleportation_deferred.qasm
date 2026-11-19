OPENQASM 3.0;
include "stdgates.inc";
// Quantum Teleportation (Deferred Measurement)
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[3] q;
bit[1] c;
h q[0];
t q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
cx q[1], q[2];
cz q[0], q[2];
tdg q[2];
h q[2];
c[0] = measure q[2];
