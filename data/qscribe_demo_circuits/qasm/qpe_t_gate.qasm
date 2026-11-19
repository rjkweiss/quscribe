OPENQASM 3.0;
include "stdgates.inc";
// Phase Estimation of the T Gate (Core of Shor's Algorithm)
// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.
qubit[4] q;
bit[3] c;
x q[3];
h q[0];
h q[1];
h q[2];
cp(pi/4) q[2], q[3];  // controlled-T
cp(pi/2) q[1], q[3];  // controlled-S
cz q[0], q[3];  // controlled-Z
// --- begin inverse QFT ---
swap q[0], q[2];
h q[2];
cp(-pi/2) q[2], q[1];
h q[1];
cp(-pi/4) q[2], q[0];
cp(-pi/2) q[1], q[0];
h q[0];
// --- end ---
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
