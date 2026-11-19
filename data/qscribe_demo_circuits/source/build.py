import json, math, os, shutil
import numpy as np
from qscribe_draw import render, to_qasm, simulate, unitary, equiv, controlled, G1, SWAP

PI = math.pi
def g(name, q): return ("g", name, q)
def cx(c, t): return ("cx", c, t)
def cz(a, b): return ("cz", a, b)
def cp(c, t, th, lab): return ("cp", c, t, th, lab)
def cu(c, t, name): return ("cu", c, t, name)
def ccx(a, b, t): return ("ccx", a, b, t)
def swap(a, b): return ("swap", a, b)
def m(q): return ("measure", q)
def oracle(label, ops, qasm_label=None): return [("group_start", label, qasm_label or "oracle")] + ops + [("group_end",)]

C = []  # circuit records

# ---------------- Deutsch (q0 input, q1 ancilla)
def deutsch(oracle_ops):
    return [g("x", 1), g("h", 0), g("h", 1)] + oracle(r"$U_f$", oracle_ops) + [g("h", 0), m(0)]

C.append(dict(id="deutsch_constant", n=2, segments=[deutsch([g("x", 1)])],
    title="Deutsch's Algorithm (Constant Oracle)",
    caption="Deutsch's algorithm with a constant oracle, f(x) = 1, implemented as an X gate on the ancilla q1. "
            "Measuring q0 yields 0, identifying f as constant.",
    claims=["f is constant", "measuring q0 yields 0 with probability 1"], expected={"0": 1.0}))
C.append(dict(id="deutsch_balanced", n=2, segments=[deutsch([cx(0, 1)])],
    title="Deutsch's Algorithm (Balanced Oracle)",
    caption="Deutsch's algorithm with a balanced oracle, f(x) = x, implemented as a CNOT from q0 to the ancilla q1. "
            "Measuring q0 yields 1, identifying f as balanced.",
    claims=["f is balanced", "measuring q0 yields 1 with probability 1"], expected={"1": 1.0}))

# ---------------- Deutsch-Jozsa n=3 (q0..q2 input, q3 ancilla)
def dj(oracle_ops):
    return ([g("x", 3)] + [g("h", q) for q in range(4)] + oracle(r"$U_f$", oracle_ops)
            + [g("h", q) for q in range(3)] + [m(q) for q in range(3)])

C.append(dict(id="dj_constant", n=4, segments=[dj([g("x", 3)])],
    title="Deutsch–Jozsa Algorithm, n = 3 (Constant Oracle)",
    caption="Deutsch–Jozsa algorithm on three input qubits with a constant oracle, f(x) = 1 for all x, "
            "implemented as an X gate on the ancilla q3. Measuring q0 q1 q2 yields 000 with probability 1.",
    claims=["f is constant", "measuring q0q1q2 yields 000 with probability 1"], expected={"000": 1.0}))
C.append(dict(id="dj_balanced", n=4,
    segments=[dj([g("x", 1), cx(0, 3), cx(1, 3), cx(2, 3), g("x", 1)])],
    title="Deutsch–Jozsa Algorithm, n = 3 (Balanced Oracle)",
    caption="Deutsch–Jozsa algorithm on three input qubits with a balanced oracle, f(x) = x0 ⊕ x1 ⊕ x2 ⊕ 1, "
            "implemented with CNOTs onto the ancilla q3 and X gates on q1. Any outcome other than 000 identifies f "
            "as balanced; this oracle yields 111 with probability 1.",
    claims=["f is balanced", "measuring q0q1q2 never yields 000", "measuring q0q1q2 yields 111 with probability 1"],
    expected={"111": 1.0}))

# ---------------- Bernstein-Vazirani n=3
C.append(dict(id="bv_secret_101", n=4, segments=[dj([cx(0, 3), cx(2, 3)])],
    title="Bernstein–Vazirani Algorithm (Secret s = 101)",
    caption="Bernstein–Vazirani algorithm with secret string s = 101, so f(x) = s · x mod 2 (a balanced function). "
            "The oracle applies a CNOT onto the ancilla q3 from each input qubit where s has a 1 (q0 and q2). "
            "Measuring q0 q1 q2 recovers s = 101 with probability 1.",
    claims=["secret string s = 101", "measuring q0q1q2 yields 101 with probability 1"], expected={"101": 1.0}))
C.append(dict(id="bv_constant", n=4, segments=[dj([g("x", 3)])],
    title="Bernstein–Vazirani Algorithm (Secret s = 000, Constant f)",
    caption="Bernstein–Vazirani algorithm with secret string s = 000 and offset b = 1, so f(x) = s · x ⊕ 1 = 1 "
            "(a constant function). The oracle is an X gate on the ancilla q3. Measuring q0 q1 q2 yields s = 000.",
    claims=["secret string s = 000", "f is constant", "measuring q0q1q2 yields 000 with probability 1"],
    expected={"000": 1.0}))

# ---------------- Simon n=2 (q0,q1 input; q2,q3 output register)
def simon(oracle_ops):
    return [g("h", 0), g("h", 1)] + oracle(r"$U_f$", oracle_ops) + [g("h", 0), g("h", 1), m(0), m(1)]

C.append(dict(id="simon_s11", n=4, segments=[simon([cx(0, 2), cx(1, 3), cx(0, 2), cx(0, 3)])],
    title="Simon's Algorithm, n = 2 (Two-to-One, s = 11)",
    caption="Simon's algorithm with a two-to-one oracle and hidden string s = 11, so f(x) = f(x ⊕ 11). "
            "The oracle copies the input into q2 q3, then applies CNOTs from q0 onto each output qubit where s has a 1. "
            "Measuring q0 q1 only yields strings y with y · s = 0 mod 2: 00 or 11, each with probability 1/2.",
    claims=["hidden string s = 11", "f is two-to-one", "outcomes y satisfy y·s = 0 mod 2", "outcomes 00 and 11 each with probability 1/2"],
    expected={"00": 0.5, "11": 0.5}))
C.append(dict(id="simon_s00", n=4, segments=[simon([cx(0, 2), cx(1, 3)])],
    title="Simon's Algorithm, n = 2 (One-to-One, s = 00)",
    caption="Simon's algorithm with a one-to-one oracle (hidden string s = 00), implemented as CNOTs copying the "
            "input q0 q1 into the output register q2 q3. Measuring q0 q1 yields all four strings uniformly, "
            "each with probability 1/4.",
    claims=["hidden string s = 00", "f is one-to-one", "all four outcomes with probability 1/4"],
    expected={"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}))

# ---------------- Shor's core: QFT and phase estimation
qft3 = [g("h", 0), cp(1, 0, PI / 2, r"$R_2$"), cp(2, 0, PI / 4, r"$R_3$"), g("h", 1),
        cp(2, 1, PI / 2, r"$R_2$"), g("h", 2), swap(0, 2)]
C.append(dict(id="qft_3qubit", n=3, segments=[qft3],
    title="Three-Qubit Quantum Fourier Transform",
    caption="Quantum Fourier transform on three qubits. R_k denotes the controlled phase rotation "
            "diag(1, exp(2πi / 2^k)); the final SWAP restores the qubit order. This transform is the core "
            "subroutine of phase estimation and Shor's algorithm.",
    claims=["implements the 3-qubit QFT", "R_k = diag(1, exp(2*pi*i/2^k))"], unitary_target="qft3"))

iqft3 = [swap(0, 2), g("h", 2), cp(2, 1, -PI / 2, r"$P(-\pi/2)$"), g("h", 1),
         cp(2, 0, -PI / 4, r"$P(-\pi/4)$"), cp(1, 0, -PI / 2, r"$P(-\pi/2)$"), g("h", 0)]
qpe = ([g("x", 3), g("h", 0), g("h", 1), g("h", 2), cu(2, 3, "t"), cu(1, 3, "s"), cu(0, 3, "z")]
       + oracle(r"$\mathrm{QFT}^{\dagger}$", iqft3, "inverse QFT") + [m(0), m(1), m(2)])
C.append(dict(id="qpe_t_gate", n=4, segments=[qpe],
    title="Phase Estimation of the T Gate (Core of Shor's Algorithm)",
    caption="Quantum phase estimation with three counting qubits (q0–q2) and target q3 prepared in the eigenstate |1⟩ "
            "of T. The counting qubits control T, T² = S, and T⁴ = Z, followed by the inverse QFT. Since T has "
            "eigenphase 1/8 = 0.001 in binary, measuring q0 q1 q2 yields 001 with probability 1.",
    claims=["eigenphase of T is 1/8", "measuring q0q1q2 yields 001 with probability 1"], expected={"001": 1.0}))

# ---------------- Elementary circuits
C.append(dict(id="bell_state", n=2, segments=[[g("h", 0), cx(0, 1), m(0), m(1)]],
    title="Bell State Preparation",
    caption="Preparation of the Bell state (|00⟩ + |11⟩)/√2 using a Hadamard followed by a CNOT. "
            "Measuring both qubits yields 00 or 11, each with probability 1/2.",
    claims=["prepares (|00> + |11>)/sqrt(2)", "outcomes 00 and 11 each with probability 1/2"],
    expected={"00": 0.5, "11": 0.5}))
C.append(dict(id="ghz_state", n=3, segments=[[g("h", 0), cx(0, 1), cx(1, 2), m(0), m(1), m(2)]],
    title="Three-Qubit GHZ State Preparation",
    caption="Preparation of the GHZ state (|000⟩ + |111⟩)/√2 using a Hadamard and a chain of CNOTs. "
            "Measuring all qubits yields 000 or 111, each with probability 1/2.",
    claims=["prepares (|000> + |111>)/sqrt(2)", "outcomes 000 and 111 each with probability 1/2"],
    expected={"000": 0.5, "111": 0.5}))
C.append(dict(id="swap_decomposition", n=2, segments=[[swap(0, 1)], [cx(0, 1), cx(1, 0), cx(0, 1)]],
    title="SWAP Gate Decomposition",
    caption="The SWAP gate (left) decomposed into three alternating CNOT gates (right).",
    claims=["right-hand side equals SWAP", "uses 3 CNOTs"], unitary_target="swap"))
C.append(dict(id="cz_decomposition", n=2, segments=[[cz(0, 1)], [g("h", 1), cx(0, 1), g("h", 1)]],
    title="Controlled-Z from a CNOT",
    caption="The controlled-Z gate (left) is equivalent to a CNOT conjugated by Hadamard gates on the target qubit (right).",
    claims=["right-hand side equals CZ"], unitary_target="cz"))
toff = [g("h", 2), cx(1, 2), g("tdg", 2), cx(0, 2), g("t", 2), cx(1, 2), g("tdg", 2), cx(0, 2),
        g("t", 1), g("t", 2), g("h", 2), cx(0, 1), g("t", 0), g("tdg", 1), cx(0, 1)]
C.append(dict(id="toffoli_decomposition", n=3, segments=[[ccx(0, 1, 2)], toff],
    title="Toffoli Gate Decomposition",
    caption="The Toffoli (CCNOT) gate (left) decomposed into six CNOTs, two Hadamards, and T / T† gates (right).",
    claims=["right-hand side equals Toffoli (CCX)", "uses 6 CNOTs"], unitary_target="ccx"))
tele = ([g("h", 0), g("t", 0)]                       # prepare |psi> = (|0> + e^{i pi/4}|1>)/sqrt2 on q0
        + [g("h", 1), cx(1, 2)]                      # Bell pair q1,q2
        + [cx(0, 1), g("h", 0)]                      # Bell-basis rotation
        + [cx(1, 2), cz(0, 2)]                       # deferred-measurement corrections
        + [g("tdg", 2), g("h", 2), m(2)])            # undo preparation on q2 and measure
C.append(dict(id="teleportation_deferred", n=3, segments=[tele],
    title="Quantum Teleportation (Deferred Measurement)",
    caption="Teleportation of the state |ψ⟩ = TH|0⟩, prepared on q0 by an H then a T gate, to q2. Measurements are "
            "deferred: the classical corrections are replaced by a CNOT from q1 and a CZ from q0. Undoing the "
            "preparation (T†, then H) on q2 returns |0⟩, so measuring q2 yields 0 with probability 1.",
    claims=["q2 ends in the state prepared on q0", "measuring q2 yields 0 with probability 1"], expected={"0": 1.0}))

# ---------------- verification
def dft(n):
    N = 2 ** n
    return np.array([[np.exp(2j * PI * x * y / N) for x in range(N)] for y in range(N)]) / math.sqrt(N)

TARGETS = {"qft3": dft(3), "swap": SWAP, "cz": controlled(G1["z"]), "ccx": controlled(G1["x"], 2)}

out = "/home/claude/qcirc/out"
shutil.rmtree(out, ignore_errors=True)
for sub in ["with_caption", "clean", "qasm", "metadata"]:
    os.makedirs(f"{out}/{sub}")

all_ok = True
for c in C:
    ref = c["segments"][-1]
    n = c["n"]
    if "expected" in c:
        got = simulate(n, ref)
        ok = got.keys() == c["expected"].keys() and all(abs(got[k] - c["expected"][k]) < 1e-6 for k in got)
        got = {k: float(v) for k, v in got.items()}
        check = f"simulated outcome distribution {got} matches expected {c['expected']}"
    else:
        U = unitary(n, ref)
        ok = equiv(U, TARGETS[c["unitary_target"]])
        if len(c["segments"]) == 2:
            ok = ok and equiv(unitary(n, c["segments"][0]), TARGETS[c["unitary_target"]])
        check = f"unitary equals {c['unitary_target']} up to global phase"
        got = None
    all_ok &= ok
    print(("PASS " if ok else "FAIL ") + c["id"], got if got else "")
    render(f"{out}/with_caption/{c['id']}.png", n, c["segments"], title=c["title"], caption=c["caption"])
    render(f"{out}/clean/{c['id']}.png", n, c["segments"])
    open(f"{out}/qasm/{c['id']}.qasm", "w").write(
        to_qasm(n, ref, comment=c["title"] + ("\nReference = right-hand side of the figure" if len(c["segments"]) == 2 else "")))
    meta = dict(figure_id=c["id"], title=c["title"], caption=c["caption"], num_qubits=n,
                category="dev_demo (textbook circuit; not benchmark data)",
                drawing_style="qscribe-matplotlib (paper-like)",
                wire_mapping={f"q[{i}]": ("top wire" if i == 0 else f"wire {i+1} from top") for i in range(n)},
                bit_order="bitstrings read top-to-bottom (q0 first)",
                figure_type="equation (LHS = RHS)" if len(c["segments"]) == 2 else "circuit",
                checkable_claims=c["claims"],
                expected_outcome=c.get("expected"), unitary_target=c.get("unitary_target"),
                reference_qasm=f"qasm/{c['id']}.qasm", reference_check=check, verified=bool(ok))
    json.dump(meta, open(f"{out}/metadata/{c['id']}.json", "w"), indent=2, ensure_ascii=False)
print("ALL OK" if all_ok else "SOME FAILED")
