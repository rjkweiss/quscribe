"""Tiny circuit toolkit: spec -> PNG (paper style), OpenQASM 3, and numpy simulation.
Convention: qubit 0 is the TOP wire and the most significant bit (big-endian).
Bitstrings are read top-to-bottom.
"""
import json, math, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc, FancyBboxPatch

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "DejaVu Serif"

# ---------------------------------------------------------------- gates
SQ = 1 / math.sqrt(2)
G1 = {
    "h": np.array([[SQ, SQ], [SQ, -SQ]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.diag([1, 1j]).astype(complex),
    "t": np.diag([1, np.exp(1j * math.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
}
LABEL1 = {"h": r"$H$", "x": r"$X$", "z": r"$Z$", "s": r"$S$", "t": r"$T$", "tdg": r"$T^{\dagger}$"}
CU_PHASE = {"t": math.pi / 4, "s": math.pi / 2, "z": math.pi}


def phase(theta):
    return np.diag([1, np.exp(1j * theta)])


def apply(state, n, U, qubits):
    """Apply k-qubit unitary U to `qubits` (big-endian, qubit 0 = MSB)."""
    k = len(qubits)
    psi = state.reshape([2] * n)
    psi = np.moveaxis(psi, qubits, list(range(k)))
    shp = psi.shape
    psi = (U @ psi.reshape(2 ** k, -1)).reshape(shp)
    psi = np.moveaxis(psi, list(range(k)), qubits)
    return psi.reshape(-1)


def controlled(U, nctrl=1):
    d = U.shape[0]
    M = np.eye(d * 2 ** nctrl, dtype=complex)
    M[-d:, -d:] = U
    return M


SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def op_unitary(op):
    kind = op[0]
    if kind == "g":
        return G1[op[1]], [op[2]]
    if kind == "cx":
        return controlled(G1["x"]), [op[1], op[2]]
    if kind == "cz":
        return controlled(G1["z"]), [op[1], op[2]]
    if kind == "cp":
        return controlled(phase(op[3])), [op[1], op[2]]
    if kind == "cu":
        return controlled(G1[op[3]]), [op[1], op[2]]
    if kind == "ccx":
        return controlled(G1["x"], 2), [op[1], op[2], op[3]]
    if kind == "swap":
        return SWAP, [op[1], op[2]]
    return None, None


def unitary(n, ops):
    U = np.eye(2 ** n, dtype=complex)
    for col in range(2 ** n):
        st = np.zeros(2 ** n, dtype=complex); st[col] = 1
        for op in ops:
            M, qs = op_unitary(op)
            if M is not None:
                st = apply(st, n, M, qs)
        U[:, col] = st
    return U


def simulate(n, ops):
    st = np.zeros(2 ** n, dtype=complex); st[0] = 1
    for op in ops:
        M, qs = op_unitary(op)
        if M is not None:
            st = apply(st, n, M, qs)
    meas = [op[1] for op in ops if op[0] == "measure"]
    probs = np.abs(st) ** 2
    out = {}
    for idx, p in enumerate(probs):
        if p < 1e-9:
            continue
        bits = format(idx, f"0{n}b")
        key = "".join(bits[q] for q in meas)
        out[key] = out.get(key, 0) + p
    return {k: round(v, 6) for k, v in sorted(out.items())}


def equiv(U, V):
    """Equal up to global phase."""
    i = np.argmax(np.abs(V))
    ph = U.flat[i] / V.flat[i]
    return abs(abs(ph) - 1) < 1e-9 and np.allclose(U, ph * V, atol=1e-9)

# ---------------------------------------------------------------- QASM

def fmt_angle(theta):
    for num, den in [(1, 1), (1, 2), (1, 4), (1, 8), (-1, 2), (-1, 4), (-1, 8), (-1, 1)]:
        if abs(theta - num * math.pi / den) < 1e-12:
            s = "pi" if den == 1 else f"pi/{den}"
            return ("-" if num < 0 else "") + s
    return repr(theta)


def to_qasm(n, ops, comment=""):
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    if comment:
        lines += [f"// {c}" for c in comment.split("\n")]
    lines.append("// Wire convention: q[0] = top wire; bitstrings read top-to-bottom.")
    lines.append(f"qubit[{n}] q;")
    meas = [op for op in ops if op[0] == "measure"]
    if meas:
        lines.append(f"bit[{len(meas)}] c;")
    mi = 0
    for op in ops:
        k = op[0]
        if k == "g":
            lines.append(f"{op[1]} q[{op[2]}];")
        elif k == "cx":
            lines.append(f"cx q[{op[1]}], q[{op[2]}];")
        elif k == "cz":
            lines.append(f"cz q[{op[1]}], q[{op[2]}];")
        elif k == "cp":
            lines.append(f"cp({fmt_angle(op[3])}) q[{op[1]}], q[{op[2]}];")
        elif k == "cu":
            if op[3] == "z":
                lines.append(f"cz q[{op[1]}], q[{op[2]}];  // controlled-Z")
            else:
                lines.append(f"cp({fmt_angle(CU_PHASE[op[3]])}) q[{op[1]}], q[{op[2]}];  // controlled-{op[3].upper()}")
        elif k == "ccx":
            lines.append(f"ccx q[{op[1]}], q[{op[2]}], q[{op[3]}];")
        elif k == "swap":
            lines.append(f"swap q[{op[1]}], q[{op[2]}];")
        elif k == "measure":
            lines.append(f"c[{mi}] = measure q[{op[1]}];"); mi += 1
        elif k == "group_start":
            lines.append(f"// --- begin {op[2] if len(op) > 2 else op[1]} ---")
        elif k == "group_end":
            lines.append("// --- end ---")
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------- layout & drawing
BOX_H = 0.62


def op_qubits(op):
    k = op[0]
    if k == "g" or k == "measure":
        return [op[op_index_q(k)]]
    if k in ("cx", "cz", "cp", "cu", "swap"):
        return [op[1], op[2]]
    if k == "ccx":
        return [op[1], op[2], op[3]]
    return []


def op_index_q(k):
    return 2 if k == "g" else 1


def op_label(op):
    k = op[0]
    if k == "g":
        return LABEL1[op[1]]
    if k == "cp":
        return op[4]
    if k == "cu":
        return LABEL1[op[3]]
    return None


def text_width(label):
    """Rough width (data units) needed for a mathtext label."""
    if label is None:
        return 0.5
    plain = (label.replace("$", "").replace("\\pi", "p").replace("\\dagger", "").replace("\\", "")
             .replace("{", "").replace("}", "").replace("^", "").replace("_", ""))
    return max(0.62, 0.21 * len(plain) + 0.32)


def schedule(n, ops):
    """ASAP schedule; group markers force a full sync. Returns cols, widths, groups."""
    level = [0] * n
    cols, groups, open_g = [], [], None
    for op in ops:
        k = op[0]
        if k == "group_start":
            m = max(level); level = [m] * n
            open_g = {"start": m, "label": op[1]}
            cols.append(None); continue
        if k == "group_end":
            m = max(level)
            if m == open_g["start"]:
                m += 1  # empty group still gets a column
            level = [m] * n
            open_g["end"] = m - 1
            groups.append(open_g); open_g = None
            cols.append(None); continue
        qs = op_qubits(op)
        lo, hi = min(qs), max(qs)
        c = max(level[lo:hi + 1])
        for w in range(lo, hi + 1):
            level[w] = c + 1
        cols.append(c)
    ncols = max(level) if n else 0
    widths = [0.62] * ncols
    for op, c in zip(ops, cols):
        if c is not None:
            widths[c] = max(widths[c], text_width(op_label(op)))
    return cols, widths, groups, ncols


def draw_segment(ax, n, ops, x0, show_labels):
    cols, widths, groups, ncols = schedule(n, ops)
    gap = 0.38
    centers, x = [], x0 + gap
    for w in widths:
        centers.append(x + w / 2); x += w + gap
    x_end = x
    # wires
    for q in range(n):
        ax.plot([x0, x_end], [-q, -q], color="black", lw=1.1, zorder=1)
    # groups (dashed oracle boxes)
    for g in groups:
        left = centers[g["start"]] - widths[g["start"]] / 2 - 0.2
        right = centers[g["end"]] + widths[g["end"]] / 2 + 0.2
        ax.add_patch(Rectangle((left, -(n - 1) - 0.48), right - left, (n - 1) + 0.96,
                               fill=False, ls=(0, (4, 3)), lw=1.0, ec="#444444", zorder=0.5))
        ax.text((left + right) / 2, 0.62, g["label"], ha="center", va="bottom", fontsize=13)
    # ops
    for op, c in zip(ops, cols):
        if c is None:
            continue
        xc, k = centers[c], op[0]
        if k in ("g",):
            box(ax, xc, -op[2], widths[c], op_label(op))
        elif k == "measure":
            meter(ax, xc, -op[1])
        elif k == "cx":
            vline(ax, xc, -op[1], -op[2]); dot(ax, xc, -op[1]); oplus(ax, xc, -op[2])
        elif k == "ccx":
            qs = [op[1], op[2], op[3]]
            vline(ax, xc, -min(qs), -max(qs)); dot(ax, xc, -op[1]); dot(ax, xc, -op[2]); oplus(ax, xc, -op[3])
        elif k == "cz":
            vline(ax, xc, -op[1], -op[2]); dot(ax, xc, -op[1]); dot(ax, xc, -op[2])
        elif k in ("cp", "cu"):
            vline(ax, xc, -op[1], -op[2]); dot(ax, xc, -op[1]); box(ax, xc, -op[2], widths[c], op_label(op))
        elif k == "swap":
            vline(ax, xc, -op[1], -op[2]); cross(ax, xc, -op[1]); cross(ax, xc, -op[2])
    if show_labels:
        for q in range(n):
            ax.text(x0 - 0.12, -q, rf"$q_{{{q}}}\ \ |0\rangle$", ha="right", va="center", fontsize=14)
    return x_end


def box(ax, x, y, w, label):
    ax.add_patch(Rectangle((x - w / 2, y - BOX_H / 2), w, BOX_H, fc="white", ec="black", lw=1.2, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=14, zorder=4)


def dot(ax, x, y):
    ax.add_patch(Circle((x, y), 0.085, fc="black", ec="black", zorder=4))


def oplus(ax, x, y, r=0.2):
    ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=1.2, zorder=4))
    ax.plot([x - r, x + r], [y, y], color="black", lw=1.2, zorder=5)
    ax.plot([x, x], [y - r, y + r], color="black", lw=1.2, zorder=5)


def cross(ax, x, y, s=0.13):
    ax.plot([x - s, x + s], [y - s, y + s], color="black", lw=1.5, zorder=4)
    ax.plot([x - s, x + s], [y + s, y - s], color="black", lw=1.5, zorder=4)


def vline(ax, x, y1, y2):
    ax.plot([x, x], [y1, y2], color="black", lw=1.2, zorder=2)


def meter(ax, x, y, w=0.62):
    ax.add_patch(Rectangle((x - w / 2, y - BOX_H / 2), w, BOX_H, fc="white", ec="black", lw=1.2, zorder=3))
    ax.add_patch(Arc((x, y - 0.14), 0.42, 0.42, theta1=0, theta2=180, lw=1.1, zorder=4))
    ax.annotate("", xy=(x + 0.17, y + 0.17), xytext=(x, y - 0.14),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black", mutation_scale=8), zorder=5)


def render(path, n, segments, title=None, caption=None, dpi=200):
    """segments: list of op-lists; consecutive segments are joined with '='."""
    fig, ax = plt.subplots()
    x = 0.0
    for i, seg in enumerate(segments):
        if i > 0:
            ax.text(x + 0.45, -(n - 1) / 2, "=", ha="center", va="center", fontsize=22)
            x += 0.9
        x = draw_segment(ax, n, seg, x, show_labels=(i == 0))
    left, right = -1.55, x + 0.15
    top, bottom = 1.1, -(n - 1) - 0.75
    cap_lines = textwrap.wrap(caption, width=max(48, int((right - left) * 7.0))) if caption else []
    extra = 0.42 * len(cap_lines) + (0.25 if cap_lines else 0)
    if title:
        ax.text((left + right) / 2, top + 0.4, title, ha="center", va="center", fontsize=15, fontweight="bold")
        top += 0.8
    if cap_lines:
        ax.text(left + 0.1, bottom - 0.2, "\n".join(cap_lines), ha="left", va="top", fontsize=11.5, linespacing=1.45)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom - extra, top)
    ax.set_aspect("equal"); ax.axis("off")
    scale = 0.62
    fig.set_size_inches((right - left) * scale, (top - bottom + extra) * scale)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
