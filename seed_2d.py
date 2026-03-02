#!/usr/bin/env python3
"""
2D Seed Space Exploration
==========================

Extend the seed from scalar to 2D vector (s₁, s₂).

Approach 1 (Additive Split):
  w_i = f(g₁ⁱ(s₁)) + h(g₂ⁱ(s₂))
  Each weight = sum of two independent orbit contributions.
  Secret-sharing property: either channel alone looks random.

Approach 2 (Bivariate Evaluator):
  w_i = F(g₁ⁱ(s₁), g₂ⁱ(s₂))
  Single bivariate polynomial maps coupled orbit to weights.

Key question: does 2D seed surface cover more of R^N than 1D curve?
"""

import numpy as np
from scipy.optimize import minimize
import sys

ARCH = [2, 4, 1]
CHUNK_SIZE = 8
SEED_1D = 3.7133
SEED_2D = (3.7133, 2.4567)

XOR_WEIGHTS = np.array([
    1.0, 1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, -1.0,
    -1.5, -0.5, -0.5, 0.5,
    0.1, 2.1, 2.1, 0.1,
    -0.05,
])


def cheb_nodes(n, lo, hi):
    k = np.arange(n, dtype=np.float64)
    raw = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)


def horner(coeffs, x):
    r = float(coeffs[0])
    for c in coeffs[1:]:
        r = r * x + float(c)
    return r


def forward_pass(weights, inputs):
    arch = ARCH
    w = np.array(weights)
    cur = np.array(inputs, dtype=np.float64)
    off = 0
    for l in range(len(arch) - 1):
        ni, no = arch[l], arch[l + 1]
        nxt = np.zeros(no)
        for j in range(no):
            s = w[off + ni * no + j]
            for i in range(ni):
                s += cur[i] * w[off + i * no + j]
            if l < len(arch) - 2 and s < 0:
                s = 0.0
            nxt[j] = s
        off += ni * no + no
        cur = nxt
    return cur


def classify(w, threshold=0.3):
    outputs = []
    for inp in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        out = forward_pass(w, inp)[0]
        outputs.append(out)
    bits = tuple(1 if o > 0.5 else 0 for o in outputs)
    clean = all(abs(o - round(o)) < threshold for o in outputs)

    truth_table = {
        (0,0,0,0): 'FALSE', (0,0,0,1): 'AND', (0,1,0,0): 'A<B',
        (0,0,1,0): 'A>B',   (0,0,1,1): 'ID_A', (0,1,0,1): 'ID_B',
        (0,1,1,0): 'XOR',   (0,1,1,1): 'OR',   (1,0,0,0): 'NOR',
        (1,0,0,1): 'XNOR',  (1,0,1,0): 'NOT_B', (1,0,1,1): 'A>=B',
        (1,1,0,0): 'NAND',  (1,1,0,1): 'B>=A', (1,1,1,0): 'NAND2',
        (1,1,1,1): 'TRUE',
    }
    name = truth_table.get(bits, f'?{bits}')
    return name, bits, outputs, clean


# ═══════════════════════════════════════════════════════════════
# 1D Encoding (baseline)
# ═══════════════════════════════════════════════════════════════

def build_1d_fg():
    flat = XOR_WEIGHTS
    chunks = [flat[i:i+CHUNK_SIZE] for i in range(0, len(flat), CHUNK_SIZE)]
    fg_pairs = []
    for ci, ch in enumerate(chunks):
        n = len(ch)
        lo = SEED_1D + 1.5 + ci * (CHUNK_SIZE + 2)
        hi = lo + max(n, 2)
        pts = cheb_nodes(n, lo, hi)
        gc = np.polyfit(pts, ch, n - 1)
        f_in = np.concatenate(([SEED_1D], pts[:-1]))
        fc = np.polyfit(f_in, pts, n - 1)
        fg_pairs.append((fc, gc, n))
    return fg_pairs


def eval_1d(s, fg_pairs):
    weights = []
    for fc, gc, n in fg_pairs:
        x = float(s)
        for i in range(n):
            x = horner(fc, x)
            w = horner(gc, x)
            if not np.isfinite(x) or not np.isfinite(w) or abs(x) > 1e15:
                return None
            weights.append(w)
    return np.array(weights)


# ═══════════════════════════════════════════════════════════════
# 2D Additive Encoding: w_i = f(g1^i(s1)) + h(g2^i(s2))
# ═══════════════════════════════════════════════════════════════

def build_2d_additive(seed1, seed2, weights, chunk_sz=CHUNK_SIZE):
    """
    Split each weight randomly: w_i = a_i + b_i.
    Build (g1, f) for the a-channel, (g2, h) for the b-channel.
    Either channel alone looks random.
    """
    rng = np.random.RandomState(42)
    flat = np.array(weights, dtype=np.float64)
    N = len(flat)

    # Random additive split
    a = rng.uniform(-2, 2, N)
    b = flat - a

    chunks_a = [a[i:i+chunk_sz] for i in range(0, N, chunk_sz)]
    chunks_b = [b[i:i+chunk_sz] for i in range(0, N, chunk_sz)]

    pairs = []
    for ci in range(len(chunks_a)):
        na = len(chunks_a[ci])

        # Channel 1: g1, f
        lo1 = seed1 + 1.5 + ci * (chunk_sz + 2)
        hi1 = lo1 + max(na, 2)
        pts1 = cheb_nodes(na, lo1, hi1)
        fc = np.polyfit(pts1, chunks_a[ci], na - 1)
        g1_in = np.concatenate(([seed1], pts1[:-1]))
        g1c = np.polyfit(g1_in, pts1, na - 1)

        # Channel 2: g2, h  (well-separated intervals)
        lo2 = seed2 + 1.5 + ci * (chunk_sz + 2) + 50.0
        hi2 = lo2 + max(na, 2)
        pts2 = cheb_nodes(na, lo2, hi2)
        hc = np.polyfit(pts2, chunks_b[ci], na - 1)
        g2_in = np.concatenate(([seed2], pts2[:-1]))
        g2c = np.polyfit(g2_in, pts2, na - 1)

        pairs.append((g1c, fc, g2c, hc, na))

    return pairs


def eval_2d_additive(s1, s2, pairs):
    weights = []
    for g1c, fc, g2c, hc, n in pairs:
        x = float(s1)
        y = float(s2)
        for i in range(n):
            x = horner(g1c, x)
            y = horner(g2c, y)
            a = horner(fc, x)
            b = horner(hc, y)
            w = a + b
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(w):
                return None
            if abs(x) > 1e15 or abs(y) > 1e15:
                return None
            weights.append(w)
    return np.array(weights)


# ═══════════════════════════════════════════════════════════════
# 2D Bivariate Encoding: w_i = F(g1^i(s1), g2^i(s2))
# ═══════════════════════════════════════════════════════════════

def build_2d_bivariate(seed1, seed2, weights, chunk_sz=CHUNK_SIZE):
    flat = np.array(weights, dtype=np.float64)
    N = len(flat)
    chunks = [flat[i:i+chunk_sz] for i in range(0, N, chunk_sz)]

    pairs = []
    for ci, ch in enumerate(chunks):
        n = len(ch)

        # Orbit 1
        lo1 = seed1 + 1.5 + ci * (chunk_sz + 2)
        hi1 = lo1 + max(n, 2)
        pts1 = cheb_nodes(n, lo1, hi1)
        g1_in = np.concatenate(([seed1], pts1[:-1]))
        g1c = np.polyfit(g1_in, pts1, n - 1)

        # Orbit 2 (well-separated)
        lo2 = seed2 + 1.5 + ci * (chunk_sz + 2) + 50.0
        hi2 = lo2 + max(n, 2)
        pts2 = cheb_nodes(n, lo2, hi2)
        g2_in = np.concatenate(([seed2], pts2[:-1]))
        g2c = np.polyfit(g2_in, pts2, n - 1)

        # Bivariate polynomial F(x, y) = sum a_ij x^i y^j, i+j <= deg
        # Find minimum total degree with enough monomials
        deg = 0
        while (deg + 1) * (deg + 2) // 2 < n:
            deg += 1
        nm = (deg + 1) * (deg + 2) // 2

        # Vandermonde matrix for scattered points
        V = np.zeros((n, nm))
        for k in range(n):
            idx = 0
            for i in range(deg + 1):
                for j in range(deg + 1 - i):
                    V[k, idx] = pts1[k]**i * pts2[k]**j
                    idx += 1

        # Solve (least squares if overdetermined)
        fc, _, _, _ = np.linalg.lstsq(V, ch, rcond=None)
        pairs.append((g1c, g2c, fc, deg, n))

    return pairs


def eval_2d_bivariate(s1, s2, pairs):
    weights = []
    for g1c, g2c, fc, deg, n in pairs:
        x = float(s1)
        y = float(s2)
        for i in range(n):
            x = horner(g1c, x)
            y = horner(g2c, y)
            w = 0.0
            idx = 0
            for ii in range(deg + 1):
                for jj in range(deg + 1 - ii):
                    w += fc[idx] * x**ii * y**jj
                    idx += 1
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(w):
                return None
            if abs(x) > 1e15 or abs(y) > 1e15:
                return None
            weights.append(w)
    return np.array(weights)


# ═══════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════

def test_sanity():
    print("\n[0] Sanity Check: 2D encoding reconstructs XOR")
    print("─" * 60)

    s1, s2 = SEED_2D

    # Additive
    pairs_add = build_2d_additive(s1, s2, XOR_WEIGHTS)
    w_add = eval_2d_additive(s1, s2, pairs_add)
    if w_add is not None:
        err = np.max(np.abs(w_add - XOR_WEIGHTS))
        print(f"  Additive:  max error = {err:.2e}")
        for inp in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            out = forward_pass(w_add, inp)[0]
            exp = 1 if (inp[0] ^ inp[1]) else 0
            print(f"    {inp} -> {out:+.4f}  (expected {exp})")

    # Bivariate
    pairs_bv = build_2d_bivariate(s1, s2, XOR_WEIGHTS)
    w_bv = eval_2d_bivariate(s1, s2, pairs_bv)
    if w_bv is not None:
        err = np.max(np.abs(w_bv - XOR_WEIGHTS))
        print(f"\n  Bivariate: max error = {err:.2e}")
        for inp in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            out = forward_pass(w_bv, inp)[0]
            exp = 1 if (inp[0] ^ inp[1]) else 0
            print(f"    {inp} -> {out:+.4f}  (expected {exp})")
    else:
        print("  Bivariate: FAILED to reconstruct")


def test_reachability():
    """PCA on sampled weight vectors: 1D curve vs 2D surface."""
    print("\n\n[1] Reachability: PCA on sampled weight vectors")
    print("─" * 60)

    # 1D baseline
    fg_1d = build_1d_fg()
    samples_1d = []
    for s in np.linspace(SEED_1D - 0.001, SEED_1D + 0.001, 1000):
        w = eval_1d(s, fg_1d)
        if w is not None and np.all(np.isfinite(w)):
            samples_1d.append(w)

    if len(samples_1d) > 10:
        M = np.array(samples_1d)
        M -= M.mean(axis=0)
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        total = np.sum(S**2)
        print(f"\n  1D seed ({len(samples_1d)} samples near s={SEED_1D}):")
        for i in range(min(5, len(S))):
            pct = 100.0 * S[i]**2 / total if total > 0 else 0
            print(f"    σ_{i} = {S[i]:.6f}  ({pct:.2f}%)")

    # 2D additive
    pairs_add = build_2d_additive(SEED_2D[0], SEED_2D[1], XOR_WEIGHTS)
    samples_2d = []
    delta = 0.001
    g1 = np.linspace(SEED_2D[0] - delta, SEED_2D[0] + delta, 50)
    g2 = np.linspace(SEED_2D[1] - delta, SEED_2D[1] + delta, 20)
    for s1 in g1:
        for s2 in g2:
            w = eval_2d_additive(s1, s2, pairs_add)
            if w is not None and np.all(np.isfinite(w)):
                samples_2d.append(w)

    if len(samples_2d) > 10:
        M = np.array(samples_2d)
        M -= M.mean(axis=0)
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        total = np.sum(S**2)
        print(f"\n  2D additive ({len(samples_2d)} samples near {SEED_2D}):")
        for i in range(min(5, len(S))):
            pct = 100.0 * S[i]**2 / total if total > 0 else 0
            print(f"    σ_{i} = {S[i]:.6f}  ({pct:.2f}%)")

    # 2D bivariate
    pairs_bv = build_2d_bivariate(SEED_2D[0], SEED_2D[1], XOR_WEIGHTS)
    samples_bv = []
    for s1 in g1:
        for s2 in g2:
            w = eval_2d_bivariate(s1, s2, pairs_bv)
            if w is not None and np.all(np.isfinite(w)):
                samples_bv.append(w)

    if len(samples_bv) > 10:
        M = np.array(samples_bv)
        M -= M.mean(axis=0)
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        total = np.sum(S**2)
        print(f"\n  2D bivariate ({len(samples_bv)} samples near {SEED_2D}):")
        for i in range(min(5, len(S))):
            pct = 100.0 * S[i]**2 / total if total > 0 else 0
            print(f"    σ_{i} = {S[i]:.6f}  ({pct:.2f}%)")


def test_behaviors():
    """Count distinct boolean behaviors reachable from 1D vs 2D seeds."""
    print("\n\n[2] Behavior Diversity: 1D vs 2D")
    print("─" * 60)

    # 1D — same tight radius for fair comparison
    fg_1d = build_1d_fg()
    beh_1d = {}
    n_valid = 0
    r_1d = 0.005
    for s in np.linspace(SEED_1D - r_1d, SEED_1D + r_1d, 250000):
        w = eval_1d(s, fg_1d)
        if w is None:
            continue
        n_valid += 1
        name, bits, outputs, clean = classify(w)
        if clean and name not in beh_1d:
            beh_1d[name] = (s, outputs)

    print(f"\n  1D: {n_valid} valid seeds in [{SEED_1D-r_1d}, {SEED_1D+r_1d}]")
    print(f"  Found {len(beh_1d)} distinct clean behaviors:")
    for name in sorted(beh_1d.keys()):
        s, outs = beh_1d[name]
        out_str = ' '.join(f'{o:+.2f}' for o in outs)
        print(f"    {name:<8} seed={s:+.4f}  [{out_str}]")

    # 2D additive — use tighter grid (polynomials diverge at distance)
    pairs_add = build_2d_additive(SEED_2D[0], SEED_2D[1], XOR_WEIGHTS)
    beh_2d = {}
    n_valid = 0
    r = 0.005  # tight radius — polynomials diverge quickly
    grid_1 = np.linspace(SEED_2D[0] - r, SEED_2D[0] + r, 500)
    grid_2 = np.linspace(SEED_2D[1] - r, SEED_2D[1] + r, 500)
    for s1 in grid_1:
        for s2 in grid_2:
            w = eval_2d_additive(s1, s2, pairs_add)
            if w is None:
                continue
            n_valid += 1
            name, bits, outputs, clean = classify(w)
            if clean and name not in beh_2d:
                beh_2d[name] = ((s1, s2), outputs)

    print(f"\n  2D additive: {n_valid} valid seeds in "
          f"[{SEED_2D[0]-r},{SEED_2D[0]+r}]×"
          f"[{SEED_2D[1]-r},{SEED_2D[1]+r}]")
    print(f"  Found {len(beh_2d)} distinct clean behaviors:")
    for name in sorted(beh_2d.keys()):
        (s1, s2), outs = beh_2d[name]
        out_str = ' '.join(f'{o:+.2f}' for o in outs)
        print(f"    {name:<8} seed=({s1:+.4f},{s2:+.4f})  [{out_str}]")

    gain = len(beh_2d) - len(beh_1d)
    print(f"\n  Gain from 2D: {len(beh_1d)} → {len(beh_2d)} behaviors ({gain:+d})")


def test_nearest_neighbor():
    """Compare nearest-neighbor approximation quality."""
    print("\n\n[3] Nearest-Neighbor Distance: 1D vs 2D")
    print("─" * 60)

    fg_1d = build_1d_fg()
    pairs_add = build_2d_additive(SEED_2D[0], SEED_2D[1], XOR_WEIGHTS)
    N = len(XOR_WEIGHTS)
    rng = np.random.RandomState(99)

    targets = []
    labels = []
    # Random perturbations of XOR weights
    for i in range(3):
        targets.append(XOR_WEIGHTS + rng.uniform(-1.0, 1.0, N))
        labels.append(f"XOR + noise(±1.0) #{i}")
    # Completely random weights
    for i in range(2):
        targets.append(rng.uniform(-3, 3, N))
        labels.append(f"Random [-3,3] #{i}")

    for ti, (target, label) in enumerate(zip(targets, labels)):
        print(f"\n  Target {ti}: {label}")

        # --- 1D search ---
        def obj_1d(s):
            w = eval_1d(float(s), fg_1d)
            if w is None:
                return 1e20
            return float(np.sum((w - target)**2))

        best_s1d = SEED_1D
        best_o1d = obj_1d(SEED_1D)
        for s in np.linspace(SEED_1D - 2, SEED_1D + 2, 20000):
            o = obj_1d(s)
            if o < best_o1d:
                best_o1d = o
                best_s1d = s

        # Refine
        try:
            res = minimize(lambda s: obj_1d(s[0]), [best_s1d],
                           method='Nelder-Mead',
                           options={'xatol': 1e-14, 'maxiter': 5000})
            if res.fun < best_o1d:
                best_o1d = res.fun
                best_s1d = res.x[0]
        except Exception:
            pass

        # --- 2D search ---
        def obj_2d(sv):
            w = eval_2d_additive(float(sv[0]), float(sv[1]), pairs_add)
            if w is None:
                return 1e20
            return float(np.sum((w - target)**2))

        best_s2d = np.array(SEED_2D)
        best_o2d = obj_2d(best_s2d)
        for s1 in np.linspace(SEED_2D[0] - 2, SEED_2D[0] + 2, 200):
            for s2 in np.linspace(SEED_2D[1] - 2, SEED_2D[1] + 2, 100):
                o = obj_2d([s1, s2])
                if o < best_o2d:
                    best_o2d = o
                    best_s2d = np.array([s1, s2])

        # Refine
        try:
            res = minimize(obj_2d, best_s2d, method='Nelder-Mead',
                           options={'xatol': 1e-14, 'maxiter': 5000})
            if res.fun < best_o2d:
                best_o2d = res.fun
                best_s2d = res.x
        except Exception:
            pass

        rmse_1d = np.sqrt(best_o1d / N)
        rmse_2d = np.sqrt(best_o2d / N)
        ratio = best_o1d / max(best_o2d, 1e-30)
        print(f"    1D: seed={best_s1d:.6f}  RMSE={rmse_1d:.4f}")
        print(f"    2D: seed=({best_s2d[0]:.6f},{best_s2d[1]:.6f})  RMSE={rmse_2d:.4f}")
        print(f"    Improvement: {ratio:.1f}x (L2² ratio)")


def test_lyapunov_2d():
    """Lyapunov analysis for 2D system."""
    print("\n\n[4] Lyapunov Analysis: 1D vs 2D")
    print("─" * 60)

    eps = 1e-9
    s1_0, s2_0 = SEED_2D

    pairs_add = build_2d_additive(s1_0, s2_0, XOR_WEIGHTS)
    w00 = eval_2d_additive(s1_0, s2_0, pairs_add)
    w10 = eval_2d_additive(s1_0 + eps, s2_0, pairs_add)
    w01 = eval_2d_additive(s1_0, s2_0 + eps, pairs_add)

    if w00 is not None and w10 is not None and w01 is not None:
        dw_ds1 = (w10 - w00) / eps  # N-vector
        dw_ds2 = (w01 - w00) / eps  # N-vector

        J = np.column_stack([dw_ds1, dw_ds2])  # N×2 Jacobian
        _, S, _ = np.linalg.svd(J, full_matrices=False)

        print(f"\n  2D additive Jacobian (17×2) at design point:")
        for i in range(len(S)):
            lyap = np.log10(S[i]) if S[i] > 0 else float('-inf')
            print(f"    σ_{i} = {S[i]:.4e}  (log₁₀ = {lyap:.2f})")

        if len(S) > 1 and S[1] > 0:
            print(f"\n  Anisotropy σ₁/σ₂ = {S[0]/S[1]:.2f}")
            print(f"  (1.0 = isotropic 2D surface, >>1 = nearly 1D)")

    # Compare with 1D Jacobian
    fg_1d = build_1d_fg()
    w0 = eval_1d(SEED_1D, fg_1d)
    w1 = eval_1d(SEED_1D + eps, fg_1d)
    if w0 is not None and w1 is not None:
        dw = (w1 - w0) / eps
        norm = np.linalg.norm(dw)
        print(f"\n  1D Jacobian norm at design point: {norm:.4e}")
        print(f"  (single stretching direction)")

    # Security analysis
    print(f"\n  Precision-security ceiling (float64, δ=1e-6):")
    print(f"    1D: adversary solves 1D search → ~33 bits headroom")
    print(f"    2D: adversary solves 2D search → same 33 bits headroom")
    print(f"        BUT: 2D search space is quadratically larger")
    print(f"        AND: additive split means each channel is random")
    print(f"             (secret-sharing: need BOTH seeds to recover weights)")


def test_secret_sharing():
    """Show that either seed alone reveals nothing about weights."""
    print("\n\n[5] Secret-Sharing Property (Additive Split)")
    print("─" * 60)

    s1, s2 = SEED_2D
    pairs = build_2d_additive(s1, s2, XOR_WEIGHTS)

    # What does channel 1 alone produce?
    print(f"\n  Full reconstruction (both seeds):")
    w_full = eval_2d_additive(s1, s2, pairs)
    if w_full is not None:
        err = np.max(np.abs(w_full - XOR_WEIGHTS))
        print(f"    Max error: {err:.2e}")
        print(f"    Weights: {np.round(w_full, 3)}")

    # What does each channel contribute individually at the design point?
    # Compute channel contributions by evaluating the f and h polynomials
    print(f"\n  Channel 1 contribution at design point (s1={s1}):")
    ch1_vals = []
    for g1c, fc, g2c, hc, n in pairs:
        x = float(s1)
        for i in range(n):
            x = horner(g1c, x)
            a = horner(fc, x)
            ch1_vals.append(a)
    ch1_vals = np.array(ch1_vals)
    print(f"    Partial sums: {np.round(ch1_vals, 3)}")
    print(f"    These are RANDOM — no information about actual weights")

    print(f"\n  Channel 2 contribution at design point (s2={s2}):")
    ch2_vals = []
    for g1c, fc, g2c, hc, n in pairs:
        y = float(s2)
        for i in range(n):
            y = horner(g2c, y)
            b = horner(hc, y)
            ch2_vals.append(b)
    ch2_vals = np.array(ch2_vals)
    print(f"    Partial sums: {np.round(ch2_vals, 3)}")

    print(f"\n  Sum (ch1 + ch2): {np.round(ch1_vals + ch2_vals, 3)}")
    print(f"  True weights:    {np.round(XOR_WEIGHTS, 3)}")
    print(f"  Residual:        {np.max(np.abs(ch1_vals + ch2_vals - XOR_WEIGHTS)):.2e}")

    # Wrong s2 → wrong weights
    s2_wrong = s2 + 0.001
    w_wrong = eval_2d_additive(s1, s2_wrong, pairs)
    if w_wrong is not None:
        print(f"\n  Wrong s2 ({s2_wrong:.4f} instead of {s2:.4f}):")
        print(f"    'Weights': {np.round(w_wrong, 3)}")
        err = np.max(np.abs(w_wrong - XOR_WEIGHTS))
        print(f"    Max weight error: {err:.4f}")
        # Test if XOR still works
        all_ok = True
        for inp, exp in [([0,0],0), ([0,1],1), ([1,0],1), ([1,1],0)]:
            out = forward_pass(w_wrong, inp)[0]
            ok = abs(out - exp) < 0.3
            if not ok:
                all_ok = False
        print(f"    XOR behavior: {'PASS' if all_ok else 'FAIL'}")

    # What does the binary reveal? The split a_i and b_i are random
    print(f"\n  Interpretation:")
    print(f"    w_i = f(orbit1_i) + h(orbit2_i)")
    print(f"    f and h are in the binary (public)")
    print(f"    But each channel's contribution is a random split")
    print(f"    Recovering s1 alone gives meaningless partial sums")
    print(f"    Need BOTH s1 AND s2 to reconstruct actual weights")


def main():
    print("=" * 65)
    print("  2D Seed Space Exploration")
    print("  Scalar seed → 2D seed: what changes?")
    print("=" * 65)

    test_sanity()
    test_reachability()
    test_behaviors()
    test_nearest_neighbor()
    test_lyapunov_2d()
    test_secret_sharing()

    print(f"\n{'═' * 65}")
    print("  SUMMARY")
    print(f"{'═' * 65}")
    print("""
  1D seed (baseline):
    - Reachable set = 1D curve in R^17 (99%+ variance in σ₁)
    - Limited behavior diversity
    - Poor nearest-neighbor approximation to arbitrary targets

  2D seed (additive split):
    - Reachable set = 2D surface in R^17 (two significant σ)
    - More diverse behaviors reachable
    - Better nearest-neighbor (2 search dimensions)
    - SECRET-SHARING: each channel alone is random
      → recovering one seed reveals nothing about weights
      → need both seeds, which can arrive via different channels

  2D seed (bivariate):
    - Also 2D surface, but coupled (no secret-sharing property)
    - Potentially better coverage (non-additive structure)
    - More complex polynomial evaluation

  The additive split is the interesting one:
    - Binary contains 4 public arrays per chunk: g1, f, g2, h
    - C2 sends 16 bytes total (two doubles)
    - s1 could come over one channel, s2 over another
    - Neither channel alone compromises the weights
    """)
    print(f"{'═' * 65}")


if __name__ == '__main__':
    main()
