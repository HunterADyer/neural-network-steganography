#!/usr/bin/env python3
"""
Nearest-Neighbor Seed Search
=============================

Given fixed (f, g) polynomial pairs already baked into a binary,
find the seed that produces the weight vector closest to a given target.

This is a 1D optimization: minimize ||F(s) - w_target||² over s ∈ ℝ.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
import sys

SEED = 3.7133
CHUNK_SIZE = 8
ARCH = [2, 4, 1]

XOR_WEIGHTS = np.array([
    1.0, 1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, -1.0,
    -1.5, -0.5, -0.5, 0.5,
    0.1, 2.1, 2.1, 0.1,
    -0.05,
])

# ---------------------------------------------------------------------------
# Encode: build the (f, g) pairs from the ORIGINAL weights + seed
# These are FIXED — they represent the binary's baked-in coefficients
# ---------------------------------------------------------------------------

def cheb_nodes(n, lo, hi):
    k = np.arange(n, dtype=np.float64)
    raw = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)


def horner(coeffs, x):
    r = float(coeffs[0])
    for c in coeffs[1:]:
        r = r * x + float(c)
    return r


def build_fixed_fg():
    """Build (f, g) pairs from the original XOR weights. These are FIXED."""
    flat = XOR_WEIGHTS
    chunks = [flat[i:i+CHUNK_SIZE] for i in range(0, len(flat), CHUNK_SIZE)]
    fg_pairs = []
    for ci, ch in enumerate(chunks):
        n = len(ch)
        lo = SEED + 1.5 + ci * (CHUNK_SIZE + 2)
        hi = lo + max(n, 2)
        pts = cheb_nodes(n, lo, hi)
        gc = np.polyfit(pts, ch, n - 1)
        f_in = np.concatenate(([SEED], pts[:-1]))
        fc = np.polyfit(f_in, pts, n - 1)
        fg_pairs.append((fc, gc, n))
    return fg_pairs


def evaluate_seed(s, fg_pairs):
    """Given a seed and fixed (f, g) pairs, produce the weight vector."""
    weights = []
    for fc, gc, n in fg_pairs:
        x = float(s)
        for i in range(n):
            x = horner(fc, x)
            w = horner(gc, x)
            if not np.isfinite(x) or not np.isfinite(w):
                return None  # diverged
            weights.append(w)
    return np.array(weights)


# ---------------------------------------------------------------------------
# Nearest-neighbor search
# ---------------------------------------------------------------------------

def find_nearest_seed(target, fg_pairs, s_init=SEED, search_range=10.0):
    """
    Find the seed that produces weights closest to target.
    Uses a combination of grid search + local optimization.
    """
    N = len(target)

    def objective(s):
        s_val = float(np.ravel(s)[0]) if hasattr(s, '__len__') else float(s)
        w = evaluate_seed(s_val, fg_pairs)
        if w is None or len(w) != N:
            return 1e20
        return float(np.sum((w - target) ** 2))

    # Phase 1: coarse grid search
    best_s = s_init
    best_obj = objective(s_init)
    grid = np.linspace(s_init - search_range, s_init + search_range, 10000)
    for s in grid:
        obj = objective(s)
        if obj < best_obj:
            best_obj = obj
            best_s = s

    # Phase 2: local optimization from best grid point
    result = minimize(objective, best_s, method='Nelder-Mead',
                      options={'xatol': 1e-15, 'fatol': 1e-15, 'maxiter': 10000})
    if result.fun < best_obj:
        best_s = result.x[0]
        best_obj = result.fun

    # Also try bounded scalar optimization near the best point
    try:
        result2 = minimize_scalar(objective,
                                  bounds=(best_s - 0.1, best_s + 0.1),
                                  method='bounded',
                                  options={'xatol': 1e-15})
        if result2.fun < best_obj:
            best_s = result2.x
            best_obj = result2.fun
    except Exception:
        pass

    return best_s, best_obj


def forward_pass(weights, inputs):
    """Run the [2,4,1] ReLU network."""
    arch = ARCH
    w = weights
    cur = np.array(inputs, dtype=np.float64)
    off = 0
    for l in range(len(arch) - 1):
        ni, no = arch[l], arch[l + 1]
        nxt = np.zeros(no)
        for j in range(no):
            s = w[off + ni * no + j]  # bias
            for i in range(ni):
                s += cur[i] * w[off + i * no + j]
            if l < len(arch) - 2 and s < 0:
                s = 0.0
            nxt[j] = s
        off += ni * no + no
        cur = nxt
    return cur


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def main():
    fg_pairs = build_fixed_fg()

    print("=" * 65)
    print("  Nearest-Neighbor Seed Search")
    print("  Fixed (f, g) from XOR weights, searching for seeds that")
    print("  approximate various target weight vectors.")
    print("=" * 65)

    # Verify: the original seed should produce the original weights exactly
    w_orig = evaluate_seed(SEED, fg_pairs)
    err_orig = np.max(np.abs(w_orig - XOR_WEIGHTS))
    print(f"\n[0] Sanity check: original seed {SEED}")
    print(f"    Max error: {err_orig:.2e}")
    print(f"    XOR outputs: ", end="")
    for inp in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        out = forward_pass(w_orig, inp)
        print(f"{inp}→{out[0]:.3f}  ", end="")
    print()

    # Experiment 1: small perturbation of original weights
    print(f"\n{'─' * 65}")
    print("[1] Target: original weights + small random perturbation (±0.1)")
    rng = np.random.RandomState(42)
    target1 = XOR_WEIGHTS + rng.uniform(-0.1, 0.1, len(XOR_WEIGHTS))
    s1, obj1 = find_nearest_seed(target1, fg_pairs)
    w1 = evaluate_seed(s1, fg_pairs)
    err1 = np.max(np.abs(w1 - target1))
    print(f"    Best seed: {s1:.10f}")
    print(f"    L2 distance²: {obj1:.6e}")
    print(f"    Max element error: {err1:.6e}")
    print(f"    XOR outputs: ", end="")
    for inp in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        out = forward_pass(w1, inp)
        print(f"{inp}→{out[0]:.3f}  ", end="")
    print()

    # Experiment 2: larger perturbation (±0.5)
    print(f"\n{'─' * 65}")
    print("[2] Target: original weights + medium perturbation (±0.5)")
    target2 = XOR_WEIGHTS + rng.uniform(-0.5, 0.5, len(XOR_WEIGHTS))
    s2, obj2 = find_nearest_seed(target2, fg_pairs)
    w2 = evaluate_seed(s2, fg_pairs)
    err2 = np.max(np.abs(w2 - target2))
    print(f"    Best seed: {s2:.10f}")
    print(f"    L2 distance²: {obj2:.6e}")
    print(f"    Max element error: {err2:.6e}")
    print(f"    XOR outputs: ", end="")
    for inp in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        out = forward_pass(w2, inp)
        print(f"{inp}→{out[0]:.3f}  ", end="")
    print()

    # Experiment 3: completely random weights
    print(f"\n{'─' * 65}")
    print("[3] Target: completely random weights in [-3, 3]")
    target3 = rng.uniform(-3, 3, len(XOR_WEIGHTS))
    s3, obj3 = find_nearest_seed(target3, fg_pairs)
    w3 = evaluate_seed(s3, fg_pairs)
    if w3 is not None:
        err3 = np.max(np.abs(w3 - target3))
        print(f"    Best seed: {s3:.10f}")
        print(f"    L2 distance²: {obj3:.6e}")
        print(f"    Max element error: {err3:.6e}")
    else:
        print(f"    Diverged at best seed")

    # Experiment 4: can we find a seed that still solves XOR
    # but with different weights?
    print(f"\n{'─' * 65}")
    print("[4] Search: any seed in [-10, 10] that solves XOR?")
    print("    Scanning 100k seeds for XOR-correct behavior...")
    xor_seeds = []
    test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    test_targets = [0.0, 1.0, 1.0, 0.0]
    grid = np.linspace(-10, 10, 100000)
    for s in grid:
        w = evaluate_seed(s, fg_pairs)
        if w is None:
            continue
        correct = True
        for inp, tgt in zip(test_inputs, test_targets):
            out = forward_pass(w, inp)[0]
            if abs(round(out) - tgt) > 0.01:
                correct = False
                break
        if correct:
            xor_seeds.append(s)

    print(f"    Found {len(xor_seeds)} seeds that solve XOR")
    if xor_seeds:
        for s in xor_seeds[:5]:
            w = evaluate_seed(s, fg_pairs)
            dist = np.linalg.norm(w - XOR_WEIGHTS)
            outputs = [forward_pass(w, inp)[0] for inp in test_inputs]
            print(f"      s={s:+.6f}: outputs={[f'{o:.3f}' for o in outputs]}, "
                  f"||w - w_orig|| = {dist:.4f}")
        if len(xor_seeds) > 5:
            print(f"      ... and {len(xor_seeds) - 5} more")

    # Experiment 5: sensitivity — how does error grow with seed distance?
    print(f"\n{'─' * 65}")
    print("[5] Sensitivity: weight error vs seed perturbation")
    deltas = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    for d in deltas:
        w_perturbed = evaluate_seed(SEED + d, fg_pairs)
        if w_perturbed is not None:
            max_err = np.max(np.abs(w_perturbed - XOR_WEIGHTS))
            l2 = np.linalg.norm(w_perturbed - XOR_WEIGHTS)
            print(f"    Δseed = {d:.0e}: max|Δw| = {max_err:.2e}, ||Δw|| = {l2:.2e}")
        else:
            print(f"    Δseed = {d:.0e}: DIVERGED")

    print(f"\n{'═' * 65}")


if __name__ == '__main__':
    main()
