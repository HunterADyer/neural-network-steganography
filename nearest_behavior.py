#!/usr/bin/env python3
"""
Nearest-Behavior Seed Search
==============================

Given fixed (f, g) polynomial pairs designed for XOR weights,
search for seeds that produce completely different network behaviors
(AND, OR, NAND, NOR, etc.)

The question: how much behavioral diversity can you get from
2 fixed equations and a single scalar seed?
"""

import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution
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

# Target behaviors: (name, [(input, expected_output), ...])
BEHAVIORS = {
    'XOR':  [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)],
    'AND':  [([0,0], 0), ([0,1], 0), ([1,0], 0), ([1,1], 1)],
    'OR':   [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 1)],
    'NAND': [([0,0], 1), ([0,1], 1), ([1,0], 1), ([1,1], 0)],
    'NOR':  [([0,0], 1), ([0,1], 0), ([1,0], 0), ([1,1], 0)],
    'XNOR': [([0,0], 1), ([0,1], 0), ([1,0], 0), ([1,1], 1)],
    'TRUE': [([0,0], 1), ([0,1], 1), ([1,0], 1), ([1,1], 1)],
    'FALSE':[([0,0], 0), ([0,1], 0), ([1,0], 0), ([1,1], 0)],
    'ID_0': [([0,0], 0), ([0,1], 0), ([1,0], 1), ([1,1], 1)],  # output = input 0
    'ID_1': [([0,0], 0), ([0,1], 1), ([1,0], 0), ([1,1], 1)],  # output = input 1
}


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


def forward_pass(weights, inputs):
    arch = ARCH
    cur = np.array(inputs, dtype=np.float64)
    off = 0
    for l in range(len(arch) - 1):
        ni, no = arch[l], arch[l + 1]
        nxt = np.zeros(no)
        for j in range(no):
            s = weights[off + ni * no + j]
            for i in range(ni):
                s += cur[i] * weights[off + i * no + j]
            if l < len(arch) - 2 and s < 0:
                s = 0.0
            nxt[j] = s
        off += ni * no + no
        cur = nxt
    return cur


def behavior_error(s, fg_pairs, target_behavior):
    """MSE between network output and target behavior."""
    s_val = float(np.ravel(s)[0]) if hasattr(s, '__len__') else float(s)
    w = evaluate_seed(s_val, fg_pairs)
    if w is None:
        return 1e10
    err = 0.0
    for inputs, expected in target_behavior:
        out = forward_pass(w, inputs)[0]
        if not np.isfinite(out):
            return 1e10
        err += (out - expected) ** 2
    return err / len(target_behavior)


def classify_behavior(w, threshold=0.3):
    """Classify the boolean behavior of a [2,4,1] network."""
    outputs = []
    for inp in [[0,0], [0,1], [1,0], [1,1]]:
        out = forward_pass(w, inp)[0]
        outputs.append(out)
    bits = tuple(1 if o > 0.5 else 0 for o in outputs)

    truth_table = {
        (0,0,0,0): 'FALSE',
        (0,0,0,1): 'AND',
        (0,0,1,0): 'ID_0∧¬ID_1',
        (0,0,1,1): 'ID_0',
        (0,1,0,0): '¬ID_0∧ID_1',
        (0,1,0,1): 'ID_1',
        (0,1,1,0): 'XOR',
        (0,1,1,1): 'OR',
        (1,0,0,0): 'NOR',
        (1,0,0,1): 'XNOR',
        (1,0,1,0): '¬ID_1',
        (1,0,1,1): 'ID_0∨¬ID_1',
        (1,1,0,0): 'NAND',
        (1,1,0,1): '¬ID_0∨ID_1',
        (1,1,1,0): '¬AND_LIKE',
        (1,1,1,1): 'TRUE',
    }

    # Check if outputs are clean (close to 0 or 1)
    clean = all(abs(o - round(o)) < threshold for o in outputs)
    name = truth_table.get(bits, f'?{bits}')
    return name, outputs, clean


def main():
    fg_pairs = build_fixed_fg()

    print("=" * 70)
    print("  Nearest-Behavior Seed Search")
    print("  Fixed (f, g) designed for XOR — can other logic gates emerge?")
    print("=" * 70)

    # First: scan a wide range and catalog all behaviors found
    print("\n[1] Scanning seeds in [-5, 12] for distinct boolean behaviors...")
    found_behaviors = {}
    grid = np.linspace(-5, 12, 500000)
    for s in grid:
        w = evaluate_seed(s, fg_pairs)
        if w is None:
            continue
        name, outputs, clean = classify_behavior(w)
        if clean and name not in found_behaviors:
            found_behaviors[name] = (s, outputs)

    print(f"    Found {len(found_behaviors)} distinct clean behaviors:\n")
    for name in sorted(found_behaviors.keys()):
        s, outputs = found_behaviors[name]
        out_str = [f"{o:+.3f}" for o in outputs]
        print(f"    {name:<12} seed={s:+.6f}  "
              f"[0,0]={out_str[0]} [0,1]={out_str[1]} "
              f"[1,0]={out_str[2]} [1,1]={out_str[3]}")

    # Second: for each target behavior, find the best seed via optimization
    print(f"\n{'─' * 70}")
    print("[2] Optimizing seed for each target behavior...\n")

    results = {}
    for name, target in BEHAVIORS.items():
        # Grid search
        best_s = SEED
        best_err = behavior_error(SEED, fg_pairs, target)
        for s in np.linspace(-5, 12, 200000):
            err = behavior_error(s, fg_pairs, target)
            if err < best_err:
                best_err = err
                best_s = s

        # Local refinement
        try:
            res = minimize_scalar(
                lambda s: behavior_error(s, fg_pairs, target),
                bounds=(best_s - 0.01, best_s + 0.01),
                method='bounded',
                options={'xatol': 1e-14})
            if res.fun < best_err:
                best_err = res.fun
                best_s = res.x
        except Exception:
            pass

        w = evaluate_seed(best_s, fg_pairs)
        if w is not None:
            outputs = [forward_pass(w, inp)[0] for inp in [[0,0],[0,1],[1,0],[1,1]]]
            expected = [t[1] for t in target]
            max_err = max(abs(o - e) for o, e in zip(outputs, expected))
            ok = max_err < 0.3

            out_str = [f"{o:+.3f}" for o in outputs]
            exp_str = [f"{e}" for e in expected]
            status = "FOUND" if ok else "CLOSE" if max_err < 1.0 else "MISS"
            results[name] = (best_s, best_err, max_err, ok)

            print(f"  {name:<6} seed={best_s:+.10f}  MSE={best_err:.4e}  "
                  f"max_err={max_err:.3f}  [{status}]")
            print(f"         outputs: {out_str}")
            print(f"         target:  {exp_str}")
            print()

    # Summary
    print(f"{'═' * 70}")
    print("  SUMMARY: Behaviors achievable from fixed (f, g) for XOR\n")
    found_count = sum(1 for v in results.values() if v[3])
    close_count = sum(1 for v in results.values() if not v[3] and v[2] < 1.0)
    miss_count = sum(1 for v in results.values() if v[2] >= 1.0)
    print(f"  FOUND (max_err < 0.3): {found_count}/{len(results)}")
    print(f"  CLOSE (max_err < 1.0): {close_count}/{len(results)}")
    print(f"  MISS  (max_err ≥ 1.0): {miss_count}/{len(results)}")
    print()
    for name in sorted(results.keys()):
        s, mse, maxe, ok = results[name]
        status = "OK" if ok else "~" if maxe < 1.0 else "X"
        print(f"    [{status}] {name:<6}  seed={s:+.6f}  max_err={maxe:.3f}")
    print(f"\n{'═' * 70}")


if __name__ == '__main__':
    main()
