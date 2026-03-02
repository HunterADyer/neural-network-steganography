#!/usr/bin/env python3
"""
Multi-Behavior Encoding
========================

Design (f, g) so that DIFFERENT seeds produce DIFFERENT network behaviors
from the same fixed binary.

Key insight: K behaviors × n chunk size = K*n interpolation constraints.
Polynomial degree = K*n - 1.

This test: encode XOR (seed s1) and OR (seed s2) into the same (f, g) pairs.
"""

import numpy as np
import json
import sys

ARCH = [2, 4, 1]

# Two different sets of weights for different behaviors
BEHAVIORS = {
    'XOR': {
        'seed': 3.7133,
        'weights': [
            1.0, 1.0, -1.0, -1.0,
            1.0, -1.0, 1.0, -1.0,
            -1.5, -0.5, -0.5, 0.5,
            0.1, 2.1, 2.1, 0.1,
            -0.05,
        ],
        'tests': [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 0)],
    },
    'AND': {
        'seed': 7.2901,
        'weights': [
            1.0, 1.0, 0.5, -0.5,
            1.0, 1.0, -0.5, 0.5,
            -1.5, -0.5, -0.3, 0.3,
            2.0, 1.0, 0.5, -0.5,
            -0.5,
        ],
        'tests': [([0,0], 0), ([0,1], 0), ([1,0], 0), ([1,1], 1)],
    },
    'OR': {
        'seed': 5.5017,
        'weights': [
            2.0, 2.0, -1.0, -1.0,
            2.0, -2.0, 2.0, -2.0,
            -0.5, -0.5, -0.5, 0.5,
            1.0, 1.0, 1.0, 0.1,
            -0.5,
        ],
        'tests': [([0,0], 0), ([0,1], 1), ([1,0], 1), ([1,1], 1)],
    },
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


def encode_multi_chunk(behavior_list, chunk_idx, chunk_sz):
    """
    Encode one chunk for K behaviors simultaneously.

    Each behavior provides (seed, weights_chunk).
    f and g polynomials of degree K*n - 1 interpolate all K orbits.

    Returns (fc, gc) polynomial coefficients.
    """
    K = len(behavior_list)
    n = len(behavior_list[0][1])  # chunk size (same for all)
    total_pts = K * n
    deg = total_pts - 1

    # For each behavior, create distinct evaluation points on non-overlapping intervals
    all_f_in = []
    all_f_out = []
    all_g_in = []
    all_g_out = []

    for bi, (seed, weights_chunk) in enumerate(behavior_list):
        # Each behavior gets its own interval, well-separated
        lo = 20.0 + bi * 15.0 + chunk_idx * (K * 15.0 + 5.0)
        hi = lo + max(n, 2) * 1.5
        pts = cheb_nodes(n, lo, hi)

        # f chain: seed -> pts[0] -> pts[1] -> ... -> pts[n-1]
        f_inputs = np.concatenate(([seed], pts[:-1]))
        f_outputs = pts

        # g evaluation: g(pts[i]) = weights[i]
        g_inputs = pts
        g_outputs = np.array(weights_chunk, dtype=np.float64)

        all_f_in.append(f_inputs)
        all_f_out.append(f_outputs)
        all_g_in.append(g_inputs)
        all_g_out.append(g_outputs)

    # Combine all interpolation constraints
    f_x = np.concatenate(all_f_in)
    f_y = np.concatenate(all_f_out)
    g_x = np.concatenate(all_g_in)
    g_y = np.concatenate(all_g_out)

    # Fit polynomials through ALL points
    fc = np.polyfit(f_x, f_y, deg)
    gc = np.polyfit(g_x, g_y, deg)

    return fc, gc


def verify_behavior(fc, gc, seed, weights_chunk):
    """Simulate reconstruction and return max error."""
    x = float(seed)
    max_err = 0.0
    recon = []
    for w_expected in weights_chunk:
        x = horner(fc, x)
        w_got = horner(gc, x)
        recon.append(w_got)
        max_err = max(max_err, abs(w_got - w_expected))
    return max_err, recon


def test_combination(names, chunk_sz=8):
    """Test encoding multiple behaviors into shared (f, g)."""
    behaviors = [BEHAVIORS[n] for n in names]
    K = len(behaviors)
    N = len(behaviors[0]['weights'])

    flat_list = [np.array(b['weights'], dtype=np.float64) for b in behaviors]
    seeds = [b['seed'] for b in behaviors]

    chunks_per = [
        [fl[i:i+chunk_sz] for i in range(0, N, chunk_sz)]
        for fl in flat_list
    ]
    n_chunks = len(chunks_per[0])

    deg = K * chunk_sz - 1
    print(f"\n  Encoding {names}")
    print(f"  K={K} behaviors, chunk_size={chunk_sz}, degree={deg}")
    print(f"  Seeds: {seeds}")

    all_fc, all_gc = [], []
    for ci in range(n_chunks):
        behavior_chunks = []
        for bi in range(K):
            behavior_chunks.append((seeds[bi], chunks_per[bi][ci]))

        fc, gc = encode_multi_chunk(behavior_chunks, ci, chunk_sz)
        all_fc.append(fc)
        all_gc.append(gc)

    # Verify each behavior
    for bi, name in enumerate(names):
        print(f"\n  --- {name} (seed={seeds[bi]}) ---")
        all_recon = []
        overall_max_err = 0.0

        for ci in range(n_chunks):
            err, recon = verify_behavior(
                all_fc[ci], all_gc[ci],
                seeds[bi], chunks_per[bi][ci])
            all_recon.extend(recon)
            overall_max_err = max(overall_max_err, err)
            print(f"    Chunk {ci}: max_err = {err:.2e}")

        print(f"    Overall max_err = {overall_max_err:.2e}")

        # Test inference
        w = np.array(all_recon)
        print(f"    Inference:")
        all_ok = True
        for inp, expected in behaviors[bi]['tests']:
            out = forward_pass(w, inp)[0]
            ok = abs(out - expected) < 0.3
            if not ok:
                all_ok = False
            print(f"      {inp} -> {out:+.4f}  (expected {expected})  "
                  f"[{'OK' if ok else 'FAIL'}]")
        print(f"    Behavior: {'PASS' if all_ok else 'FAIL'}")

    return deg


def main():
    print("=" * 65)
    print("  Multi-Behavior Encoding")
    print("  Same (f, g) polynomials, different seeds → different behaviors")
    print("=" * 65)

    # Test K=2 with chunk_size=8 (degree 15)
    print(f"\n{'━' * 65}")
    print("TEST 1: K=2, chunk_size=8 → degree 15 (borderline float64)")
    print(f"{'━' * 65}")
    test_combination(['XOR', 'OR'], chunk_sz=8)

    # Test K=2 with chunk_size=4 (degree 7 — safe)
    print(f"\n{'━' * 65}")
    print("TEST 2: K=2, chunk_size=4 → degree 7 (safe float64)")
    print(f"{'━' * 65}")
    test_combination(['XOR', 'OR'], chunk_sz=4)

    # Test K=3 with chunk_size=4 (degree 11)
    print(f"\n{'━' * 65}")
    print("TEST 3: K=3, chunk_size=4 → degree 11 (should be ok)")
    print(f"{'━' * 65}")
    test_combination(['XOR', 'OR', 'AND'], chunk_sz=4)

    # Test K=3 with chunk_size=8 (degree 23 — likely unstable)
    print(f"\n{'━' * 65}")
    print("TEST 4: K=3, chunk_size=8 → degree 23 (likely unstable)")
    print(f"{'━' * 65}")
    test_combination(['XOR', 'OR', 'AND'], chunk_sz=8)

    # Test K=2 with chunk_size=2 (degree 3 — very safe)
    print(f"\n{'━' * 65}")
    print("TEST 5: K=2, chunk_size=2 → degree 3 (very safe)")
    print(f"{'━' * 65}")
    test_combination(['XOR', 'AND'], chunk_sz=2)

    # Summary
    print(f"\n{'━' * 65}")
    print("SCALING: max behaviors vs chunk size at degree budget ≤ 15")
    print(f"{'━' * 65}")
    for n in [2, 3, 4, 5, 6, 8]:
        k_max = (15 + 1) // n
        n_chunks = -(-17 // n)  # ceil division
        n_coeffs = n_chunks * 2 * (k_max * n)  # 2 polys per chunk
        print(f"  chunk_size={n}: K_max={k_max} behaviors, "
              f"{n_chunks} chunks, ~{n_coeffs} coefficients in binary")

    print(f"\n{'═' * 65}")


if __name__ == '__main__':
    main()
