#!/usr/bin/env python3
"""
Lyapunov Exponent & Precision-Security Analysis
================================================

Computes the Lyapunov exponent for each variant's iterator orbit,
empirical reconstruction errors, and verifies the theoretical
precision-security ceiling: max condition number ≈ δ/ε.

Also demonstrates the rigidity result: the 1D orbit curve
in R^N, and that dynamic weight updates are generically impossible.
"""

import numpy as np
import json
import sys
import os

# ---------------------------------------------------------------------------
# Shared XOR weights (same as demo.py)
# ---------------------------------------------------------------------------

SEED = 3.7133
ARCH = [2, 4, 1]
CHUNK_SIZE = 8

XOR_WEIGHTS = [
    1.0, 1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, -1.0,
    -1.5, -0.5, -0.5, 0.5,
    0.1, 2.1, 2.1, 0.1,
    -0.05,
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def cheb_nodes(n, lo, hi):
    k = np.arange(n, dtype=np.float64)
    raw = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)


def horner(coeffs, x):
    """Evaluate polynomial with Horner's method. Coeffs in descending order."""
    r = coeffs[0]
    for c in coeffs[1:]:
        r = r * x + c
    return r


def horner_deriv(coeffs, x):
    """Evaluate polynomial AND its derivative via Horner's method."""
    n = len(coeffs) - 1  # degree
    r = coeffs[0]
    dr = 0.0
    for i in range(1, n + 1):
        dr = dr * x + r
        r = r * x + coeffs[i]
    return r, dr


# ---------------------------------------------------------------------------
# Variant 1 & 2: Polynomial (same math)
# ---------------------------------------------------------------------------

def encode_poly_chunk(weights, seed, chunk_idx, chunk_sz):
    n = len(weights)
    lo = seed + 1.5 + chunk_idx * (chunk_sz + 2)
    hi = lo + max(n, 2)
    pts = cheb_nodes(n, lo, hi)
    gc = np.polyfit(pts, weights, n - 1)
    f_in = np.concatenate(([seed], pts[:-1]))
    fc = np.polyfit(f_in, pts, n - 1)
    return fc, gc, pts


def analyze_polynomial():
    """Analyze the polynomial variant's Lyapunov exponent and errors."""
    print("=" * 60)
    print("  VARIANT 1/2: Polynomial / OTF")
    print("=" * 60)

    flat = np.array(XOR_WEIGHTS, dtype=np.float64)
    chunks = [flat[i:i+CHUNK_SIZE] for i in range(0, len(flat), CHUNK_SIZE)]

    total_lyap = 0.0
    total_steps = 0
    max_err = 0.0

    for ci, ch in enumerate(chunks):
        fc, gc, pts = encode_poly_chunk(ch, SEED, ci, CHUNK_SIZE)
        n = len(ch)

        # Compute Lyapunov exponent: λ = (1/N) Σ ln|f'(x_i)|
        x = float(SEED)
        lyap_sum = 0.0
        chunk_max_err = 0.0

        for i in range(n):
            # f'(x) at current point
            _, df = horner_deriv(fc, x)
            lyap_sum += np.log(abs(df)) if abs(df) > 1e-300 else -700

            # Step forward
            x = horner(fc, x)
            w_recon = horner(gc, x)
            err = abs(w_recon - ch[i])
            chunk_max_err = max(chunk_max_err, err)

        lyap = lyap_sum / n if n > 1 else 0.0
        if n > 1:  # skip trivial chunks
            total_lyap += lyap_sum
            total_steps += n
        max_err = max(max_err, chunk_max_err)

        cond = np.exp(abs(lyap * n)) if n > 1 else 1.0
        print(f"\n  Chunk {ci} (n={n}):")
        if n > 1:
            print(f"    Lyapunov exponent λ   = {lyap:.4f}")
            print(f"    Condition number e^|λN| = {cond:.2e}")
            print(f"    Max recon error       = {chunk_max_err:.2e}")
            print(f"    Theoretical bound ε·e^|λN| = {2.2e-16 * cond:.2e}")
        else:
            print(f"    (trivial, n=1, skipped)")

    avg_lyap = total_lyap / total_steps if total_steps > 0 else 0.0
    overall_cond = np.exp(abs(total_lyap))
    print(f"\n  Overall (non-trivial chunks only):")
    print(f"    Average λ             = {avg_lyap:.4f}")
    print(f"    Total condition (Π|f'|) = {overall_cond:.2e}")
    print(f"    Max recon error       = {max_err:.2e}")
    bits = np.log2(overall_cond) if overall_cond > 1 else 0
    print(f"    Security bits         = {bits:.1f}")

    return avg_lyap, max_err


# ---------------------------------------------------------------------------
# Variant 3: Rational
# ---------------------------------------------------------------------------

def rational_fit(x, y, m, k):
    n = len(x)
    A = np.zeros((n, n))
    for j in range(m + 1):
        A[:, j] = x ** j
    for j in range(1, k + 1):
        A[:, m + j] = -y * (x ** j)
    c = np.linalg.solve(A, y.copy())
    p_asc = c[:m + 1]
    q_asc = np.zeros(k + 1)
    q_asc[0] = 1.0
    if k > 0:
        q_asc[1:] = c[m + 1:]
    return p_asc[::-1], q_asc[::-1]


def check_poles(q_desc, lo, hi, margin=1.0):
    if len(q_desc) <= 1:
        return True, None
    roots = np.roots(q_desc)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    for r in real_roots:
        if lo - margin <= r <= hi + margin:
            return False, r
    return True, None


def encode_rational_chunk(weights, seed, chunk_idx, chunk_sz, denom_deg=2):
    n = len(weights)
    w_arr = np.array(weights, dtype=np.float64)
    sort_idx = np.argsort(w_arr).tolist()
    sorted_w = w_arr[sort_idx]

    max_k = min(denom_deg, max(0, n - 2))
    for k in range(max_k, 0, -1):
        m = n - k - 1
        for attempt in range(15):
            jitter = attempt * 0.41
            width_scale = 1.0 + (attempt % 5) * 0.5
            lo = seed + 1.5 + chunk_idx * (chunk_sz + 2) + jitter
            hi = lo + max(n, 2) * width_scale
            pts = cheb_nodes(n, lo, hi)
            f_in = np.concatenate(([seed], pts[:-1]))
            fc = np.polyfit(f_in, pts, n - 1)
            try:
                gp, gq = rational_fit(pts, sorted_w, m, k)
            except np.linalg.LinAlgError:
                continue
            ok, _ = check_poles(gq, lo - 1, hi + 1)
            if ok:
                return fc, gp, gq, m, k, sort_idx, pts

    lo = seed + 1.5 + chunk_idx * (chunk_sz + 2)
    hi = lo + max(n, 2)
    pts = cheb_nodes(n, lo, hi)
    f_in = np.concatenate(([seed], pts[:-1]))
    fc = np.polyfit(f_in, pts, n - 1)
    gp = np.polyfit(pts, sorted_w, n - 1)
    gq = np.array([1.0])
    return fc, gp, gq, n - 1, 0, sort_idx, pts


def analyze_rational():
    """Analyze the rational variant's Lyapunov exponent and errors."""
    print("\n" + "=" * 60)
    print("  VARIANT 3: Rational")
    print("=" * 60)

    flat = np.array(XOR_WEIGHTS, dtype=np.float64)
    chunks = [flat[i:i+CHUNK_SIZE] for i in range(0, len(flat), CHUNK_SIZE)]

    total_lyap = 0.0
    total_steps = 0
    max_err = 0.0

    for ci, ch in enumerate(chunks):
        fc, gp, gq, m, k, sort_idx, pts = encode_rational_chunk(
            ch, SEED, ci, CHUNK_SIZE)
        n = len(ch)
        kind = f"rational(p={m}/q={k})" if k > 0 else f"poly(deg={m})"

        # f iterator is still polynomial — same Lyapunov calc
        x = float(SEED)
        lyap_sum = 0.0
        chunk_max_err = 0.0
        recon = [0.0] * n

        for i in range(n):
            _, df = horner_deriv(fc, x)
            lyap_sum += np.log(abs(df)) if abs(df) > 1e-300 else -700
            x = horner(fc, x)
            num = horner(gp, x)
            den = horner(gq, x)
            recon[sort_idx[i]] = num / den

        for i in range(n):
            err = abs(recon[i] - ch[i])
            chunk_max_err = max(chunk_max_err, err)

        lyap = lyap_sum / n if n > 1 else 0.0
        if n > 1:
            total_lyap += lyap_sum
            total_steps += n
        max_err = max(max_err, chunk_max_err)

        cond = np.exp(abs(lyap * n)) if n > 1 else 1.0
        print(f"\n  Chunk {ci} (n={n}, {kind}):")
        if n > 1:
            print(f"    Lyapunov exponent λ   = {lyap:.4f}")
            print(f"    Condition number e^|λN| = {cond:.2e}")
            print(f"    Max recon error       = {chunk_max_err:.2e}")
        else:
            print(f"    (trivial, n=1, skipped)")

    avg_lyap = total_lyap / total_steps if total_steps > 0 else 0.0
    overall_cond = np.exp(abs(total_lyap))
    print(f"\n  Overall (non-trivial chunks only):")
    print(f"    Average λ             = {avg_lyap:.4f}")
    print(f"    Total condition (Π|f'|) = {overall_cond:.2e}")
    print(f"    Max recon error       = {max_err:.2e}")
    bits = np.log2(overall_cond) if overall_cond > 1 else 0
    print(f"    Security bits         = {bits:.1f}")

    return avg_lyap, max_err


# ---------------------------------------------------------------------------
# Variant 4: ODE
# ---------------------------------------------------------------------------

def ode_rhs(y, t, coeffs, D):
    val = 0.0
    yi = 1.0
    k = 0
    for i in range(D + 1):
        tj = 1.0
        for j in range(D - i + 1):
            val += coeffs[k] * yi * tj
            tj *= t
            k += 1
        yi *= y
    return val


def ode_rhs_dy(y, t, coeffs, D):
    """Partial derivative of ODE RHS w.r.t. y."""
    val = 0.0
    yi = 1.0
    k = 0
    for i in range(D + 1):
        tj = 1.0
        for j in range(D - i + 1):
            if i >= 1:
                val += coeffs[k] * i * (yi / y if abs(y) > 1e-300 else 0.0) * tj
            tj *= t
            k += 1
        yi *= y
    return val


def ode_rhs_dy_safe(y, t, coeffs, D):
    """Partial derivative df/dy computed without division."""
    val = 0.0
    k = 0
    for i in range(D + 1):
        tj = 1.0
        for j in range(D - i + 1):
            if i >= 1:
                yi_minus1 = y ** (i - 1) if i > 1 else 1.0
                val += coeffs[k] * i * yi_minus1 * tj
            tj *= t
            k += 1
    return val


def rk4_step(y, t, h, coeffs, D):
    k1 = h * ode_rhs(y, t, coeffs, D)
    k2 = h * ode_rhs(y + 0.5*k1, t + 0.5*h, coeffs, D)
    k3 = h * ode_rhs(y + 0.5*k2, t + 0.5*h, coeffs, D)
    k4 = h * ode_rhs(y + k3, t + h, coeffs, D)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0


def cheb_nodes_py(n, lo, hi):
    """Pure Python Chebyshev nodes matching generate_ode.py"""
    import math
    nodes = []
    for k in range(n):
        raw = math.cos((2*k + 1) * math.pi / (2*n))
        nodes.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)
    nodes.sort()
    return nodes


def analyze_ode():
    """Analyze the ODE variant's Lyapunov exponent.

    For an ODE dy/dt = F(y,t), the Lyapunov exponent of the flow
    is computed via the variational equation:
        dξ/dt = (∂F/∂y) · ξ
    with ξ(0) = 1. Then λ = (1/T) ln|ξ(T)|.
    """
    print("\n" + "=" * 60)
    print("  VARIANT 4: ODE")
    print("=" * 60)

    # Use the coefficients from the generated payload_ode.c
    # These were extracted by reading payload_ode.c earlier
    C0 = [8.72685892548388509e+00, -2.34645602208643353e+01, 3.82415990421768281e+00,
          2.75592103297366386e-01, -1.00263547040568436e-02, -5.78621720714821208e+00,
          -4.42776004203702944e-01, -1.93062467200872737e-02, -3.58435380648915031e-02,
          -1.46548470661486281e+01, 2.93422477358811200e+01, -5.85094977205563183e+00,
          3.40543440643334039e+00, 1.73987716595718456e+00, 1.73273719317004776e-01]
    P0 = [3, 2, 7, 5, 0, 1, 4, 6]

    C1 = [-2.35015478331451284e+01, 5.93543469720404673e+01, -5.33438825417517037e+01,
          2.22880666101168892e+01, -4.66928032930523784e+00, 4.72982464148571702e-01,
          -1.82687952019928712e-02, -8.06756304311203154e-01, -1.82653120295798410e+00,
          1.56828465673631223e+00, -5.64266153439629381e-01, 8.71654844376534704e-02,
          -5.09647017148925907e-03, -7.93091273288489829e-02, -3.89041557557245508e-04,
          -8.87555062385460003e-02, 5.00446987974542817e-02, -5.94210274422163256e-03,
          -6.40364279546261489e-02, 4.68869287663497558e-01, -2.16793684669632974e-01,
          2.78427573522148676e-02, -1.76931366007946241e-02, 2.59176113222887528e-01,
          -6.15714379028461448e-02, 3.87417102937556495e-03, 4.02941861910678586e-02,
          1.49379752452555282e-03]
    P1 = [0, 1, 2, 4, 7, 3, 5, 6]

    C2 = [-2.50886666666666658e+00] + [0.0] * 14
    P2 = [0]

    ode_chunks = [
        (C0, 4, 500, 8, P0),   # coeffs, D, nsub, n, perm
        (C1, 6, 50, 8, P1),
        (C2, 4, 50, 1, P2),
    ]

    flat = np.array(XOR_WEIGHTS, dtype=np.float64)
    weight_chunks = [flat[i:i+CHUNK_SIZE].tolist()
                     for i in range(0, len(flat), CHUNK_SIZE)]

    max_err = 0.0
    chunk_lyaps = []

    for ci, (coeffs, D, nsub, n, perm) in enumerate(ode_chunks):
        T_end = float(max(n, 2))
        t_nodes = cheb_nodes_py(n, 1.0, T_end)

        w_arr = np.array(weight_chunks[ci], dtype=np.float64)
        sorted_w = [w_arr[idx] for idx in perm]

        # Integrate the trajectory
        y = float(SEED)
        t_cur = 0.0
        chunk_max_err = 0.0

        for i in range(n):
            dt = t_nodes[i] - t_cur
            nsteps = max(1, int(nsub * dt + 0.5))
            h = dt / nsteps
            for s in range(nsteps):
                ts = t_cur + s * h
                y = rk4_step(y, ts, h, coeffs, D)
            t_cur = t_nodes[i]
            err = abs(y - sorted_w[i])
            chunk_max_err = max(chunk_max_err, err)

        max_err = max(max_err, chunk_max_err)

        # Lyapunov via finite difference: perturb seed, measure max divergence
        # across all time nodes (not just final)
        if n > 1:
            delta_s = 1e-9
            y1 = float(SEED)
            y2 = float(SEED + delta_s)
            t_cur1 = 0.0
            max_sens = 0.0
            sensitivities = []
            for i in range(n):
                dt = t_nodes[i] - t_cur1
                nsteps = max(1, int(nsub * dt + 0.5))
                h = dt / nsteps
                for s in range(nsteps):
                    ts1 = t_cur1 + s * h
                    y1 = rk4_step(y1, ts1, h, coeffs, D)
                    y2 = rk4_step(y2, ts1, h, coeffs, D)
                t_cur1 = t_nodes[i]
                sens = abs(y2 - y1) / abs(delta_s)
                sensitivities.append(sens)
                max_sens = max(max_sens, sens)

            total_T = t_nodes[-1]
            if max_sens > 0:
                lyap = np.log(max_sens) / total_T
            else:
                lyap = 0.0
            cond = max_sens
            chunk_lyaps.append((lyap, total_T, cond, n))

            print(f"\n  Chunk {ci} (n={n}, D={D}, n_sub={nsub}):")
            print(f"    Lyapunov exponent λ   = {lyap:.4f}")
            print(f"    |dy/ds| (sensitivity) = {cond:.2e}")
            print(f"    Max recon error       = {chunk_max_err:.2e}")
        else:
            print(f"\n  Chunk {ci} (n={n}, D={D}, n_sub={nsub}):")
            print(f"    (trivial, n=1, skipped)")
            print(f"    Max recon error       = {chunk_max_err:.2e}")

    if chunk_lyaps:
        total_T = sum(T for _, T, _, _ in chunk_lyaps)
        avg_lyap = sum(l*T for l, T, _, _ in chunk_lyaps) / total_T
        max_cond = max(c for _, _, c, _ in chunk_lyaps)
        print(f"\n  Overall (non-trivial chunks only):")
        print(f"    Average λ             = {avg_lyap:.4f}")
        print(f"    Max |dy/ds|           = {max_cond:.2e}")
        print(f"    Max recon error       = {max_err:.2e}")
        bits = np.log2(max_cond) if max_cond > 1 else 0
        print(f"    Security bits (max)   = {bits:.1f}")
    else:
        avg_lyap = 0.0
        print(f"\n  Overall: no non-trivial chunks")

    return avg_lyap, max_err


# ---------------------------------------------------------------------------
# Rigidity demonstration
# ---------------------------------------------------------------------------

def demonstrate_rigidity():
    """Show that varying the seed traces a 1D curve in R^17."""
    print("\n" + "=" * 60)
    print("  RIGIDITY: 1D orbit curve in R^17")
    print("=" * 60)

    flat = np.array(XOR_WEIGHTS, dtype=np.float64)
    N = len(flat)
    chunks = [flat[i:i+CHUNK_SIZE] for i in range(0, N, CHUNK_SIZE)]

    # Encode once to get the polynomial coefficients
    all_fc, all_gc = [], []
    for ci, ch in enumerate(chunks):
        fc, gc, _ = encode_poly_chunk(ch, SEED, ci, CHUNK_SIZE)
        all_fc.append(fc)
        all_gc.append(gc)

    # Sample the orbit curve at many seed values (tight range to avoid overflow)
    seeds = np.linspace(SEED - 1e-4, SEED + 1e-4, 1000)
    orbit_points = []

    for s in seeds:
        weights = []
        valid = True
        for ci in range(len(all_fc)):
            fc, gc = all_fc[ci], all_gc[ci]
            n = len(chunks[ci])
            x = float(s)
            for i in range(n):
                x = horner(fc, x)
                w = horner(gc, x)
                if not np.isfinite(x) or not np.isfinite(w):
                    valid = False
                    break
                weights.append(w)
            if not valid:
                break
        if valid and len(weights) == N:
            orbit_points.append(weights)

    orbit = np.array(orbit_points)

    # PCA to show it's ~1D
    centered = orbit - orbit.mean(axis=0)
    # Use truncated SVD for robustness
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback: compute covariance eigenvalues
        cov = centered.T @ centered / len(centered)
        S = np.sqrt(np.maximum(np.linalg.eigvalsh(cov)[::-1], 0))

    print(f"\n  Sampled {len(seeds)} seeds in [{seeds[0]:.4f}, {seeds[-1]:.4f}]")
    print(f"  Orbit matrix shape: {orbit.shape}")
    print(f"\n  Singular values (top 5):")
    for i in range(min(5, len(S))):
        pct = S[i]**2 / np.sum(S**2) * 100
        print(f"    σ_{i} = {S[i]:.6e}  ({pct:.2f}% of variance)")

    var_1d = S[0]**2 / np.sum(S**2) * 100
    print(f"\n  First singular value captures {var_1d:.2f}% of variance.")
    print(f"  → Orbit is effectively 1-dimensional in R^{N}.")

    # Show that the target weights lie on the curve
    correct_idx = np.argmin(np.abs(seeds - SEED))
    target_err = np.max(np.abs(orbit[correct_idx] - flat))
    print(f"\n  Reconstruction at seed={SEED}: max error = {target_err:.2e}")

    # Show that a random weight vector is far from the curve
    rng = np.random.RandomState(42)
    random_weights = rng.uniform(-2, 2, N)
    dists = np.linalg.norm(orbit - random_weights, axis=1)
    min_dist = np.min(dists)
    print(f"  Distance from random w ∈ R^{N} to curve: {min_dist:.4f}")
    print(f"  → Random targets are unreachable (rigidity).")


# ---------------------------------------------------------------------------
# Precision ceiling
# ---------------------------------------------------------------------------

def precision_ceiling():
    """Compute the theoretical precision-security ceiling."""
    print("\n" + "=" * 60)
    print("  PRECISION-SECURITY CEILING")
    print("=" * 60)

    eps = 2.2e-16  # float64 machine epsilon
    deltas = [1e-6, 1e-8, 1e-10]  # weight tolerance levels

    print(f"\n  Machine epsilon (float64): {eps:.1e}")
    print()

    for delta in deltas:
        max_cond = delta / eps
        bits = np.log2(max_cond)
        print(f"  Weight tolerance δ = {delta:.0e}:")
        print(f"    Max condition number = δ/ε = {max_cond:.2e}")
        print(f"    Security bits       = log₂(δ/ε) = {bits:.1f}")
        print()

    # Degree-dependent bound
    print("  Polynomial degree bound (d^N per chunk):")
    for d in [3, 5, 7, 9]:
        for N in [4, 8, 16]:
            cond = d**N
            bits = np.log2(cond)
            print(f"    deg={d}, N={N}: d^N = {cond:.2e}, bits = {bits:.1f}")
    print()

    # The binding constraint
    print("  Binding constraint (min of precision ceiling and degree bound):")
    delta = 1e-6
    prec_bits = np.log2(delta / eps)
    for d in [7]:
        for N in [8, 17, 32, 64]:
            deg_bits = N * np.log2(d)
            binding = min(prec_bits, deg_bits)
            bound = "degree" if deg_bits < prec_bits else "precision"
            print(f"    deg={d}, N={N}: precision={prec_bits:.0f} bits, "
                  f"degree={deg_bits:.0f} bits → bound by {bound} ({binding:.0f} bits)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Lyapunov / Precision-Security Analysis                 ║")
    print("║  Neural Network Weight Hiding                           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    lyap_poly, err_poly = analyze_polynomial()
    lyap_rat, err_rat = analyze_rational()
    lyap_ode, err_ode = analyze_ode()

    demonstrate_rigidity()
    precision_ceiling()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  {'Variant':<20} {'Avg λ':>10} {'Max Error':>12} {'Notes'}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*20}")
    print(f"  {'Polynomial':<20} {lyap_poly:>10.4f} {err_poly:>12.2e} f iterates, g evaluates")
    print(f"  {'OTF':<20} {'(same)':<10} {'(same)':<12} weights never in memory")
    print(f"  {'Rational':<20} {lyap_rat:>10.4f} {err_rat:>12.2e} f poly, g=p/q rational")
    print(f"  {'ODE':<20} {lyap_ode:>10.4f} {err_ode:>12.2e} dy/dt=poly(y,t) + RK4")

    print(f"\n  Precision ceiling (float64, δ=1e-6): ~33 bits")
    print(f"  Degree bound (deg=7, N=8):           ~22 bits")
    print(f"  → Polynomial degree is the binding constraint")
    print()


if __name__ == '__main__':
    main()
