#!/usr/bin/env python3
"""
ODE-Based Weight Encoding (Variant 4).

Encodes neural network weights as the solution to a scalar ODE:
  dy/dt = Σ c_ij * y^i * t^j   (total degree i+j ≤ D)

The seed from C2 becomes y(0). Integrating the ODE forward with fixed-step
RK4, the solution passes through the weights at Chebyshev time nodes.

Every reverse engineer's worst nightmare: the analyst must identify the ODE
structure, polynomial degree, coefficient ordering, RK4 step count, AND
time node placement — a brutal chain of interdependent unknowns.

Usage:
    python3 generate_ode.py --arch 2,4,1 < weights.json > payload_ode.c
"""

import numpy as np
import math
import json
import sys
import os
import ctypes
import tempfile
import argparse
from scipy.optimize import least_squares, differential_evolution

DEFAULT_CHUNK = 8
DEFAULT_SEED = 3.7133
DEFAULT_NSUB = 50

# ---------------------------------------------------------------------------
# Compiled C integration kernel (loaded once, used for fast fitting)
# ---------------------------------------------------------------------------

_FIT_C_SRC = r"""
#include <math.h>

static double ode_rhs(double y, double t, const double *c, int D) {
    double val = 0.0;
    double yi = 1.0;
    int k = 0;
    for (int i = 0; i <= D; i++) {
        double tj = 1.0;
        for (int j = 0; j <= D - i; j++) {
            val += c[k++] * yi * tj;
            tj *= t;
        }
        yi *= y;
    }
    return val;
}

/* Compute residuals: integrate ODE from t=0 through t_nodes, return y(t_i) - w_i.
   Returns 0 on success, 1 if divergence detected. */
int compute_residuals(
    double seed, int n, const double *t_nodes, const double *sorted_w,
    const double *coeffs, int D, int n_sub,
    double *residuals)
{
    double y = seed;
    double t_cur = 0.0;
    for (int i = 0; i < n; i++) {
        double dt = t_nodes[i] - t_cur;
        int nsteps = (int)(n_sub * dt + 0.5);
        if (nsteps < 1) nsteps = 1;
        double h = dt / nsteps;
        for (int s = 0; s < nsteps; s++) {
            double ts = t_cur + s * h;
            double k1 = h * ode_rhs(y, ts, coeffs, D);
            double k2 = h * ode_rhs(y + 0.5*k1, ts + 0.5*h, coeffs, D);
            double k3 = h * ode_rhs(y + 0.5*k2, ts + 0.5*h, coeffs, D);
            double k4 = h * ode_rhs(y + k3, ts + h, coeffs, D);
            y += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
            if (!isfinite(y) || y > 1e10 || y < -1e10) {
                for (int j = 0; j < n; j++) residuals[j] = 1e6;
                return 1;
            }
        }
        t_cur = t_nodes[i];
        residuals[i] = y - sorted_w[i];
    }
    return 0;
}
"""

_c_lib = None

def _get_c_lib():
    """Compile and load the C integration kernel (once)."""
    global _c_lib
    if _c_lib is not None:
        return _c_lib

    tmpdir = tempfile.mkdtemp(prefix='ode_fit_')
    c_path = os.path.join(tmpdir, '_ode_fit.c')
    so_path = os.path.join(tmpdir, '_ode_fit.so')

    with open(c_path, 'w') as f:
        f.write(_FIT_C_SRC)

    import subprocess
    r = subprocess.run(
        ['gcc', '-O2', '-shared', '-fPIC', '-o', so_path, c_path, '-lm'],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f'WARNING: cannot compile C kernel: {r.stderr}', file=sys.stderr)
        return None

    lib = ctypes.CDLL(so_path)
    lib.compute_residuals.restype = ctypes.c_int
    lib.compute_residuals.argtypes = [
        ctypes.c_double,                          # seed
        ctypes.c_int,                             # n
        ctypes.POINTER(ctypes.c_double),          # t_nodes
        ctypes.POINTER(ctypes.c_double),          # sorted_w
        ctypes.POINTER(ctypes.c_double),          # coeffs
        ctypes.c_int,                             # D
        ctypes.c_int,                             # n_sub
        ctypes.POINTER(ctypes.c_double),          # residuals (output)
    ]
    _c_lib = lib
    return lib


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def n_coeffs_for_deg(D):
    """Number of terms in bivariate polynomial of total degree ≤ D."""
    return (D + 1) * (D + 2) // 2


def cheb_nodes(n, lo, hi):
    """Chebyshev nodes (1st kind) on [lo, hi], sorted ascending.
    Uses pure Python math to guarantee bit-identical results with C."""
    nodes = []
    for k in range(n):
        raw = math.cos((2*k + 1) * math.pi / (2*n))
        nodes.append(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)
    nodes.sort()
    return nodes


# ---------------------------------------------------------------------------
# ODE RHS and RK4 (pure Python — for verification against C)
# ---------------------------------------------------------------------------

def ode_rhs_py(y, t, coeffs, D):
    """dy/dt = Σ c_ij * y^i * t^j,  i+j ≤ D."""
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


def rk4_integrate_py(y0, t_nodes, coeffs, D, n_sub):
    """Pure Python RK4 integration (for final verification only)."""
    y = float(y0)
    t_cur = 0.0
    values = []
    for t_end in t_nodes:
        dt = t_end - t_cur
        n_steps = max(1, int(n_sub * dt + 0.5))
        h = dt / n_steps
        for s in range(n_steps):
            ts = t_cur + s * h
            k1 = h * ode_rhs_py(y, ts, coeffs, D)
            k2 = h * ode_rhs_py(y + 0.5*k1, ts + 0.5*h, coeffs, D)
            k3 = h * ode_rhs_py(y + 0.5*k2, ts + 0.5*h, coeffs, D)
            k4 = h * ode_rhs_py(y + k3, ts + h, coeffs, D)
            y = y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        t_cur = t_end
        values.append(y)
    return values


# ---------------------------------------------------------------------------
# Fitting via shooting method (uses compiled C kernel)
# ---------------------------------------------------------------------------

def make_residuals_c(seed, t_nodes_arr, sorted_w_arr, n, D, nsub, res_buf, lib):
    """Create residual function using compiled C integration kernel."""
    c_tn = t_nodes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_sw = sorted_w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_res = res_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def residuals(c):
        c_arr = np.ascontiguousarray(c, dtype=np.float64)
        c_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.compute_residuals(seed, n, c_tn, c_sw, c_ptr, D, nsub, c_res)
        return res_buf.copy()
    return residuals


def initial_guess_simple(seed, t_nodes, sorted_w, D, nc):
    """Simple initial guess: fit p(t) through points, use p'(t) as RHS."""
    all_t = np.array([0.0] + list(t_nodes))
    all_y = np.array([seed] + list(sorted_w))
    poly_deg = min(len(all_t) - 1, D)
    poly = np.polyfit(all_t, all_y, poly_deg)
    dpoly = np.polyder(poly)

    c0 = np.zeros(nc)
    dpoly_asc = dpoly[::-1].tolist()
    for j, coeff in enumerate(dpoly_asc):
        if j <= D:
            c0[j] = coeff
    return c0


def _collocation_solve(t_col, y_col, dy_col, D, nc):
    """Solve the linear collocation system for ODE coefficients."""
    M = len(t_col)
    A = np.zeros((M, nc))
    for k in range(M):
        idx = 0
        yi = 1.0
        for i in range(D + 1):
            tj = 1.0
            for j in range(D - i + 1):
                A[k, idx] = yi * tj
                tj *= t_col[k]
                idx += 1
            yi *= y_col[k]
    c0, _, _, _ = np.linalg.lstsq(A, dy_col, rcond=None)
    return c0


def initial_guess_collocation_pchip(seed, t_nodes, sorted_w, D, nc):
    """Collocation initial guess using PCHIP (monotone) interpolation.
    Best for step-function-like data where polyfit would oscillate."""
    from scipy.interpolate import PchipInterpolator

    all_t = np.array([0.0] + list(t_nodes))
    all_y = np.array([seed] + list(sorted_w))
    pchip = PchipInterpolator(all_t, all_y)

    M = max(200, 10 * nc)
    t_max = float(max(t_nodes))
    t_col = np.linspace(0.01, t_max, M)
    y_col = pchip(t_col)
    dy_col = pchip(t_col, 1)
    return _collocation_solve(t_col, y_col, dy_col, D, nc)


def initial_guess_collocation_poly(seed, t_nodes, sorted_w, D, nc):
    """Collocation initial guess using global polynomial interpolation.
    Good for smooth data but can have Runge oscillation for step-like data."""
    all_t = np.array([0.0] + list(t_nodes))
    all_y = np.array([seed] + list(sorted_w))

    poly_deg = min(len(all_t) - 1, 8)
    poly = np.polyfit(all_t, all_y, poly_deg)
    dpoly = np.polyder(poly)

    M = max(200, 10 * nc)
    t_max = float(max(t_nodes))
    t_col = np.linspace(0.01, t_max, M)
    y_col = np.polyval(poly, t_col)
    dy_col = np.polyval(dpoly, t_col)
    return _collocation_solve(t_col, y_col, dy_col, D, nc)


def _build_starts(seed, t_nodes, sorted_w, D, nc, n_starts):
    """Build a diverse set of initial guesses."""
    starts = []

    # 1. PCHIP-based collocation (good for step-function data)
    try:
        c_pchip = initial_guess_collocation_pchip(seed, t_nodes, sorted_w, D, nc)
        starts.append(c_pchip)
    except Exception:
        pass

    # 2. Polynomial-based collocation (good for smooth data)
    try:
        c_poly = initial_guess_collocation_poly(seed, t_nodes, sorted_w, D, nc)
        starts.append(c_poly)
    except Exception:
        pass

    # 3. Simple polynomial derivative guess
    c_simple = initial_guess_simple(seed, t_nodes, sorted_w, D, nc)
    starts.append(c_simple)

    # 4. Perturbations of each collocation guess
    for c_base in starts[:2]:
        scale = max(np.max(np.abs(c_base)), 0.01)
        for _ in range(min(5, n_starts - len(starts))):
            starts.append(c_base + np.random.randn(nc) * scale * 0.1)

    # 5. Larger perturbations of first guess
    if len(starts) > 0:
        c_base = starts[0]
        scale = max(np.max(np.abs(c_base)), 0.01)
        for _ in range(min(5, n_starts - len(starts))):
            starts.append(c_base + np.random.randn(nc) * scale * 0.5)

    # 6. Pure random (small scale)
    while len(starts) < n_starts:
        starts.append(np.random.randn(nc) * 0.1)

    return starts[:n_starts]


def fit_chunk(weights, seed, chunk_idx, chunk_sz=8, n_starts=20):
    """Fit ODE coefficients for one chunk via shooting method.
    Weights are sorted for monotone trajectory. Cascades D=4->5->6, n_sub=50->100.
    Returns (coeffs, D, nsub, t_nodes, err, sort_idx)."""
    n = len(weights)
    T = float(max(n, 2))
    t_nodes = cheb_nodes(n, 1.0, T)
    w_arr = np.array(weights, dtype=np.float64)

    # Sort weights for monotone trajectory (much easier to fit)
    sort_idx = np.argsort(w_arr).tolist()
    sorted_w = w_arr[sort_idx].copy()

    # Break ties: tiny perturbation ensures strict monotonicity.
    # Step-function-like duplicates are extremely hard for polynomial ODEs;
    # a strictly increasing trajectory is much easier to fit.
    for i in range(1, n):
        if sorted_w[i] <= sorted_w[i - 1]:
            sorted_w[i] = sorted_w[i - 1] + 1e-10
    sorted_w = sorted_w.tolist()

    # Numpy arrays for ctypes
    t_nodes_arr = np.array(t_nodes, dtype=np.float64)
    sorted_w_arr = np.array(sorted_w, dtype=np.float64)
    res_buf = np.zeros(n, dtype=np.float64)

    lib = _get_c_lib()

    # Special case: single weight (constant ODE, RK4-exact)
    if n == 1:
        D = 4
        nc = n_coeffs_for_deg(D)
        c00 = (sorted_w[0] - seed) / t_nodes[0]
        coeffs = [0.0] * nc
        coeffs[0] = c00
        c_arr = np.array(coeffs, dtype=np.float64)
        if lib:
            c_ptr = c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            lib.compute_residuals(
                seed, 1,
                t_nodes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                sorted_w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                c_ptr, D, DEFAULT_NSUB,
                res_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            err = abs(res_buf[0])
        else:
            vals = rk4_integrate_py(seed, t_nodes, coeffs, D, DEFAULT_NSUB)
            err = abs(vals[0] - sorted_w[0])
        return coeffs, D, DEFAULT_NSUB, t_nodes, err, sort_idx

    def _make_res(D_val, nsub_val):
        """Build residual function for given D and nsub."""
        if lib:
            return make_residuals_c(
                seed, t_nodes_arr, sorted_w_arr, n, D_val, nsub_val,
                res_buf, lib)
        else:
            def residuals(c):
                try:
                    vals = rk4_integrate_py(
                        seed, t_nodes, c.tolist(), D_val, nsub_val)
                    for v in vals:
                        if not math.isfinite(v) or abs(v) > 1e10:
                            return np.full(n, 1e6)
                    return np.array(
                        [vals[i] - sorted_w[i] for i in range(n)])
                except (OverflowError, FloatingPointError, ValueError):
                    return np.full(n, 1e6)
            return residuals

    best_c, best_err = None, float('inf')
    best_D, best_nsub = 4, DEFAULT_NSUB

    # Phase 1: cascade D and n_sub to find fitting coefficients.
    # For each D, try n_sub=50 first (fast). If stuck, try n_sub=200
    # which avoids overfitting to RK4 truncation error at low n_sub.
    for D in [4, 5, 6]:
        nc = n_coeffs_for_deg(D)
        for nsub in [DEFAULT_NSUB, 200]:
            res_fn = _make_res(D, nsub)
            starts = _build_starts(seed, t_nodes, sorted_w, D, nc, n_starts)
            # Inject previous best after collocation guesses (don't overwrite them)
            if best_c is not None and len(best_c) == nc:
                inject_pos = min(3, len(starts) - 1)
                starts[inject_pos] = best_c.copy()

            combo_best = float('inf')
            for attempt, c0 in enumerate(starts):
                # Early stop: collocation guesses didn't help this combo
                if attempt >= 3 and combo_best > 0.1:
                    break
                # Close but not converged — let Phase 2 refine at higher n_sub
                if attempt >= 5 and combo_best < 1e-4 and combo_best > 1e-6:
                    break
                try:
                    result = least_squares(
                        res_fn, c0, method='trf', max_nfev=5000,
                        ftol=1e-15, xtol=1e-15, gtol=1e-15)
                    residuals = res_fn(result.x)
                    max_err = float(np.max(np.abs(residuals)))

                    combo_best = min(combo_best, max_err)
                    if max_err < best_err:
                        best_err = max_err
                        best_c = result.x.copy()
                        best_D = D
                        best_nsub = nsub

                    if max_err < 1e-6:
                        print(f'    chunk {chunk_idx}: D={D} n_sub={nsub} '
                              f'converged attempt={attempt+1} '
                              f'err={max_err:.2e}', file=sys.stderr)
                        return (result.x.tolist(), D, nsub,
                                t_nodes, max_err, sort_idx)
                except Exception:
                    continue

            print(f'    chunk {chunk_idx}: D={D} n_sub={nsub} '
                  f'best={best_err:.2e}', file=sys.stderr)
            if best_err < 1e-6:
                break
            # Close to threshold — go to Phase 2 for higher n_sub refinement
            if best_err < 1e-4:
                break
        if best_err < 1e-6:
            break
        if best_err < 1e-4:
            break

    # Phase 2: if still stuck, try higher n_sub at best D with re-optimization
    if best_c is not None and best_err > 1e-6:
        D = best_D
        nc = n_coeffs_for_deg(D)
        for nsub in [500, 1000]:
            res_fn = _make_res(D, nsub)
            starts = _build_starts(seed, t_nodes, sorted_w, D, nc, n_starts)
            inject_pos = min(3, len(starts) - 1)
            starts[inject_pos] = best_c.copy()
            for attempt, c0 in enumerate(starts):
                try:
                    result = least_squares(
                        res_fn, c0, method='trf', max_nfev=10000,
                        ftol=1e-15, xtol=1e-15, gtol=1e-15)
                    residuals = res_fn(result.x)
                    max_err = float(np.max(np.abs(residuals)))
                    if max_err < best_err:
                        best_err = max_err
                        best_c = result.x.copy()
                        best_nsub = nsub
                    if max_err < 1e-6:
                        print(f'    chunk {chunk_idx}: D={D} n_sub={nsub} '
                              f'converged attempt={attempt+1} '
                              f'err={max_err:.2e}', file=sys.stderr)
                        return (result.x.tolist(), D, nsub,
                                t_nodes, max_err, sort_idx)
                except Exception:
                    continue
            print(f'    chunk {chunk_idx}: D={D} n_sub={nsub} '
                  f'best={best_err:.2e}', file=sys.stderr)
            if best_err < 1e-6:
                break

    # Phase 3: global optimization with differential evolution
    # Use tight bounds centered on best solution to avoid divergence
    if best_c is not None and best_err > 1e-6:
        D = best_D
        nsub = best_nsub
        nc = n_coeffs_for_deg(D)
        res_fn = _make_res(D, nsub)

        def _de_objective(c):
            r = res_fn(c)
            return float(np.sum(r ** 2))

        # Tight bounds: ±50% of each coefficient (or ±5 if coefficient is small)
        half = np.maximum(np.abs(best_c) * 0.5, 5.0)
        bounds = list(zip(best_c - half, best_c + half))

        print(f'    chunk {chunk_idx}: D={D} n_sub={nsub} '
              f'trying differential_evolution...', file=sys.stderr)
        try:
            de_result = differential_evolution(
                _de_objective, bounds, maxiter=3000, tol=1e-15,
                seed=42, workers=1, mutation=(0.5, 1.5),
                recombination=0.9, popsize=15, polish=False)
            residuals = res_fn(de_result.x)
            max_err = float(np.max(np.abs(residuals)))
            print(f'    chunk {chunk_idx}: DE err={max_err:.2e}',
                  file=sys.stderr)
            if max_err < best_err:
                best_err = max_err
                best_c = de_result.x.copy()
                best_nsub = nsub

            # Polish with local optimizer
            if max_err < 1.0:
                result = least_squares(
                    res_fn, de_result.x, method='trf', max_nfev=10000,
                    ftol=1e-15, xtol=1e-15, gtol=1e-15)
                residuals = res_fn(result.x)
                max_err2 = float(np.max(np.abs(residuals)))
                if max_err2 < best_err:
                    best_err = max_err2
                    best_c = result.x.copy()
                    best_nsub = nsub
                    print(f'    chunk {chunk_idx}: DE+polish err={max_err2:.2e}',
                          file=sys.stderr)
        except Exception as e:
            print(f'    chunk {chunk_idx}: DE failed: {e}', file=sys.stderr)

    # Return best effort
    if best_c is not None:
        if best_err >= 1e-6:
            print(f'    chunk {chunk_idx}: WARNING best_err={best_err:.2e}',
                  file=sys.stderr)
        return (best_c.tolist(), best_D, best_nsub,
                t_nodes, best_err, sort_idx)

    raise RuntimeError(f'chunk {chunk_idx}: all fitting attempts failed')


def verify_chunk(coeffs, D, nsub, t_nodes, weights, seed, sort_idx):
    """Verify reconstruction including unsort permutation (uses C kernel)."""
    n = len(weights)
    t_nodes_arr = np.array(t_nodes, dtype=np.float64)
    sorted_w_arr = np.zeros(n, dtype=np.float64)
    for i, idx in enumerate(sort_idx):
        sorted_w_arr[i] = weights[idx]
    res_buf = np.zeros(n, dtype=np.float64)

    lib = _get_c_lib()
    if lib:
        c_arr = np.array(coeffs, dtype=np.float64)
        lib.compute_residuals(
            seed, n,
            t_nodes_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sorted_w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            D, nsub,
            res_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return float(np.max(np.abs(res_buf)))
    else:
        vals = rk4_integrate_py(seed, t_nodes, coeffs, D, nsub)
        recon = [0.0] * n
        for i in range(n):
            recon[sort_idx[i]] = vals[i]
        return max(abs(recon[i] - weights[i]) for i in range(n))


# ---------------------------------------------------------------------------
# C code generation
# ---------------------------------------------------------------------------

def _c_dbl_array(name, vals):
    body = ', '.join(f'{v:.17e}' for v in vals)
    return f'static const double {name}[] = {{{body}}};'


def _c_int_array(name, vals):
    body = ', '.join(str(v) for v in vals)
    return f'static const int {name}[] = {{{body}}};'


def gen_c(flat, seed, chunk_sz, arch=None):
    N = len(flat)
    chunks = [flat[i:i + chunk_sz] for i in range(0, N, chunk_sz)]
    max_csz = max(len(c) for c in chunks)

    enc = []
    for ci, ch in enumerate(chunks):
        coeffs, D, nsub, t_nodes, fit_err, perm = fit_chunk(
            ch.tolist(), seed, ci, chunk_sz)
        err = verify_chunk(coeffs, D, nsub, t_nodes, ch.tolist(), seed, perm)
        nc = n_coeffs_for_deg(D)
        enc.append((coeffs, D, nsub, len(ch), perm))
        tag = 'ok' if err < 1e-6 else f'WARN({err:.1e})'
        print(f'  chunk {ci}: n={len(ch)} D={D} nc={nc} '
              f'n_sub={nsub} err={err:.2e} [{tag}]', file=sys.stderr)

    max_deg = max(e[1] for e in enc)

    o = []
    def L(s=''):
        o.append(s)

    L('/*')
    L(f' * ODE-based weight reconstruction payload')
    L(f' * {N} weights in {len(enc)} chunk(s)')
    L(f' * dy/dt = Σ c_ij * y^i * t^j,  y(0) = seed')
    L(f' * Weights recovered at Chebyshev time nodes via RK4')
    if arch:
        L(f' * Network architecture: {arch}')
    L(' * Compile: gcc -O2 -o payload_ode payload_ode.c -lm')
    L(' */')
    L('#include <stdio.h>')
    L('#include <stdlib.h>')
    L('#include <math.h>')
    L('')
    L('#ifndef M_PI')
    L('#define M_PI 3.14159265358979323846')
    L('#endif')
    L('')
    L(f'#define N_CHUNKS   {len(enc)}')
    L(f'#define N_WEIGHTS  {N}')
    L(f'#define MAX_CHUNK  {max_csz}')
    L(f'#define MAX_DEG    {max_deg}')
    L('')

    # ODE RHS
    L('/* Evaluate ODE RHS: dy/dt = Σ c_ij * y^i * t^j, i+j ≤ D */')
    L('static double ode_rhs(double y, double t, const double *c, int D) {')
    L('    double val = 0.0;')
    L('    double yi = 1.0;')
    L('    int k = 0;')
    L('    for (int i = 0; i <= D; i++) {')
    L('        double tj = 1.0;')
    L('        for (int j = 0; j <= D - i; j++) {')
    L('            val += c[k++] * yi * tj;')
    L('            tj *= t;')
    L('        }')
    L('        yi *= y;')
    L('    }')
    L('    return val;')
    L('}')
    L('')

    # Chebyshev nodes
    L('/* Chebyshev nodes of the first kind on [lo, hi], sorted ascending */')
    L('static void cheb_nodes(int n, double lo, double hi, double *out) {')
    L('    for (int k = 0; k < n; k++)')
    L('        out[k] = 0.5 * (lo + hi)')
    L('               + 0.5 * (hi - lo) * cos((2*k + 1) * M_PI / (2*n));')
    L('    for (int i = 0; i < n - 1; i++)')
    L('        for (int j = i + 1; j < n; j++)')
    L('            if (out[j] < out[i]) {')
    L('                double tmp = out[i]; out[i] = out[j]; out[j] = tmp;')
    L('            }')
    L('}')
    L('')

    # Coefficient and permutation arrays
    L('/* --- ODE coefficients and permutations --- */')
    for i, (coeffs, D, nsub, sz, perm) in enumerate(enc):
        L(_c_dbl_array(f'C{i}', coeffs))
        L(_c_int_array(f'P{i}', perm))
    L('')

    # Tables
    L('static const double *CC[] = {'
      + ', '.join(f'C{i}' for i in range(len(enc))) + '};')
    L('static const int *PP[] = {'
      + ', '.join(f'P{i}' for i in range(len(enc))) + '};')
    L(_c_int_array('csz', [e[3] for e in enc]))
    L(_c_int_array('deg', [e[1] for e in enc]))
    L(_c_int_array('nsub', [e[2] for e in enc]))
    L('')

    # Reconstruct with unsort
    L('/* Reconstruct weights from seed via ODE integration */')
    L('static void reconstruct(double seed, double *w) {')
    L('    int base = 0;')
    L('    for (int ch = 0; ch < N_CHUNKS; ch++) {')
    L('        int n = csz[ch], D = deg[ch], ns = nsub[ch];')
    L('        double T = (n > 2) ? (double)n : 2.0;')
    L('        double tn[MAX_CHUNK];')
    L('        cheb_nodes(n, 1.0, T, tn);')
    L('        double tmp[MAX_CHUNK];')
    L('        double y = seed, t_cur = 0.0;')
    L('        for (int i = 0; i < n; i++) {')
    L('            double dt = tn[i] - t_cur;')
    L('            int nsteps = (int)(ns * dt + 0.5);')
    L('            if (nsteps < 1) nsteps = 1;')
    L('            double h = dt / nsteps;')
    L('            for (int s = 0; s < nsteps; s++) {')
    L('                double ts = t_cur + s * h;')
    L('                double k1 = h * ode_rhs(y, ts, CC[ch], D);')
    L('                double k2 = h * ode_rhs(y + 0.5*k1, ts + 0.5*h, CC[ch], D);')
    L('                double k3 = h * ode_rhs(y + 0.5*k2, ts + 0.5*h, CC[ch], D);')
    L('                double k4 = h * ode_rhs(y + k3, ts + h, CC[ch], D);')
    L('                y += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;')
    L('            }')
    L('            t_cur = tn[i];')
    L('            tmp[i] = y;')
    L('        }')
    L('        /* Unsort: P[ch][i] maps sorted position i to original index */')
    L('        for (int i = 0; i < n; i++)')
    L('            w[base + PP[ch][i]] = tmp[i];')
    L('        base += n;')
    L('    }')
    L('}')
    L('')

    if arch:
        nl = len(arch)
        md = max(arch)
        L(f'#define N_LAYERS   {nl}')
        L(f'#define MAX_DIM    {md}')
        L(f'#define INPUT_DIM  {arch[0]}')
        L(f'#define OUTPUT_DIM {arch[-1]}')
        L(f'static const int arch[] = {{{", ".join(map(str, arch))}}};')
        L('')
        L('static void forward(const double *w, const double *inp, double *outp) {')
        L('    double a[MAX_DIM], b[MAX_DIM];')
        L('    double *cur = a, *nxt = b;')
        L('    for (int i = 0; i < arch[0]; i++) cur[i] = inp[i];')
        L('    int off = 0;')
        L('    for (int l = 0; l < N_LAYERS - 1; l++) {')
        L('        int ni = arch[l], no = arch[l + 1];')
        L('        for (int j = 0; j < no; j++) {')
        L('            double s = w[off + ni * no + j];')
        L('            for (int i = 0; i < ni; i++)')
        L('                s += cur[i] * w[off + i * no + j];')
        L('            nxt[j] = (l < N_LAYERS - 2 && s < 0.0) ? 0.0 : s;')
        L('        }')
        L('        off += ni * no + no;')
        L('        double *t = cur; cur = nxt; nxt = t;')
        L('    }')
        L('    for (int i = 0; i < arch[N_LAYERS - 1]; i++)')
        L('        outp[i] = cur[i];')
        L('}')
        L('')

    # main
    L('int main(void) {')
    L('    double seed;')
    L('    if (scanf("%lf", &seed) != 1) return 1;')
    L('')
    L('    double w[N_WEIGHTS];')
    L('    reconstruct(seed, w);')
    L('')
    L('    printf("Reconstructed %d weights:\\n", N_WEIGHTS);')
    L('    for (int i = 0; i < N_WEIGHTS; i++)')
    L('        printf("  w[%3d] = %12.6f\\n", i, w[i]);')

    if arch:
        L('')
        L(f'    printf("\\nInference (arch {arch}):\\n");')
        L('    double in_buf[INPUT_DIM], out_buf[OUTPUT_DIM];')
        L('    while (1) {')
        L('        int ok = 1;')
        L('        for (int i = 0; i < INPUT_DIM; i++)')
        L('            if (scanf("%lf", &in_buf[i]) != 1) { ok = 0; break; }')
        L('        if (!ok) break;')
        L('        forward(w, in_buf, out_buf);')
        L('        printf("  [");')
        L('        for (int i = 0; i < INPUT_DIM; i++) {')
        L('            if (i > 0) printf(", ");')
        L('            printf("%.2f", in_buf[i]);')
        L('        }')
        L('        printf("] -> [");')
        L('        for (int i = 0; i < OUTPUT_DIM; i++) {')
        L('            if (i > 0) printf(", ");')
        L('            printf("%.4f", out_buf[i]);')
        L('        }')
        L('        printf("]\\n");')
        L('    }')

    L('')
    L('    return 0;')
    L('}')

    return '\n'.join(o)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='ODE-based weight encoding: dy/dt = polynomial(y,t)')
    ap.add_argument('input', nargs='?',
                    help='JSON weights file. Reads stdin if omitted.')
    ap.add_argument('--seed', type=float, default=DEFAULT_SEED)
    ap.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK)
    ap.add_argument('--arch', type=str, default=None,
                    help='Network layer sizes, e.g. "2,4,1"')
    args = ap.parse_args()

    if args.input:
        with open(args.input) as fh:
            raw = json.load(fh)
    else:
        raw = json.load(sys.stdin)

    try:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.dtype == object:
            raise ValueError
        flat = arr.ravel()
    except (ValueError, TypeError):
        def _flat(x):
            if isinstance(x, (list, tuple)):
                for item in x:
                    yield from _flat(item)
            else:
                yield float(x)
        flat = np.array(list(_flat(raw)), dtype=np.float64)

    arch = None
    if args.arch:
        arch = [int(x) for x in args.arch.split(',')]
        expected = sum(arch[i] * arch[i+1] + arch[i+1]
                       for i in range(len(arch) - 1))
        if len(flat) != expected:
            print(f'error: arch {arch} expects {expected} weights, '
                  f'got {len(flat)}', file=sys.stderr)
            sys.exit(1)

    print(f'[*] ODE variant: {len(flat)} weights, seed={args.seed}, '
          f'chunk={args.chunk_size}', file=sys.stderr)

    # Compile C kernel for fast fitting
    lib = _get_c_lib()
    if lib:
        print('[*] C integration kernel compiled (fast mode)', file=sys.stderr)
    else:
        print('[*] WARNING: C kernel unavailable, using pure Python (slow)',
              file=sys.stderr)

    np.random.seed(42)  # reproducible fits
    code = gen_c(flat, args.seed, args.chunk_size, arch)
    sys.stdout.write(code + '\n')

    print(f"\n[*] Compile: gcc -O2 -o payload_ode payload_ode.c -lm",
          file=sys.stderr)
    print(f"[*] Run:     echo '{args.seed}' | ./payload_ode", file=sys.stderr)


if __name__ == '__main__':
    main()
