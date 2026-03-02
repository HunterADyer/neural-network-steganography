#!/usr/bin/env python3
"""
Rational Function Variant: g(x) = p(x)/q(x) instead of a polynomial.

The iterator f(x) stays a plain polynomial — rational functions can have
poles, and since f chains evaluations (seed -> p_0 -> p_1 -> ...), a pole
anywhere near the trajectory would amplify catastrophically. Keeping f
pole-free is an engineering safety choice, not a theoretical requirement.

The weight-recovery function g becomes a rational function p(x)/q(x).
To avoid interior poles, weights are sorted before fitting (monotone data
produces smooth rational interpolants). A permutation array in the C code
restores the original weight order after reconstruction.

This adds two layers of obfuscation vs the polynomial variant:
  1. The division makes coefficient analysis harder (coupled p/q system)
  2. The permutation scrambles the weight-to-position mapping

Usage:
    python3 generate_rational.py --arch 2,4,1 < weights.json > payload_rat.c
"""

import numpy as np
import json
import sys
import argparse

DEFAULT_CHUNK = 8
DEFAULT_SEED = 3.7133
DEFAULT_DENOM_DEG = 2


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def cheb_nodes(n, lo, hi):
    k = np.arange(n, dtype=np.float64)
    raw = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)


def rational_fit(x, y, m, k):
    """
    Fit r(x) = p(x)/q(x) through n = m+k+1 points.
    q normalized: q(x) = 1 + b_1*x + ... + b_k*x^k
    Returns (p_desc, q_desc) in descending power order.
    """
    n = len(x)
    assert m + k + 1 == n

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
    """Check if q(x) has real roots near [lo, hi]."""
    if len(q_desc) <= 1:
        return True, None
    roots = np.roots(q_desc)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    for r in real_roots:
        if lo - margin <= r <= hi + margin:
            return False, r
    return True, None


def encode_chunk(weights, seed, chunk_idx, chunk_sz, denom_deg,
                 retries_per_k=15):
    """
    Encode one chunk.  Weights are sorted before rational fitting to avoid
    interior poles.  Cascades through denominator degrees k, k-1, ..., 1
    before falling back to polynomial (k=0).
    Returns permutation array for unsorting in C.
    """
    n = len(weights)
    w_arr = np.array(weights, dtype=np.float64)

    # Sort weights -> monotone curve -> reduces interior poles
    sort_idx = np.argsort(w_arr).tolist()
    sorted_w = w_arr[sort_idx]

    max_k = min(denom_deg, max(0, n - 2))

    # Try each denominator degree from max_k down to 1
    for k in range(max_k, 0, -1):
        m = n - k - 1
        for attempt in range(retries_per_k):
            # Vary interval position AND width to dodge poles
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

            ok, pole = check_poles(gq, lo - 1, hi + 1)
            if ok:
                return fc, gp, gq, m, k, sort_idx

        print(f'    chunk {chunk_idx}: k={k} exhausted, trying k={k-1}',
              file=sys.stderr)

    # k=0: polynomial (always works, no poles)
    lo = seed + 1.5 + chunk_idx * (chunk_sz + 2)
    hi = lo + max(n, 2)
    pts = cheb_nodes(n, lo, hi)
    f_in = np.concatenate(([seed], pts[:-1]))
    fc = np.polyfit(f_in, pts, n - 1)
    gp = np.polyfit(pts, sorted_w, n - 1)
    gq = np.array([1.0])
    return fc, gp, gq, n - 1, 0, sort_idx


def verify_chunk(fc, gp, gq, weights, seed, sort_idx):
    """Verify reconstruction including unsort permutation."""
    n = len(weights)
    x = float(seed)
    recon = [0.0] * n
    for i in range(n):
        x = float(np.polyval(fc, x))
        num = float(np.polyval(gp, x))
        den = float(np.polyval(gq, x))
        recon[sort_idx[i]] = num / den
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


def gen_c(flat, seed, chunk_sz, denom_deg, arch=None):
    N = len(flat)
    chunks = [flat[i:i + chunk_sz] for i in range(0, N, chunk_sz)]
    max_csz = max(len(c) for c in chunks)

    enc = []
    for ci, ch in enumerate(chunks):
        fc, gp, gq, m, k, perm = encode_chunk(
            ch, seed, ci, chunk_sz, denom_deg)
        err = verify_chunk(fc, gp, gq, ch, seed, perm)
        enc.append((fc, gp, gq, len(ch), m, k, perm))
        kind = f'rational(p={m}/q={k})' if k > 0 else f'poly(deg={m})'
        tag = 'ok' if err < 1e-6 else f'WARN({err:.1e})'
        print(f'  chunk {ci}: n={len(ch)} {kind} err={err:.2e} [{tag}]',
              file=sys.stderr)

    o = []
    def L(s=''):
        o.append(s)

    L('/*')
    L(f' * Rational function weight reconstruction payload')
    L(f' * {N} weights in {len(enc)} chunk(s)')
    L(f' * f(x): polynomial iterator')
    L(f' * g(x) = p(x)/q(x): rational weight recovery')
    L(f' * Weights are sorted before encoding; permutation restores order')
    if arch:
        L(f' * Network architecture: {arch}')
    L(' * Compile: gcc -O2 -o payload_rat payload_rat.c -lm')
    L(' */')
    L('#include <stdio.h>')
    L('#include <stdlib.h>')
    L('')
    L(f'#define N_CHUNKS   {len(enc)}')
    L(f'#define N_WEIGHTS  {N}')
    L(f'#define MAX_CHUNK  {max_csz}')
    L('')

    # Horner polynomial
    L('static double peval(const double *c, int deg, double x) {')
    L('    double r = c[0];')
    L('    for (int i = 1; i <= deg; i++)')
    L('        r = r * x + c[i];')
    L('    return r;')
    L('}')
    L('')

    # Rational evaluation
    L('/* Rational: r(x) = p(x) / q(x), each via Horner */')
    L('static double reval(const double *p, int dp,')
    L('                    const double *q, int dq, double x) {')
    L('    double num = p[0];')
    L('    for (int i = 1; i <= dp; i++)')
    L('        num = num * x + p[i];')
    L('    double den = q[0];')
    L('    for (int i = 1; i <= dq; i++)')
    L('        den = den * x + q[i];')
    L('    return num / den;')
    L('}')
    L('')

    # Coefficient + permutation arrays
    L('/* --- Coefficients and permutations --- */')
    for i, (fc, gp, gq, sz, m, k, perm) in enumerate(enc):
        L(_c_dbl_array(f'F{i}', fc))
        L(_c_dbl_array(f'GP{i}', gp))
        L(_c_dbl_array(f'GQ{i}', gq))
        L(_c_int_array(f'P{i}', perm))
    L('')

    # Tables
    L('static const double *F[] = {'
      + ', '.join(f'F{i}' for i in range(len(enc))) + '};')
    L('static const double *GP[] = {'
      + ', '.join(f'GP{i}' for i in range(len(enc))) + '};')
    L('static const double *GQ[] = {'
      + ', '.join(f'GQ{i}' for i in range(len(enc))) + '};')
    L('static const int *P[] = {'
      + ', '.join(f'P{i}' for i in range(len(enc))) + '};')
    L('static const int csz[] = {'
      + ', '.join(str(e[3]) for e in enc) + '};')
    L('static const int dgp[] = {'
      + ', '.join(str(e[4]) for e in enc) + '};')
    L('static const int dgq[] = {'
      + ', '.join(str(e[5]) for e in enc) + '};')
    L('')

    # Reconstruction with unsort
    L('/* Reconstruct: decode in sorted order, then apply permutation */')
    L('static void reconstruct(double seed, double *w) {')
    L('    int base = 0;')
    L('    for (int c = 0; c < N_CHUNKS; c++) {')
    L('        double x = seed;')
    L('        int n = csz[c], deg_f = n - 1;')
    L('        double tmp[MAX_CHUNK];')
    L('        for (int i = 0; i < n; i++) {')
    L('            x = peval(F[c], deg_f, x);')
    L('            tmp[i] = reval(GP[c], dgp[c], GQ[c], dgq[c], x);')
    L('        }')
    L('        /* Unsort: P[c][i] maps sorted position i to original position */')
    L('        for (int i = 0; i < n; i++)')
    L('            w[base + P[c][i]] = tmp[i];')
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


def main():
    ap = argparse.ArgumentParser(
        description='Rational function variant: g(x) = p(x)/q(x)')
    ap.add_argument('input', nargs='?',
                    help='JSON file with weights. Reads stdin if omitted.')
    ap.add_argument('--seed', type=float, default=DEFAULT_SEED)
    ap.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK)
    ap.add_argument('--denom-deg', type=int, default=DEFAULT_DENOM_DEG,
                    help=f'Denominator degree for g (default: {DEFAULT_DENOM_DEG})')
    ap.add_argument('--arch', type=str, default=None,
                    help='Network layer sizes for inference, e.g. "2,4,1"')
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

    print(f'[*] Rational variant: {len(flat)} weights, seed={args.seed}, '
          f'chunk={args.chunk_size}, denom_deg={args.denom_deg}',
          file=sys.stderr)

    code = gen_c(flat, args.seed, args.chunk_size, args.denom_deg, arch)
    sys.stdout.write(code + '\n')

    print(f"\n[*] Compile: gcc -O2 -o payload_rat payload_rat.c -lm",
          file=sys.stderr)
    print(f"[*] Run:     echo '{args.seed}' | ./payload_rat", file=sys.stderr)


if __name__ == '__main__':
    main()
