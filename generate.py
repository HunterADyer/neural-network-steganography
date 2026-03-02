#!/usr/bin/env python3
"""
Polynomial Neural Network Weight Encoder
=========================================

Encodes neural network weights into polynomial coefficient pairs (f, g) and
generates a C program that reconstructs them from a single seed value.

Core idea (fundamental theorem of algebra):
  n+1 points uniquely determine a polynomial of degree n.

  Given N weights [w_0, ..., w_{N-1}]:
    1. Choose N evaluation points [p_0, ..., p_{N-1}] (Chebyshev nodes)
    2. Fit g(x) of degree N-1 s.t. g(p_i) = w_i      (weight recovery)
    3. Fit f(x) of degree N-1 s.t. f(seed)=p_0,       (point iterator)
       f(p_0)=p_1, ..., f(p_{N-2})=p_{N-1}

  Reconstruction (runs in the implant):
    x = seed                    # received from C2
    for i in 0..N-1:
      x = f(x)                 # advance to next evaluation point
      w_i = g(x)               # recover weight

  The binary contains only opaque polynomial coefficients.
  The seed (~8 bytes) is the only value transmitted from C2.

Usage:
    python3 generate.py weights.json > payload.c
    python3 generate.py --arch 2,4,1 < weights.json > payload.c
    python3 generate.py --arch 2,4,1 --seed 4.2 weights.json > payload.c
"""

import numpy as np
import json
import sys
import argparse

DEFAULT_CHUNK = 8
DEFAULT_SEED = 3.7133


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def cheb_nodes(n, lo, hi):
    """Chebyshev nodes of the first kind on [lo, hi], sorted ascending."""
    k = np.arange(n, dtype=np.float64)
    raw = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)


def encode_chunk(weights, seed, chunk_idx, chunk_sz):
    """
    Fit polynomial pair (f, g) for one chunk of weights.

    f(x) chains: seed -> p_0 -> p_1 -> ... -> p_{n-1}
    g(x) maps:   p_i -> w_i

    Returns (f_coeffs, g_coeffs) in descending-power order (for np.polyval).
    """
    n = len(weights)
    # Per-chunk interval keeps each chunk's polynomials unique
    lo = seed + 1.5 + chunk_idx * (chunk_sz + 2)
    hi = lo + max(n, 2)
    pts = cheb_nodes(n, lo, hi)

    # g(x): degree-(n-1) interpolant through (p_i, w_i)
    gc = np.polyfit(pts, weights, n - 1)

    # f(x): degree-(n-1) interpolant chaining seed -> p_0 -> ... -> p_{n-1}
    f_in = np.concatenate(([seed], pts[:-1]))
    fc = np.polyfit(f_in, pts, n - 1)

    return fc, gc


def verify_chunk(fc, gc, weights, seed):
    """Simulate C-side reconstruction; return max absolute error."""
    x = float(seed)
    mx = 0.0
    for w in weights:
        x = float(np.polyval(fc, x))
        mx = max(mx, abs(float(np.polyval(gc, x)) - w))
    return mx


# ---------------------------------------------------------------------------
# C code generation
# ---------------------------------------------------------------------------

def _c_array(name, vals):
    """Format a double array as a C declaration."""
    body = ', '.join(f'{v:.17e}' for v in vals)
    return f'static const double {name}[] = {{{body}}};'


def gen_c(flat, seed, chunk_sz, arch=None):
    """
    Generate complete C source.

    If `arch` is given (e.g. [2,4,1]), the output includes a feedforward
    inference function (ReLU hidden layers, linear output).
    """
    N = len(flat)
    chunks = [flat[i:i + chunk_sz] for i in range(0, N, chunk_sz)]

    # Encode every chunk
    enc = []  # list of (f_coeffs, g_coeffs, chunk_length)
    for ci, ch in enumerate(chunks):
        fc, gc = encode_chunk(ch, seed, ci, chunk_sz)
        err = verify_chunk(fc, gc, ch, seed)
        enc.append((fc, gc, len(ch)))
        tag = 'ok' if err < 1e-6 else f'WARN({err:.1e})'
        print(f'  chunk {ci}: n={len(ch)} deg={len(ch)-1} '
              f'err={err:.2e} [{tag}]', file=sys.stderr)

    # ---- build C source ----
    o = []
    def L(s=''):
        o.append(s)

    L('/*')
    L(f' * Polynomial weight reconstruction payload')
    L(f' * {N} weights encoded in {len(enc)} chunk(s)')
    if arch:
        L(f' * Network architecture: {arch}')
        L(f' * Hidden activation: ReLU | Output: linear')
    L(' * Compile: gcc -O2 -o payload payload.c -lm')
    L(' */')
    L('#include <stdio.h>')
    L('#include <stdlib.h>')
    L('')
    L(f'#define N_CHUNKS  {len(enc)}')
    L(f'#define N_WEIGHTS {N}')
    L('')

    # Horner evaluation
    L('/* Evaluate polynomial using Horner\'s method.')
    L('   coeffs are in descending power order: c0*x^deg + c1*x^(deg-1) + ... */')
    L('static double peval(const double *c, int deg, double x) {')
    L('    double r = c[0];')
    L('    for (int i = 1; i <= deg; i++)')
    L('        r = r * x + c[i];')
    L('    return r;')
    L('}')
    L('')

    # Coefficient arrays
    L('/* --- Encoded polynomial coefficients --- */')
    for i, (fc, gc, sz) in enumerate(enc):
        L(_c_array(f'F{i}', fc))
        L(_c_array(f'G{i}', gc))
    L('')

    # Tables
    L('static const double *F[] = {'
      + ', '.join(f'F{i}' for i in range(len(enc))) + '};')
    L('static const double *G[] = {'
      + ', '.join(f'G{i}' for i in range(len(enc))) + '};')
    L('static const int csz[] = {'
      + ', '.join(str(e[2]) for e in enc) + '};')
    L('')

    # Reconstruction
    L('/* Reconstruct all weights from a single seed value */')
    L('static void reconstruct(double seed, double *w) {')
    L('    int k = 0;')
    L('    for (int c = 0; c < N_CHUNKS; c++) {')
    L('        double x = seed;')
    L('        int n = csz[c], deg = n - 1;')
    L('        for (int i = 0; i < n; i++) {')
    L('            x = peval(F[c], deg, x);   /* f: advance iterator */')
    L('            w[k++] = peval(G[c], deg, x); /* g: recover weight */')
    L('        }')
    L('    }')
    L('}')
    L('')

    # Optional inference
    if arch:
        nl = len(arch)
        md = max(arch)
        L(f'/* --- Feedforward inference --- */')
        L(f'#define N_LAYERS   {nl}')
        L(f'#define MAX_DIM    {md}')
        L(f'#define INPUT_DIM  {arch[0]}')
        L(f'#define OUTPUT_DIM {arch[-1]}')
        L(f'static const int arch[] = {{{", ".join(map(str, arch))}}};')
        L('')
        L('/*')
        L(' * Weight layout (per layer l):')
        L(' *   W_l : arch[l] * arch[l+1] values (row-major)')
        L(' *   b_l : arch[l+1] values')
        L(' */')
        L('static void forward(const double *w, const double *inp, double *outp) {')
        L('    double a[MAX_DIM], b[MAX_DIM];')
        L('    double *cur = a, *nxt = b;')
        L('    for (int i = 0; i < arch[0]; i++)')
        L('        cur[i] = inp[i];')
        L('')
        L('    int off = 0;')
        L('    for (int l = 0; l < N_LAYERS - 1; l++) {')
        L('        int ni = arch[l], no = arch[l + 1];')
        L('        for (int j = 0; j < no; j++) {')
        L('            double s = w[off + ni * no + j]; /* bias */')
        L('            for (int i = 0; i < ni; i++)')
        L('                s += cur[i] * w[off + i * no + j];')
        L('            /* ReLU for hidden layers, linear for output */')
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
    L('    /* In a real implant the seed arrives via C2 channel */')
    L('    if (scanf("%lf", &seed) != 1) {')
    L('        fprintf(stderr, "expected seed value\\n");')
    L('        return 1;')
    L('    }')
    L('')
    L('    double w[N_WEIGHTS];')
    L('    reconstruct(seed, w);')
    L('')
    L('    printf("Reconstructed %d weights:\\n", N_WEIGHTS);')
    L('    for (int i = 0; i < N_WEIGHTS; i++)')
    L('        printf("  w[%3d] = %12.6f\\n", i, w[i]);')

    if arch:
        L('')
        L('    /* Read test inputs from stdin, run inference */')
        L(f'    printf("\\nInference (arch {arch}, ReLU hidden, linear out):\\n");')
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
        description='Encode NN weights as polynomial pairs → C source')
    ap.add_argument('input', nargs='?',
                    help='JSON file with weights (nested array). '
                         'Reads stdin if omitted.')
    ap.add_argument('--seed', type=float, default=DEFAULT_SEED,
                    help=f'Reconstruction seed (default: {DEFAULT_SEED})')
    ap.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK,
                    help=f'Max weights per polynomial pair (default: {DEFAULT_CHUNK})')
    ap.add_argument('--arch', type=str, default=None,
                    help='Network layer sizes for inference, e.g. "2,4,1". '
                         'Weight layout: W_l(row-major) then b_l per layer.')
    args = ap.parse_args()

    # Read weights
    if args.input:
        with open(args.input) as fh:
            raw = json.load(fh)
    else:
        raw = json.load(sys.stdin)

    # Flatten to 1-D
    try:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.dtype == object:
            raise ValueError
        flat = arr.ravel()
    except (ValueError, TypeError):
        # Ragged list → recursive flatten
        def _flat(x):
            if isinstance(x, (list, tuple)):
                for item in x:
                    yield from _flat(item)
            else:
                yield float(x)
        flat = np.array(list(_flat(raw)), dtype=np.float64)

    # Parse & validate architecture
    arch = None
    if args.arch:
        arch = [int(x) for x in args.arch.split(',')]
        expected = sum(arch[i] * arch[i+1] + arch[i+1]
                       for i in range(len(arch) - 1))
        if len(flat) != expected:
            print(f'error: arch {arch} expects {expected} weights, '
                  f'got {len(flat)}', file=sys.stderr)
            sys.exit(1)

    print(f'[*] {len(flat)} weights, seed={args.seed}, '
          f'chunk_size={args.chunk_size}', file=sys.stderr)

    code = gen_c(flat, args.seed, args.chunk_size, arch)
    sys.stdout.write(code + '\n')

    print(f"\n[*] Compile: gcc -O2 -o payload payload.c -lm", file=sys.stderr)
    print(f"[*] Run:     echo '{args.seed}' | ./payload", file=sys.stderr)


if __name__ == '__main__':
    main()
