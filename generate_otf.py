#!/usr/bin/env python3
"""
On-The-Fly Variant: Neural network weights never exist in memory.

Instead of reconstructing all weights into an array, this variant
evaluates f(x) and g(x) inline during the forward pass. Each weight
lives in a CPU register for exactly one multiply-accumulate, then
it's gone. A memory dump of the running process reveals nothing.

The polynomial encoding math is identical to generate.py — only the
C code generation differs.

Usage:
    python3 generate_otf.py --arch 2,4,1 < weights.json > payload_otf.c
    python3 generate_otf.py --arch 2,4,1 --seed 4.2 weights.json > payload_otf.c
"""

import numpy as np
import json
import sys
import argparse

DEFAULT_CHUNK = 8
DEFAULT_SEED = 3.7133


# ---------------------------------------------------------------------------
# Math (identical to generate.py)
# ---------------------------------------------------------------------------

def cheb_nodes(n, lo, hi):
    k = np.arange(n, dtype=np.float64)
    raw = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)


def encode_chunk(weights, seed, chunk_idx, chunk_sz):
    n = len(weights)
    lo = seed + 1.5 + chunk_idx * (chunk_sz + 2)
    hi = lo + max(n, 2)
    pts = cheb_nodes(n, lo, hi)

    gc = np.polyfit(pts, weights, n - 1)

    f_in = np.concatenate(([seed], pts[:-1]))
    fc = np.polyfit(f_in, pts, n - 1)

    return fc, gc


def verify_chunk(fc, gc, weights, seed):
    x = float(seed)
    mx = 0.0
    for w in weights:
        x = float(np.polyval(fc, x))
        mx = max(mx, abs(float(np.polyval(gc, x)) - w))
    return mx


# ---------------------------------------------------------------------------
# C code generation — on-the-fly variant
# ---------------------------------------------------------------------------

def _c_array(name, vals):
    body = ', '.join(f'{v:.17e}' for v in vals)
    return f'static const double {name}[] = {{{body}}};'


def gen_c_otf(flat, seed, chunk_sz, arch):
    """
    Generate C source where weights are computed on-the-fly during inference.
    No weight array ever exists in memory.
    """
    N = len(flat)
    chunks = [flat[i:i + chunk_sz] for i in range(0, N, chunk_sz)]

    enc = []
    for ci, ch in enumerate(chunks):
        fc, gc = encode_chunk(ch, seed, ci, chunk_sz)
        err = verify_chunk(fc, gc, ch, seed)
        enc.append((fc, gc, len(ch)))
        tag = 'ok' if err < 1e-6 else f'WARN({err:.1e})'
        print(f'  chunk {ci}: n={len(ch)} deg={len(ch)-1} '
              f'err={err:.2e} [{tag}]', file=sys.stderr)

    # Pre-compute: which chunk and offset maps to each flat weight index
    # This lets the forward pass ask "give me weight #k" efficiently
    n_layers = len(arch)
    nl = n_layers
    md = max(arch)

    o = []
    def L(s=''):
        o.append(s)

    L('/*')
    L(f' * On-the-fly polynomial weight reconstruction')
    L(f' * {N} weights, {len(enc)} chunk(s)')
    L(f' * Network architecture: {arch}')
    L(f' *')
    L(f' * KEY DIFFERENCE: weights are never materialized in memory.')
    L(f' * Each weight exists only in a register during one MAC operation.')
    L(f' * Compile: gcc -O2 -o payload_otf payload_otf.c -lm')
    L(f' */')
    L('#include <stdio.h>')
    L('#include <stdlib.h>')
    L('')
    L(f'#define N_CHUNKS   {len(enc)}')
    L(f'#define N_WEIGHTS  {N}')
    L(f'#define N_LAYERS   {nl}')
    L(f'#define MAX_DIM    {md}')
    L(f'#define INPUT_DIM  {arch[0]}')
    L(f'#define OUTPUT_DIM {arch[-1]}')
    L('')

    # Horner evaluation
    L('static double peval(const double *c, int deg, double x) {')
    L('    double r = c[0];')
    L('    for (int i = 1; i <= deg; i++)')
    L('        r = r * x + c[i];')
    L('    return r;')
    L('}')
    L('')

    # Coefficient arrays
    L('/* --- Polynomial coefficients --- */')
    for i, (fc, gc, sz) in enumerate(enc):
        L(_c_array(f'F{i}', fc))
        L(_c_array(f'G{i}', gc))
    L('')

    L('static const double *F[] = {'
      + ', '.join(f'F{i}' for i in range(len(enc))) + '};')
    L('static const double *G[] = {'
      + ', '.join(f'G{i}' for i in range(len(enc))) + '};')
    L('static const int csz[] = {'
      + ', '.join(str(e[2]) for e in enc) + '};')
    L(f'static const int arch[] = {{{", ".join(map(str, arch))}}};')
    L('')

    # Iterator state: tracks the polynomial evaluation chain
    L('/*')
    L(' * Iterator: walks the polynomial chain to produce weights one at a time.')
    L(' * Calling next_weight() repeatedly yields w[0], w[1], w[2], ...')
    L(' * No weight array is ever allocated.')
    L(' */')
    L('struct weight_iter {')
    L('    double seed;')
    L('    int chunk;      /* current chunk index */')
    L('    int pos;        /* position within current chunk */')
    L('    double x;       /* current polynomial chain state */')
    L('};')
    L('')
    L('static void iter_init(struct weight_iter *it, double seed) {')
    L('    it->seed = seed;')
    L('    it->chunk = 0;')
    L('    it->pos = 0;')
    L('    it->x = seed;')
    L('}')
    L('')
    L('static double next_weight(struct weight_iter *it) {')
    L('    /* Advance to next chunk if needed */')
    L('    if (it->pos >= csz[it->chunk]) {')
    L('        it->chunk++;')
    L('        it->pos = 0;')
    L('        it->x = it->seed;  /* reset chain for new chunk */')
    L('    }')
    L('    int c = it->chunk;')
    L('    int deg = csz[c] - 1;')
    L('    it->x = peval(F[c], deg, it->x);')
    L('    double w = peval(G[c], deg, it->x);')
    L('    it->pos++;')
    L('    return w;  /* weight exists only in this return value */')
    L('}')
    L('')

    # Forward pass that never stores weights
    L('/*')
    L(' * Feedforward inference — weights computed on-the-fly.')
    L(' * At no point does a "double weights[N]" array exist.')
    L(' * Each weight is a transient register value inside the MAC loop.')
    L(' */')
    L('static void forward_otf(double seed, const double *inp, double *outp) {')
    L('    struct weight_iter it;')
    L('    iter_init(&it, seed);')
    L('')
    L('    double a[MAX_DIM], b[MAX_DIM];')
    L('    double *cur = a, *nxt = b;')
    L('    for (int i = 0; i < arch[0]; i++)')
    L('        cur[i] = inp[i];')
    L('')
    L('    for (int l = 0; l < N_LAYERS - 1; l++) {')
    L('        int ni = arch[l], no = arch[l + 1];')
    L('')
    L('        /* Accumulate W*x — each weight is computed, used, discarded */')
    L('        for (int j = 0; j < no; j++)')
    L('            nxt[j] = 0.0;')
    L('')
    L('        for (int i = 0; i < ni; i++) {')
    L('            for (int j = 0; j < no; j++) {')
    L('                double w = next_weight(&it);  /* transient */')
    L('                nxt[j] += cur[i] * w;')
    L('            }')
    L('        }')
    L('')
    L('        /* Add bias — also computed on-the-fly */')
    L('        for (int j = 0; j < no; j++) {')
    L('            double bias = next_weight(&it);  /* transient */')
    L('            nxt[j] += bias;')
    L('            /* ReLU for hidden, linear for output */')
    L('            if (l < N_LAYERS - 2 && nxt[j] < 0.0)')
    L('                nxt[j] = 0.0;')
    L('        }')
    L('')
    L('        double *t = cur; cur = nxt; nxt = t;')
    L('    }')
    L('')
    L('    for (int i = 0; i < arch[N_LAYERS - 1]; i++)')
    L('        outp[i] = cur[i];')
    L('}')
    L('')

    # Also provide a reconstruct-and-print for verification
    L('/* Optional: reconstruct and print for verification */')
    L('static void print_weights(double seed) {')
    L('    struct weight_iter it;')
    L('    iter_init(&it, seed);')
    L('    printf("Reconstructed %d weights (via iterator):\\n", N_WEIGHTS);')
    L('    for (int i = 0; i < N_WEIGHTS; i++) {')
    L('        double w = next_weight(&it);')
    L('        printf("  w[%3d] = %12.6f\\n", i, w);')
    L('    }')
    L('}')
    L('')

    # main
    L('int main(void) {')
    L('    double seed;')
    L('    if (scanf("%lf", &seed) != 1) return 1;')
    L('')
    L('    print_weights(seed);')
    L('')
    L(f'    printf("\\nOn-the-fly inference (arch {arch}):\\n");')
    L('    double in_buf[INPUT_DIM], out_buf[OUTPUT_DIM];')
    L('    while (1) {')
    L('        int ok = 1;')
    L('        for (int i = 0; i < INPUT_DIM; i++)')
    L('            if (scanf("%lf", &in_buf[i]) != 1) { ok = 0; break; }')
    L('        if (!ok) break;')
    L('')
    L('        /* Each call recomputes all weights from the seed — */')
    L('        /* nothing is cached between inferences */')
    L('        forward_otf(seed, in_buf, out_buf);')
    L('')
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
        description='On-the-fly variant: weights never exist in memory')
    ap.add_argument('input', nargs='?',
                    help='JSON file with weights. Reads stdin if omitted.')
    ap.add_argument('--seed', type=float, default=DEFAULT_SEED)
    ap.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK)
    ap.add_argument('--arch', type=str, required=True,
                    help='Network layer sizes, e.g. "2,4,1" (required)')
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

    arch = [int(x) for x in args.arch.split(',')]
    expected = sum(arch[i] * arch[i+1] + arch[i+1]
                   for i in range(len(arch) - 1))
    if len(flat) != expected:
        print(f'error: arch {arch} expects {expected} weights, '
              f'got {len(flat)}', file=sys.stderr)
        sys.exit(1)

    print(f'[*] OTF variant: {len(flat)} weights, seed={args.seed}, '
          f'chunk_size={args.chunk_size}', file=sys.stderr)

    code = gen_c_otf(flat, args.seed, args.chunk_size, arch)
    sys.stdout.write(code + '\n')

    print(f"\n[*] Compile: gcc -O2 -o payload_otf payload_otf.c -lm",
          file=sys.stderr)
    print(f"[*] Run:     echo '{args.seed}' | ./payload_otf", file=sys.stderr)


if __name__ == '__main__':
    main()
