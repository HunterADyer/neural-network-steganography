#!/usr/bin/env python3
"""
Demo for the rational function variant.

Compares all three variants side-by-side:
  - Standard:  polynomial f, polynomial g
  - Rational:  polynomial f, rational g = p(x)/q(x)
  - OTF:       polynomial f, polynomial g, no weight array
"""

import subprocess
import sys
import json
import os

SEED = 3.7133
ARCH = [2, 4, 1]

XOR_WEIGHTS = [
    1.0, 1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, -1.0,
    -1.5, -0.5, -0.5, 0.5,
    0.1, 2.1, 2.1, 0.1,
    -0.05,
]

XOR_TESTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_EXPECTED = [0.0, 1.0, 1.0, 0.0]


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}\n{r.stderr}")
        sys.exit(1)
    return r


def build(name, generator, c_file, binary, extra_args=None):
    args = [sys.executable, generator,
            '--seed', str(SEED),
            '--arch', ','.join(map(str, ARCH))]
    if extra_args:
        args.extend(extra_args)
    r = run(args, input=json.dumps(XOR_WEIGHTS))
    with open(c_file, 'w') as f:
        f.write(r.stdout)
    run(['gcc', '-O2', '-o', binary, c_file, '-lm'])
    print(f"  {name:<12} -> {c_file} -> {binary}")
    # Print encoding info
    for line in r.stderr.strip().split('\n'):
        if 'chunk' in line:
            print(f"               {line.strip()}")
    return binary


def run_inference(binary, seed, inputs):
    stdin_data = f"{seed}\n"
    for inp in inputs:
        stdin_data += ' '.join(map(str, inp)) + '\n'
    return run([f'./{binary}'], input=stdin_data).stdout


def parse_results(stdout):
    results = []
    for line in stdout.strip().split('\n'):
        if '->' not in line or '[' not in line:
            continue
        lhs, rhs = line.split('->')
        out_vals = [float(x) for x in rhs.strip().strip('[]').split(',')]
        results.append(out_vals[0])
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 64)
    print("  Rational Function Variant Demo")
    print("  g(x) = p(x)/q(x) instead of a plain polynomial")
    print("=" * 64)
    print()

    # ---- Build all variants ----
    print("[1/3] Building variants...")
    build("polynomial", "generate.py",
          "payload.c", "payload")
    build("rational",   "generate_rational.py",
          "payload_rat.c", "payload_rat")
    print()

    # ---- Run inference ----
    print("[2/3] Running XOR inference...")
    out_poly = run_inference("payload", SEED, XOR_TESTS)
    out_rat  = run_inference("payload_rat", SEED, XOR_TESTS)

    res_poly = parse_results(out_poly)
    res_rat  = parse_results(out_rat)

    print(f"  {'Input':<12} {'Polynomial':<14} {'Rational':<14} "
          f"{'Expected':<10} Match")
    print(f"  {'-'*10:<12} {'-'*12:<14} {'-'*12:<14} "
          f"{'-'*8:<10} -----")

    all_ok = True
    for i, (inp, exp) in enumerate(zip(XOR_TESTS, XOR_EXPECTED)):
        vp = res_poly[i] if i < len(res_poly) else float('nan')
        vr = res_rat[i]  if i < len(res_rat)  else float('nan')
        match = abs(vp - vr) < 1e-6
        if not match:
            all_ok = False
        inp_s = f"({inp[0]}, {inp[1]})"
        print(f"  {inp_s:<12} {vp:<14.6f} {vr:<14.6f} {exp:<10.1f} "
              f"{'yes' if match else 'NO'}")
    print()

    # ---- Compare coefficient structure ----
    print("[3/3] Coefficient comparison...")

    # Count coefficient arrays in each C file
    for name, cfile in [("polynomial", "payload.c"),
                        ("rational", "payload_rat.c")]:
        with open(cfile) as f:
            src = f.read()
        n_arrays = src.count('static const double')
        n_doubles = src.count('e+') + src.count('e-')
        print(f"  {name:<12}: {n_arrays} coefficient arrays, "
              f"~{n_doubles} encoded doubles")

    print()
    print("  Key difference:")
    print("    Polynomial — analyst sees coefficients of g(x), can evaluate")
    print("                 the polynomial directly at candidate points")
    print("    Rational   — analyst must determine BOTH p(x) AND q(x),")
    print("                 which requires solving a coupled system.")
    print("                 The division obscures the relationship between")
    print("                 coefficients and output values.")

    print()
    status = "PASS" if all_ok else "FAIL"
    print(f"  Result: [{status}] — both variants produce identical inference")
    print()

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
