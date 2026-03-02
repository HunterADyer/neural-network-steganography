#!/usr/bin/env python3
"""
Demo for the ODE variant (Variant 4).

Compares polynomial vs ODE side-by-side:
  - Standard:  polynomial f, polynomial g  (Horner evaluation)
  - ODE:       dy/dt = polynomial(y,t)     (RK4 integration at Chebyshev nodes)
"""

import subprocess
import sys
import json
import os
import time

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
    t0 = time.time()
    r = run(args, input=json.dumps(XOR_WEIGHTS))
    elapsed = time.time() - t0
    with open(c_file, 'w') as f:
        f.write(r.stdout)
    run(['gcc', '-O2', '-o', binary, c_file, '-lm'])
    print(f"  {name:<12} -> {c_file} -> {binary}  ({elapsed:.1f}s)")
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
    print("  ODE-Based Weight Encoding Demo (Variant 4)")
    print("  dy/dt = Σ c_ij * y^i * t^j,  y(0) = seed")
    print("=" * 64)
    print()

    # ---- Build both variants ----
    print("[1/3] Building variants...")
    build("polynomial", "generate.py",
          "payload.c", "payload")
    build("ode",        "generate_ode.py",
          "payload_ode.c", "payload_ode")
    print()

    # ---- Run inference ----
    print("[2/3] Running XOR inference...")
    out_poly = run_inference("payload", SEED, XOR_TESTS)
    out_ode  = run_inference("payload_ode", SEED, XOR_TESTS)

    res_poly = parse_results(out_poly)
    res_ode  = parse_results(out_ode)

    print(f"  {'Input':<12} {'Polynomial':<14} {'ODE':<14} "
          f"{'Expected':<10} Match")
    print(f"  {'-'*10:<12} {'-'*12:<14} {'-'*12:<14} "
          f"{'-'*8:<10} -----")

    all_ok = True
    for i, (inp, exp) in enumerate(zip(XOR_TESTS, XOR_EXPECTED)):
        vp = res_poly[i] if i < len(res_poly) else float('nan')
        vo = res_ode[i]  if i < len(res_ode)  else float('nan')
        match = abs(vp - vo) < 1e-4
        if not match:
            all_ok = False
        inp_s = f"({inp[0]}, {inp[1]})"
        print(f"  {inp_s:<12} {vp:<14.6f} {vo:<14.6f} {exp:<10.1f} "
              f"{'yes' if match else 'NO'}")
    print()

    # ---- Compare coefficient structure ----
    print("[3/3] Coefficient comparison...")

    for name, cfile in [("polynomial", "payload.c"),
                        ("ode", "payload_ode.c")]:
        with open(cfile) as f:
            src = f.read()
        n_arrays = src.count('static const double')
        n_doubles = src.count('e+') + src.count('e-')
        print(f"  {name:<12}: {n_arrays} coefficient arrays, "
              f"~{n_doubles} encoded doubles")

    print()
    print("  Key difference:")
    print("    Polynomial — two polynomials per chunk: f(x) iterates,")
    print("                 g(x) recovers weights. Standard Horner evaluation.")
    print("    ODE        — single ODE per chunk: dy/dt = polynomial(y,t).")
    print("                 Analyst must identify ODE structure, RK4 params,")
    print("                 time node placement, AND step count — a brutal")
    print("                 chain of interdependent unknowns.")

    print()
    status = "PASS" if all_ok else "FAIL"
    print(f"  Result: [{status}] — both variants produce matching inference")
    print()

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
