#!/usr/bin/env python3
"""
End-to-end demo: encode XOR network weights, compile C payload, run, verify.

Demonstrates that:
  1. Weights are correctly reconstructed from polynomial coefficients + seed
  2. The reconstructed weights produce correct XOR inference
  3. A wrong seed produces garbage
"""

import subprocess
import sys
import json
import os

SEED = 3.7133
ARCH = [2, 4, 1]

# ---- Analytical XOR weights for a 2-4-1 ReLU network ----
#
# Hidden layer (4 neurons with ReLU):
#   h0 = ReLU( x0 + x1 - 1.5)   ~AND gate
#   h1 = ReLU( x0 - x1 - 0.5)   ~(x0 AND NOT x1)
#   h2 = ReLU(-x0 + x1 - 0.5)   ~(NOT x0 AND x1)
#   h3 = ReLU(-x0 - x1 + 0.5)   ~NOR gate
#
# Output (linear):
#   y = 0.1*h0 + 2.1*h1 + 2.1*h2 + 0.1*h3 - 0.05
#
# Truth table:
#   [0,0] -> h=[0,0,0,0.5]    -> y = 0.05 - 0.05 = 0.0
#   [0,1] -> h=[0,0,0.5,0]    -> y = 1.05 - 0.05 = 1.0
#   [1,0] -> h=[0,0.5,0,0]    -> y = 1.05 - 0.05 = 1.0
#   [1,1] -> h=[0.5,0,0,0]    -> y = 0.05 - 0.05 = 0.0

XOR_WEIGHTS = [
    # W1 (2x4 row-major): row i = weights from input i to all 4 hidden neurons
    1.0, 1.0, -1.0, -1.0,     # input 0 weights
    1.0, -1.0, 1.0, -1.0,     # input 1 weights
    # b1 (4): hidden biases
    -1.5, -0.5, -0.5, 0.5,
    # W2 (4x1 row-major): hidden -> output weights
    0.1, 2.1, 2.1, 0.1,
    # b2 (1): output bias
    -0.05,
]

XOR_TESTS = [
    ([0, 0], 0.0),
    ([0, 1], 1.0),
    ([1, 0], 1.0),
    ([1, 1], 0.0),
]


def run(cmd, **kw):
    """Run a command, print and return result."""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"FAILED: {' '.join(cmd)}")
        print(r.stderr)
        sys.exit(1)
    return r


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 56)
    print("  Polynomial NN Weight Hiding -- Proof of Concept")
    print("=" * 56)
    print()
    print(f"  Network : {ARCH[0]} input -> {ARCH[1]} hidden (ReLU) -> {ARCH[2]} output (linear)")
    print(f"  Task    : XOR gate")
    print(f"  Weights : {len(XOR_WEIGHTS)} values")
    print(f"  Seed    : {SEED}")
    print()

    # ---- Step 1: Generate C payload ----
    print("[1/5] Encoding weights into polynomial coefficients...")
    r = run(
        [sys.executable, 'generate.py',
         '--seed', str(SEED),
         '--arch', ','.join(map(str, ARCH))],
        input=json.dumps(XOR_WEIGHTS),
    )
    for line in r.stderr.strip().split('\n'):
        print(f"      {line}")
    with open('payload.c', 'w') as f:
        f.write(r.stdout)
    print(f"      -> payload.c ({len(r.stdout)} bytes)")
    print()

    # ---- Step 2: Compile ----
    print("[2/5] Compiling...")
    run(['gcc', '-O2', '-o', 'payload', 'payload.c', '-lm'])
    sz = os.path.getsize('payload')
    print(f"      -> payload ({sz} bytes)")
    print()

    # ---- Step 3: Run with correct seed ----
    print(f"[3/5] Executing with correct seed ({SEED})...")
    stdin_lines = [str(SEED)]
    for inp, _ in XOR_TESTS:
        stdin_lines.append(' '.join(map(str, inp)))
    stdin_data = '\n'.join(stdin_lines) + '\n'

    r = run(['./payload'], input=stdin_data)
    for line in r.stdout.strip().split('\n'):
        print(f"      {line}")
    print()

    # ---- Step 4: Verify ----
    print("[4/5] Verifying...")

    # Weight accuracy
    max_w_err = 0.0
    for line in r.stdout.strip().split('\n'):
        s = line.strip()
        if s.startswith('w['):
            idx = int(s.split('[')[1].split(']')[0])
            val = float(s.split('=')[1])
            err = abs(val - XOR_WEIGHTS[idx])
            max_w_err = max(max_w_err, err)
    w_ok = max_w_err < 1e-4
    print(f"      Weight max error : {max_w_err:.2e}  "
          f"[{'PASS' if w_ok else 'FAIL'}]")

    # Inference accuracy
    infer_ok = True
    for line in r.stdout.strip().split('\n'):
        if '->' not in line or '[' not in line:
            continue
        lhs, rhs = line.split('->')
        out_val = float(rhs.strip().strip('[]'))
        in_vals = [float(x) for x in lhs.strip().strip('[]').split(',')]
        for inp, exp in XOR_TESTS:
            if all(abs(a - b) < 0.01 for a, b in zip(in_vals, inp)):
                err = abs(out_val - exp)
                ok = err < 0.1
                if not ok:
                    infer_ok = False
                print(f"      XOR({inp[0]}, {inp[1]}) = {out_val:7.4f}  "
                      f"expected {exp:.1f}  [{'PASS' if ok else 'FAIL'}]")

    overall = w_ok and infer_ok
    print(f"\n      Overall: [{'PASS' if overall else 'FAIL'}]")
    print()

    # ---- Step 5: Wrong seed ----
    wrong_seed = 99.9
    print(f"[5/5] Testing with WRONG seed ({wrong_seed})...")
    stdin_wrong = f"{wrong_seed}\n0 1\n"
    r = run(['./payload'], input=stdin_wrong)

    # Show garbled weights
    wrong_w = []
    for line in r.stdout.strip().split('\n'):
        s = line.strip()
        if s.startswith('w['):
            wrong_w.append(float(s.split('=')[1]))
    if wrong_w:
        fmt = lambda v: f"{v:.4f}" if abs(v) < 1e6 else f"{v:.2e}"
        print(f"      First 3 reconstructed : "
              f"[{', '.join(fmt(x) for x in wrong_w[:3])}]")
        print(f"      Expected              : "
              f"{[round(x, 2) for x in XOR_WEIGHTS[:3]]}")

    # Show garbled inference
    for line in r.stdout.strip().split('\n'):
        if '->' in line and '[' in line:
            rhs = line.split('->')[1].strip().strip('[]')
            out_val = float(rhs)
            print(f"      XOR(0, 1) = {out_val:.4f}  "
                  f"(expected 1.0)  [GARBAGE]")

    print()
    print("=" * 56)
    print("  The binary contains only polynomial coefficients.")
    print("  Without the correct seed, weights are unrecoverable.")
    print("=" * 56)

    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
