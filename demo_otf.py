#!/usr/bin/env python3
"""
Demo for the on-the-fly variant.

Compares both variants side-by-side:
  - Standard:  reconstruct all weights into array, then forward pass
  - OTF:       compute each weight in a register during forward pass, never store
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


def build_variant(name, generator, c_file, binary):
    """Generate, compile, and return the binary path."""
    print(f"  [{name}] Generating {c_file}...")
    r = run(
        [sys.executable, generator,
         '--seed', str(SEED),
         '--arch', ','.join(map(str, ARCH))],
        input=json.dumps(XOR_WEIGHTS),
    )
    with open(c_file, 'w') as f:
        f.write(r.stdout)

    print(f"  [{name}] Compiling {binary}...")
    run(['gcc', '-O2', '-o', binary, c_file, '-lm'])
    return binary


def run_inference(binary, seed, inputs):
    """Run binary with seed + inputs, return stdout."""
    stdin_data = f"{seed}\n"
    for inp in inputs:
        stdin_data += ' '.join(map(str, inp)) + '\n'
    r = run([f'./{binary}'], input=stdin_data)
    return r.stdout


def parse_inference(stdout):
    """Extract inference results as list of (input, output) pairs."""
    results = []
    for line in stdout.strip().split('\n'):
        if '->' not in line or '[' not in line:
            continue
        lhs, rhs = line.split('->')
        in_vals = [float(x) for x in lhs.strip().strip('[]').split(',')]
        out_vals = [float(x) for x in rhs.strip().strip('[]').split(',')]
        results.append((in_vals, out_vals))
    return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("  On-The-Fly Variant Demo")
    print("  Weights never exist in memory — only in CPU registers")
    print("=" * 60)
    print()

    # ---- Build both variants ----
    print("[1/3] Building both variants...")
    build_variant("standard", "generate.py", "payload.c", "payload")
    build_variant("otf",      "generate_otf.py", "payload_otf.c", "payload_otf")
    print()

    # ---- Run both ----
    print("[2/3] Running inference...")
    out_std = run_inference("payload", SEED, XOR_TESTS)
    out_otf = run_inference("payload_otf", SEED, XOR_TESTS)

    res_std = parse_inference(out_std)
    res_otf = parse_inference(out_otf)

    print(f"  {'Input':<12} {'Standard':<14} {'OTF':<14} {'Expected':<10} Match")
    print(f"  {'-'*10:<12} {'-'*12:<14} {'-'*12:<14} {'-'*8:<10} -----")
    all_ok = True
    for i, (inp, exp) in enumerate(zip(XOR_TESTS, XOR_EXPECTED)):
        v_std = res_std[i][1][0] if i < len(res_std) else float('nan')
        v_otf = res_otf[i][1][0] if i < len(res_otf) else float('nan')
        match = abs(v_std - v_otf) < 1e-10
        if not match:
            all_ok = False
        inp_s = f"({inp[0]}, {inp[1]})"
        print(f"  {inp_s:<12} {v_std:<14.6f} {v_otf:<14.6f} {exp:<10.1f} "
              f"{'yes' if match else 'NO'}")
    print()

    # ---- Compare binaries ----
    print("[3/3] Binary comparison...")
    sz_std = os.path.getsize("payload")
    sz_otf = os.path.getsize("payload_otf")
    print(f"  Standard binary : {sz_std} bytes")
    print(f"  OTF binary      : {sz_otf} bytes")

    # Check for weight-like strings
    for name in ["payload", "payload_otf"]:
        r = run(["strings", name])
        found = any(f"{w:.4f}" in r.stdout for w in XOR_WEIGHTS if w != 0)
        print(f"  {name:<18}: weight strings in binary? "
              f"{'YES (bad)' if found else 'no (good)'}")

    # Key difference explanation
    print()
    print("  Key forensic difference:")
    print("    Standard — weights exist as a contiguous double[] in process memory")
    print("    OTF      — weights only exist in registers during MAC operations")
    print("               A memory dump of the OTF process reveals nothing")

    print()
    status = "PASS" if all_ok else "FAIL"
    print(f"  Result: [{status}] — both variants produce identical output")
    print()

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
