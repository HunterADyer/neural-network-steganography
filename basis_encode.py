#!/usr/bin/env python3
"""
Basis Vector Encoding
======================

Seed → PRNG expand → basis coefficients → linear combination → output bytes.

No iteration, no Lyapunov amplification. Full float64 precision preserved.

The binary contains d basis vectors (disguised as numerical data).
The seed selects a point in the d-dimensional subspace they span.

Test: can we recover arbitrary byte sequences (shellcode, files, etc.)
from a small seed + fixed basis?
"""

import numpy as np
import struct
import sys
import os


def prng_expand(seed_bytes, n_coeffs):
    """
    Expand a short seed into n_coeffs floats in [-1, 1].
    Uses seed as numpy PRNG seed — in a real binary this would be
    a custom LCG/xorshift that looks like normal arithmetic.
    """
    # Interpret seed bytes as a 64-bit integer
    seed_int = int.from_bytes(seed_bytes, 'little') % (2**32)
    rng = np.random.RandomState(seed_int)
    return rng.uniform(-1, 1, n_coeffs)


def build_basis(n_output, d, master_seed=0xDEADBEEF):
    """
    Generate d basis vectors in R^n_output.
    These are baked into the binary as opaque double arrays.

    In practice you'd want these to look like filter coefficients,
    polynomial tables, etc.
    """
    rng = np.random.RandomState(master_seed)
    # Random orthogonal-ish basis (not exactly orthogonal — that would
    # be suspicious. Real numerical data isn't perfectly orthogonal.)
    basis = rng.standard_normal((d, n_output))
    return basis


def encode(target_bytes, basis, seed_bytes):
    """
    Given target bytes, find basis coefficients that reconstruct them.

    target = Σ αᵢ · basis[i]  (solved via least squares)

    Returns the residual — if zero, exact reconstruction is possible.
    """
    n_output = len(target_bytes)
    target = np.array(list(target_bytes), dtype=np.float64)

    # The seed determines the coefficients
    d = basis.shape[0]
    alpha = prng_expand(seed_bytes, d)

    # What the seed currently produces
    produced = alpha @ basis  # matrix-vector: (d,) @ (d, n_output) → (n_output,)

    return produced, alpha


def decode(basis, seed_bytes):
    """
    Reconstruct bytes from seed + basis (what the binary does at runtime).
    """
    d = basis.shape[0]
    alpha = prng_expand(seed_bytes, d)
    raw = alpha @ basis
    # Round to nearest byte
    out = np.clip(np.round(raw), 0, 255).astype(np.uint8)
    return bytes(out)


def find_seed_for_target(target_bytes, basis, max_seeds=2**20):
    """
    Brute-force search: find a seed whose PRNG expansion,
    combined with the basis, produces the target bytes.

    This is the BUILDER's problem: given desired output, find a seed.
    """
    n_output = len(target_bytes)
    target = np.array(list(target_bytes), dtype=np.float64)
    d = basis.shape[0]

    best_seed = None
    best_err = float('inf')

    for i in range(max_seeds):
        seed = struct.pack('<Q', i)[:8]
        alpha = prng_expand(seed, d)
        produced = alpha @ basis
        # Error: how far are we from integer byte values?
        rounded = np.clip(np.round(produced), 0, 255)
        err = np.sum((rounded - target) ** 2)
        if err < best_err:
            best_err = err
            best_seed = seed
            if err == 0:
                break

    return best_seed, best_err


def find_seed_leastsq(target_bytes, basis):
    """
    Analytical approach: solve for α such that α @ basis ≈ target,
    then find a seed whose PRNG produces that α.

    Step 1: α* = argmin ||α @ basis - target||²  (least squares)
    Step 2: find seed whose prng_expand ≈ α*       (search)
    """
    target = np.array(list(target_bytes), dtype=np.float64)
    d = basis.shape[0]
    n = len(target_bytes)

    # Step 1: solve for ideal coefficients
    # basis is (d, n), we want α (d,) such that α @ basis = target
    # This is: basis.T @ α = target, i.e., solve (n × d) system
    alpha_star, residuals, rank, sv = np.linalg.lstsq(basis.T, target, rcond=None)

    if len(residuals) > 0:
        residual_norm = np.sqrt(residuals[0])
    else:
        residual_norm = np.linalg.norm(alpha_star @ basis - target)

    return alpha_star, residual_norm


def test_basic():
    """Basic test: can we reconstruct known bytes?"""
    print("\n[1] Basic Encoding Test")
    print("─" * 60)

    target = b"Hello, World!"
    n = len(target)
    d = n  # square system: d = n means we can hit any target exactly

    basis = build_basis(n, d)

    # Analytical solution
    alpha_star, residual = find_seed_leastsq(target, basis)
    reconstructed = alpha_star @ basis
    rounded = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
    print(f"  Target:        {target}")
    print(f"  Reconstructed: {bytes(rounded)}")
    print(f"  Residual norm: {residual:.2e}")
    print(f"  Exact match:   {bytes(rounded) == target}")
    print(f"  Basis size:    {d} × {n} = {d*n} doubles = {d*n*8} bytes in binary")
    print(f"  Seed info:     {d} coefficients (need seed that produces them)")


def test_shellcode():
    """Test with actual x86-64 shellcode-like bytes."""
    print("\n\n[2] Shellcode Reconstruction")
    print("─" * 60)

    # Classic x86-64 execve("/bin/sh") shellcode (48 bytes)
    shellcode = bytes([
        0x48, 0x31, 0xf6, 0x56, 0x48, 0xbf, 0x2f, 0x62,
        0x69, 0x6e, 0x2f, 0x2f, 0x73, 0x68, 0x57, 0x54,
        0x5f, 0x6a, 0x3b, 0x58, 0x99, 0x0f, 0x05, 0x90,
        0x48, 0x31, 0xf6, 0x56, 0x48, 0xbf, 0x2f, 0x62,
        0x69, 0x6e, 0x2f, 0x2f, 0x73, 0x68, 0x57, 0x54,
        0x5f, 0x6a, 0x3b, 0x58, 0x99, 0x0f, 0x05, 0x90,
    ])
    n = len(shellcode)

    for d in [8, 16, 24, 32, 48, 64]:
        basis = build_basis(n, d)
        alpha_star, residual = find_seed_leastsq(shellcode, basis)
        reconstructed = alpha_star @ basis
        rounded = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
        n_correct = sum(a == b for a, b in zip(rounded, shellcode))
        exact = bytes(rounded) == shellcode

        print(f"  d={d:3d}: residual={residual:8.2f}, "
              f"correct={n_correct}/{n}, exact={exact}, "
              f"binary_cost={d*n*8} bytes")


def test_scaling():
    """How does basis dimension d need to scale with output size n?"""
    print("\n\n[3] Scaling: d vs n for exact reconstruction")
    print("─" * 60)

    rng = np.random.RandomState(777)

    for n in [16, 32, 64, 128, 256, 512]:
        target = bytes(rng.randint(0, 256, n).tolist())

        # Find minimum d for exact reconstruction
        for d in range(1, n + 1):
            basis = build_basis(n, d, master_seed=0xCAFE0000 + n)
            alpha_star, residual = find_seed_leastsq(target, basis)
            reconstructed = alpha_star @ basis
            rounded = np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)
            if bytes(rounded) == target:
                binary_cost = d * n * 8
                ratio = binary_cost / n
                print(f"  n={n:4d} bytes: need d≥{d:4d}, "
                      f"binary={binary_cost:8d} bytes "
                      f"({ratio:.0f}x overhead)")
                break
        else:
            print(f"  n={n:4d} bytes: need d={n} (square system), "
                  f"binary={n*n*8} bytes")


def test_seed_bruteforce():
    """
    The real question: given a fixed basis, can we find a SMALL seed
    (8 bytes) that produces desired output?

    This tests the actual C2 scenario: binary has basis, C2 sends 8 bytes.
    """
    print("\n\n[4] Brute-Force Seed Search (the real test)")
    print("─" * 60)

    target = b"ABCD"  # small target for tractable search
    n = len(target)
    d = n

    basis = build_basis(n, d, master_seed=0xBEEF)

    print(f"  Target: {target} ({n} bytes)")
    print(f"  Basis:  {d}×{n} doubles in binary")
    print(f"  Searching 2^20 seeds...")

    best_seed, best_err = find_seed_for_target(target, basis, max_seeds=2**20)
    if best_seed is not None:
        result = decode(basis, best_seed)
        seed_int = int.from_bytes(best_seed, 'little') % (2**32)
        print(f"  Best seed: {seed_int} (0x{seed_int:08x})")
        print(f"  L2² error: {best_err:.1f}")
        print(f"  Produced:  {result}")
        print(f"  Match:     {result == target}")

    # The fundamental problem:
    print(f"\n  The problem: PRNG maps 32-bit seed → {d} floats in [-1,1]")
    print(f"  That's {d} continuous DOFs quantized to 2^32 discrete seeds")
    print(f"  For {n} target bytes (each 0-255), need to hit a specific")
    print(f"  point in a {d}-dim hypercube — probability ≈ 0")


def test_designed_basis():
    """
    The RIGHT approach: design the basis AND seed together.

    Builder picks seed first, computes α = prng_expand(seed),
    then solves for basis vectors such that α @ basis = target.
    """
    print("\n\n[5] Designed Basis (builder controls both)")
    print("─" * 60)

    rng = np.random.RandomState(42)

    for label, target in [
        ("Hello, World!", b"Hello, World!"),
        ("x86-64 NOP sled", b"\x90" * 32),
        ("Mixed binary", bytes(range(32))),
        ("Random 64 bytes", bytes(rng.randint(0, 256, 64).tolist())),
    ]:
        n = len(target)
        d = n  # square system

        # Builder picks a seed
        seed = struct.pack('<Q', 0x41414141)

        # Compute what α the seed produces
        alpha = prng_expand(seed, d)

        # Solve for basis such that α @ basis = target
        # basis is (d, n), α is (d,)
        # For each output position j: Σ_i α_i * basis[i,j] = target[j]
        # This is d equations in d unknowns (one per basis column)
        # But we have d basis vectors, each of length n
        # Simplest: make basis diagonal-ish
        target_vec = np.array(list(target), dtype=np.float64)

        # α @ basis = target means basis = outer product structure
        # For a square system: basis[i,j] is determined by the constraint
        # Use: basis = (1/α) ⊗ target (rank-1, but works)
        # More generally: any basis where α @ basis = target

        # Clean solution: basis[i,:] = (target / α[i]) * e_i for diagonal
        # But that's too obvious. Use random basis + correction.
        basis = rng.standard_normal((d, n))
        # Current output: alpha @ basis
        current = alpha @ basis
        # Correct the first basis vector to fix the residual
        residual = target_vec - current
        # Add residual/alpha[0] to basis[0]
        basis[0] += residual / alpha[0]

        # Verify
        result = decode(basis, seed)
        match = result == target

        print(f"  {label} ({n} bytes): match={match}, "
              f"seed=0x41414141, "
              f"binary_cost={d*n*8} bytes ({d*n*8/n:.0f}x)")


def test_compression_ratio():
    """
    Summary: what's the actual compression ratio?

    For n output bytes, the binary needs d×n×8 bytes of basis data.
    The seed is 8 bytes.

    Total cost = binary_overhead + seed_over_c2
    """
    print("\n\n[6] Compression Analysis")
    print("─" * 60)

    print(f"  {'n (output)':>12} {'d (basis)':>10} {'binary':>10} "
          f"{'seed':>6} {'ratio':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*6} {'─'*8}")

    for n in [17, 48, 128, 256, 512, 1024, 4096]:
        d = n  # need d=n for exact reconstruction of arbitrary data
        binary_cost = d * n * 8
        seed_cost = 8
        # Ratio: binary cost / output size
        ratio = binary_cost / n
        print(f"  {n:>12} {d:>10} {binary_cost:>10} {seed_cost:>6} {ratio:>7.0f}x")

    print(f"\n  The ratio is always ~8n (storing n vectors of n doubles).")
    print(f"  For n=17 (our NN):  binary=2,312 bytes, seed=8 bytes")
    print(f"  For n=4096 (1 page): binary=134 MB — impractical")
    print(f"\n  Compare: original polynomial scheme for 17 weights:")
    print(f"    binary = ~272 bytes (2 chunks × 8 coeffs × 2 polys × 8 bytes)")
    print(f"    seed = 8 bytes")
    print(f"    ratio = 16x (vs 8×17=136x for basis method)")


def main():
    print("=" * 65)
    print("  Basis Vector Encoding")
    print("  Seed → PRNG → coefficients → basis combination → output")
    print("=" * 65)

    test_basic()
    test_shellcode()
    test_scaling()
    test_seed_bruteforce()
    test_designed_basis()
    test_compression_ratio()

    print(f"\n{'═' * 65}")
    print("  VERDICT")
    print(f"{'═' * 65}")
    print("""
  The basis method CAN encode arbitrary bytes (shellcode, whatever)
  with exact reconstruction. The builder designs both seed and basis.

  But the cost is brutal:
    - Binary overhead: O(n²) for n output bytes (n basis vectors × n doubles)
    - For shellcode (48 bytes): 18 KB of basis data in the binary
    - For a page (4096 bytes): 134 MB — absurd

  The polynomial iteration scheme is MUCH more compact:
    - Binary overhead: O(n) for n output values
    - 17 weights → 272 bytes of coefficients
    - But limited to ~2^10 effective seed states (Lyapunov ceiling)

  Fundamental tradeoff:
    Polynomial iteration:  O(n) binary, ~10 bits effective seed
    Basis method:          O(n²) binary, full seed precision
    Just send the data:    0 binary, O(n) over C2

  The polynomial scheme wins on binary compactness.
  The basis scheme wins on output diversity.
  Neither beats just sending the data, if you have the channel.
""")
    print(f"{'═' * 65}")


if __name__ == '__main__':
    main()
