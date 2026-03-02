# Neural Network Steganography

Hiding neural network weights in binaries by encoding them as coefficients of mathematical functions. No weight arrays appear in the compiled binary — a secret seed reconstructs everything at runtime.

## Variants

| Variant | Encoder | Generated C | Description |
|---------|---------|-------------|-------------|
| Polynomial | `generate.py` | `payload.c` | Horner-evaluated polynomials f(x) and g(x); iterative orbit from seed |
| On-the-fly | `generate_otf.py` | `payload_otf.c` | Same math, weights never materialized as an array in memory |
| Rational | `generate_rational.py` | `payload_rat.c` | Rational function encoding (polynomial ratio) |
| ODE | `generate_ode.py` | `payload_ode.c` | Weights recovered via RK4 integration of dy/dt = P(y,t) |

All four variants correctly reconstruct a 2-4-1 XOR network (17 weights, seed `3.7133`).

## Quick Start

```bash
pip install numpy scipy

# Generate and run the polynomial variant
echo '[1,1,-1,-1,1,-1,1,-1,-1.5,-0.5,-0.5,0.5,0.1,2.1,2.1,0.1,-0.05]' | \
    python3 generate.py --arch 2,4,1 > payload.c
gcc -O2 -o payload payload.c -lm
echo '3.7133 0 1' | ./payload

# Run all demos (side-by-side comparison of Python vs generated C)
python3 demo.py            # polynomial
python3 demo_otf.py        # on-the-fly
python3 demo_rational.py   # rational functions
python3 demo_ode.py        # ODE integration
```

## Analysis & Extensions

```bash
python3 analyze_lyapunov.py    # Lyapunov exponents, precision-security ceiling
python3 nearest_seed.py        # seed space exploration
python3 nearest_behavior.py    # boolean behavior diversity from fixed (f,g)
python3 encode_bytes.py        # encode arbitrary bytes (shellcode, configs, etc.)
python3 multi_behavior.py      # different seeds → different network behaviors
python3 seed_2d.py             # 2D seed space exploration
python3 basis_encode.py        # basis vector encoding (no iteration)
```

## How It Works

Instead of storing weights directly, encode them as coefficients of mathematical functions. The binary contains only opaque `double` arrays and arithmetic loops.

```
seed  →  g(seed)  →  g²(seed)  →  g³(seed)  →  ...
              ↓            ↓            ↓
           f(g(s))     f(g²(s))    f(g³(s))
              ↓            ↓            ↓
             w₀           w₁           w₂         ...
```

A reverse engineer sees polynomial evaluation loops and numerical constants — indistinguishable from DSP, spline interpolation, or physics simulation code.

## Dependencies

- Python 3, numpy, scipy
- Any C99 compiler (generated C depends only on `<math.h>`)
