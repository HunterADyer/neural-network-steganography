/*
 * ODE-based weight reconstruction payload
 * 17 weights in 3 chunk(s)
 * dy/dt = Σ c_ij * y^i * t^j,  y(0) = seed
 * Weights recovered at Chebyshev time nodes via RK4
 * Network architecture: [2, 4, 1]
 * Compile: gcc -O2 -o payload_ode payload_ode.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_CHUNKS   3
#define N_WEIGHTS  17
#define MAX_CHUNK  8
#define MAX_DEG    6

/* Evaluate ODE RHS: dy/dt = Σ c_ij * y^i * t^j, i+j ≤ D */
static double ode_rhs(double y, double t, const double *c, int D) {
    double val = 0.0;
    double yi = 1.0;
    int k = 0;
    for (int i = 0; i <= D; i++) {
        double tj = 1.0;
        for (int j = 0; j <= D - i; j++) {
            val += c[k++] * yi * tj;
            tj *= t;
        }
        yi *= y;
    }
    return val;
}

/* Chebyshev nodes of the first kind on [lo, hi], sorted ascending */
static void cheb_nodes(int n, double lo, double hi, double *out) {
    for (int k = 0; k < n; k++)
        out[k] = 0.5 * (lo + hi)
               + 0.5 * (hi - lo) * cos((2*k + 1) * M_PI / (2*n));
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (out[j] < out[i]) {
                double tmp = out[i]; out[i] = out[j]; out[j] = tmp;
            }
}

/* --- ODE coefficients and permutations --- */
static const double C0[] = {8.72685892548388509e+00, -2.34645602208643353e+01, 3.82415990421768281e+00, 2.75592103297366386e-01, -1.00263547040568436e-02, -5.78621720714821208e+00, -4.42776004203702944e-01, -1.93062467200872737e-02, -3.58435380648915031e-02, -1.46548470661486281e+01, 2.93422477358811200e+01, -5.85094977205563183e+00, 3.40543440643334039e+00, 1.73987716595718456e+00, 1.73273719317004776e-01};
static const int P0[] = {3, 2, 7, 5, 0, 1, 4, 6};
static const double C1[] = {-2.35015478331451284e+01, 5.93543469720404673e+01, -5.33438825417517037e+01, 2.22880666101168892e+01, -4.66928032930523784e+00, 4.72982464148571702e-01, -1.82687952019928712e-02, -8.06756304311203154e-01, -1.82653120295798410e+00, 1.56828465673631223e+00, -5.64266153439629381e-01, 8.71654844376534704e-02, -5.09647017148925907e-03, -7.93091273288489829e-02, -3.89041557557245508e-04, -8.87555062385460003e-02, 5.00446987974542817e-02, -5.94210274422163256e-03, -6.40364279546261489e-02, 4.68869287663497558e-01, -2.16793684669632974e-01, 2.78427573522148676e-02, -1.76931366007946241e-02, 2.59176113222887528e-01, -6.15714379028461448e-02, 3.87417102937556495e-03, 4.02941861910678586e-02, 1.49379752452555282e-03};
static const int P1[] = {0, 1, 2, 4, 7, 3, 5, 6};
static const double C2[] = {-2.50886666666666658e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00};
static const int P2[] = {0};

static const double *CC[] = {C0, C1, C2};
static const int *PP[] = {P0, P1, P2};
static const int csz[] = {8, 8, 1};
static const int deg[] = {4, 6, 4};
static const int nsub[] = {500, 50, 50};

/* Reconstruct weights from seed via ODE integration */
static void reconstruct(double seed, double *w) {
    int base = 0;
    for (int ch = 0; ch < N_CHUNKS; ch++) {
        int n = csz[ch], D = deg[ch], ns = nsub[ch];
        double T = (n > 2) ? (double)n : 2.0;
        double tn[MAX_CHUNK];
        cheb_nodes(n, 1.0, T, tn);
        double tmp[MAX_CHUNK];
        double y = seed, t_cur = 0.0;
        for (int i = 0; i < n; i++) {
            double dt = tn[i] - t_cur;
            int nsteps = (int)(ns * dt + 0.5);
            if (nsteps < 1) nsteps = 1;
            double h = dt / nsteps;
            for (int s = 0; s < nsteps; s++) {
                double ts = t_cur + s * h;
                double k1 = h * ode_rhs(y, ts, CC[ch], D);
                double k2 = h * ode_rhs(y + 0.5*k1, ts + 0.5*h, CC[ch], D);
                double k3 = h * ode_rhs(y + 0.5*k2, ts + 0.5*h, CC[ch], D);
                double k4 = h * ode_rhs(y + k3, ts + h, CC[ch], D);
                y += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
            }
            t_cur = tn[i];
            tmp[i] = y;
        }
        /* Unsort: P[ch][i] maps sorted position i to original index */
        for (int i = 0; i < n; i++)
            w[base + PP[ch][i]] = tmp[i];
        base += n;
    }
}

#define N_LAYERS   3
#define MAX_DIM    4
#define INPUT_DIM  2
#define OUTPUT_DIM 1
static const int arch[] = {2, 4, 1};

static void forward(const double *w, const double *inp, double *outp) {
    double a[MAX_DIM], b[MAX_DIM];
    double *cur = a, *nxt = b;
    for (int i = 0; i < arch[0]; i++) cur[i] = inp[i];
    int off = 0;
    for (int l = 0; l < N_LAYERS - 1; l++) {
        int ni = arch[l], no = arch[l + 1];
        for (int j = 0; j < no; j++) {
            double s = w[off + ni * no + j];
            for (int i = 0; i < ni; i++)
                s += cur[i] * w[off + i * no + j];
            nxt[j] = (l < N_LAYERS - 2 && s < 0.0) ? 0.0 : s;
        }
        off += ni * no + no;
        double *t = cur; cur = nxt; nxt = t;
    }
    for (int i = 0; i < arch[N_LAYERS - 1]; i++)
        outp[i] = cur[i];
}

int main(void) {
    double seed;
    if (scanf("%lf", &seed) != 1) return 1;

    double w[N_WEIGHTS];
    reconstruct(seed, w);

    printf("Reconstructed %d weights:\n", N_WEIGHTS);
    for (int i = 0; i < N_WEIGHTS; i++)
        printf("  w[%3d] = %12.6f\n", i, w[i]);

    printf("\nInference (arch [2, 4, 1]):\n");
    double in_buf[INPUT_DIM], out_buf[OUTPUT_DIM];
    while (1) {
        int ok = 1;
        for (int i = 0; i < INPUT_DIM; i++)
            if (scanf("%lf", &in_buf[i]) != 1) { ok = 0; break; }
        if (!ok) break;
        forward(w, in_buf, out_buf);
        printf("  [");
        for (int i = 0; i < INPUT_DIM; i++) {
            if (i > 0) printf(", ");
            printf("%.2f", in_buf[i]);
        }
        printf("] -> [");
        for (int i = 0; i < OUTPUT_DIM; i++) {
            if (i > 0) printf(", ");
            printf("%.4f", out_buf[i]);
        }
        printf("]\n");
    }

    return 0;
}
