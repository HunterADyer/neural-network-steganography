/*
 * Rational function weight reconstruction payload
 * 17 weights in 3 chunk(s)
 * f(x): polynomial iterator
 * g(x) = p(x)/q(x): rational weight recovery
 * Weights are sorted before encoding; permutation restores order
 * Network architecture: [2, 4, 1]
 * Compile: gcc -O2 -o payload_rat payload_rat.c -lm
 */
#include <stdio.h>
#include <stdlib.h>

#define N_CHUNKS   3
#define N_WEIGHTS  17
#define MAX_CHUNK  8

static double peval(const double *c, int deg, double x) {
    double r = c[0];
    for (int i = 1; i <= deg; i++)
        r = r * x + c[i];
    return r;
}

/* Rational: r(x) = p(x) / q(x), each via Horner */
static double reval(const double *p, int dp,
                    const double *q, int dq, double x) {
    double num = p[0];
    for (int i = 1; i <= dp; i++)
        num = num * x + p[i];
    double den = q[0];
    for (int i = 1; i <= dq; i++)
        den = den * x + q[i];
    return num / den;
}

/* --- Coefficients and permutations --- */
static const double F0[] = {-2.78563124966526340e-04, 1.65988507480255530e-02, -4.16392659796086750e-01, 5.69298050067785955e+00, -4.57462696256783019e+01, 2.15598294343785824e+02, -5.49003569682041075e+02, 5.84157631286303967e+02};
static const double GP0[] = {-3.61742592601146275e-05, 1.66642151420579357e-03, -2.82267992497080254e-02, 2.14369834947513854e-01, -6.53422800122071989e-01, 2.92904603022594578e-01};
static const double GQ0[] = {1.16818037495929911e-02, -2.15255924972232643e-01, 1.00000000000000000e+00};
static const int P0[] = {3, 2, 7, 5, 0, 1, 4, 6};
static const double F1[] = {-2.43700902948071830e-05, 2.90829541180302048e-03, -1.45941896983471181e-01, 3.97037644404381007e+00, -6.27058445103055817e+01, 5.66505088630622595e+02, -2.63236638629852723e+03, 4.53013768334879660e+03};
static const double GP1[] = {-7.44092302231084361e-03, 8.52443452801661672e-01, -4.05211438910615200e+01, 1.02306262235085489e+03, -1.44700711420380630e+04, 1.08714592197348335e+05, -3.38981032039766200e+05};
static const double GQ1[] = {9.55939967067084007e-02, 1.00000000000000000e+00};
static const int P1[] = {0, 1, 2, 4, 7, 3, 5, 6};
static const double F2[] = {2.62133000000000003e+01};
static const double GP2[] = {-5.00000000000000028e-02};
static const double GQ2[] = {1.00000000000000000e+00};
static const int P2[] = {0};

static const double *F[] = {F0, F1, F2};
static const double *GP[] = {GP0, GP1, GP2};
static const double *GQ[] = {GQ0, GQ1, GQ2};
static const int *P[] = {P0, P1, P2};
static const int csz[] = {8, 8, 1};
static const int dgp[] = {5, 6, 0};
static const int dgq[] = {2, 1, 0};

/* Reconstruct: decode in sorted order, then apply permutation */
static void reconstruct(double seed, double *w) {
    int base = 0;
    for (int c = 0; c < N_CHUNKS; c++) {
        double x = seed;
        int n = csz[c], deg_f = n - 1;
        double tmp[MAX_CHUNK];
        for (int i = 0; i < n; i++) {
            x = peval(F[c], deg_f, x);
            tmp[i] = reval(GP[c], dgp[c], GQ[c], dgq[c], x);
        }
        /* Unsort: P[c][i] maps sorted position i to original position */
        for (int i = 0; i < n; i++)
            w[base + P[c][i]] = tmp[i];
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
