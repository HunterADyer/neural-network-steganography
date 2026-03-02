/*
 * On-the-fly polynomial weight reconstruction
 * 17 weights, 3 chunk(s)
 * Network architecture: [2, 4, 1]
 *
 * KEY DIFFERENCE: weights are never materialized in memory.
 * Each weight exists only in a register during one MAC operation.
 * Compile: gcc -O2 -o payload_otf payload_otf.c -lm
 */
#include <stdio.h>
#include <stdlib.h>

#define N_CHUNKS   3
#define N_WEIGHTS  17
#define N_LAYERS   3
#define MAX_DIM    4
#define INPUT_DIM  2
#define OUTPUT_DIM 1

static double peval(const double *c, int deg, double x) {
    double r = c[0];
    for (int i = 1; i <= deg; i++)
        r = r * x + c[i];
    return r;
}

/* --- Polynomial coefficients --- */
static const double F0[] = {-2.78563124966526340e-04, 1.65988507480255530e-02, -4.16392659796086750e-01, 5.69298050067785955e+00, -4.57462696256783019e+01, 2.15598294343785824e+02, -5.49003569682041075e+02, 5.84157631286303967e+02};
static const double G0[] = {-2.29663203619631402e-03, 1.40899110725681620e-01, -3.62847665485718363e+00, 5.07857971740482910e+01, -4.16861769781887972e+02, 2.00570313866703395e+03, -5.23853468320937554e+03, 5.73470235600655906e+03};
static const double F1[] = {-2.43700902948071830e-05, 2.90829541180302048e-03, -1.45941896983471181e-01, 3.97037644404381007e+00, -6.27058445103055817e+01, 5.66505088630622595e+02, -2.63236638629852723e+03, 4.53013768334879660e+03};
static const double G1[] = {1.38847365420055667e-03, -1.88234983225918473e-01, 1.08913251953263650e+01, -3.48647444163544833e+02, 6.66881411507473968e+03, -7.62215993725151202e+04, 4.82020905835384387e+05, -1.30114809603011352e+06};
static const double F2[] = {2.62133000000000003e+01};
static const double G2[] = {-5.00000000000000028e-02};

static const double *F[] = {F0, F1, F2};
static const double *G[] = {G0, G1, G2};
static const int csz[] = {8, 8, 1};
static const int arch[] = {2, 4, 1};

/*
 * Iterator: walks the polynomial chain to produce weights one at a time.
 * Calling next_weight() repeatedly yields w[0], w[1], w[2], ...
 * No weight array is ever allocated.
 */
struct weight_iter {
    double seed;
    int chunk;      /* current chunk index */
    int pos;        /* position within current chunk */
    double x;       /* current polynomial chain state */
};

static void iter_init(struct weight_iter *it, double seed) {
    it->seed = seed;
    it->chunk = 0;
    it->pos = 0;
    it->x = seed;
}

static double next_weight(struct weight_iter *it) {
    /* Advance to next chunk if needed */
    if (it->pos >= csz[it->chunk]) {
        it->chunk++;
        it->pos = 0;
        it->x = it->seed;  /* reset chain for new chunk */
    }
    int c = it->chunk;
    int deg = csz[c] - 1;
    it->x = peval(F[c], deg, it->x);
    double w = peval(G[c], deg, it->x);
    it->pos++;
    return w;  /* weight exists only in this return value */
}

/*
 * Feedforward inference — weights computed on-the-fly.
 * At no point does a "double weights[N]" array exist.
 * Each weight is a transient register value inside the MAC loop.
 */
static void forward_otf(double seed, const double *inp, double *outp) {
    struct weight_iter it;
    iter_init(&it, seed);

    double a[MAX_DIM], b[MAX_DIM];
    double *cur = a, *nxt = b;
    for (int i = 0; i < arch[0]; i++)
        cur[i] = inp[i];

    for (int l = 0; l < N_LAYERS - 1; l++) {
        int ni = arch[l], no = arch[l + 1];

        /* Accumulate W*x — each weight is computed, used, discarded */
        for (int j = 0; j < no; j++)
            nxt[j] = 0.0;

        for (int i = 0; i < ni; i++) {
            for (int j = 0; j < no; j++) {
                double w = next_weight(&it);  /* transient */
                nxt[j] += cur[i] * w;
            }
        }

        /* Add bias — also computed on-the-fly */
        for (int j = 0; j < no; j++) {
            double bias = next_weight(&it);  /* transient */
            nxt[j] += bias;
            /* ReLU for hidden, linear for output */
            if (l < N_LAYERS - 2 && nxt[j] < 0.0)
                nxt[j] = 0.0;
        }

        double *t = cur; cur = nxt; nxt = t;
    }

    for (int i = 0; i < arch[N_LAYERS - 1]; i++)
        outp[i] = cur[i];
}

/* Optional: reconstruct and print for verification */
static void print_weights(double seed) {
    struct weight_iter it;
    iter_init(&it, seed);
    printf("Reconstructed %d weights (via iterator):\n", N_WEIGHTS);
    for (int i = 0; i < N_WEIGHTS; i++) {
        double w = next_weight(&it);
        printf("  w[%3d] = %12.6f\n", i, w);
    }
}

int main(void) {
    double seed;
    if (scanf("%lf", &seed) != 1) return 1;

    print_weights(seed);

    printf("\nOn-the-fly inference (arch [2, 4, 1]):\n");
    double in_buf[INPUT_DIM], out_buf[OUTPUT_DIM];
    while (1) {
        int ok = 1;
        for (int i = 0; i < INPUT_DIM; i++)
            if (scanf("%lf", &in_buf[i]) != 1) { ok = 0; break; }
        if (!ok) break;

        /* Each call recomputes all weights from the seed — */
        /* nothing is cached between inferences */
        forward_otf(seed, in_buf, out_buf);

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
