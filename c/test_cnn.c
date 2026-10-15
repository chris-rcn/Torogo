/* test_cnn.c — checks for cnn.c: backward against central finite differences on
 * every parameter of a small net (all weights nonzero, so the residual paths
 * carry gradient), exact translation invariance on the torus, and a save/load
 * round trip.
 *
 * Built twice (c/build.sh): test_cnn.bin as is, where float32 loss resolution
 * and ReLU kinks inside +-eps make single-parameter comparisons unreliable, so
 * the gradient test is the cosine between the analytic and numeric gradient
 * vectors (> 0.999, a coarse sanity check); and test_cnn_double.bin with
 * -Dfloat=double -Dexpf=exp -DCNN_TEST_DOUBLE, where eps 1e-5 leaves both
 * negligible and every parameter with a gradient above 1e-5 is checked to 1e-4
 * relative.  Both a 4-channel net (the generic kernels) and a 32-channel one
 * (the register-blocked kernels) are checked. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "game2.h"
#include "cnn.h"

static int passed = 0, failed = 0;
static void check(const char *name, int ok) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) passed++; else failed++;
}

/* A position reached by random legal play (seeded), with stones of every liberty class. */
static void random_position(Game2 *g, int N, int moves, long seed) {
    Rng r; rng_seed(&r, seed);
    g2_new(g, N);
    for (int i = 0; i < moves && !g->game_over; i++) g2_play(g, g2_random_legal_move(g, &r));
}

static double bce(float v, float t) { return -(t * log(v) + (1 - t) * log(1 - v)); }

/* C = 4 runs the generic kernels, C = 32 the register-blocked ones. */
static void test_gradients(int C, int B, int H) {
    const int N = 7, area = N * N;
    g2_init_topology(N);
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    const VnetLayout L = vnet_layout(C, B, H);
    float *w = malloc(sizeof(float) * L.n_params), *G = calloc(L.n_params, sizeof(float));
    vnet_init(&L, w, 99);
    Rng r; rng_seed(&r, 5);
    /* Every weight nonzero (the init zeroes each block's second conv and the
     * biases), scaled to its layer's fan-in so activations stay O(1) at any
     * width: uniform with the He-normal standard deviation. */
    const float s_in = sqrtf(2.0f / (9 * VNET_PLANES)), s_c = sqrtf(2.0f / (9 * C)), u = sqrtf(12.0f);
    for (int i = L.stem_b; i < L.stem_b + C; i++) w[i] = 0.1f * (rng_float(&r) - 0.5f);
    for (int i = 0; i < 9 * VNET_PLANES * C; i++) w[L.stem_w + i] += 0.2f * s_in * u * (rng_float(&r) - 0.5f);
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < 9 * C * C; i++) w[L.blk_w2[b] + i] = 0.7f * s_c * u * (rng_float(&r) - 0.5f);
        for (int i = 0; i < C; i++) { w[L.blk_b1[b] + i] = 0.1f * (rng_float(&r) - 0.5f); w[L.blk_b2[b] + i] = 0.1f * (rng_float(&r) - 0.5f); }
    }
    /* Positive head biases keep the head units active, so gradient reaches the body. */
    for (int i = 0; i < H; i++) w[L.head_b + i] = 0.5f + 0.1f * (rng_float(&r) - 0.5f);
    /* Output weights O(1) (not the init's 0.1/sqrt(H)), so the gradients clear
     * the float loss's finite-difference noise floor (~5e-5 at eps 1e-3). */
    for (int j = 0; j < L.H; j++) w[L.out_w + j] = 1.5f * (rng_float(&r) + 0.5f);
    Game2 g; random_position(&g, N, 30, 3);
    VnetWork *k = vnet_work_new(&L, area);
    vnet_encode(&g, NULL, vnet_work_input(k));
    const float t = 0.3f;
    const float v = vnet_forward(&L, w, nbr9, area, k);
    vnet_backward(&L, w, nbr9, area, k, v - t, G);
    int checked = 0, bad = 0;
    double worst = 0, dot = 0, nn = 0, na = 0;
#ifdef CNN_TEST_DOUBLE
    /* eps 1e-5: smaller lets double rounding noise (it grows as 1/eps: 89 /
     * 613 / 3868 failures at 1e-6 / 1e-7 / 1e-8 on the 32-channel net) swamp
     * the comparison; kinks inside +-1e-5 stay rare (0 failures). */
    const float eps = 1e-5, tol = 1e-4, floor_ = 1e-5;
#else
    const float eps = 1e-3f, tol = 0.02f, floor_ = 1e-3f;
#endif
    for (int i = 0; i < L.n_params; i++) {
        const float keep = w[i];
        w[i] = keep + eps; const double lp = bce(vnet_forward(&L, w, nbr9, area, k), t);
        w[i] = keep - eps; const double lm = bce(vnet_forward(&L, w, nbr9, area, k), t);
        w[i] = keep;
        const double num = (lp - lm) / (2 * eps), ana = G[i];
        dot += num * ana; nn += num * num; na += ana * ana;
        if (fabs(num) + fabs(ana) < floor_) continue;   /* too near the noise floor to compare singly */
        checked++;
        const double rel = fabs(num - ana) / (fabs(num) + fabs(ana));
        if (rel > worst) worst = rel;
        if (rel > tol) bad++;
    }
    const double cosine = dot / sqrt(nn * na);
    printf("  gradient check, %d channels %d blocks (%s, eps %g): %d of %d parameters checked singly, %d over %g relative error (worst %.2e); "
           "cosine(analytic, numeric) over all %d = %.8f\n", C, B, sizeof(float) == 8 ? "double" : "float",
           (double)eps, checked, L.n_params, bad, (double)tol, worst, L.n_params, cosine);
#ifdef CNN_TEST_DOUBLE
    check("backward matches finite differences, every parameter", checked > L.n_params / 2 && bad == 0);
#else
    /* A coarse sanity check only — float finite differences are noisy at this
     * scale; the double build is the exact test. */
    check("backward matches finite differences (gradient vectors' cosine > 0.999)", cosine > 0.999);
#endif
    vnet_work_free(k); free(w); free(G);
}

static void test_translation_invariance(void) {
    const int N = 9, area = N * N;
    g2_init_topology(N);
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    const VnetLayout L = vnet_layout(8, 2, 4);
    float *w = malloc(sizeof(float) * L.n_params);
    vnet_init(&L, w, 7);
    Rng r; rng_seed(&r, 6);
    for (int i = 0; i < L.n_params; i++) w[i] += 0.2f * (rng_float(&r) - 0.5f);
    Game2 g; random_position(&g, N, 40, 11);
    VnetWork *k = vnet_work_new(&L, area);
    vnet_encode(&g, NULL, vnet_work_input(k));
    const float v0 = vnet_forward(&L, w, nbr9, area, k);
    double worst = 0;
    int32_t map[MAX_CAP];
    for (int dy = 0; dy < N; dy++) for (int dx = 0; dx < N; dx++) {
        for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) map[y * N + x] = ((y + dy) % N) * N + (x + dx) % N;
        vnet_encode(&g, map, vnet_work_input(k));
        const double d = fabs(vnet_forward(&L, w, nbr9, area, k) - v0);
        if (d > worst) worst = d;
    }
    printf("  translation invariance: value %.6f, worst |difference| over all %d shifts %.2e\n", v0, area, worst);
    check("value is translation-invariant on the torus", worst < 1e-5);
    vnet_work_free(k); free(w);
}

static void test_save_load(void) {
    const VnetLayout L = vnet_layout(8, 3, 5);
    float *w = malloc(sizeof(float) * L.n_params);
    vnet_init(&L, w, 3);
    const char *path = "/tmp/test_cnn_vnet.bin";
    int ok = vnet_save(path, &L, w);
    VnetLayout L2;
    float *w2 = ok ? vnet_load(path, &L2) : NULL;
    ok = w2 && L2.C == 8 && L2.B == 3 && L2.H == 5 && memcmp(w, w2, sizeof(float) * L.n_params) == 0;
    check("save / load round trip", ok);
    remove(path);
    free(w); free(w2);
}

int main(void) {
    test_gradients(4, 2, 8);
    test_gradients(32, 2, 8);
    test_translation_invariance();
    test_save_load();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0;
}
