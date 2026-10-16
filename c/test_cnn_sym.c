/* test_cnn_sym.c — checks for cnn_sym.c (the symmetric-kernel experiment):
 * backward against central finite differences on every parameter, exact
 * invariance under all 8 reflections/rotations and all translations of the
 * torus, the grown net computing what it did, and the VSYM1 save/load (which
 * refuses a cnn.c VNET1 file).  Built twice, as test_cnn: float (the
 * gradient vectors' cosine) and -Dfloat=double -Dexpf=exp -DCNN_TEST_DOUBLE
 * (every parameter to 1e-4 relative). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "game2.h"
#include "game3.h"
#include "cnn.h"

static int passed = 0, failed = 0;
static void check(const char *name, int ok) {
    printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) passed++; else failed++;
}

static void random_position(Game2 *g, int N, int moves, long seed) {
    Rng r; rng_seed(&r, seed);
    g2_new(g, N);
    for (int i = 0; i < moves && !g->game_over; i++) g2_play(g, g2_random_legal_move(g, &r));
}

static double bce(float v, float t) { return -(t * log(v) + (1 - t) * log(1 - v)); }

/* Every weight nonzero (the init zeroes each block's second conv and the
 * biases), O(1) activations, output centred near 0.5. */
static float *test_weights(const VnetLayout *L, const int32_t *nbr9, int area, VnetWork *k, long seed) {
    float *w = malloc(sizeof(float) * L->n_params);
    vnet_init(L, w, (uint64_t)seed);
    Rng r; rng_seed(&r, seed + 1);
    const int C = L->C, H = L->H;
    const float s_c = sqrtf(2.0f / (9 * C)), u = sqrtf(12.0f);
    for (int i = L->stem_b; i < L->stem_b + C; i++) w[i] = 0.1f * (rng_float(&r) - 0.5f);
    for (int b = 0; b < L->B; b++) {
        for (int i = 0; i < 3 * C * C; i++) w[L->blk_w2[b] + i] = 0.7f * s_c * u * (rng_float(&r) - 0.5f);
        for (int i = 0; i < C; i++) { w[L->blk_b1[b] + i] = 0.1f * (rng_float(&r) - 0.5f); w[L->blk_b2[b] + i] = 0.1f * (rng_float(&r) - 0.5f); }
    }
    for (int i = 0; i < H; i++) w[L->head_b + i] = 0.5f + 0.1f * (rng_float(&r) - 0.5f);
    for (int j = 0; j < H; j++) w[L->out_w + j] = 1.5f * (rng_float(&r) + 0.5f);
    /* Halve the output weights until the output is off its rails (summing 4
     * correlated neighbours per weight grows activations faster than He
     * assumes), then centre it: a saturated sigmoid's logit is not finite. */
    float v0 = vnet_forward(L, w, nbr9, area, k);
    for (int it = 0; it < 60 && (v0 > 0.99f || v0 < 0.01f); it++) {
        for (int j = 0; j < H; j++) w[L->out_w + j] *= 0.5f;
        v0 = vnet_forward(L, w, nbr9, area, k);
    }
    w[L->out_b] -= logf(v0 / (1 - v0));
    return w;
}

static void test_gradients(int C, int B, int H) {
    const int N = 7, area = N * N;
    g2_init_topology(N);
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    const VnetLayout L = vnet_layout(C, B, H);
    VnetWork *k = vnet_work_new(&L, area);
    Game2 g; random_position(&g, N, 30, 3);
    Game3 *g3 = g3_new(N);
    vnet_encode(&g, g3, NULL, vnet_work_input(k));
    g3_free(g3);
    float *w = test_weights(&L, nbr9, area, k, 99), *G = calloc(L.n_params, sizeof(float));
    const float t = 0.3f;
    const float v = vnet_forward(&L, w, nbr9, area, k);
    vnet_backward(&L, w, nbr9, area, k, v - t, G);
#ifdef CNN_TEST_DOUBLE
    const float eps = 1e-5, tol = 1e-4, floor_ = 1e-5;
#else
    const float eps = 1e-3f, tol = 0.02f, floor_ = 1e-3f;
#endif
    int checked = 0, bad = 0;
    double worst = 0, dot = 0, nn = 0, na = 0;
    for (int i = 0; i < L.n_params; i++) {
        const float keep = w[i];
        w[i] = keep + eps; const double lp = bce(vnet_forward(&L, w, nbr9, area, k), t);
        w[i] = keep - eps; const double lm = bce(vnet_forward(&L, w, nbr9, area, k), t);
        w[i] = keep;
        const double num = (lp - lm) / (2 * eps), ana = G[i];
        dot += num * ana; nn += num * num; na += ana * ana;
        if (fabs(num) + fabs(ana) < floor_) continue;
        checked++;
        const double rel = fabs(num - ana) / (fabs(num) + fabs(ana));
        if (rel > worst) worst = rel;
        if (rel > tol) bad++;
    }
    const double cosine = dot / sqrt(nn * na);
    printf("  gradient check, %d channels %d blocks (%s): %d of %d checked singly, %d over %g (worst %.2e); cosine %.8f\n",
           C, B, sizeof(float) == 8 ? "double" : "float", checked, L.n_params, bad, (double)tol, worst, cosine);
#ifdef CNN_TEST_DOUBLE
    check("backward matches finite differences, every parameter", checked > L.n_params / 2 && bad == 0);
#else
    check("backward matches finite differences (cosine > 0.999)", cosine > 0.999);
#endif
    vnet_work_free(k); free(w); free(G);
}

/* The value under every reflection/rotation and every translation of a
 * position: equal up to float summation order. */
static void test_invariance(void) {
    const int N = 9, area = N * N;
    g2_init_topology(N);
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    static int32_t sym[8 * MAX_CAP]; vnet_sym_map(N, sym);
    const VnetLayout L = vnet_layout(32, 3, 16);
    VnetWork *k = vnet_work_new(&L, area);
    Game3 *g3 = g3_new(N);
    double worst_d4 = 0, worst_tr = 0;
    for (int pi = 0; pi < 10; pi++) {
        Game2 g; random_position(&g, N, 15 + 4 * pi, 200 + pi);
        vnet_encode(&g, g3, NULL, vnet_work_input(k));
        float *w = test_weights(&L, nbr9, area, k, 7 + pi);
        vnet_encode(&g, g3, NULL, vnet_work_input(k));
        const double v0 = vnet_forward(&L, w, nbr9, area, k);
        for (int t = 1; t < 8; t++) {
            vnet_encode(&g, g3, sym + (size_t)t * area, vnet_work_input(k));
            const double d = fabs(vnet_forward(&L, w, nbr9, area, k) - v0);
            if (d > worst_d4) worst_d4 = d;
        }
        int32_t map[MAX_CAP];
        for (int dy = 0; dy < N; dy += 3) for (int dx = 0; dx < N; dx += 2) {
            for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) map[y * N + x] = ((y + dy) % N) * N + (x + dx) % N;
            vnet_encode(&g, g3, map, vnet_work_input(k));
            const double d = fabs(vnet_forward(&L, w, nbr9, area, k) - v0);
            if (d > worst_tr) worst_tr = d;
        }
        free(w);
    }
    printf("  invariance over 10 positions: worst |difference| %.2e under the 8 reflections/rotations, %.2e under translations\n", worst_d4, worst_tr);
    check("value is invariant under every reflection/rotation", worst_d4 < 1e-5);
    check("value is invariant under translations", worst_tr < 1e-5);
    g3_free(g3); vnet_work_free(k);
}

static void test_grow_blocks(void) {
    const int N = 9, area = N * N;
    g2_init_topology(N);
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    const VnetLayout L = vnet_layout(32, 2, 8);
    VnetWork *k = vnet_work_new(&L, area);
    Game3 *g3 = g3_new(N);
    Game2 pos0; random_position(&pos0, N, 20, 50);
    vnet_encode(&pos0, g3, NULL, vnet_work_input(k));
    float *w = test_weights(&L, nbr9, area, k, 4);
    VnetLayout G;
    float *g = vnet_grow_blocks(&L, w, 5, 9, &G);
    VnetWork *kg = vnet_work_new(&G, area);
    double worst = 0;
    for (int p = 0; p < 20; p++) {
        Game2 pos; random_position(&pos, N, 10 + 3 * p, 100 + p);
        vnet_encode(&pos, g3, NULL, vnet_work_input(k));
        vnet_encode(&pos, g3, NULL, vnet_work_input(kg));
        const double d = fabs((double)vnet_forward(&L, w, nbr9, area, k) - vnet_forward(&G, g, nbr9, area, kg));
        if (d > worst) worst = d;
    }
    printf("  grow 2 -> 5 blocks: worst |value difference| over 20 positions %.2e\n", worst);
    check("grown net computes exactly what it did", G.B == 5 && worst == 0);
    g3_free(g3); vnet_work_free(k); vnet_work_free(kg); free(w); free(g);
}

static void test_save_load(void) {
    const VnetLayout L = vnet_layout(8, 3, 5);
    float *w = malloc(sizeof(float) * L.n_params);
    vnet_init(&L, w, 3);
    const char *path = "/tmp/test_cnn_sym.bin", *vpath = "/tmp/test_cnn_sym_vnet1.bin";
    int ok = vnet_save(path, &L, w);
    VnetLayout L2;
    float *w2 = ok ? vnet_load(path, &L2) : NULL;
    ok = w2 && L2.C == 8 && L2.B == 3 && L2.H == 5 && memcmp(w, w2, sizeof(float) * L.n_params) == 0;
    check("save / load round trip (VSYM1)", ok);
    /* A cnn.c file: same header layout, VNET1 magic. */
    FILE *f = fopen(vpath, "wb");
    const int32_t hdr[5] = { 8, 3, 5, VNET_PLANES, L.n_params };
    fwrite("VNET1", 1, 5, f); fwrite(hdr, sizeof hdr, 1, f); fwrite(w, sizeof(float), L.n_params, f); fclose(f);
    VnetLayout L3;
    float *w3 = vnet_load(vpath, &L3);
    check("a VNET1 file is refused", w3 == NULL);
    remove(path); remove(vpath);
    free(w); free(w2); free(w3);
}

int main(void) {
    test_gradients(4, 2, 8);
    test_gradients(32, 1, 8);
    test_invariance();
    test_grow_blocks();
    test_save_load();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0;
}
