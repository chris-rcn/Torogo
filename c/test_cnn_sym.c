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

/* The neighbour-arrangement planes (after cnn.c's VNET_PLANES): exactly one
 * per point, and two own stones opposite vs adjacent land on different ones. */
static void test_orbit_planes(void) {
#ifdef VSYM_DIAG
    const int N = 9, area = N * N, PL = VNET_PLANES + 42;
#elif defined(VSYM_CORNER)
    const int N = 9, area = N * N, PL = VNET_PLANES + 21 + 18;
#else
    const int N = 9, area = N * N, PL = VNET_PLANES + 21;
#endif
    g2_init_topology(N);
    Game2 g;
    g2_parse_board(&g,
        "9 . . . . . . . . .\n"
        "8 . . . . . . . . .\n"
        "7 . . X . . . . . .\n"
        "6 . . . . . . X . .\n"
        "5 . . X . . X . . .\n"
        "4 . . . . . . . . .\n"
        "3 . . . . . . . . .\n"
        "2 . . . . . . . . .\n"
        "1 . . . . . . . . .\n",
        BLACK);
    const int opposite = 5 * N + 2, adjacent = 4 * N + 6;   /* c6 (X above and below), g5 (X above and left) */
    Game3 *g3 = g3_new(N);
    static float x[MAX_CAP * (VNET_PLANES + 42)];   /* the largest variant */
    vnet_encode(&g, g3, NULL, x);
    int one_each = 1, po = -1, pa = -1;
    for (int p = 0; p < area; p++) {
        int n = 0;
        for (int q = VNET_PLANES; q < VNET_PLANES + 21; q++) if (x[p * PL + q] != 0) { n++; if (p == opposite) po = q; if (p == adjacent) pa = q; }
        if (n != 1) one_each = 0;
#ifdef VSYM_DIAG
        n = 0;
        for (int q = VNET_PLANES + 21; q < PL; q++) if (x[p * PL + q] != 0) n++;
        if (n != 1) one_each = 0;
#endif
    }
    printf("  orbit planes: one per point %s; c6 (own above + below) plane %d, g5 (own above + left) plane %d\n",
           one_each ? "yes" : "NO", po - VNET_PLANES, pa - VNET_PLANES);
    check("exactly one neighbour-arrangement plane per point", one_each);
    check("own stones opposite vs adjacent: different planes", po >= 0 && pa >= 0 && po != pa);
#ifdef VSYM_CORNER
    /* Corner L-shapes: 1 to 4 classes set per point; own stones at N + NE
     * (one corner) vs N + SW (two corners) set different planes. */
    {
        int ok_count = 1;
        for (int p = 0; p < area; p++) {
            int n = 0;
            for (int q = VNET_PLANES + 21; q < PL; q++) n += x[p * PL + q] != 0;
            if (n < 1 || n > 4) ok_count = 0;
        }
        check("1 to 4 corner classes per point", ok_count);
        Game2 h;
        g2_parse_board(&h,
            "9 . . . . . . . . .\n"
            "8 . . . . . . . . .\n"
            "7 . . X X . . . . .\n"
            "6 . . . . . . X . .\n"
            "5 . . . . . . . . .\n"
            "4 . . . . . X . . .\n"
            "3 . . . . . . . . .\n"
            "2 . . . . . . . . .\n"
            "1 . . . . . . . . .\n",
            BLACK);
        const int a = 5 * N + 2, b = 4 * N + 6;   /* c6: X at c7 (N) and d7 (NE); g5: X at g6 (N) and f4 (SW) */
        vnet_encode(&h, g3, NULL, x);
        int same = 1;
        for (int q = VNET_PLANES + 21; q < PL; q++) if ((x[a * PL + q] != 0) != (x[b * PL + q] != 0)) same = 0;
        printf("  corner planes: N + NE and N + SW %s\n", same ? "SAME" : "differ");
        check("own stones at N + NE vs N + SW: different corner planes", !same);
    }
#endif
#ifdef VSYM_DIAG
    /* Diagonals: own stones on opposite corners vs on the same side. */
    {
        Game2 h;
        g2_parse_board(&h,
            "9 . . . . . . . . .\n"
            "8 . . . . . . . . .\n"
            "7 . X . . . . . . .\n"
            "6 . . . . . X . X .\n"
            "5 . . . X . . . . .\n"
            "4 . . . . . . . . .\n"
            "3 . . . . . . . . .\n"
            "2 . . . . . . . . .\n"
            "1 . . . . . . . . .\n",
            BLACK);
        const int dopp = 5 * N + 2, dsame = 4 * N + 6;   /* c6 (X at b7 and d5: opposite corners), g5 (X at f6 and h6: same side) */
        vnet_encode(&h, g3, NULL, x);
        int qo = -1, qs = -1;
        for (int q = VNET_PLANES + 21; q < PL; q++) { if (x[dopp * PL + q] != 0) qo = q; if (x[dsame * PL + q] != 0) qs = q; }
        printf("  diagonal orbit planes: c6 (opposite corners) plane %d, g5 (same side) plane %d\n", qo - VNET_PLANES - 21, qs - VNET_PLANES - 21);
        check("own stones on opposite vs same-side diagonals: different planes", qo >= 0 && qs >= 0 && qo != qs);
    }
#endif
    g3_free(g3);
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
    test_orbit_planes();
    test_save_load();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0;
}
