/*
 * cnn_sym.c — EXPERIMENT: the value network of cnn.h with symmetric 3x3
 * kernels.  Every kernel has three weights per (input, output) channel pair:
 * the centre, one shared by the 4 orthogonal neighbours, one shared by the 4
 * diagonals.  Such a kernel looks the same under all 8 rotations/reflections,
 * so — with the per-point ReLUs, the residual adds and the head's mean over
 * the board — the whole net is exactly D4-invariant for any weights, as cnn.c
 * is translation-invariant; the trainer's symmetry augmentation becomes a
 * no-op.  The price: a single kernel cannot see direction.
 *
 * A drop-in for cnn.c: the same cnn.h API, so train_vnet_supervised.c and
 * vnet_tool.c build against it unchanged (c/build.sh: train_vnet_sym.bin).
 * VnetLayout keeps cnn.c's field names; the conv weights are W[3][Cin][Cout]
 * (tap class: 0 centre, 1 orthogonal, 2 diagonal).  Files are "VSYM1", which
 * cnn.c's loader refuses and this one requires, so the two kinds never mix.
 *
 * Each conv is a dense product over 3*Cin features per point — the point's
 * input, the sum over its orthogonal neighbours, the sum over its diagonals —
 * a third of cnn.c's 9*Cin.  The board encoding is cnn.c's, copied.
 */
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cnn.h"
#include "ladder2.h"
#include "fatal.h"

#define P VNET_PLANES
#define T 3   /* tap classes */

VnetLayout vnet_layout(int C, int B, int H) {
    VnetLayout L;
    memset(&L, 0, sizeof L);
    L.C = C; L.B = B; L.H = H;
    int o = 0;
    L.stem_w = o; o += T * P * C;
    L.stem_b = o; o += C;
    for (int i = 0; i < B; i++) {
        L.blk_w1[i] = o; o += T * C * C;
        L.blk_b1[i] = o; o += C;
        L.blk_w2[i] = o; o += T * C * C;
        L.blk_b2[i] = o; o += C;
    }
    L.head_w = o; o += C * H;
    L.head_b = o; o += H;
    L.out_w  = o; o += H;
    L.out_b  = o; o += 1;
    L.n_params = o;
    return L;
}

static uint64_t sm_next(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static double sm_normal(uint64_t *s) {
    double u1 = ((sm_next(s) >> 11) + 1.0) / 9007199254740993.0;
    double u2 = (sm_next(s) >> 11) / 9007199254740992.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* He-normal: a class's weight reaches 1 (centre) or 4 points, 9 in all, so
 * each weight gets cnn.c's variance 2 / (9 * fan-in channels). */
void vnet_init(const VnetLayout *L, float *w, uint64_t seed) {
    memset(w, 0, sizeof(float) * (size_t)L->n_params);
    uint64_t s = seed ? seed : 1;
    const int C = L->C, H = L->H;
    double sd = sqrt(2.0 / (9.0 * P));
    for (int i = 0; i < T * P * C; i++) w[L->stem_w + i] = (float)(sd * sm_normal(&s));
    sd = sqrt(2.0 / (9.0 * C));
    for (int b = 0; b < L->B; b++)
        for (int i = 0; i < T * C * C; i++) w[L->blk_w1[b] + i] = (float)(sd * sm_normal(&s));
        /* blk_w2 stays zero: every block starts as the identity */
    sd = sqrt(2.0 / C);
    for (int i = 0; i < C * H; i++) w[L->head_w + i] = (float)(sd * sm_normal(&s));
    sd = 0.1 * sqrt(1.0 / H);
    for (int i = 0; i < H; i++) w[L->out_w + i] = (float)(sd * sm_normal(&s));
}

float *vnet_grow_blocks(const VnetLayout *L, const float *w, int B, uint64_t seed, VnetLayout *G) {
    *G = vnet_layout(L->C, B, L->H);
    float *g = malloc(sizeof(float) * (size_t)G->n_params);
    vnet_init(G, g, seed);
    const size_t C = (size_t)L->C, H = (size_t)L->H, conv = T * C * C;
    memcpy(g + G->stem_w, w + L->stem_w, sizeof(float) * T * P * C);
    memcpy(g + G->stem_b, w + L->stem_b, sizeof(float) * C);
    for (int b = 0; b < L->B; b++) {
        memcpy(g + G->blk_w1[b], w + L->blk_w1[b], sizeof(float) * conv);
        memcpy(g + G->blk_b1[b], w + L->blk_b1[b], sizeof(float) * C);
        memcpy(g + G->blk_w2[b], w + L->blk_w2[b], sizeof(float) * conv);
        memcpy(g + G->blk_b2[b], w + L->blk_b2[b], sizeof(float) * C);
    }
    memcpy(g + G->head_w, w + L->head_w, sizeof(float) * C * H);
    memcpy(g + G->head_b, w + L->head_b, sizeof(float) * H);
    memcpy(g + G->out_w, w + L->out_w, sizeof(float) * H);
    memcpy(g + G->out_b, w + L->out_b, sizeof(float));
    return g;
}

struct VnetWork {
    int area, C, B, H;
    float *x, *h, *a, *g, *z, y;
    float *dh, *dpre, *da, *dz;
    float *F;                      /* one conv's features, area x 3*max(P, C) */
    float *gF;                     /* their gradient */
};

VnetWork *vnet_work_new(const VnetLayout *L, int area) {
    VnetWork *w = calloc(1, sizeof *w);
    w->area = area; w->C = L->C; w->B = L->B; w->H = L->H;
    const size_t AC = (size_t)area * L->C, mx = L->C > P ? L->C : P;
    w->x    = calloc((size_t)area * P, sizeof(float));
    w->h    = calloc(AC * (L->B + 1), sizeof(float));
    w->a    = calloc(AC * (L->B ? L->B : 1), sizeof(float));
    w->g    = calloc(L->C, sizeof(float));
    w->z    = calloc(L->H, sizeof(float));
    w->dh   = calloc(AC, sizeof(float));
    w->dpre = calloc(AC, sizeof(float));
    w->da   = calloc(AC, sizeof(float));
    w->dz   = calloc(L->H, sizeof(float));
    w->F    = calloc((size_t)area * T * mx, sizeof(float));
    w->gF   = calloc((size_t)area * T * mx, sizeof(float));
    return w;
}

void vnet_work_free(VnetWork *w) {
    if (!w) return;
    free(w->x); free(w->h); free(w->a); free(w->g); free(w->z);
    free(w->dh); free(w->dpre); free(w->da); free(w->dz); free(w->F); free(w->gF); free(w);
}

float *vnet_work_input(VnetWork *w) { return w->x; }

/* ── Board encoding: cnn.c's, copied ───────────────────────────────────── */

void vnet_topology(int N, int32_t *nbr9) {
    const int area = N * N;
    for (int p = 0; p < area; p++) {
        nbr9[p * 9] = p;
        for (int d = 0; d < 4; d++) {
            nbr9[p * 9 + 1 + d] = g2_nbr[p * 4 + d];
            nbr9[p * 9 + 5 + d] = g2_dnbr[p * 4 + d];
        }
    }
}

void vnet_sym_map(int N, int32_t *map) {
    const int area = N * N;
    for (int t = 0; t < 8; t++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                int a = x, b = y;
                if (t & 4) { int tmp = a; a = b; b = tmp; }
                if (t & 1) a = N - 1 - a;
                if (t & 2) b = N - 1 - b;
                map[t * area + y * N + x] = b * N + a;
            }
}

void vnet_compact(const Game2 *g, Game3 *g3, int8_t *code, VnetMeta *meta) {
    const int8_t cur = g->current;
    for (int p = 0; p < g->cap; p++) {
        const int8_t c = g->cells[p];
        if (c == EMPTY) { code[p] = 0; continue; }
        const int lib = g->ls[g->gid[p]];
        const int cls = lib >= 3 ? 2 : lib - 1;
        code[p] = (int8_t)((c == cur ? 1 : 4) + cls);
    }
    if (!g3_from_game2(g3, g)) FATAL("vnet_compact: position not reachable by legal play\n");
    Ladder2Result st[MAX_CAP];
    const int n = ladder2_get_all_statuses(g3, 1, st, MAX_CAP);
    /* Own chain: not urgent = safe, urgent = 2 if a move saves it else 1.
     * Opponent chain: not urgent = dead even moving first (1), urgent = 2 if a
     * move captures it, else safe. */
    for (int i = 0; i < n; i++) {
        const Ladder2Status *s = &st[i].status;
        const int urgent = s->urgent_count > 0 || !s->mover_succeeds;
        const int own = st[i].color == cur;
        const int lad = own ? (!urgent ? 0 : s->mover_succeeds ? 2 : 1)
                            : (!urgent ? 1 : s->mover_succeeds ? 2 : 0);
        if (!lad) continue;
        for (int p = 0; p < g3->cap; p++)
            if (g3->gid[p] == st[i].gid) code[p] |= (int8_t)(lad << 3);
    }
    meta->ko = g->ko;
    meta->last = g->last_move;
    meta->black_to_move = cur == BLACK;
}

void vnet_encode_compact(const int8_t *code, const VnetMeta *meta, int area, const int32_t *map, float *x) {
    memset(x, 0, sizeof(float) * (size_t)area * P);
    for (int p = 0; p < area; p++) {
        float *xp = x + (size_t)(map ? map[p] : p) * P;
        const int c = code[p] & 7, lad = code[p] >> 3;
        if (c == 0) xp[6] = 1;
        else xp[c - 1] = 1;                       /* codes 1-6 -> planes 0-5 */
        if (meta->black_to_move) xp[9] = 1;
        if (lad) xp[10 + (c >= 4 ? 2 : 0) + (lad - 1)] = 1;   /* own lost / urgent, opponent dead / urgent */
    }
    if (meta->ko >= 0)   x[(size_t)(map ? map[meta->ko] : meta->ko) * P + 7] = 1;
    if (meta->last >= 0) x[(size_t)(map ? map[meta->last] : meta->last) * P + 8] = 1;
}

void vnet_encode(const Game2 *g, Game3 *g3, const int32_t *map, float *x) {
    int8_t code[MAX_CAP];
    VnetMeta meta;
    vnet_compact(g, g3, code, &meta);
    vnet_encode_compact(code, &meta, g->cap, map, x);
}

/* ── Symmetric convolution ─────────────────────────────────────────────── */

/* F[p] = (in[p], sum of in over p's 4 orthogonal neighbours, sum over its 4
 * diagonals): 3*Cin features per point. */
static void sym_feats(const float *restrict in, int Cin, const int32_t *nbr9, int area, float *restrict F) {
    for (int p = 0; p < area; p++) {
        float *restrict f = F + (size_t)p * T * Cin;
        const float *restrict c = in + (size_t)p * Cin;
        for (int ci = 0; ci < Cin; ci++) { f[ci] = c[ci]; f[Cin + ci] = 0; f[2 * Cin + ci] = 0; }
        for (int k = 1; k <= 8; k++) {
            const float *restrict q = in + (size_t)nbr9[p * 9 + k] * Cin;
            float *restrict d = f + (k <= 4 ? Cin : 2 * Cin);
            for (int ci = 0; ci < Cin; ci++) d[ci] += q[ci];
        }
    }
}

/* out[p] = b + F[p] . W, W[3*Cin][Cout]. */
#define CW 32
static void sym_conv_fwd(const float *restrict in, int Cin, float *restrict out, int Cout,
                         const float *restrict W, const float *restrict b, const int32_t *nbr9, int area,
                         float *restrict F) {
    sym_feats(in, Cin, nbr9, area, F);
    const int J = T * Cin;
    for (int p = 0; p < area; p++) {
        const float *restrict f = F + (size_t)p * J;
        float *restrict o = out + (size_t)p * Cout;
        if (Cout == CW) {   /* the 32 sums in registers */
            float acc[CW];
            for (int co = 0; co < CW; co++) acc[co] = b[co];
            for (int j = 0; j < J; j++) {
                const float a = f[j];
                if (a == 0) continue;   /* the input planes, their sums and ReLU zeros */
                const float *restrict wr = W + (size_t)j * CW;
                for (int co = 0; co < CW; co++) acc[co] += a * wr[co];
            }
            for (int co = 0; co < CW; co++) o[co] = acc[co];
        } else {
            for (int co = 0; co < Cout; co++) o[co] = b[co];
            for (int j = 0; j < J; j++) {
                const float a = f[j];
                if (a == 0) continue;
                const float *restrict wr = W + (size_t)j * Cout;
                for (int co = 0; co < Cout; co++) o[co] += a * wr[co];
            }
        }
    }
}

/* Gradients of sym_conv_fwd for dout: ADDS dW, db, and (din != NULL) the
 * input gradient: each feature's gradient goes back to the points that
 * summed into it — p itself, its orthogonal and its diagonal neighbours (the
 * neighbour relation is symmetric on the torus). */
static void sym_conv_bwd(const float *restrict in, int Cin, const float *restrict dout, int Cout,
                         const float *restrict W, float *restrict dW, float *restrict db, float *restrict din,
                         const int32_t *nbr9, int area, float *restrict F, float *restrict gF) {
    sym_feats(in, Cin, nbr9, area, F);
    const int J = T * Cin;
    for (int p = 0; p < area; p++) {
        const float *restrict f = F + (size_t)p * J;
        const float *restrict d = dout + (size_t)p * Cout;
        for (int co = 0; co < Cout; co++) db[co] += d[co];
        for (int j = 0; j < J; j++) {
            const float a = f[j];
            if (a == 0) continue;
            float *restrict gw = dW + (size_t)j * Cout;
            for (int co = 0; co < Cout; co++) gw[co] += a * d[co];
        }
        if (din) {
            float *restrict g = gF + (size_t)p * J;
            for (int j = 0; j < J; j++) {
                const float *restrict wr = W + (size_t)j * Cout;
                float s = 0;
                for (int co = 0; co < Cout; co++) s += wr[co] * d[co];
                g[j] = s;
            }
        }
    }
    if (!din) return;
    for (int p = 0; p < area; p++) {
        float *restrict o = din + (size_t)p * Cin;
        const float *restrict g = gF + (size_t)p * J;
        for (int ci = 0; ci < Cin; ci++) o[ci] += g[ci];
        for (int k = 1; k <= 8; k++) {
            const float *restrict q = gF + (size_t)nbr9[p * 9 + k] * J + (k <= 4 ? Cin : 2 * Cin);
            for (int ci = 0; ci < Cin; ci++) o[ci] += q[ci];
        }
    }
}

float vnet_forward(const VnetLayout *L, const float *w, const int32_t *nbr9, int area, VnetWork *k) {
    const int C = L->C, B = L->B, H = L->H;
    const size_t AC = (size_t)area * C;
    float *h0 = k->h;
    sym_conv_fwd(k->x, P, h0, C, w + L->stem_w, w + L->stem_b, nbr9, area, k->F);
    for (size_t j = 0; j < AC; j++) if (h0[j] < 0) h0[j] = 0;
    for (int i = 0; i < B; i++) {
        float *hin = k->h + AC * i, *a = k->a + AC * i, *hout = k->h + AC * (i + 1);
        sym_conv_fwd(hin, C, a, C, w + L->blk_w1[i], w + L->blk_b1[i], nbr9, area, k->F);
        for (size_t j = 0; j < AC; j++) if (a[j] < 0) a[j] = 0;
        sym_conv_fwd(a, C, hout, C, w + L->blk_w2[i], w + L->blk_b2[i], nbr9, area, k->F);
        for (size_t j = 0; j < AC; j++) { const float v = hout[j] + hin[j]; hout[j] = v > 0 ? v : 0; }
    }
    const float *hB = k->h + AC * B;
    for (int c = 0; c < C; c++) k->g[c] = 0;
    for (int p = 0; p < area; p++) for (int c = 0; c < C; c++) k->g[c] += hB[(size_t)p * C + c];
    for (int c = 0; c < C; c++) k->g[c] /= area;
    float y = w[L->out_b];
    for (int j = 0; j < H; j++) {
        float s = w[L->head_b + j];
        for (int c = 0; c < C; c++) s += k->g[c] * w[L->head_w + c * H + j];
        k->z[j] = s > 0 ? s : 0;
        y += w[L->out_w + j] * k->z[j];
    }
    k->y = y;
    return 1.0f / (1.0f + expf(-y));
}

void vnet_backward(const VnetLayout *L, const float *w, const int32_t *nbr9, int area,
                   VnetWork *k, float dy, float *G) {
    const int C = L->C, B = L->B, H = L->H;
    const size_t AC = (size_t)area * C;
    G[L->out_b] += dy;
    for (int j = 0; j < H; j++) {
        G[L->out_w + j] += dy * k->z[j];
        k->dz[j] = k->z[j] > 0 ? dy * w[L->out_w + j] : 0;
        G[L->head_b + j] += k->dz[j];
        for (int c = 0; c < C; c++) G[L->head_w + c * H + j] += k->g[c] * k->dz[j];
    }
    for (int c = 0; c < C; c++) {
        float s = 0;
        for (int j = 0; j < H; j++) s += w[L->head_w + c * H + j] * k->dz[j];
        s /= area;
        for (int p = 0; p < area; p++) k->dh[(size_t)p * C + c] = s;
    }
    for (int i = B - 1; i >= 0; i--) {
        const float *hin = k->h + AC * i, *a = k->a + AC * i, *hout = k->h + AC * (i + 1);
        for (size_t j = 0; j < AC; j++) k->dpre[j] = hout[j] > 0 ? k->dh[j] : 0;
        memset(k->da, 0, sizeof(float) * AC);
        sym_conv_bwd(a, C, k->dpre, C, w + L->blk_w2[i], G + L->blk_w2[i], G + L->blk_b2[i], k->da, nbr9, area, k->F, k->gF);
        for (size_t j = 0; j < AC; j++) if (a[j] <= 0) k->da[j] = 0;
        memcpy(k->dh, k->dpre, sizeof(float) * AC);   /* the skip path */
        sym_conv_bwd(hin, C, k->da, C, w + L->blk_w1[i], G + L->blk_w1[i], G + L->blk_b1[i], k->dh, nbr9, area, k->F, k->gF);
    }
    const float *h0 = k->h;
    for (size_t j = 0; j < AC; j++) k->dpre[j] = h0[j] > 0 ? k->dh[j] : 0;
    sym_conv_bwd(k->x, P, k->dpre, C, w + L->stem_w, G + L->stem_w, G + L->stem_b, NULL, nbr9, area, k->F, k->gF);
}

#ifndef VNET_NO_FILE_IO
/* Atomic, as cnn.c's: <path>.tmp, then rename. */
int vnet_save(const char *path, const VnetLayout *L, const float *w) {
    char tmp[1024];
    if (snprintf(tmp, sizeof tmp, "%s.tmp", path) >= (int)sizeof tmp) { fprintf(stderr, "vnet_save: path too long: %s\n", path); return 0; }
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "vnet_save: cannot open %s: %s\n", tmp, strerror(errno)); return 0; }
    const int32_t hdr[5] = { L->C, L->B, L->H, P, L->n_params };
    int ok = fwrite("VSYM1", 1, 5, f) == 5 && fwrite(hdr, sizeof hdr, 1, f) == 1 &&
             fwrite(w, sizeof(float), (size_t)L->n_params, f) == (size_t)L->n_params;
    ok &= fclose(f) == 0;
    if (!ok) { fprintf(stderr, "vnet_save: write to %s failed; %s not updated\n", tmp, path); remove(tmp); return 0; }
    if (rename(tmp, path) != 0) { fprintf(stderr, "vnet_save: rename %s -> %s failed: %s\n", tmp, path, strerror(errno)); remove(tmp); return 0; }
    return 1;
}

float *vnet_load(const char *path, VnetLayout *L) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vnet_load: cannot open %s\n", path); return NULL; }
    char magic[5]; int32_t hdr[5];
    if (fread(magic, 1, 5, f) != 5 || fread(hdr, sizeof hdr, 1, f) != 1) { fprintf(stderr, "vnet_load: %s: short file\n", path); fclose(f); return NULL; }
    if (memcmp(magic, "VSYM1", 5) != 0) {
        fprintf(stderr, "vnet_load: %s is not a symmetric-kernel net (VSYM1)%s\n", path,
                memcmp(magic, "VNET1", 5) == 0 ? ": it is a cnn.c net (VNET1)" : "");
        fclose(f); return NULL;
    }
    if (hdr[3] != P) { fprintf(stderr, "vnet_load: %s has %d input planes, this build %d\n", path, hdr[3], P); fclose(f); return NULL; }
    if (hdr[1] < 0 || hdr[1] > VNET_MAX_BLOCKS) { fprintf(stderr, "vnet_load: %s has %d blocks (max %d)\n", path, hdr[1], VNET_MAX_BLOCKS); fclose(f); return NULL; }
    *L = vnet_layout(hdr[0], hdr[1], hdr[2]);
    if (L->n_params != hdr[4]) { fprintf(stderr, "vnet_load: %s: %d parameters, layout says %d\n", path, hdr[4], L->n_params); fclose(f); return NULL; }
    float *w = malloc(sizeof(float) * (size_t)L->n_params);
    if (fread(w, sizeof(float), (size_t)L->n_params, f) != (size_t)L->n_params) { fprintf(stderr, "vnet_load: %s: short file\n", path); free(w); fclose(f); return NULL; }
    fclose(f);
    return w;
}
#endif /* VNET_NO_FILE_IO */
