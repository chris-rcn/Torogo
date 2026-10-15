/* cnn.c — the toroidal residual value network (see cnn.h). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cnn.h"

#define P VNET_PLANES

VnetLayout vnet_layout(int C, int B, int H) {
    VnetLayout L;
    memset(&L, 0, sizeof L);
    L.C = C; L.B = B; L.H = H;
    int o = 0;
    L.stem_w = o; o += 9 * P * C;
    L.stem_b = o; o += C;
    for (int i = 0; i < B; i++) {
        L.blk_w1[i] = o; o += 9 * C * C;
        L.blk_b1[i] = o; o += C;
        L.blk_w2[i] = o; o += 9 * C * C;
        L.blk_b2[i] = o; o += C;
    }
    L.head_w = o; o += C * H;
    L.head_b = o; o += H;
    L.out_w  = o; o += H;
    L.out_b  = o; o += 1;
    L.n_params = o;
    return L;
}

/* splitmix64 and Box-Muller, for initialisation only. */
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

void vnet_init(const VnetLayout *L, float *w, uint64_t seed) {
    memset(w, 0, sizeof(float) * (size_t)L->n_params);
    uint64_t s = seed ? seed : 1;
    const int C = L->C, H = L->H;
    double sd = sqrt(2.0 / (9.0 * P));
    for (int i = 0; i < 9 * P * C; i++) w[L->stem_w + i] = (float)(sd * sm_normal(&s));
    sd = sqrt(2.0 / (9.0 * C));
    for (int b = 0; b < L->B; b++)
        for (int i = 0; i < 9 * C * C; i++) w[L->blk_w1[b] + i] = (float)(sd * sm_normal(&s));
        /* blk_w2 stays zero: every block starts as the identity */
    sd = sqrt(2.0 / C);
    for (int i = 0; i < C * H; i++) w[L->head_w + i] = (float)(sd * sm_normal(&s));
    sd = 0.1 * sqrt(1.0 / H);
    for (int i = 0; i < H; i++) w[L->out_w + i] = (float)(sd * sm_normal(&s));
}

struct VnetWork {
    int area, C, B, H;
    float *x, *h, *a, *g, *z, y;
    float *dh, *dpre, *da, *dz;
    float *wt;                     /* one block conv's weights transposed per tap */
};

VnetWork *vnet_work_new(const VnetLayout *L, int area) {
    VnetWork *w = calloc(1, sizeof *w);
    w->area = area; w->C = L->C; w->B = L->B; w->H = L->H;
    const size_t AC = (size_t)area * L->C;
    w->x    = calloc((size_t)area * P, sizeof(float));
    w->h    = calloc(AC * (L->B + 1), sizeof(float));
    w->a    = calloc(AC * (L->B ? L->B : 1), sizeof(float));
    w->g    = calloc(L->C, sizeof(float));
    w->z    = calloc(L->H, sizeof(float));
    w->dh   = calloc(AC, sizeof(float));
    w->dpre = calloc(AC, sizeof(float));
    w->da   = calloc(AC, sizeof(float));
    w->dz   = calloc(L->H, sizeof(float));
    w->wt   = calloc((size_t)9 * L->C * L->C, sizeof(float));
    return w;
}

void vnet_work_free(VnetWork *w) {
    if (!w) return;
    free(w->x); free(w->h); free(w->a); free(w->g); free(w->z);
    free(w->dh); free(w->dpre); free(w->da); free(w->dz); free(w->wt); free(w);
}

float *vnet_work_input(VnetWork *w) { return w->x; }

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

void vnet_compact(const Game2 *g, int8_t *code, VnetMeta *meta) {
    const int8_t cur = g->current;
    for (int p = 0; p < g->cap; p++) {
        const int8_t c = g->cells[p];
        if (c == EMPTY) { code[p] = 0; continue; }
        const int lib = g->ls[g->gid[p]];
        const int cls = lib >= 3 ? 2 : lib - 1;
        code[p] = (int8_t)((c == cur ? 1 : 4) + cls);
    }
    meta->ko = g->ko;
    meta->last = g->last_move;
    meta->black_to_move = cur == BLACK;
}

void vnet_encode_compact(const int8_t *code, const VnetMeta *meta, int area, const int32_t *map, float *x) {
    memset(x, 0, sizeof(float) * (size_t)area * P);
    for (int p = 0; p < area; p++) {
        float *xp = x + (size_t)(map ? map[p] : p) * P;
        const int c = code[p];
        if (c == 0) xp[6] = 1;
        else xp[c - 1] = 1;                       /* codes 1-6 -> planes 0-5 */
        if (meta->black_to_move) xp[9] = 1;
    }
    if (meta->ko >= 0)   x[(size_t)(map ? map[meta->ko] : meta->ko) * P + 7] = 1;
    if (meta->last >= 0) x[(size_t)(map ? map[meta->last] : meta->last) * P + 8] = 1;
}

void vnet_encode(const Game2 *g, const int32_t *map, float *x) {
    int8_t code[MAX_CAP];
    VnetMeta meta;
    vnet_compact(g, code, &meta);
    vnet_encode_compact(code, &meta, g->cap, map, x);
}

/* conv_fwd for 32 output channels: each point's 32 sums stay in registers
 * through the whole window.  b == NULL accumulates into out instead of
 * starting from a bias. */
#define CW 32
static void conv_fwd32(const float *restrict in, int Cin, float *restrict out,
                       const float *restrict W, const float *restrict b, const int32_t *nbr9, int area) {
    for (int p = 0; p < area; p++) {
        float acc[CW];
        float *restrict o = out + (size_t)p * CW;
        for (int co = 0; co < CW; co++) acc[co] = b ? b[co] : o[co];
        for (int k = 0; k < 9; k++) {
            const float *restrict ip = in + (size_t)nbr9[p * 9 + k] * Cin;
            const float *restrict Wk = W + (size_t)k * Cin * CW;
            if (Cin == P) {   /* the sparse input planes: skip zeros */
                for (int ci = 0; ci < Cin; ci++) {
                    const float a = ip[ci];
                    if (a == 0) continue;
                    const float *restrict wr = Wk + (size_t)ci * CW;
                    for (int co = 0; co < CW; co++) acc[co] += a * wr[co];
                }
            } else {          /* dense activations: branchless (ReLU zeros are unpredictable) */
                /* Two input channels at a time into two accumulator sets: twice
                 * the independent multiply-add chains in flight. */
                float acc2[CW];
                for (int co = 0; co < CW; co++) acc2[co] = 0;
                int ci = 0;
                for (; ci + 1 < Cin; ci += 2) {
                    const float a0 = ip[ci], a1 = ip[ci + 1];
                    const float *restrict w0 = Wk + (size_t)ci * CW, *restrict w1 = w0 + CW;
                    for (int co = 0; co < CW; co++) { acc[co] += a0 * w0[co]; acc2[co] += a1 * w1[co]; }
                }
                for (; ci < Cin; ci++) {
                    const float a = ip[ci];
                    const float *restrict wr = Wk + (size_t)ci * CW;
                    for (int co = 0; co < CW; co++) acc[co] += a * wr[co];
                }
                for (int co = 0; co < CW; co++) acc[co] += acc2[co];
            }
        }
        for (int co = 0; co < CW; co++) o[co] = acc[co];
    }
}

/* Weight and bias gradients for 32 output channels: per (tap, input channel),
 * the 32 sums over points stay in registers. */
static void conv_dw32(const float *restrict in, int Cin, const float *restrict dout,
                      float *restrict dW, float *restrict db, const int32_t *nbr9, int area) {
    for (int p = 0; p < area; p++) for (int co = 0; co < CW; co++) db[co] += dout[(size_t)p * CW + co];
    for (int k = 0; k < 9; k++) {
        int ci = 0;
        /* Two input channels at a time (two accumulator sets) on dense inputs. */
        if (Cin != P)
            for (; ci + 1 < Cin; ci += 2) {
                float acc0[CW], acc1[CW];
                for (int co = 0; co < CW; co++) acc0[co] = acc1[co] = 0;
                for (int p = 0; p < area; p++) {
                    const float *restrict ip = in + (size_t)nbr9[p * 9 + k] * Cin + ci;
                    const float a0 = ip[0], a1 = ip[1];
                    const float *restrict d = dout + (size_t)p * CW;
                    for (int co = 0; co < CW; co++) { acc0[co] += a0 * d[co]; acc1[co] += a1 * d[co]; }
                }
                float *restrict dw0 = dW + ((size_t)k * Cin + ci) * CW, *restrict dw1 = dw0 + CW;
                for (int co = 0; co < CW; co++) { dw0[co] += acc0[co]; dw1[co] += acc1[co]; }
            }
        for (; ci < Cin; ci++) {
            float acc[CW];
            for (int co = 0; co < CW; co++) acc[co] = 0;
            for (int p = 0; p < area; p++) {
                const float a = in[(size_t)nbr9[p * 9 + k] * Cin + ci];
                if (Cin == P && a == 0) continue;   /* skip only on the sparse input planes */
                const float *restrict d = dout + (size_t)p * CW;
                for (int co = 0; co < CW; co++) acc[co] += a * d[co];
            }
            float *restrict dwr = dW + ((size_t)k * Cin + ci) * CW;
            for (int co = 0; co < CW; co++) dwr[co] += acc[co];
        }
    }
}

/* opp[k]: the tap pointing back, nbr9(nbr9(p, k), opp[k]) == p (the torus
 * window is symmetric), found from the table itself. */
static void tap_opposites(const int32_t *nbr9, int *opp) {
    for (int k = 0; k < 9; k++) {
        opp[k] = -1;
        for (int j = 0; j < 9; j++) if (nbr9[nbr9[k] * 9 + j] == 0) { opp[k] = j; break; }
        if (opp[k] < 0) { fprintf(stderr, "cnn: neighbour table is not symmetric\n"); exit(1); }
    }
}

/* out[p][co] = b[co] + sum_k sum_ci in[nbr9(p,k)][ci] * W[k][ci][co] */
static void conv_fwd(const float *restrict in, int Cin, float *restrict out, int Cout,
                     const float *restrict W, const float *restrict b, const int32_t *nbr9, int area) {
    if (Cout == CW) { conv_fwd32(in, Cin, out, W, b, nbr9, area); return; }
    for (int p = 0; p < area; p++) {
        float *restrict o = out + (size_t)p * Cout;
        for (int co = 0; co < Cout; co++) o[co] = b[co];
        for (int k = 0; k < 9; k++) {
            const float *restrict ip = in + (size_t)nbr9[p * 9 + k] * Cin;
            const float *restrict Wk = W + (size_t)k * Cin * Cout;
            for (int ci = 0; ci < Cin; ci++) {
                const float a = ip[ci];
                if (a == 0) continue;
                const float *restrict wr = Wk + (size_t)ci * Cout;
                for (int co = 0; co < Cout; co++) o[co] += a * wr[co];
            }
        }
    }
}

static void conv_bwd(const float *restrict in, int Cin, const float *restrict dout, int Cout,
                     const float *restrict WT, float *restrict dW, float *restrict db,
                     float *restrict din, const int32_t *nbr9, int area);

/* WT[k][co][ci] = W[k][ci][co]. */
static void transpose_taps(const float *restrict W, int Cin, int Cout, float *restrict WT) {
    for (int k = 0; k < 9; k++)
        for (int ci = 0; ci < Cin; ci++)
            for (int co = 0; co < Cout; co++)
                WT[((size_t)k * Cout + co) * Cin + ci] = W[((size_t)k * Cin + ci) * Cout + co];
}

/* V[j][co][ci] = W[opp[j]][ci][co]: the input gradient of a conv with W is
 * then the forward conv of dout with V — din[p][ci] = sum_j sum_co
 * dout[nbr9(p,j)][co] * W[opp[j]][ci][co]. */
static void flip_transpose_taps(const float *restrict W, int C, const int *opp, float *restrict V) {
    for (int j = 0; j < 9; j++)
        for (int ci = 0; ci < C; ci++)
            for (int co = 0; co < C; co++)
                V[((size_t)j * C + co) * C + ci] = W[((size_t)opp[j] * C + ci) * C + co];
}

/* One block conv's backward with C in = C out: dW, db, and din += W^T dout. */
static void block_conv_bwd(const float *in, const float *dout, int C, const float *W, float *dW, float *db,
                           float *din, const int32_t *nbr9, int area, float *scratch) {
    if (C == CW) {
        int opp[9]; tap_opposites(nbr9, opp);
        conv_dw32(in, C, dout, dW, db, nbr9, area);
        flip_transpose_taps(W, C, opp, scratch);
        conv_fwd32(dout, C, din, scratch, NULL, nbr9, area);   /* accumulates */
        return;
    }
    transpose_taps(W, C, C, scratch);
    conv_bwd(in, C, dout, C, scratch, dW, db, din, nbr9, area);
}

/* Given dout = dL/dout: dW += in (x) dout, db += dout, and din += W^T dout
 * (din may be NULL), with WT = W transposed per tap, [k][co][ci], so the
 * input gradient is a broadcast-add over ci like the forward pass. */
static void conv_bwd(const float *restrict in, int Cin, const float *restrict dout, int Cout,
                     const float *restrict WT, float *restrict dW, float *restrict db,
                     float *restrict din, const int32_t *nbr9, int area) {
    for (int p = 0; p < area; p++) {
        const float *restrict d = dout + (size_t)p * Cout;
        int any = 0;
        for (int co = 0; co < Cout; co++) any |= d[co] != 0;
        if (!any) continue;
        for (int co = 0; co < Cout; co++) db[co] += d[co];
        for (int k = 0; k < 9; k++) {
            const int nb = nbr9[p * 9 + k];
            const float *restrict ip = in + (size_t)nb * Cin;
            float *restrict dWk = dW + (size_t)k * Cin * Cout;
            for (int ci = 0; ci < Cin; ci++) {
                const float a = ip[ci];
                if (a == 0) continue;
                float *restrict dwr = dWk + (size_t)ci * Cout;
                for (int co = 0; co < Cout; co++) dwr[co] += a * d[co];
            }
            if (din) {
                float *restrict dip = din + (size_t)nb * Cin;
                const float *restrict WTk = WT + (size_t)k * Cout * Cin;
                for (int co = 0; co < Cout; co++) {
                    const float dc = d[co];
                    if (dc == 0) continue;
                    const float *restrict wt = WTk + (size_t)co * Cin;
                    for (int ci = 0; ci < Cin; ci++) dip[ci] += dc * wt[ci];
                }
            }
        }
    }
}

float vnet_forward(const VnetLayout *L, const float *w, const int32_t *nbr9, int area, VnetWork *k) {
    const int C = L->C, B = L->B, H = L->H;
    const size_t AC = (size_t)area * C;
    float *h0 = k->h;
    conv_fwd(k->x, P, h0, C, w + L->stem_w, w + L->stem_b, nbr9, area);
    for (size_t j = 0; j < AC; j++) if (h0[j] < 0) h0[j] = 0;
    for (int i = 0; i < B; i++) {
        float *hin = k->h + AC * i, *a = k->a + AC * i, *hout = k->h + AC * (i + 1);
        conv_fwd(hin, C, a, C, w + L->blk_w1[i], w + L->blk_b1[i], nbr9, area);
        for (size_t j = 0; j < AC; j++) if (a[j] < 0) a[j] = 0;
        conv_fwd(a, C, hout, C, w + L->blk_w2[i], w + L->blk_b2[i], nbr9, area);
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
    /* head */
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
    /* blocks, last first: dh holds dL/d(block output) */
    for (int i = B - 1; i >= 0; i--) {
        const float *hin = k->h + AC * i, *a = k->a + AC * i, *hout = k->h + AC * (i + 1);
        for (size_t j = 0; j < AC; j++) k->dpre[j] = hout[j] > 0 ? k->dh[j] : 0;
        memset(k->da, 0, sizeof(float) * AC);
        block_conv_bwd(a, k->dpre, C, w + L->blk_w2[i], G + L->blk_w2[i], G + L->blk_b2[i], k->da, nbr9, area, k->wt);
        for (size_t j = 0; j < AC; j++) if (a[j] <= 0) k->da[j] = 0;
        memcpy(k->dh, k->dpre, sizeof(float) * AC);   /* the skip path */
        block_conv_bwd(hin, k->da, C, w + L->blk_w1[i], G + L->blk_w1[i], G + L->blk_b1[i], k->dh, nbr9, area, k->wt);
    }
    /* stem */
    const float *h0 = k->h;
    for (size_t j = 0; j < AC; j++) k->dpre[j] = h0[j] > 0 ? k->dh[j] : 0;
    if (C == CW) conv_dw32(k->x, P, k->dpre, G + L->stem_w, G + L->stem_b, nbr9, area);
    else conv_bwd(k->x, P, k->dpre, C, NULL, G + L->stem_w, G + L->stem_b, NULL, nbr9, area);   /* no input gradient */
}

int vnet_save(const char *path, const VnetLayout *L, const float *w) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "vnet_save: cannot open %s\n", path); return 0; }
    const int32_t hdr[5] = { L->C, L->B, L->H, P, L->n_params };
    int ok = fwrite("VNET1", 1, 5, f) == 5 && fwrite(hdr, sizeof hdr, 1, f) == 1 &&
             fwrite(w, sizeof(float), (size_t)L->n_params, f) == (size_t)L->n_params;
    ok &= fclose(f) == 0;
    if (!ok) fprintf(stderr, "vnet_save: write to %s failed\n", path);
    return ok;
}

float *vnet_load(const char *path, VnetLayout *L) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vnet_load: cannot open %s\n", path); return NULL; }
    char magic[5]; int32_t hdr[5];
    if (fread(magic, 1, 5, f) != 5 || memcmp(magic, "VNET1", 5) != 0 || fread(hdr, sizeof hdr, 1, f) != 1) {
        fprintf(stderr, "vnet_load: %s is not a VNET1 file\n", path); fclose(f); return NULL;
    }
    if (hdr[3] != P) { fprintf(stderr, "vnet_load: %s has %d input planes, this build %d\n", path, hdr[3], P); fclose(f); return NULL; }
    if (hdr[1] > VNET_MAX_BLOCKS) { fprintf(stderr, "vnet_load: %s has %d blocks (max %d)\n", path, hdr[1], VNET_MAX_BLOCKS); fclose(f); return NULL; }
    *L = vnet_layout(hdr[0], hdr[1], hdr[2]);
    if (L->n_params != hdr[4]) { fprintf(stderr, "vnet_load: %s header says %d params, layout %d\n", path, hdr[4], L->n_params); fclose(f); return NULL; }
    float *w = malloc(sizeof(float) * (size_t)L->n_params);
    if (fread(w, sizeof(float), (size_t)L->n_params, f) != (size_t)L->n_params) {
        fprintf(stderr, "vnet_load: %s is truncated\n", path); free(w); fclose(f); return NULL;
    }
    fclose(f);
    return w;
}
