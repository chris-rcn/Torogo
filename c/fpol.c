/* fpol.c — featurepol stone8AdjLib3 policy (see fpol.h). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "fpol.h"

/* ── Hashing: featurepol-lib.js's _mix32 / _hashCombine / _hashStr ─────────── */

static uint32_t mix32(uint32_t x) {
    x = (x ^ (x >> 16)) * 0x45d9f3bu;
    x = (x ^ (x >> 16)) * 0x45d9f3bu;
    return x ^ (x >> 16);
}
static uint32_t hash_combine(uint32_t h, uint32_t v) { return mix32(h ^ mix32(v)); }
static uint32_t hash_str(const char *s) {
    uint32_t h = 0x811c9dc5u;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 0x01000193u;
    return h;
}

/* ── Model: key -> weight, open addressing ──────────────────────────────────── */

#define FPOL_CAP  3                       /* liberty cap of stone8AdjLib3 */
#define FPOL_R    (2 * FPOL_CAP + 1)      /* orthogonal radix */

static uint32_t *tab_keys = NULL;          /* 0 slot = empty (key 0 kept aside) */
static float    *tab_vals = NULL;
static uint32_t  tab_mask = 0;
static bool      has_zero_key = false;
static float     zero_key_val = 0.0f;
static uint32_t  salt_space, salt_term;

static void tab_put(uint32_t k, float v) {
    if (k == 0) { has_zero_key = true; zero_key_val = v; return; }
    uint32_t i = mix32(k) & tab_mask;
    while (tab_keys[i] != 0 && tab_keys[i] != k) i = (i + 1) & tab_mask;
    tab_keys[i] = k; tab_vals[i] = v;
}
static float tab_get(uint32_t k) {
    if (k == 0) return has_zero_key ? zero_key_val : 0.0f;
    uint32_t i = mix32(k) & tab_mask;
    while (tab_keys[i] != 0) { if (tab_keys[i] == k) return tab_vals[i]; i = (i + 1) & tab_mask; }
    return 0.0f;                           /* unknown pattern: no weight */
}

static int b64_val(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}
/* Decode base64 text [s, s+len) into out; returns bytes written. */
static size_t b64_decode(const char *s, size_t len, uint8_t *out) {
    uint32_t acc = 0; int bits = 0; size_t n = 0;
    for (size_t i = 0; i < len; i++) {
        int v = b64_val((unsigned char)s[i]);
        if (v < 0) continue;               /* '=' padding */
        acc = (acc << 6) | (uint32_t)v; bits += 6;
        if (bits >= 8) { bits -= 8; out[n++] = (uint8_t)(acc >> bits); }
    }
    return n;
}

static const char *field(const char *buf, const char *name, const char *path) {
    const char *p = strstr(buf, name);
    if (!p) { fprintf(stderr, "fpol: %s has no '%s'\n", path, name); exit(1); }
    return p + strlen(name);
}

bool fpol_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "fpol: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "fpol: short read on %s\n", path); exit(1); }
    buf[len] = '\0';
    fclose(f);

    const char *sp = field(buf, "const spec = \"", path);
    if (strncmp(sp, "stone8AdjLib3\"", 14) != 0) {
        const char *e = strchr(sp, '"');
        fprintf(stderr, "fpol: %s has spec '%.*s'; only stone8AdjLib3 is supported\n", path, e ? (int)(e - sp) : 20, sp);
        exit(1);
    }
    const int count = atoi(field(buf, "const count = ", path));
    const double scale = strtod(field(buf, "const scale = ", path), NULL);
    const char *b = field(buf, "const b64 = '", path);
    const char *e = strchr(b, '\'');
    if (count <= 0 || !(scale > 0) || !e) { fprintf(stderr, "fpol: %s: bad count/scale/b64\n", path); exit(1); }

    uint8_t *bytes = malloc((size_t)(e - b));
    const size_t nb = b64_decode(b, (size_t)(e - b), bytes);
    if (nb < (size_t)count * 6) { fprintf(stderr, "fpol: %s: b64 holds %zu bytes, need %d\n", path, nb, count * 6); exit(1); }

    uint32_t cap = 1; while (cap < (uint32_t)count * 2) cap <<= 1;
    free(tab_keys); free(tab_vals);
    tab_keys = calloc(cap, sizeof *tab_keys); tab_vals = calloc(cap, sizeof *tab_vals);
    tab_mask = cap - 1; has_zero_key = false;
    const double inv = 1.0 / scale;       /* as the JS loader: qval * (1 / scale), stored as float32 */
    for (int i = 0; i < count; i++) {
        int32_t k; int16_t q;
        memcpy(&k, bytes + 4 * i, 4);                       /* little-endian, as the JS typed arrays */
        memcpy(&q, bytes + (size_t)count * 4 + 2 * i, 2);
        tab_put((uint32_t)k, (float)((double)q * inv));
    }
    free(bytes); free(buf);
    salt_space = hash_str("space:stone8AdjLib3");
    salt_term  = hash_str("stone8AdjLib3");
    return true;
}

/* ── Nearest cells: featurepol-lib.js's _NEAR_OFFSETS[0..7] (dy, dx) ─────────── */

static const int NEAR_DY[8] = { -1, 0, 1, 0, -1, 1, 1, -1 };
static const int NEAR_DX[8] = { 0, 1, 0, -1, 1, 1, -1, -1 };
static int32_t near_tab[MAX_CAP * 8];
static int     near_N = 0;

static void ensure_near(int N) {
    if (near_N == N) return;
    for (int idx = 0; idx < N * N; idx++) {
        const int r = idx / N, c = idx % N;
        for (int k = 0; k < 8; k++)
            near_tab[idx * 8 + k] = ((r + NEAR_DY[k] + N) % N) * N + (c + NEAR_DX[k] + N) % N;
    }
    near_N = N;
}

/* _canon8AdjLib: min over the 8 D4 symmetries of the mixed-radix encoding. */
static uint32_t canon8adjlib(const int *c) {
    const uint32_t R = FPOL_R;
#define ENC(a, b, d, e, f, g, h, i) ((((((((uint32_t)c[a] * R + c[b]) * R + c[d]) * R + c[e]) * 3 + c[f]) * 3 + c[g]) * 3 + c[h]) * 3 + c[i])
    uint32_t best = ENC(0,1,2,3, 4,5,6,7), v;
    v = ENC(1,2,3,0, 5,6,7,4); if (v < best) best = v;
    v = ENC(2,3,0,1, 6,7,4,5); if (v < best) best = v;
    v = ENC(3,0,1,2, 7,4,5,6); if (v < best) best = v;
    v = ENC(0,3,2,1, 7,6,5,4); if (v < best) best = v;
    v = ENC(2,1,0,3, 5,4,7,6); if (v < best) best = v;
    v = ENC(3,2,1,0, 6,5,4,7); if (v < best) best = v;
    v = ENC(1,0,3,2, 4,7,6,5); if (v < best) best = v;
#undef ENC
    return best;
}

/* Not vectorised: under -ffast-math GCC would call glibc's vector exp
 * (libmvec), which is off by an ulp from the scalar exp and V8's Math.exp,
 * and that ulp flips float32 priors often enough to fork a search from JS. */
__attribute__((optimize("no-tree-vectorize")))
int fpol_eval(const Game2 *g, FpolState *st, double temperature) {
    ensure_near(g->N);
    const int8_t cur = g->current;
    int n = 0;
    double maxL = -DBL_MAX; int maxI = 0;
    for (int ei = 0; ei < g->empty_count; ei++) {
        const int idx = g->empty_cells[ei];
        if (!g2_is_legal(g, idx) || g2_is_true_eye_at(g, idx)) continue;
        const int32_t *nn = near_tab + idx * 8;
        int cv[8];
        for (int i = 0; i < 4; i++) {                     /* orthogonal: liberty-aware */
            const int ni = nn[i], c = g->cells[ni];
            if (c == EMPTY) { cv[i] = 0; continue; }
            int lib = g->ls[g->gid[ni]]; if (lib > FPOL_CAP) lib = FPOL_CAP;
            cv[i] = (c == cur) ? lib : FPOL_CAP + lib;
        }
        for (int i = 4; i < 8; i++) {                     /* diagonal: shape only */
            const int c = g->cells[nn[i]];
            cv[i] = c == EMPTY ? 0 : (c == cur ? 1 : 2);
        }
        const uint32_t key = hash_combine(salt_space, hash_combine(salt_term, canon8adjlib(cv)));
        const double s = (double)tab_get(key);
        st->moves[n] = idx; st->logits[n] = s;
        if (s > maxL) { maxL = s; maxI = n; }
        n++;
    }
    st->n = n;
    if (n == 0) return 0;
    if (temperature == 0) {
        for (int i = 0; i < n; i++) st->probs[i] = 0.0;
        st->probs[maxI] = 1.0;
        return n;
    }
    const double invT = 1.0 / temperature;
    double sum = 0;
    for (int i = 0; i < n; i++) { st->probs[i] = exp((st->logits[i] - maxL) * invT); sum += st->probs[i]; }
    const double inv = 1.0 / sum;
    for (int i = 0; i < n; i++) st->probs[i] *= inv;
    return n;
}

int32_t fpol_sample(const FpolState *st, Rng *rng) {
    if (st->n == 0) return PASS;
    double u = rng_random(rng);
    for (int i = 0; i < st->n; i++) { u -= st->probs[i]; if (u < 0) return st->moves[i]; }
    return st->moves[st->n - 1];           /* rounding: the last candidate */
}

static const FpolState *rank_st;
static int rank_cmp(const void *a, const void *b) {
    const int i = *(const int *)a, j = *(const int *)b;
    const double li = rank_st->logits[i], lj = rank_st->logits[j];
    if (li != lj) return li > lj ? -1 : 1;
    return i - j;                          /* stable: candidate order */
}
int fpol_rank(const FpolState *st, int *order) {
    for (int i = 0; i < st->n; i++) order[i] = i;
    rank_st = st;
    qsort(order, (size_t)st->n, sizeof *order, rank_cmp);
    return st->n;
}
