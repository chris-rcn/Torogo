#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "vpat.h"

/* ── Model state ──────────────────────────────────────────────────────────── */

#define VPAT_MAX_SPECS 8

typedef struct { int size; int max_libs; int phase_bins; } VpatSpec;

static VpatSpec vp_specs[VPAT_MAX_SPECS];
static int      vp_num_specs;
static int      vp_has_turn, vp_turn_bins;
static int      vp_max_libs_list[VPAT_MAX_SPECS];  /* distinct, descending */
static int      vp_num_max_libs;
static int      vp_do2[VPAT_MAX_SPECS], vp_do3[VPAT_MAX_SPECS]; /* per max_libs entry */

/* Open-addressing int32 -> double weight table (linear probe, power of two). */
static int32_t *vp_keys;
static double  *vp_vals;
static uint8_t *vp_used;
static uint32_t vp_mask;

/* Optional baked truncation block: trunc: { delta: D }. */
static int    vp_trunc_has;
static double vp_trunc_delta;

static double vp_lookup(int32_t key) {
    uint32_t h = (uint32_t)key * 0x9E3779B1u;
    for (uint32_t i = h & vp_mask; ; i = (i + 1) & vp_mask) {
        if (!vp_used[i]) return 0.0;
        if (vp_keys[i] == key) return vp_vals[i];
    }
}

static void vp_insert(int32_t key, double val) {
    uint32_t h = (uint32_t)key * 0x9E3779B1u;
    for (uint32_t i = h & vp_mask; ; i = (i + 1) & vp_mask) {
        if (!vp_used[i]) { vp_used[i] = 1; vp_keys[i] = key; vp_vals[i] = val; return; }
        if (vp_keys[i] == key) { vp_vals[i] = val; return; }  /* last write wins, like Map */
    }
}

/* ── Hash primitives (bit-identical to vpatterns.js / hpatterns.js) ───────── */

static inline int32_t vp_uh(int32_t a, int32_t b) {
    return (int32_t)(1234567u + (uint32_t)a + (uint32_t)b + (uint32_t)a * (uint32_t)b);
}
static inline int32_t vp_xh4(int32_t tl, int32_t tr, int32_t bl, int32_t br) {
    return vp_uh(vp_uh(tl, br), vp_uh(tr, bl));
}
/* mixTag(h, tag) = uh(h, tag) */

#define VPAT_TURN_TAG  (16 << 3)
#define VPAT_TURN_SALT ((int32_t)0x5ce7a13b)

/* Leaf mapping: prime - 1, so 1 + leaf is prime (unique pair products). */
static const int32_t vp_leaf_tab[31] = {
    1, 2, 4, 6, 10, 12, 16, 18, 22, 28, 30, 36, 40, 42, 46, 52,
    58, 60, 66, 70, 72, 78, 82, 88, 96, 100, 102, 106, 108, 112, 126 };

/* ── Loading ──────────────────────────────────────────────────────────────── */

static char *vp_read_file(const char *path, long *len_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "vpat: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "vpat: short read on %s\n", path); exit(1); }
    buf[len] = '\0';
    fclose(f);
    *len_out = len;
    return buf;
}

/* Parse one integer field ("name":<int>) inside the spec object at [p, end). */
static int vp_spec_field(const char *p, const char *end, const char *name, int dflt) {
    const char *q = p;
    size_t nlen = strlen(name);
    while ((q = memchr(q, '"', (size_t)(end - q))) != NULL) {
        if ((size_t)(end - q) > nlen + 2 && strncmp(q + 1, name, nlen) == 0 && q[1 + nlen] == '"') {
            const char *v = q + nlen + 2;
            while (v < end && (*v == ':' || *v == ' ')) v++;
            return (int)strtol(v, NULL, 10);
        }
        q++;
    }
    return dflt;
}

/* Size the open-addressing table for n entries (load factor <= 0.5). */
static void vp_table_init(long n) {
    uint32_t cap = 64;
    while (cap < (uint32_t)(n * 2 + 1)) cap <<= 1;
    free(vp_keys); free(vp_vals); free(vp_used);
    vp_keys = malloc(cap * sizeof(int32_t));
    vp_vals = malloc(cap * sizeof(double));
    vp_used = calloc(cap, 1);
    vp_mask = cap - 1;
}

/* Standard base64 (A-Z a-z 0-9 + /, '=' padding) of len chars; returns a
 * malloc'd buffer and its length in *out_len.  Exits on a bad character. */
static unsigned char *vp_b64_decode(const char *s, long len, long *out_len) {
    static signed char tab[256];
    static int tab_ready = 0;
    if (!tab_ready) {
        for (int i = 0; i < 256; i++) tab[i] = -1;
        const char *alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) tab[(unsigned char)alpha[i]] = (signed char)i;
        tab_ready = 1;
    }
    unsigned char *out = malloc((size_t)(len / 4 + 1) * 3);
    long o = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (long i = 0; i < len; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '=') break;
        const int v = tab[c];
        if (v < 0) { fprintf(stderr, "vpat: bad base64 character 0x%02x in weightsQ6\n", c); exit(1); }
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out[o++] = (unsigned char)(acc >> bits); }
    }
    *out_len = o;
    return out;
}

bool vpat_load(const char *path) {
    long len;
    char *buf = vp_read_file(path, &len);

    /* specs: [...] — parse each {...} object */
    char *sp = strstr(buf, "specs:");
    if (!sp) { fprintf(stderr, "vpat: no specs in %s\n", path); exit(1); }
    char *sp_end = strchr(sp, ']');
    if (!sp_end) { fprintf(stderr, "vpat: unterminated specs in %s\n", path); exit(1); }
    vp_num_specs = 0; vp_has_turn = 0; vp_turn_bins = 1; vp_num_max_libs = 0;
    for (char *o = strchr(sp, '{'); o && o < sp_end; o = strchr(o + 1, '{')) {
        char *o_end = strchr(o, '}');
        if (!o_end || o_end > sp_end) { fprintf(stderr, "vpat: bad spec object in %s\n", path); exit(1); }
        if (vp_num_specs >= VPAT_MAX_SPECS) { fprintf(stderr, "vpat: too many specs in %s\n", path); exit(1); }
        VpatSpec *s = &vp_specs[vp_num_specs++];
        s->size       = vp_spec_field(o, o_end, "size", -1);
        s->max_libs   = vp_spec_field(o, o_end, "maxLibs", -999);
        s->phase_bins = vp_spec_field(o, o_end, "phaseBins", 1);
        if (s->size == 5) {
            vp_has_turn = 1; vp_turn_bins = s->phase_bins > 1 ? s->phase_bins : 1;
        } else if ((s->size == 2 || s->size == 3) &&
                   s->max_libs >= 1 && s->max_libs <= 15 && s->phase_bins <= 1) {
            int mi = -1;
            for (int i = 0; i < vp_num_max_libs; i++) if (vp_max_libs_list[i] == s->max_libs) mi = i;
            if (mi < 0) {
                /* insert keeping the list descending (the clamp chain relies on it) */
                mi = vp_num_max_libs++;
                while (mi > 0 && vp_max_libs_list[mi - 1] < s->max_libs) {
                    vp_max_libs_list[mi] = vp_max_libs_list[mi - 1];
                    vp_do2[mi] = vp_do2[mi - 1]; vp_do3[mi] = vp_do3[mi - 1];
                    mi--;
                }
                vp_max_libs_list[mi] = s->max_libs; vp_do2[mi] = 0; vp_do3[mi] = 0;
            }
            if (s->size == 2) vp_do2[mi] = 1; else vp_do3[mi] = 1;
        } else {
            fprintf(stderr, "vpat: unsupported spec {size:%d, maxLibs:%d, phaseBins:%d} in %s "
                            "(this port supports 't[pN]' and '2:<ml>'/'3:<ml>' only)\n",
                    s->size, s->max_libs, s->phase_bins, path);
            exit(1);
        }
    }

    long loaded = 0;
    char *q6 = strstr(buf, "weightsQ6:");
    if (q6) {
        /* weightsQ6: { count: N, b64: '...' } -- base64 of N int32 keys then N
         * int32 values in millionths, little-endian (vpatterns.js). */
        char *cp = strstr(q6, "count:");
        char *bp = strstr(q6, "b64:");
        if (!cp || !bp) { fprintf(stderr, "vpat: bad weightsQ6 in %s\n", path); exit(1); }
        long n = strtol(cp + 6, NULL, 10);
        char *b64 = strchr(bp, '\'');
        char *b64_end = b64 ? strchr(b64 + 1, '\'') : NULL;
        if (n < 0 || !b64_end) { fprintf(stderr, "vpat: bad weightsQ6 in %s\n", path); exit(1); }
        b64++;
        long nbytes = 0;
        unsigned char *bytes = vp_b64_decode(b64, b64_end - b64, &nbytes);
        if (nbytes != n * 8) {
            fprintf(stderr, "vpat: weightsQ6 holds %ld bytes, expected %ld for %ld weights in %s\n", nbytes, n * 8, n, path);
            exit(1);
        }
        vp_table_init(n);
        for (long i = 0; i < n; i++) {
            const unsigned char *kb = bytes + i * 4, *vb = bytes + n * 4 + i * 4;
            int32_t key = (int32_t)((uint32_t)kb[0] | (uint32_t)kb[1] << 8 | (uint32_t)kb[2] << 16 | (uint32_t)kb[3] << 24);
            int32_t qv  = (int32_t)((uint32_t)vb[0] | (uint32_t)vb[1] << 8 | (uint32_t)vb[2] << 16 | (uint32_t)vb[3] << 24);
            vp_insert(key, (double)qv / 1e6);   /* exactly the double the 6-decimal text parsed to */
            loaded++;
        }
        free(bytes);
    } else {
    /* older files: weights: new Map([[k,v],...]) */
    char *w = strstr(buf, "new Map([");
    if (!w) { fprintf(stderr, "vpat: no weightsQ6 or weights Map in %s\n", path); exit(1); }
    w += (long)strlen("new Map([");
    /* count entries for table sizing */
    long n = 0;
    for (const char *q = w; *q && !(q[0] == ']' && q[1] == ')'); q++) if (*q == '[') n++;
    vp_table_init(n);
    char *q = w;
    while (*q) {
        while (*q && *q != '[' && !(q[0] == ']' && q[1] == ')')) q++;
        if (*q != '[') break;
        q++;
        char *endp;
        long key = strtol(q, &endp, 10);
        if (endp == q || *endp != ',') { fprintf(stderr, "vpat: bad weight entry in %s\n", path); exit(1); }
        q = endp + 1;
        double val = strtod(q, &endp);
        if (endp == q) { fprintf(stderr, "vpat: bad weight value in %s\n", path); exit(1); }
        q = endp;
        vp_insert((int32_t)key, val);
        loaded++;
    }
    if (loaded != n) { fprintf(stderr, "vpat: parsed %ld of %ld weight entries in %s\n", loaded, n, path); exit(1); }
    }

    /* Optional baked truncation default: trunc: { delta: D }.  The JS consumers
     * read it as a default; here it lets the trainer default --trunc-delta.
     * (Older files may carry an offset too; it is ignored.) */
    vp_trunc_has = 0;
    char *tr = strstr(buf, "trunc:");
    if (tr) {
        char *tr_end = strchr(tr, '}');
        char *dp = strstr(tr, "delta:");
        if (dp && (!tr_end || dp < tr_end)) {
            vp_trunc_delta = strtod(dp + 6, NULL);
            vp_trunc_has = 1;
        }
    }

    free(buf);
    fprintf(stderr, "vpat: loaded %ld weights from %s (%d specs%s)\n",
            loaded, path, vp_num_specs, vp_has_turn ? ", turn" : "");
    return true;
}

/* Baked truncation default from the model file (trunc: {delta}).  Returns true
 * and fills *delta if the loaded model carried the block; false if it had none. */
bool vpat_trunc(double *delta) {
    if (!vp_trunc_has) return false;
    if (delta) *delta = vp_trunc_delta;
    return true;
}

/* ── Liberty counts from cells alone (flood fill, mirrors _cellLibCounts) ─── */

static int32_t vp_libs[MAX_CAP];
static int32_t vp_stack[MAX_CAP], vp_group[MAX_CAP];
static int32_t vp_seen[MAX_CAP], vp_mark[MAX_CAP];
static int32_t vp_stamp;

static void vp_cell_lib_counts(const Game2 *g) {
    const int area = g->N * g->N;
    const int8_t *cells = g->cells;
    if (vp_stamp > 0x7fff0000) { memset(vp_seen, 0, sizeof vp_seen); memset(vp_mark, 0, sizeof vp_mark); vp_stamp = 0; }
    const int32_t seen_stamp = ++vp_stamp;
    for (int i = 0; i < area; i++) {
        if (cells[i] == EMPTY) { vp_libs[i] = 0; continue; }
        if (vp_seen[i] == seen_stamp) continue;
        const int32_t mark_stamp = ++vp_stamp;
        const int8_t color = cells[i];
        int top = 0, size = 0, libs = 0;
        vp_stack[top++] = i; vp_seen[i] = seen_stamp;
        while (top > 0) {
            const int c = vp_stack[--top];
            vp_group[size++] = c;
            const int b = c * 4;
            for (int k = 0; k < 4; k++) {
                const int nb = g2_nbr[b + k];
                if (cells[nb] == EMPTY) {
                    if (vp_mark[nb] != mark_stamp) { vp_mark[nb] = mark_stamp; libs++; }
                } else if (cells[nb] == color && vp_seen[nb] != seen_stamp) {
                    vp_seen[nb] = seen_stamp; vp_stack[top++] = nb;
                }
            }
        }
        for (int k = 0; k < size; k++) vp_libs[vp_group[k]] = libs;
    }
}

/* ── Evaluation ───────────────────────────────────────────────────────────── */

static int8_t  vp_raw[MAX_CAP];
static int32_t vp_lN[MAX_CAP], vp_lI[MAX_CAP];
static int32_t vp_h2N[MAX_CAP], vp_h2I[MAX_CAP];

double vpat_evaluate(const Game2 *g) {
    return 1.0 / (1.0 + exp(-vpat_evaluate_z(g)));
}

double vpat_evaluate_z(const Game2 *g) {
    const int N = g->N, area = N * N;
    const int8_t *cells = g->cells;
    double z = 0.0;

    /* Turn feature, keyed by phase bucket. */
    if (vp_has_turn) {
        int32_t salt = 0;
        if (vp_turn_bins > 1) {
            const double phase = 1.0 - (double)g->empty_count / area;
            int b = (int)(phase * vp_turn_bins);
            if (b >= vp_turn_bins) b = vp_turn_bins - 1;
            /* imul((b+1) ^ imul(tag+1, 131), 0x9E3779B1) */
            const int32_t t1 = (int32_t)((uint32_t)(VPAT_TURN_TAG + 1) * 131u);
            salt = (int32_t)((uint32_t)((b + 1) ^ t1) * 0x9E3779B1u);
        }
        const int32_t key = vp_uh(VPAT_TURN_SALT, VPAT_TURN_TAG) ^ salt;
        z += (g->current == BLACK ? 1.0 : -1.0) * vp_lookup(key);
    }

    /* Pattern families, max_libs descending with in-place clamping. */
    int have_raw = 0;
    for (int mi = 0; mi < vp_num_max_libs; mi++) {
        const int ml = vp_max_libs_list[mi];
        if (!have_raw) {
            if (ml == 1) {
                for (int i = 0; i < area; i++) vp_raw[i] = cells[i];
            } else {
                vp_cell_lib_counts(g);
                for (int i = 0; i < area; i++) {
                    const int8_t c = cells[i];
                    vp_raw[i] = c == EMPTY ? 0
                              : (int8_t)(vp_libs[i] < ml ? c * vp_libs[i] : c * ml);
                }
            }
            have_raw = 1;
        } else {
            for (int i = 0; i < area; i++) {
                if      (vp_raw[i] >  ml) vp_raw[i] = (int8_t)ml;
                else if (vp_raw[i] < -ml) vp_raw[i] = (int8_t)-ml;
            }
        }
        /* Leaf planes (normal + colour-inverted), then 2x2 X-hash planes. */
        for (int i = 0; i < area; i++) {
            vp_lN[i] = vp_leaf_tab[vp_raw[i] + ml];
            vp_lI[i] = vp_leaf_tab[ml - vp_raw[i]];
        }
        for (int y = 0; y < N; y++) {
            const int r0 = y * N, r1 = (y + 1 < N ? y + 1 : 0) * N;
            for (int x = 0; x < N; x++) {
                const int x1 = x + 1 < N ? x + 1 : 0;
                const int i = r0 + x;
                vp_h2N[i] = vp_xh4(vp_lN[r0 + x], vp_lN[r0 + x1], vp_lN[r1 + x], vp_lN[r1 + x1]);
                vp_h2I[i] = vp_xh4(vp_lI[r0 + x], vp_lI[r0 + x1], vp_lI[r1 + x], vp_lI[r1 + x1]);
            }
        }
        const int tag_base = ml;   /* tagBaseOf(positive ml) = ml */
        if (vp_do2[mi]) {
            const int32_t tag = (int32_t)((tag_base << 3) | 2);
            for (int i = 0; i < area; i++) {
                const int32_t kN = vp_h2N[i], kI = vp_h2I[i];
                if (kN == kI) continue;                 /* colour-twin: zero value */
                const int32_t key = vp_uh(kN < kI ? kN : kI, tag);
                z += (kN < kI ? 1.0 : -1.0) * vp_lookup(key);
            }
        }
        if (vp_do3[mi]) {
            const int32_t tag = (int32_t)((tag_base << 3) | 3);
            for (int y = 0; y < N; y++) {
                const int r0 = y * N, r1 = (y + 1 < N ? y + 1 : 0) * N;
                for (int x = 0; x < N; x++) {
                    const int x1 = x + 1 < N ? x + 1 : 0;
                    const int32_t kN = vp_xh4(vp_h2N[r0 + x], vp_h2N[r0 + x1], vp_h2N[r1 + x], vp_h2N[r1 + x1]);
                    const int32_t kI = vp_xh4(vp_h2I[r0 + x], vp_h2I[r0 + x1], vp_h2I[r1 + x], vp_h2I[r1 + x1]);
                    if (kN == kI) continue;
                    const int32_t key = vp_uh(kN < kI ? kN : kI, tag);
                    z += (kN < kI ? 1.0 : -1.0) * vp_lookup(key);
                }
            }
        }
    }
    return z;
}
