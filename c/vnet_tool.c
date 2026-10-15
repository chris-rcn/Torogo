/* vnet_tool.c — see vnet_tool.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "vnet_tool.h"
#include "fpol.h"

const char *vt_prog = "vnet";

/* Every flag the program asked about, for args_check. */
enum { ARG_VALUE, ARG_OPTIONAL, ARG_BARE };
static struct { const char *flag; int kind; } known[128];
static int n_known;
static void note_flag(const char *flag, int kind) {
    for (int i = 0; i < n_known; i++) if (!strcmp(known[i].flag, flag)) return;
    if (n_known == (int)(sizeof known / sizeof known[0])) { fprintf(stderr, "%s: too many flags for args_check\n", vt_prog); exit(1); }
    known[n_known].flag = flag; known[n_known++].kind = kind;
}

const char *arg_str(int argc, char **argv, const char *flag, const char *def) {
    note_flag(flag, ARG_VALUE);
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], flag)) {
            if (i + 1 >= argc || !strncmp(argv[i + 1], "--", 2)) {
                fprintf(stderr, "%s: %s needs a value\n", vt_prog, flag); exit(1);
            }
            return argv[i + 1];
        }
    return def;
}
const char *arg_opt(int argc, char **argv, const char *flag, const char *def) {
    note_flag(flag, ARG_OPTIONAL);
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], flag)) return (i + 1 < argc && strncmp(argv[i + 1], "--", 2)) ? argv[i + 1] : def;
    return NULL;
}
int arg_flag(int argc, char **argv, const char *flag) {
    note_flag(flag, ARG_BARE);
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}
void args_check(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        int k = -1;
        for (int j = 0; j < n_known; j++) if (!strcmp(known[j].flag, argv[i])) { k = j; break; }
        if (k < 0) {
            if (!strncmp(argv[i], "--", 2)) fprintf(stderr, "%s: unknown flag %s (see --help)\n", vt_prog, argv[i]);
            else fprintf(stderr, "%s: unexpected argument '%s' (see --help)\n", vt_prog, argv[i]);
            exit(1);
        }
        if (known[k].kind == ARG_VALUE || (known[k].kind == ARG_OPTIONAL && i + 1 < argc && strncmp(argv[i + 1], "--", 2))) i++;
    }
}
double wall_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
double cpu_s(void)  { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
double thread_cpu_s(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
const char *fmt_time(double s, char *buf) {
    if (s < 1e-3) snprintf(buf, 16, "%.0fus", s * 1e6);
    else if (s < 1) snprintf(buf, 16, "%.1fms", s * 1e3);
    else snprintf(buf, 16, "%.1fs", s);
    return buf;
}

/* ── Data ───────────────────────────────────────────────────────────────── */


static int32_t parse_move(const char *s, int N) {
    if (s[0] == 'p') return PASS;
    return (atoi(s + 1) - 1) * N + (s[0] - 'a');
}

/* Load gen-agent-evals records "<size> <phase> <moves> <winRatio>" (comment
 * lines skipped).  One board size per run; a record that does not parse or
 * replay is reported, counted and dropped, never trained on.  Lines are read
 * in chunks on this thread; each chunk's replays and vnet_compact (the ladder
 * reads) run on `threads` threads, kept in file order. */
#define LOAD_CHUNK 16384
typedef struct { long moves; int lineno; float label; } Pending;   /* moves: arena offset, -1 = did not parse */

void load_set(const char *path, Set *S, int *size, int limit, int threads) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "%s: cannot open %s\n", vt_prog, path); exit(1); }
    static char line[16384], moves[16384];
    int lineno = 0, bad = 0, eof = 0;
    Pending *pend = malloc(sizeof(Pending) * LOAD_CHUNK);
    unsigned char *ok = malloc(LOAD_CHUNK);
    size_t arena_cap = 1 << 22;
    char *arena = malloc(arena_cap);
    Game3 **g3 = calloc(threads, sizeof(Game3 *));   /* ladder-read scratch for vnet_compact, per thread */
    while (!eof && (limit <= 0 || S->n < limit)) {
        int np = 0;
        size_t used = 0;
        while (np < LOAD_CHUNK) {
            if (!fgets(line, sizeof line, f)) { eof = 1; break; }
            lineno++;
            if (line[0] == '#' || line[0] == '\n') continue;
            int n; double label;
            Pending *P = &pend[np++];
            P->lineno = lineno;
            if (sscanf(line, "%d %*s %16383s %lf", &n, moves, &label) != 3) { P->moves = -1; continue; }
            if (*size == 0) { *size = n; g2_init_topology(n); }
            else if (n != *size) { fprintf(stderr, "%s: %s line %d: size %d, run is size %d\n", vt_prog, path, lineno, n, *size); exit(1); }
            const size_t L = strlen(moves) + 1;
            if (used + L > arena_cap) { while (used + L > arena_cap) arena_cap *= 2; arena = realloc(arena, arena_cap); }
            memcpy(arena + used, moves, L);
            P->moves = (long)used; P->label = (float)label;
            used += L;
        }
        if (np == 0) break;
        if (S->n + np > S->cap) { while (S->n + np > S->cap) S->cap = S->cap ? 2 * S->cap : 65536; S->r = realloc(S->r, sizeof(Record) * (size_t)S->cap); }
        const int N = *size, base = S->n;
        /* Replay and compact into the slots after the records so far; the
         * in-order pass below closes the gaps the dropped ones leave. */
        #pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
        for (int i = 0; i < np; i++) {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            ok[i] = 0;
            if (pend[i].moves < 0) continue;
            Game2 g; g2_new(&g, N);
            const char *m = arena + pend[i].moves;
            int good = 1;
            while (*m) {
                if (!g2_play(&g, parse_move(m, N))) { good = 0; break; }
                while (*m && *m != ',') m++;
                if (*m == ',') m++;
            }
            if (!good) continue;
            if (!g3[t]) g3[t] = g3_new(N);
            Record *R = &S->r[base + i];
            vnet_compact(&g, g3[t], R->code, &R->meta);
            R->label = pend[i].label;
            ok[i] = 1;
        }
        for (int i = 0; i < np && (limit <= 0 || S->n < limit); i++) {
            if (!ok[i]) {
                if (++bad <= 5) fprintf(stderr, pend[i].moves < 0 ? "%s: %s line %d: not '<size> <phase> <moves> <winRatio>'\n"
                                                                   : "%s: %s line %d: an illegal move on replay\n", vt_prog, path, pend[i].lineno);
                continue;
            }
            if (S->n != base + i) S->r[S->n] = S->r[base + i];
            S->n++;
        }
    }
    fclose(f);
    for (int t = 0; t < threads; t++) if (g3[t]) g3_free(g3[t]);
    free(g3); free(pend); free(ok); free(arena);
    if (bad) fprintf(stderr, "%s: %s: %d record(s) dropped (see above)\n", vt_prog, path, bad);
    if (S->n == 0) { fprintf(stderr, "%s: no records in %s\n", vt_prog, path); exit(1); }
}

/* ── MD 1-ply test (--md) ───────────────────────────────────────────────── */

typedef struct { char m[8]; float wr; } Cand;           /* wr < 0: no rating (terminal) */
typedef struct { Game2 g; double phase; int nc; Cand *c; } MdPos;
typedef struct { MdPos *p; int n; char *rows; long evals; } MdSet;   /* rows: the data lines, NUL-separated, for the fingerprint; evals: net evaluations one test needs */

static int32_t coord_to_idx(const char *s, int N) { return s[0] == 'p' ? PASS : (atoi(s + 1) - 1) * N + (s[0] - 'a'); }
static void idx_to_coord(int32_t m, int N, char *out) {
    if (m == PASS) strcpy(out, "pass");
    else snprintf(out, 8, "%c%d", 'a' + m % N, m / N + 1);
}

/* f1 <size> <phase> <history> <candidates>: history replays from an EMPTY board
 * (it starts with the centre stone); candidates "move:kwr" best-first. */
/* The moves the test values at g: featurepol's top fp_width (0 = every legal
 * non-true-eye move, index order) into out; returns the count. */
static int md_cands(const Game2 *g, int fp_width, int32_t *out) {
    int n = 0;
    if (fp_width > 0) {
        FpolState fs; int order[MAX_CAP];
        fpol_eval(g, &fs, 1.0);
        fpol_rank(&fs, order);
        for (int i = 0; i < fs.n && i < fp_width; i++) out[n++] = fs.moves[order[i]];
    } else
        for (int m = 0; m < g->cap; m++)
            if (g->cells[m] == EMPTY && g2_is_legal(g, m) && !g2_is_true_eye_at(g, m)) out[n++] = m;
    return n;
}

static void load_md(const char *path, int N, int fp_width, MdSet *M) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "%s: cannot open --md %s\n", vt_prog, path); exit(1); }
    static char line[65536], hist[32768], cands[32768];
    size_t rows_len = 0, rows_cap = 1 << 20;
    M->rows = malloc(rows_cap);
    int cap = 0, lineno = 0;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (L == 0 || line[0] == '#') continue;
        if (rows_len + L + 2 > rows_cap) { rows_cap = 2 * (rows_len + L + 2); M->rows = realloc(M->rows, rows_cap); }
        memcpy(M->rows + rows_len, line, L + 1); rows_len += L + 1;
        char id[8]; int n; double phase;
        if (sscanf(line, "%7s %d %lf %32767s %32767s", id, &n, &phase, hist, cands) != 5 || strcmp(id, "f1") != 0) {
            fprintf(stderr, "%s: --md %s line %d: not an f1 row\n", vt_prog, path, lineno); exit(1);
        }
        if (n != N) { fprintf(stderr, "%s: --md %s is size %d, training is size %d\n", vt_prog, path, n, N); exit(1); }
        if (M->n == cap) { cap = cap ? 2 * cap : 1024; M->p = realloc(M->p, sizeof(MdPos) * (size_t)cap); }
        MdPos *P = &M->p[M->n];
        g2_new_empty(&P->g, N);
        if (strcmp(hist, "-") != 0)
            for (char *t = strtok(hist, ","); t; t = strtok(NULL, ","))
                if (!g2_play(&P->g, coord_to_idx(t, N))) { fprintf(stderr, "%s: --md %s line %d: illegal history move %s\n", vt_prog, path, lineno, t); exit(1); }
        P->phase = phase;
        P->nc = 0; P->c = NULL;
        int ccap = 0;
        if (strcmp(cands, "-") != 0)
            for (char *t = strtok(cands, ","); t; t = strtok(NULL, ",")) {
                char *colon = strchr(t, ':');
                if (!colon) { fprintf(stderr, "%s: --md %s line %d: candidate %s has no ':'\n", vt_prog, path, lineno, t); exit(1); }
                if (P->nc == ccap) { ccap = ccap ? 2 * ccap : 64; P->c = realloc(P->c, sizeof(Cand) * (size_t)ccap); }
                Cand *C = &P->c[P->nc++];
                const int ml = (int)(colon - t) < 7 ? (int)(colon - t) : 7;
                memcpy(C->m, t, ml); C->m[ml] = 0;
                C->wr = colon[1] ? atoi(colon + 1) / 1000.0f : -1.0f;
            }
        if (P->nc == 0 || P->c[0].wr < 0) { fprintf(stderr, "%s: --md %s line %d: no rated top candidate\n", vt_prog, path, lineno); exit(1); }
        int32_t cand[MAX_CAP];
        M->evals += md_cands(&P->g, fp_width, cand);   /* the moves md_eval will evaluate */
        M->n++;
    }
    fclose(f);
    if (M->n == 0) { fprintf(stderr, "%s: no positions in --md %s\n", vt_prog, path); exit(1); }
}

/* SHA-256 (FIPS 180-4), for evalmovedetails' MD fingerprint. */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; int nb; } Sha256;
static const uint32_t SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha_block(Sha256 *S, const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = ROR(w[i-15], 7) ^ ROR(w[i-15], 18) ^ (w[i-15] >> 3);
        const uint32_t s1 = ROR(w[i-2], 17) ^ ROR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = S->h[0], b = S->h[1], c = S->h[2], d = S->h[3], e = S->h[4], f = S->h[5], g = S->h[6], h = S->h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t t1 = h + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + SHA_K[i] + w[i];
        const uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    S->h[0] += a; S->h[1] += b; S->h[2] += c; S->h[3] += d; S->h[4] += e; S->h[5] += f; S->h[6] += g; S->h[7] += h;
}
static void sha_init(Sha256 *S) {
    static const uint32_t h0[8] = { 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19 };
    memcpy(S->h, h0, sizeof h0); S->len = 0; S->nb = 0;
}
static void sha_update(Sha256 *S, const void *data, size_t n) {
    const uint8_t *p = data;
    S->len += n;
    while (n--) { S->buf[S->nb++] = *p++; if (S->nb == 64) { sha_block(S, S->buf); S->nb = 0; } }
}
static void sha_hex16(Sha256 *S, char *out) {
    const uint64_t bits = S->len * 8;
    uint8_t pad = 0x80; sha_update(S, &pad, 1);
    pad = 0; while (S->nb != 56) sha_update(S, &pad, 1);
    for (int i = 7; i >= 0; i--) { const uint8_t b = (uint8_t)(bits >> (8 * i)); sha_update(S, &b, 1); }
    for (int i = 0; i < 8; i++) snprintf(out + 2 * i, 3, "%02x", (S->h[i / 4] >> (24 - 8 * (i % 4))) & 0xff);
    out[16] = 0;
}
static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
/* sha256 of the data rows, sorted, joined by '\n': its first 16 hex digits. */
static void md_fingerprint(const MdSet *M, char *out) {
    char **rows = malloc(sizeof(char *) * (size_t)M->n);
    char *r = M->rows;
    for (int i = 0; i < M->n; i++) { rows[i] = r; r += strlen(r) + 1; }
    qsort(rows, (size_t)M->n, sizeof(char *), cmp_str);
    Sha256 S; sha_init(&S);
    for (int i = 0; i < M->n; i++) { if (i) sha_update(&S, "\n", 1); sha_update(&S, rows[i], strlen(rows[i])); }
    sha_hex16(&S, out);
    free(rows);
}

/* The elo map (md-phase-fit.js --save): intercept, the mae curve's A and k,
 * phaseBuckets, the bands, and the fingerprint of the MD file it was fit on. */
typedef struct { double intercept, A, k; int buckets, nb; double lo[16], hi[16]; char fp[17]; char md[512]; } EloMap;
static const char *json_after(const char *s, const char *key) { const char *p = strstr(s, key); return p ? p + strlen(key) : NULL; }
static EloMap load_elo_map(const char *path) {
    EloMap E; memset(&E, 0, sizeof E);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "%s: cannot open --elo-map %s\n", vt_prog, path); exit(1); }
    static char js[1 << 16];
    const size_t n = fread(js, 1, sizeof js - 1, f); js[n] = 0; fclose(f);
    const char *p, *mae = json_after(js, "\"mae\"");
    if (!(p = json_after(js, "\"intercept\":"))) goto bad;
    E.intercept = atof(p);
    if (!(p = json_after(js, "\"phaseBuckets\":"))) goto bad;
    E.buckets = atoi(p);
    if (!mae || !(p = json_after(mae, "\"A\":"))) goto bad;
    E.A = atof(p);
    if (!(p = json_after(mae, "\"k\":"))) goto bad;
    E.k = atof(p);
    if (!(p = json_after(js, "\"mdFingerprint\":"))) goto bad;
    while (*p && *p != '"') p++;
    if (*p != '"') goto bad;
    memcpy(E.fp, p + 1, 16); E.fp[16] = 0;
    if (!(p = json_after(js, "\"mdFile\":"))) goto bad;
    while (*p && *p != '"') p++;
    if (*p != '"') goto bad;
    {   const char *q = strchr(p + 1, '"');
        if (!q || q - p - 1 >= (int)sizeof E.md) goto bad;
        memcpy(E.md, p + 1, q - p - 1); E.md[q - p - 1] = 0; }
    {   const char *b = json_after(js, "\"bands\":"), *end = b ? strchr(b, ']') : NULL;
        if (!b || !end) goto bad;
        for (const char *q = b; (q = strstr(q, "\"lo\":")) && q < end && E.nb < 16; ) {
            E.lo[E.nb] = atof(q + 5);
            const char *h = strstr(q, "\"hi\":"); if (!h) goto bad;
            E.hi[E.nb++] = atof(h + 5);
            q = h;
        }
    }
    if (E.nb == 0 || E.buckets <= 0 || E.buckets > 16) goto bad;
    return E;
bad:
    fprintf(stderr, "%s: --elo-map %s: missing intercept / phaseBuckets / bands / curves.mae / mdFingerprint / mdFile\n", vt_prog, path);
    exit(1);
}
static int band_of(double phase, int nb) { int b = (int)floor(phase * nb); return b < 0 ? 0 : b >= nb ? nb - 1 : b; }

/* The 1-ply player's mean gap over the MD set, with per-band sums for the elo. */
static double md_eval(const MdSet *M, const VnetLayout *L, const float *w, const int32_t *nbr9, int area,
                      VnetWork **WK, Game3 **G3, int fp_width, int buckets, double *band_sum, int *band_n) {
    double gsum = 0;
    for (int b = 0; b < buckets; b++) { band_sum[b] = 0; band_n[b] = 0; }
    #pragma omp parallel for schedule(dynamic, 4) reduction(+:gsum)
    for (int i = 0; i < M->n; i++) {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        const MdPos *P = &M->p[i];
        const int N = P->g.N;
        int32_t best = PASS; float best_v = -1;
        int32_t cand[MAX_CAP];
        const int nc = md_cands(&P->g, fp_width, cand);
        for (int c = 0; c < nc; c++) {
            Game2 child; g2_clone(&child, &P->g); g2_play(&child, cand[c]);
            vnet_encode(&child, G3[t], NULL, vnet_work_input(WK[t]));
            const float v = 1 - vnet_forward(L, w, nbr9, area, WK[t]);   /* the opponent is to move in child */
            if (v > best_v) { best_v = v; best = cand[c]; }
        }
        char ms[8]; idx_to_coord(best, N, ms);
        float wr = -1;
        for (int c = 0; c < P->nc; c++) if (strcmp(P->c[c].m, ms) == 0) { wr = P->c[c].wr; break; }
        if (wr < 0) for (int c = P->nc - 1; c >= 0; c--) if (P->c[c].wr >= 0) { wr = P->c[c].wr; break; }
        const double gap = P->c[0].wr - wr;
        gsum += gap;
        const int b = band_of(P->phase, buckets);
        #pragma omp atomic
        band_sum[b] += gap;
        #pragma omp atomic
        band_n[b]++;
    }
    return gsum / M->n;
}
uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t z = a * 0x9e3779b9u ^ (b + 0x7f4a7c15u);
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return z ^ (z >> 16);
}

Adam adam_new(int n) {
    Adam a = { calloc(n, sizeof(float)), calloc(n, sizeof(float)), 0, n };
    return a;
}

void adam_step(Adam *a, float *w, const float *G, float lr) {
    a->step++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    const float c1 = 1 - powf(b1, (float)a->step), c2 = 1 - powf(b2, (float)a->step);
    for (int j = 0; j < a->n; j++) {
        a->m[j] = b1 * a->m[j] + (1 - b1) * G[j];
        a->v[j] = b2 * a->v[j] + (1 - b2) * G[j] * G[j];
        w[j] -= lr * (a->m[j] / c1) / (sqrtf(a->v[j] / c2) + eps);
    }
}

struct MdTest {
    MdSet md;
    EloMap elo;
    const char *md_f, *elo_f;
    int fp_width;
    int elo_valid;
    double total;   /* wall time spent in the test */
    double scale;   /* last run's actual time / its estimate */
};

MdTest *md_test_open(const char *elo_f, const char *md_f, int N, int fp_width) {
    MdTest *T = calloc(1, sizeof *T);
    T->fp_width = fp_width;
    T->elo = load_elo_map(elo_f);
    T->elo_f = elo_f;
    T->md_f = md_f ? md_f : T->elo.md;
    load_md(T->md_f, N, fp_width, &T->md);   /* on this thread: fpol's first use builds its tables */
    char fp[17]; md_fingerprint(&T->md, fp);
    T->elo_valid = strcmp(fp, T->elo.fp) == 0;
    if (!T->elo_valid) fprintf(stderr, "%s: %s (fingerprint %s) is not the elo map's MD file (%s): elo not estimated\n", vt_prog, T->md_f, fp, T->elo.fp);
    T->scale = 1;
    return T;
}

void md_test_describe(const MdTest *T) {
    char fw[32] = "full width";
    if (T->fp_width > 0) snprintf(fw, sizeof fw, "featurepol top %d", T->fp_width);
    printf("md: %s (%d positions, %s: %ld net evaluations per test), elo map: %s%s\n", T->md_f, T->md.n, fw, T->md.evals, T->elo_f, T->elo_valid ? "" : " (not this MD file: elo -)");
}

int md_test_row(MdTest *T, const VnetLayout *L, const float *w, const int32_t *nbr9, int area,
                VnetWork **WK, Game3 **G3, int threads, double infer, double now, double *best, char *col) {
    const double raw = T->md.evals * infer / threads, est = raw * T->scale;
    if (T->total + est > 0.20 * (now + est)) { snprintf(col, 40, "  %6s %6s", "-", "-"); return 0; }
    const double m0 = wall_s();
    double bs[16]; int bn[16];
    const EloMap *E = &T->elo;
    const double mae = md_eval(&T->md, L, w, nbr9, area, WK, G3, T->fp_width, E->buckets, bs, bn);
    double e = E->intercept; int have = T->elo_valid;
    for (int j = 0; j < E->nb && have; j++) {
        const int b = band_of((E->lo[j] + E->hi[j]) / 2, E->buckets);
        if (bn[b] == 0) have = 0;
        else e -= E->A * exp(E->k * (E->lo[j] + E->hi[j]) / 2) * (bs[b] / bn[b]);
    }
    const int is_best = best && mae < *best;
    if (is_best) *best = mae;
    if (have) snprintf(col, 40, "  %6.4f%c%6.0f", mae, is_best ? '*' : ' ', e);
    else snprintf(col, 40, "  %6.4f%c     -", mae, is_best ? '*' : ' ');
    const double took = wall_s() - m0;
    T->total += took;
    if (raw > 0) T->scale = took / raw;
    return is_best;
}
