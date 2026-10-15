/*
 * train_vnet.c — supervised training of the toroidal residual value network
 * (cnn.h) on gen-agent-evals labels: each record's position (replayed from the
 * board with the free centre stone, as gen-agent-evals writes it) against its
 * win ratio, P(side to move wins).
 *
 * Each epoch shuffles the training records and gives every one a random board
 * symmetry (of the 8 reflections/rotations; translations need none — the value
 * is translation-invariant by construction).  Loss: cross-entropy against the
 * soft label.  Optimiser: Adam.  Rows (positions geometric x1.4, plus every
 * epoch end) report the interval's training loss and MSE, the test-file MSE
 * (unaugmented) — the same quantity as train-vpat-supervised's teMSE on the
 * same split — and infer, one test record's forward pass (encoding included)
 * in CPU time: the cost on one core whatever --threads is.  pos/s is positions
 * trained over total elapsed time, test passes included.  The checkpoint is
 * written at every row and -best on a new best teMSE.
 *
 * --md FILE adds a 1-ply test of the net as a player: at every MD position
 * (movedetails, *.md) each legal move except filling one's own true eye is
 * played and valued 1 - V(child) (the opponent is to move there), and the best
 * is charged its gap to the file's top move — the worst rated candidate when it
 * has no rating, as evalmovedetails does.  mdMae is the mean gap; elo maps the
 * per-band gaps to a CGOS Elo estimate with --elo-map's curves (default
 * out/elo-map-trunc30k-827.json: elo = intercept - A * sum_b exp(k*mid_b) * mae_b),
 * printed only when the MD file is the map's own (sha256 fingerprint of its
 * sorted rows, as evalmovedetails checks).  About 100 net evaluations per MD
 * position at every row, spread over --threads.
 *
 *   train_vnet.bin --evals TRAIN --test-evals TEST [--channels 32] [--blocks 6]
 *       [--hidden 32] [--batch 256] [--lr 0.001] [--epochs 10] [--threads 1]
 *       [--train-pos N] [--test-pos N] [--seed 1] [--save PATH] [--load PATH]
 *       [--md FILE] [--elo-map FILE]
 *
 * --threads parallelises each batch with OpenMP (and the test pass); results
 * depend only on the seed and the thread count.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "game2.h"
#include "cnn.h"

static const char *arg_str(int argc, char **argv, const char *flag, const char *def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}
static int arg_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}
static double wall_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu_s(void)  { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* A duration as "850us" / "2.2ms" / "1.3s". */
static const char *fmt_time(double s, char *buf) {
    if (s < 1e-3) snprintf(buf, 16, "%.0fus", s * 1e6);
    else if (s < 1) snprintf(buf, 16, "%.1fms", s * 1e3);
    else snprintf(buf, 16, "%.1fs", s);
    return buf;
}

/* ── Data ───────────────────────────────────────────────────────────────── */

typedef struct { int8_t code[MAX_CAP]; VnetMeta meta; float label; } Record;
typedef struct { Record *r; int n, cap; } Set;

static int32_t parse_move(const char *s, int N) {
    if (s[0] == 'p') return PASS;
    return (atoi(s + 1) - 1) * N + (s[0] - 'a');
}

/* Load gen-agent-evals records "<size> <phase> <moves> <winRatio>" (comment
 * lines skipped).  One board size per run; a record that does not replay is
 * reported, counted and dropped, never trained on. */
static void load_set(const char *path, Set *S, int *size, int limit) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "train_vnet: cannot open %s\n", path); exit(1); }
    static char line[16384], moves[16384];
    int lineno = 0, bad = 0;
    while (fgets(line, sizeof line, f) && (limit <= 0 || S->n < limit)) {
        lineno++;
        if (line[0] == '#' || line[0] == '\n') continue;
        int n; double label;
        if (sscanf(line, "%d %*s %16383s %lf", &n, moves, &label) != 3) {
            if (++bad <= 5) fprintf(stderr, "train_vnet: %s line %d: not '<size> <phase> <moves> <winRatio>'\n", path, lineno);
            continue;
        }
        if (*size == 0) { *size = n; g2_init_topology(n); }
        else if (n != *size) { fprintf(stderr, "train_vnet: %s line %d: size %d, run is size %d\n", path, lineno, n, *size); exit(1); }
        Game2 g; g2_new(&g, n);
        int ok = 1;
        for (char *tok = strtok(moves, ","); tok; tok = strtok(NULL, ","))
            if (!g2_play(&g, parse_move(tok, n))) { ok = 0; break; }
        if (!ok) { if (++bad <= 5) fprintf(stderr, "train_vnet: %s line %d: an illegal move on replay\n", path, lineno); continue; }
        if (S->n == S->cap) { S->cap = S->cap ? 2 * S->cap : 65536; S->r = realloc(S->r, sizeof(Record) * (size_t)S->cap); }
        Record *R = &S->r[S->n++];
        vnet_compact(&g, R->code, &R->meta);
        R->label = (float)label;
    }
    fclose(f);
    if (bad) fprintf(stderr, "train_vnet: %s: %d record(s) dropped (see above)\n", path, bad);
    if (S->n == 0) { fprintf(stderr, "train_vnet: no records in %s\n", path); exit(1); }
}

/* ── MD 1-ply test (--md) ───────────────────────────────────────────────── */

typedef struct { char m[8]; float wr; } Cand;           /* wr < 0: no rating (terminal) */
typedef struct { Game2 g; double phase; int nc; Cand *c; } MdPos;
typedef struct { MdPos *p; int n; char *rows; } MdSet;   /* rows: the data lines, NUL-separated, for the fingerprint */

static int32_t coord_to_idx(const char *s, int N) { return s[0] == 'p' ? PASS : (atoi(s + 1) - 1) * N + (s[0] - 'a'); }
static void idx_to_coord(int32_t m, int N, char *out) {
    if (m == PASS) strcpy(out, "pass");
    else snprintf(out, 8, "%c%d", 'a' + m % N, m / N + 1);
}

/* f1 <size> <phase> <history> <candidates>: history replays from an EMPTY board
 * (it starts with the centre stone); candidates "move:kwr" best-first. */
static void load_md(const char *path, int N, MdSet *M) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "train_vnet: cannot open --md %s\n", path); exit(1); }
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
            fprintf(stderr, "train_vnet: --md %s line %d: not an f1 row\n", path, lineno); exit(1);
        }
        if (n != N) { fprintf(stderr, "train_vnet: --md %s is size %d, training is size %d\n", path, n, N); exit(1); }
        if (M->n == cap) { cap = cap ? 2 * cap : 1024; M->p = realloc(M->p, sizeof(MdPos) * (size_t)cap); }
        MdPos *P = &M->p[M->n];
        g2_new_empty(&P->g, N);
        if (strcmp(hist, "-") != 0)
            for (char *t = strtok(hist, ","); t; t = strtok(NULL, ","))
                if (!g2_play(&P->g, coord_to_idx(t, N))) { fprintf(stderr, "train_vnet: --md %s line %d: illegal history move %s\n", path, lineno, t); exit(1); }
        P->phase = phase;
        P->nc = 0; P->c = NULL;
        int ccap = 0;
        if (strcmp(cands, "-") != 0)
            for (char *t = strtok(cands, ","); t; t = strtok(NULL, ",")) {
                char *colon = strchr(t, ':');
                if (!colon) { fprintf(stderr, "train_vnet: --md %s line %d: candidate %s has no ':'\n", path, lineno, t); exit(1); }
                if (P->nc == ccap) { ccap = ccap ? 2 * ccap : 64; P->c = realloc(P->c, sizeof(Cand) * (size_t)ccap); }
                Cand *C = &P->c[P->nc++];
                const int ml = (int)(colon - t) < 7 ? (int)(colon - t) : 7;
                memcpy(C->m, t, ml); C->m[ml] = 0;
                C->wr = colon[1] ? atoi(colon + 1) / 1000.0f : -1.0f;
            }
        if (P->nc == 0 || P->c[0].wr < 0) { fprintf(stderr, "train_vnet: --md %s line %d: no rated top candidate\n", path, lineno); exit(1); }
        M->n++;
    }
    fclose(f);
    if (M->n == 0) { fprintf(stderr, "train_vnet: no positions in --md %s\n", path); exit(1); }
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
typedef struct { double intercept, A, k; int buckets, nb; double lo[16], hi[16]; char fp[17]; } EloMap;
static const char *json_after(const char *s, const char *key) { const char *p = strstr(s, key); return p ? p + strlen(key) : NULL; }
static EloMap load_elo_map(const char *path) {
    EloMap E; memset(&E, 0, sizeof E);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "train_vnet: cannot open --elo-map %s\n", path); exit(1); }
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
    fprintf(stderr, "train_vnet: --elo-map %s: missing intercept / phaseBuckets / bands / curves.mae / mdFingerprint\n", path);
    exit(1);
}
static int band_of(double phase, int nb) { int b = (int)floor(phase * nb); return b < 0 ? 0 : b >= nb ? nb - 1 : b; }

/* The 1-ply player's mean gap over the MD set, with per-band sums for the elo. */
static double md_eval(const MdSet *M, const VnetLayout *L, const float *w, const int32_t *nbr9, int area,
                      VnetWork **WK, int buckets, double *band_sum, int *band_n) {
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
        for (int m = 0; m < area; m++) {
            if (P->g.cells[m] != EMPTY || !g2_is_legal(&P->g, m) || g2_is_true_eye_at(&P->g, m)) continue;
            Game2 child; g2_clone(&child, &P->g); g2_play(&child, m);
            vnet_encode(&child, NULL, vnet_work_input(WK[t]));
            const float v = 1 - vnet_forward(L, w, nbr9, area, WK[t]);   /* the opponent is to move in child */
            if (v > best_v) { best_v = v; best = m; }
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

/* ── Training ───────────────────────────────────────────────────────────── */

static uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t z = a * 0x9e3779b9u ^ (b + 0x7f4a7c15u);
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return z ^ (z >> 16);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);   /* every line as it is printed, into a file or pipe too */
    const char *train_f = arg_str(argc, argv, "--evals", NULL), *test_f = arg_str(argc, argv, "--test-evals", NULL);
    if (arg_flag(argc, argv, "--help") || !train_f || !test_f) {
        fprintf(stderr, "usage: train_vnet.bin --evals TRAIN --test-evals TEST [--channels 32] [--blocks 6] [--hidden 32]\n"
                        "         [--batch 256] [--lr 0.001] [--epochs 10] [--threads 1] [--train-pos N] [--test-pos N]\n"
                        "         [--seed 1] [--save PATH] [--load PATH] [--md FILE] [--elo-map FILE]\n");
        return 1;
    }
    const int C = atoi(arg_str(argc, argv, "--channels", "32")), B = atoi(arg_str(argc, argv, "--blocks", "6"));
    const int Hd = atoi(arg_str(argc, argv, "--hidden", "32")), batch = atoi(arg_str(argc, argv, "--batch", "256"));
    const float lr = (float)atof(arg_str(argc, argv, "--lr", "0.001"));
    const int epochs = atoi(arg_str(argc, argv, "--epochs", "10")), threads = atoi(arg_str(argc, argv, "--threads", "1"));
    const int train_cap = atoi(arg_str(argc, argv, "--train-pos", "0")), test_cap = atoi(arg_str(argc, argv, "--test-pos", "0"));
    const uint32_t seed = (uint32_t)atol(arg_str(argc, argv, "--seed", "1"));
    const char *load_f = arg_str(argc, argv, "--load", NULL);
    const char *md_f = arg_str(argc, argv, "--md", NULL);
    const char *elo_f = arg_str(argc, argv, "--elo-map", "out/elo-map-trunc30k-827.json");
    char save_buf[256];
    const char *save_f = arg_str(argc, argv, "--save", NULL);
    if (!save_f) { snprintf(save_buf, sizeof save_buf, "out/vnet-%08x.bin", mix((uint32_t)time(NULL), seed)); save_f = save_buf; }
    char best_f[300];
    {   const char *dot = strrchr(save_f, '.');
        if (dot && dot > strrchr(save_f, '/')) snprintf(best_f, sizeof best_f, "%.*s-best%s", (int)(dot - save_f), save_f, dot);
        else snprintf(best_f, sizeof best_f, "%s-best", save_f); }
    if (C < 1 || B < 0 || B > VNET_MAX_BLOCKS || Hd < 1 || batch < 1 || threads < 1 || epochs < 1) { fprintf(stderr, "train_vnet: bad size/batch/threads/epochs\n"); return 1; }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    if (threads > 1) { fprintf(stderr, "train_vnet: built without OpenMP; --threads must be 1\n"); return 1; }
#endif

    int N = 0;
    Set tr = {0}, te = {0};
    const double t_load = wall_s();
    load_set(train_f, &tr, &N, train_cap);
    load_set(test_f, &te, &N, test_cap);
    const int area = N * N;
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    MdSet md = {0};
    EloMap elo = {0};
    int elo_valid = 0;
    if (md_f) {
        load_md(md_f, N, &md);
        elo = load_elo_map(elo_f);
        char fp[17]; md_fingerprint(&md, fp);
        elo_valid = strcmp(fp, elo.fp) == 0;
        if (!elo_valid) fprintf(stderr, "train_vnet: %s (fingerprint %s) is not the elo map's MD file (%s): elo not estimated\n", md_f, fp, elo.fp);
    }
    static int32_t sym[8 * MAX_CAP]; vnet_sym_map(N, sym);

    VnetLayout L;
    float *w;
    if (load_f) {
        w = vnet_load(load_f, &L);
        if (!w) return 1;
    } else {
        L = vnet_layout(C, B, Hd);
        w = malloc(sizeof(float) * (size_t)L.n_params);
        vnet_init(&L, w, seed);
    }
    const int np = L.n_params;
    float *m = calloc(np, sizeof(float)), *v2 = calloc(np, sizeof(float)), *G = calloc(np, sizeof(float));
    float **TG = malloc(sizeof(float *) * threads);
    VnetWork **WK = malloc(sizeof(VnetWork *) * threads);
    for (int t = 0; t < threads; t++) { TG[t] = calloc(np, sizeof(float)); WK[t] = vnet_work_new(&L, area); }

    printf("train: %s (%d records)  test: %s (%d records)  size %d  [%.1fs to load]\n",
           train_f, tr.n, test_f, te.n, N, wall_s() - t_load);
    printf("model: %d channels, %d blocks, head %d: %d parameters%s%s  batch %d  lr %g  epochs %d  threads %d  seed %u\n",
           L.C, L.B, L.H, np, load_f ? "  loaded from " : "", load_f ? load_f : "", batch, (double)lr, epochs, threads, seed);
    printf("save: %s (best: %s)\n", save_f, best_f);
    if (md_f) printf("md: %s (%d positions), elo map: %s%s\n", md_f, md.n, elo_f, elo_valid ? "" : " (not this MD file: elo -)");
    printf("\n%10s %6s %9s %7s %8s %8s %8s %7s%s\n", "positions", "epoch", "elapsed", "pos/s", "trLoss", "trMSE", "teMSE", "infer",
           md_f ? "   mdMae    elo" : "");

    int *order = malloc(sizeof(int) * tr.n);
    for (int i = 0; i < tr.n; i++) order[i] = i;
    long done = 0, next_row = batch;
    double iv_loss = 0, iv_se = 0; long iv_n = 0;
    double best = 1e30;
    const double t0 = wall_s();
    long step = 0;
    for (int ep = 0; ep < epochs; ep++) {
        /* Fisher-Yates from a per-epoch stream */
        uint32_t s = mix(seed, 1000003u + ep);
        for (int i = tr.n - 1; i > 0; i--) { s = mix(s, i); const int j = s % (uint32_t)(i + 1); const int tmp = order[i]; order[i] = order[j]; order[j] = tmp; }
        for (int b0 = 0; b0 < tr.n; b0 += batch) {
            const int bn = b0 + batch <= tr.n ? batch : tr.n - b0;
            double loss = 0, se = 0;
            #pragma omp parallel for schedule(static) reduction(+:loss, se)
            for (int i = 0; i < bn; i++) {
                int t = 0;
#ifdef _OPENMP
                t = omp_get_thread_num();
#endif
                const Record *R = &tr.r[order[b0 + i]];
                const int sy = (int)(mix(seed ^ (uint32_t)ep * 2654435761u, (uint32_t)order[b0 + i]) & 7);
                vnet_encode_compact(R->code, &R->meta, area, sym + (size_t)sy * area, vnet_work_input(WK[t]));
                const float v = vnet_forward(&L, w, nbr9, area, WK[t]);
                const float vc = v < 1e-7f ? 1e-7f : v > 1 - 1e-7f ? 1 - 1e-7f : v;
                loss += -(R->label * log(vc) + (1 - R->label) * log(1 - vc));
                se += (double)(v - R->label) * (v - R->label);
                vnet_backward(&L, w, nbr9, area, WK[t], v - R->label, TG[t]);
            }
            /* Reduce in thread order (deterministic), then Adam. */
            for (int j = 0; j < np; j++) { float g = 0; for (int t = 0; t < threads; t++) { g += TG[t][j]; TG[t][j] = 0; } G[j] = g / bn; }
            step++;
            const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
            const float c1 = 1 - powf(b1, (float)step), c2 = 1 - powf(b2, (float)step);
            for (int j = 0; j < np; j++) {
                m[j] = b1 * m[j] + (1 - b1) * G[j];
                v2[j] = b2 * v2[j] + (1 - b2) * G[j] * G[j];
                w[j] -= lr * (m[j] / c1) / (sqrtf(v2[j] / c2) + eps);
            }
            done += bn; iv_loss += loss; iv_se += se; iv_n += bn;
            const int epoch_end = b0 + bn >= tr.n;
            if (done >= next_row || epoch_end) {
                double tse = 0;
                const double c0 = cpu_s();
                #pragma omp parallel for schedule(static) reduction(+:tse)
                for (int i = 0; i < te.n; i++) {
                    int t = 0;
#ifdef _OPENMP
                    t = omp_get_thread_num();
#endif
                    vnet_encode_compact(te.r[i].code, &te.r[i].meta, area, NULL, vnet_work_input(WK[t]));
                    const double d = vnet_forward(&L, w, nbr9, area, WK[t]) - te.r[i].label;
                    tse += d * d;
                }
                const double infer = (cpu_s() - c0) / te.n;
                const double temse = tse / te.n, el = wall_s() - t0;
                char ibuf[16];
                const int is_best = temse < best;
                if (is_best) { best = temse; vnet_save(best_f, &L, w); }
                vnet_save(save_f, &L, w);
                char mdcol[40] = "";
                if (md_f) {
                    double bs[16]; int bn_[16];
                    const double mae = md_eval(&md, &L, w, nbr9, area, WK, elo.buckets, bs, bn_);
                    double e = elo.intercept; int have = elo_valid;
                    for (int j = 0; j < elo.nb && have; j++) {
                        const int b = band_of((elo.lo[j] + elo.hi[j]) / 2, elo.buckets);
                        if (bn_[b] == 0) have = 0;
                        else e -= elo.A * exp(elo.k * (elo.lo[j] + elo.hi[j]) / 2) * (bs[b] / bn_[b]);
                    }
                    if (have) snprintf(mdcol, sizeof mdcol, "  %6.4f %6.0f", mae, e);
                    else snprintf(mdcol, sizeof mdcol, "  %6.4f      -", mae);
                }
                printf("%10ld %6.2f %8.1fs %7.0f %8.5f %8.5f %8.5f%s %7s%s\n", done, ep + (double)(b0 + bn) / tr.n, el,
                       done / el, iv_loss / iv_n, iv_se / iv_n, temse, is_best ? "*" : " ", fmt_time(infer, ibuf), mdcol);
                fflush(stdout);
                iv_loss = iv_se = 0; iv_n = 0;
                while (next_row <= done) next_row = (long)(next_row * 1.4) + 1;
            }
        }
    }
    printf("\nbest teMSE %.5f  (%s)\n", best, best_f);
    return 0;
}
