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
 * epoch end) report the interval's training loss and MSE and the test-file MSE
 * (unaugmented) — the same quantity as train-vpat-supervised's teMSE on the
 * same split.  The checkpoint is written at every row and -best on a new best
 * teMSE.
 *
 *   train_vnet.bin --evals TRAIN --test-evals TEST [--channels 32] [--blocks 6]
 *       [--hidden 32] [--batch 256] [--lr 0.001] [--epochs 10] [--threads 1]
 *       [--train-pos N] [--test-pos N] [--seed 1] [--save PATH] [--load PATH]
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

/* ── Training ───────────────────────────────────────────────────────────── */

static uint32_t mix(uint32_t a, uint32_t b) {
    uint32_t z = a * 0x9e3779b9u ^ (b + 0x7f4a7c15u);
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return z ^ (z >> 16);
}

int main(int argc, char **argv) {
    const char *train_f = arg_str(argc, argv, "--evals", NULL), *test_f = arg_str(argc, argv, "--test-evals", NULL);
    if (arg_flag(argc, argv, "--help") || !train_f || !test_f) {
        fprintf(stderr, "usage: train_vnet.bin --evals TRAIN --test-evals TEST [--channels 32] [--blocks 6] [--hidden 32]\n"
                        "         [--batch 256] [--lr 0.001] [--epochs 10] [--threads 1] [--train-pos N] [--test-pos N]\n"
                        "         [--seed 1] [--save PATH] [--load PATH]\n");
        return 1;
    }
    const int C = atoi(arg_str(argc, argv, "--channels", "32")), B = atoi(arg_str(argc, argv, "--blocks", "6"));
    const int Hd = atoi(arg_str(argc, argv, "--hidden", "32")), batch = atoi(arg_str(argc, argv, "--batch", "256"));
    const float lr = (float)atof(arg_str(argc, argv, "--lr", "0.001"));
    const int epochs = atoi(arg_str(argc, argv, "--epochs", "10")), threads = atoi(arg_str(argc, argv, "--threads", "1"));
    const int train_cap = atoi(arg_str(argc, argv, "--train-pos", "0")), test_cap = atoi(arg_str(argc, argv, "--test-pos", "0"));
    const uint32_t seed = (uint32_t)atol(arg_str(argc, argv, "--seed", "1"));
    const char *load_f = arg_str(argc, argv, "--load", NULL);
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
    printf("save: %s (best: %s)\n\n", save_f, best_f);
    printf("%10s %6s %9s %7s %8s %8s %8s\n", "positions", "epoch", "elapsed", "pos/s", "trLoss", "trMSE", "teMSE");

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
                const double temse = tse / te.n, el = wall_s() - t0;
                const int is_best = temse < best;
                if (is_best) { best = temse; vnet_save(best_f, &L, w); }
                vnet_save(save_f, &L, w);
                printf("%10ld %6.2f %8.1fs %7.0f %8.5f %8.5f %8.5f%s\n", done, ep + (double)(b0 + bn) / tr.n, el,
                       done / el, iv_loss / iv_n, iv_se / iv_n, temse, is_best ? "*" : "");
                fflush(stdout);
                iv_loss = iv_se = 0; iv_n = 0;
                while (next_row <= done) next_row = (long)(next_row * 1.4) + 1;
            }
        }
    }
    printf("\nbest teMSE %.5f  (%s)\n", best, best_f);
    return 0;
}
