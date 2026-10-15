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
 * --elo-map [FILE] (default out/elo-map-trunc30k-827.json) adds a 1-ply test
 * of the net as a player on the map's own MD file (its mdFile); --md FILE
 * runs the test on another MD file instead.  The test: at every MD position
 * (movedetails, *.md) each legal move except filling one's own true eye (or,
 * with --test-fp-width W > 0, featurepol's top W of them, the --fp-model) is
 * played and valued 1 - V(child) (the opponent is to move there), and the best
 * is charged its gap to the file's top move — the worst rated candidate when it
 * has no rating, as evalmovedetails does.  mdMae is the mean gap; elo maps the
 * per-band gaps to a CGOS Elo estimate with --elo-map's curves (default
 * out/elo-map-trunc30k-827.json: elo = intercept - A * sum_b exp(k*mid_b) * mae_b),
 * printed only when the MD file is the map's own (sha256 fingerprint of its
 * sorted rows, as evalmovedetails checks).  About 100 net evaluations per MD
 * position at full width, spread over --threads, so it runs only at a row where it would
 * keep the MD test's share of the run's wall time within 20%.  Its cost is
 * estimated before it runs: the evaluations it needs (the moves it values,
 * counted at load) x the row's measured CPU time per forward
 * pass / --threads, scaled by how far the last run's actual time was off that
 * estimate.  Other rows print '-'.
 *
 *   train_vnet.bin --evals TRAIN --test-evals TEST [--channels 32] [--blocks 6]
 *       [--hidden 32] [--batch 256] [--lr 0.001] [--epochs 10] [--threads 1]
 *       [--train-pos N] [--test-pos N] [--seed 1] [--save PATH] [--load PATH]
 *       [--elo-map [FILE]] [--md FILE] [--test-fp-width 0] [--fp-model FILE]
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
#include "game3.h"
#include "cnn.h"
#include "fpol.h"
#include "vnet_tool.h"

int main(int argc, char **argv) {
    vt_prog = "train_vnet";
    setvbuf(stdout, NULL, _IOLBF, 0);   /* every line as it is printed, into a file or pipe too */
    const char *train_f = arg_str(argc, argv, "--evals", NULL), *test_f = arg_str(argc, argv, "--test-evals", NULL);
    if (arg_flag(argc, argv, "--help") || !train_f || !test_f) {
        fprintf(stderr, "usage: train_vnet.bin --evals TRAIN --test-evals TEST [--channels 32] [--blocks 6] [--hidden 32]\n"
                        "         [--batch 256] [--lr 0.001] [--epochs 10] [--threads 1] [--train-pos N] [--test-pos N]\n"
                        "         [--seed 1] [--save PATH] [--load PATH] [--elo-map [FILE]] [--md FILE]\n"
                        "         [--test-fp-width 0] [--fp-model ref/ref-fp-stone8AdjLib3.js]\n");
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
    const char *elo_f = arg_opt(argc, argv, "--elo-map", "out/elo-map-trunc30k-827.json");
    if (md_f && !elo_f) elo_f = "out/elo-map-trunc30k-827.json";   /* --md alone: the default map */
    char save_buf[256];
    const char *save_f = arg_str(argc, argv, "--save", NULL);
    const int test_fp_width = atoi(arg_str(argc, argv, "--test-fp-width", "0"));
    const char *fp_f = arg_str(argc, argv, "--fp-model", "ref/ref-fp-stone8AdjLib3.js");
    args_check(argc, argv);
    if (test_fp_width < 0 || (test_fp_width > 0 && !elo_f)) { fprintf(stderr, "train_vnet: --test-fp-width needs W >= 0 and the MD test (--elo-map / --md)\n"); return 1; }
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
    load_set(train_f, &tr, &N, train_cap, threads);
    load_set(test_f, &te, &N, test_cap, threads);
    const int area = N * N;
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    if (test_fp_width > 0) fpol_load(fp_f);   /* exits loudly on a missing file or another spec */
    MdTest *md = elo_f ? md_test_open(elo_f, md_f, N, test_fp_width) : NULL;   /* md_f NULL: the map's own MD file */
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
    float *G = calloc(np, sizeof(float));
    Adam adam = adam_new(np);
    float **TG = malloc(sizeof(float *) * threads);
    VnetWork **WK = malloc(sizeof(VnetWork *) * threads);
    Game3 **G3 = malloc(sizeof(Game3 *) * threads);
    for (int t = 0; t < threads; t++) { TG[t] = calloc(np, sizeof(float)); WK[t] = vnet_work_new(&L, area); G3[t] = g3_new(N); }

    printf("train: %s (%d records)  test: %s (%d records)  size %d  [%.1fs to load]\n",
           train_f, tr.n, test_f, te.n, N, wall_s() - t_load);
    printf("model: %d channels, %d blocks, head %d: %d parameters%s%s  batch %d  lr %g  epochs %d  threads %d  seed %u\n",
           L.C, L.B, L.H, np, load_f ? "  loaded from " : "", load_f ? load_f : "", batch, (double)lr, epochs, threads, seed);
    printf("save: %s (best: %s)\n", save_f, best_f);
    if (md) md_test_describe(md);
    printf("\n%10s %6s %9s %7s %8s %8s %8s %7s%s\n", "positions", "epoch", "elapsed", "pos/s", "trLoss", "trMSE", "teMSE", "infer",
           md ? "   mdMae    elo" : "");

    int *order = malloc(sizeof(int) * tr.n);
    for (int i = 0; i < tr.n; i++) order[i] = i;
    long done = 0, next_row = batch;
    double iv_loss = 0, iv_se = 0; long iv_n = 0;
    double best = 1e30;
    const double t0 = wall_s();
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
            adam_step(&adam, w, G, lr);
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
                if (md) md_test_row(md, &L, w, nbr9, area, WK, G3, threads, infer, wall_s() - t0, mdcol);
                printf("%10ld %6.2f %8.2fm %7.0f %8.5f %8.5f %8.5f%s %7s%s\n", done, ep + (double)(b0 + bn) / tr.n, el / 60,
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
