/*
 * train_vnet_supervised.c — supervised training of the toroidal residual value network
 * (cnn.h) on gen-agent-evals labels: each record's position (replayed from the
 * board with the free centre stone, as gen-agent-evals writes it) against its
 * win ratio, P(side to move wins).
 *
 * Each epoch shuffles the training records and gives every one a random board
 * symmetry (of the 8 reflections/rotations; translations need none — the value
 * is translation-invariant by construction).  Loss: cross-entropy against the
 * soft label.  Optimiser: Adam.  Rows (positions geometric x1.3, plus the
 * end of an --epochs N run) report the interval's training loss and MSE, the test-file MSE
 * (unaugmented) — the same quantity as train-vpat-supervised's teMSE on the
 * same split — and infer, one test record's forward pass (encoding included)
 * in CPU time: the cost on one core whatever --threads is.  pos/s is positions
 * trained over total elapsed time, test passes included.  The checkpoint is
 * written at every row, and -best on a new best elo when the MD test runs on
 * the elo map's own MD file, a new best mdMae when it runs on another (rows it
 * skips cannot be best), else on a new best teMSE.  Without
 * --test-evals there is no teMSE column, and infer is timed on the interval's
 * training forwards (encoding included, augmentation's symmetry map too).
 *
 * --elo-map [FILE] (default out/elo-map-trunc30k-827.json) adds a 1-ply test
 * of the net as a player on the map's own MD file (its mdFile); --md FILE
 * runs the test on another MD file instead.  The test: at every MD position
 * (movedetails, *.md) each legal move except filling one's own true eye (or,
 * with --test-fp-width W > 0, default 7, featurepol's top W of them, the
 * --fp-model; 0 = all) is
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
 * --lr-warmup STEPS: the learning rate ramps linearly up to --lr over the
 * first STEPS Adam steps (default 500; 0 = off).  Adam's first steps move every
 * weight by about the rate whatever its gradient's size, which on --load
 * (fresh moments) wrecks the loaded net — and a grown net's zero-initialised
 * blocks — at the full rate.
 *
 * --smooth-weights A: a Polyak average of the weights (train-vpat-td's knob),
 * updated after every Adam step as ema = A*ema + (1-A)*w; the test pass, the MD
 * test, the checkpoint and -best all use it, training continues on w.  auto
 * (the default): A = 1 - 4/step, a window of about the last quarter of the
 * run's steps; 0 = off (the raw weights throughout).
 *
 *   train_vnet_supervised.bin --evals TRAIN [--test-evals TEST] [--channels 32] [--blocks 6]
 *       [--hidden 32] [--batch 256] [--lr 0.001] [--epochs 0] [--threads 1]
 *       [--train-pos N] [--test-pos N] [--seed N] [--save PATH] [--load PATH]
 *       [--elo-map [FILE]] [--md FILE] [--test-fp-width 7] [--fp-model FILE]
 *       [--smooth-weights auto] [--lr-warmup 500]
 *
 * --load takes the file's layout: --channels / --hidden, when given, must
 * match it, and --blocks above its count grows the net (vnet_grow_blocks:
 * identity blocks appended, so the first rows start at the loaded net's
 * strength); --blocks below it is an error.
 *
 * --epochs N stops after N epochs; 0 (the default) trains until stopped.
 *
 * --threads parallelises each batch with OpenMP (and the test pass); results
 * depend only on the seed and the thread count.  The seed is random unless
 * --seed is given; the model line prints it.
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
    vt_prog = "train_vnet_supervised";
    setvbuf(stdout, NULL, _IOLBF, 0);   /* every line as it is printed, into a file or pipe too */
    const char *train_f = arg_str(argc, argv, "--evals", NULL), *test_f = arg_str(argc, argv, "--test-evals", NULL);
    if (arg_flag(argc, argv, "--help") || !train_f) {
        fprintf(stderr, "usage: train_vnet_supervised.bin --evals TRAIN [--test-evals TEST] [--channels 32] [--blocks 6] [--hidden 32]\n"
                        "         [--batch 256] [--lr 0.001] [--epochs 0] [--threads 1] [--train-pos N] [--test-pos N]\n"
                        "         [--seed N] [--save PATH] [--load PATH] [--elo-map [FILE]] [--md FILE]\n"
                        "         [--test-fp-width 7] [--fp-model ref/ref-fp-stone8AdjLib3.js] [--smooth-weights auto]\n"
                        "         [--lr-warmup 500]\n");
        return 1;
    }
    const char *C_s = arg_str(argc, argv, "--channels", NULL), *B_s = arg_str(argc, argv, "--blocks", NULL);
    const char *H_s = arg_str(argc, argv, "--hidden", NULL);
    const int C = atoi(C_s ? C_s : "32"), B = atoi(B_s ? B_s : "6"), Hd = atoi(H_s ? H_s : "32");
    const int batch = atoi(arg_str(argc, argv, "--batch", "256"));
    const float lr = (float)atof(arg_str(argc, argv, "--lr", "0.001"));
    const int warmup = atoi(arg_str(argc, argv, "--lr-warmup", "500"));
    if (warmup < 0) { fprintf(stderr, "train_vnet_supervised: --lr-warmup must be >= 0 (Adam steps)\n"); return 1; }
    const char *smooth_s = arg_str(argc, argv, "--smooth-weights", "auto");
    const int smooth_auto = !strcmp(smooth_s, "auto");
    char *smooth_end;
    const double smooth_a = smooth_auto ? 0 : strtod(smooth_s, &smooth_end);
    if (!smooth_auto && (*smooth_end || !(smooth_a >= 0 && smooth_a < 1))) {
        fprintf(stderr, "train_vnet_supervised: --smooth-weights takes auto or a decay in [0, 1) (got '%s')\n", smooth_s); return 1;
    }
    const int smooth = smooth_auto || smooth_a > 0;
    const int epochs = atoi(arg_str(argc, argv, "--epochs", "0")), threads = atoi(arg_str(argc, argv, "--threads", "1"));
    const int train_cap = atoi(arg_str(argc, argv, "--train-pos", "0")), test_cap = atoi(arg_str(argc, argv, "--test-pos", "0"));
    const uint32_t seed = arg_seed(argc, argv);   /* default: random (printed) */
    const char *load_f = arg_str(argc, argv, "--load", NULL);
    const char *md_f = arg_str(argc, argv, "--md", NULL);
    const char *elo_f = arg_opt(argc, argv, "--elo-map", "out/elo-map-trunc30k-827.json");
    if (md_f && !elo_f) elo_f = "out/elo-map-trunc30k-827.json";   /* --md alone: the default map */
    char save_buf[256];
    const char *save_f = arg_str(argc, argv, "--save", NULL);
    const char *test_fp_s = arg_str(argc, argv, "--test-fp-width", NULL);
    const int test_fp_width = atoi(test_fp_s ? test_fp_s : "7");
    const char *fp_f = arg_str(argc, argv, "--fp-model", "ref/ref-fp-stone8AdjLib3.js");
    args_check(argc, argv);
    if (test_fp_width < 0 || (test_fp_s && !elo_f)) { fprintf(stderr, "train_vnet_supervised: --test-fp-width needs W >= 0 and the MD test (--elo-map / --md)\n"); return 1; }
    if (!save_f) { snprintf(save_buf, sizeof save_buf, "out/vnet-%08x.bin", mix((uint32_t)time(NULL), seed)); save_f = save_buf; }
    char best_f[300];
    {   const char *dot = strrchr(save_f, '.');
        if (dot && dot > strrchr(save_f, '/')) snprintf(best_f, sizeof best_f, "%.*s-best%s", (int)(dot - save_f), save_f, dot);
        else snprintf(best_f, sizeof best_f, "%s-best", save_f); }
    if (C < 1 || B < 0 || B > VNET_MAX_BLOCKS || Hd < 1 || batch < 1 || threads < 1 || epochs < 0) { fprintf(stderr, "train_vnet_supervised: bad size/batch/threads/epochs\n"); return 1; }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    if (threads > 1) { fprintf(stderr, "train_vnet_supervised: built without OpenMP; --threads must be 1\n"); return 1; }
#endif

    int N = 0;
    Set tr = {0}, te = {0};
    const double t_load = wall_s();
    load_set(train_f, &tr, &N, train_cap, threads);
    if (test_f) load_set(test_f, &te, &N, test_cap, threads);
    const int area = N * N;
    int32_t nbr9[MAX_CAP * 9]; vnet_topology(N, nbr9);
    if (elo_f && test_fp_width > 0) fpol_load(fp_f);   /* exits loudly on a missing file or another spec */
    MdTest *md = elo_f ? md_test_open(elo_f, md_f, N, test_fp_width) : NULL;   /* md_f NULL: the map's own MD file */
    static int32_t sym[8 * MAX_CAP]; vnet_sym_map(N, sym);

    VnetLayout L;
    float *w;
    int grown_from = 0;   /* the loaded net's block count when --blocks grew it */
    if (load_f) {
        w = vnet_load(load_f, &L);
        if (!w) return 1;
        /* --load takes the file's layout; --blocks above its count grows it
         * (identity blocks appended); any other size flag must match. */
        if ((C_s && C != L.C) || (H_s && Hd != L.H) || (B_s && B < L.B)) {
            fprintf(stderr, "train_vnet_supervised: %s is %d channels, %d blocks, head %d: --channels / --hidden must match it, "
                            "--blocks must be at least %d (more grows it)\n", load_f, L.C, L.B, L.H, L.B);
            return 1;
        }
        if (B_s && B > L.B) {
            VnetLayout G;
            float *g = vnet_grow_blocks(&L, w, B, seed, &G);
            free(w);
            grown_from = L.B;
            w = g; L = G;
        }
    } else {
        L = vnet_layout(C, B, Hd);
        w = malloc(sizeof(float) * (size_t)L.n_params);
        vnet_init(&L, w, seed);
    }
    const int np = L.n_params;
    /* ew: the weights tested and saved — the Polyak average, or w itself when off */
    float *ema = smooth ? malloc(sizeof(float) * (size_t)np) : NULL;
    if (ema) memcpy(ema, w, sizeof(float) * (size_t)np);
    const float *ew = ema ? ema : w;
    float *G = calloc(np, sizeof(float));
    Adam adam = adam_new(np);
    float **TG = malloc(sizeof(float *) * threads);
    VnetWork **WK = malloc(sizeof(VnetWork *) * threads);
    Game3 **G3 = malloc(sizeof(Game3 *) * threads);
    for (int t = 0; t < threads; t++) { TG[t] = calloc(np, sizeof(float)); WK[t] = vnet_work_new(&L, area); G3[t] = g3_new(N); }

    printf("train: %s (%d records)  ", train_f, tr.n);
    if (test_f) printf("test: %s (%d records)  ", test_f, te.n);
    else printf("test: (none)  ");
    printf("size %d  [%.1fs to load]\n", N, wall_s() - t_load);
    printf("model: %d channels, %d blocks, head %d: %d parameters%s%s  batch %d  lr %g  epochs %d  threads %d  seed %u  smooth-weights %s  lr-warmup %d\n",
           L.C, L.B, L.H, np, load_f ? "  loaded from " : "", load_f ? load_f : "", batch, (double)lr, epochs, threads, seed,
           smooth_auto ? "auto" : smooth ? smooth_s : "off", warmup);
    const char *best_by = md ? (md_test_elo_valid(md) ? "elo" : "mdMae") : "teMSE";
    if (grown_from) printf("grown: %d -> %d blocks (the new ones start as the identity)\n", grown_from, L.B);
    if (md || test_f) printf("save: %s (best by %s: %s)\n", save_f, best_by, best_f);
    else printf("save: %s\n", save_f);
    if (md) md_test_describe(md);
    printf("\n%10s %6s %9s %7s %8s %8s%s %7s%s\n", "positions", "epoch", "elapsed", "pos/s", "trLoss", "trMSE", test_f ? "    teMSE " : "", "infer",
           md ? md_test_header(md) : "");

    int *order = malloc(sizeof(int) * tr.n);
    for (int i = 0; i < tr.n; i++) order[i] = i;
    long done = 0, next_row = batch;
    double iv_loss = 0, iv_se = 0, iv_fcpu = 0; long iv_n = 0;   /* iv_fcpu: training forwards' CPU, timed without a test set */
    double best = 1e30;
    const double t0 = wall_s();
    for (int ep = 0; epochs == 0 || ep < epochs; ep++) {   /* --epochs 0: until stopped */
        /* Fisher-Yates from a per-epoch stream */
        uint32_t s = mix(seed, 1000003u + ep);
        for (int i = tr.n - 1; i > 0; i--) { s = mix(s, i); const int j = s % (uint32_t)(i + 1); const int tmp = order[i]; order[i] = order[j]; order[j] = tmp; }
        for (int b0 = 0; b0 < tr.n; b0 += batch) {
            const int bn = b0 + batch <= tr.n ? batch : tr.n - b0;
            double loss = 0, se = 0, fcpu = 0;
            #pragma omp parallel for schedule(static) reduction(+:loss, se, fcpu)
            for (int i = 0; i < bn; i++) {
                int t = 0;
#ifdef _OPENMP
                t = omp_get_thread_num();
#endif
                const Record *R = &tr.r[order[b0 + i]];
                const int sy = (int)(mix(seed ^ (uint32_t)ep * 2654435761u, (uint32_t)order[b0 + i]) & 7);
                const double c0 = test_f ? 0 : thread_cpu_s();
                vnet_encode_compact(R->code, &R->meta, area, sym + (size_t)sy * area, vnet_work_input(WK[t]));
                const float v = vnet_forward(&L, w, nbr9, area, WK[t]);
                if (!test_f) fcpu += thread_cpu_s() - c0;
                const float vc = v < 1e-7f ? 1e-7f : v > 1 - 1e-7f ? 1 - 1e-7f : v;
                loss += -(R->label * log(vc) + (1 - R->label) * log(1 - vc));
                se += (double)(v - R->label) * (v - R->label);
                vnet_backward(&L, w, nbr9, area, WK[t], v - R->label, TG[t]);
            }
            /* Reduce in thread order (deterministic), then Adam. */
            for (int j = 0; j < np; j++) { float g = 0; for (int t = 0; t < threads; t++) { g += TG[t][j]; TG[t][j] = 0; } G[j] = g / bn; }
            /* Warm-up: Adam's first steps move every weight by about the rate
             * whatever its gradient's size (fresh moments), which wrecks a
             * loaded net; ramping the rate up keeps those steps small. */
            adam_step(&adam, w, G, adam.step < warmup ? lr * (float)(adam.step + 1) / (float)warmup : lr);
            if (ema) {
                const float a = smooth_auto ? (adam.step > 4 ? 1.0f - 4.0f / (float)adam.step : 0.0f) : (float)smooth_a;
                for (int j = 0; j < np; j++) ema[j] = a * ema[j] + (1 - a) * w[j];
            }
            done += bn; iv_loss += loss; iv_se += se; iv_fcpu += fcpu; iv_n += bn;
            const int run_end = epochs > 0 && ep == epochs - 1 && b0 + bn >= tr.n;
            if (done >= next_row || run_end) {
                double tse = 0;
                const double c0 = cpu_s();
                #pragma omp parallel for schedule(static) reduction(+:tse)
                for (int i = 0; i < te.n; i++) {
                    int t = 0;
#ifdef _OPENMP
                    t = omp_get_thread_num();
#endif
                    vnet_encode_compact(te.r[i].code, &te.r[i].meta, area, NULL, vnet_work_input(WK[t]));
                    const double d = vnet_forward(&L, ew, nbr9, area, WK[t]) - te.r[i].label;
                    tse += d * d;
                }
                const double infer = test_f ? (cpu_s() - c0) / te.n : iv_fcpu / iv_n;
                const double temse = test_f ? tse / te.n : 0, el = wall_s() - t0;
                char ibuf[16], tcol[16] = "";
                /* -best: by the MD test (elo, else mdMae) when it runs, else by teMSE */
                const int te_best = !md && test_f && temse < best;
                if (te_best) { best = temse; vnet_save(best_f, &L, ew); }
                if (test_f) snprintf(tcol, sizeof tcol, " %8.5f%s", temse, te_best ? "*" : " ");
                vnet_save(save_f, &L, ew);
                char mdcol[40] = "";
                if (md && md_test_row(md, &L, ew, nbr9, area, WK, G3, threads, infer, wall_s() - t0, &best, mdcol))
                    vnet_save(best_f, &L, ew);
                printf("%10ld %6.2f %8.2fm %7.0f %8.5f %8.5f%s %7s%s\n", done, ep + (double)(b0 + bn) / tr.n, el / 60,
                       done / el, iv_loss / iv_n, iv_se / iv_n, tcol, fmt_time(infer, ibuf), mdcol);
                fflush(stdout);
                iv_loss = iv_se = iv_fcpu = 0; iv_n = 0;
                while (next_row <= done) next_row = (long)(next_row * 1.3) + 1;
            }
        }
    }
    if (best < 1e30) {
        if (md && md_test_elo_valid(md)) printf("\nbest elo %.0f  (%s)\n", -best, best_f);
        else printf("\nbest %s %.5f  (%s)\n", best_by, best, best_f);
    }
    return 0;
}
