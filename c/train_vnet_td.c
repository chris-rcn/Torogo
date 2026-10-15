/*
 * train_vnet_td.c — TD(0) self-play training of the toroidal residual value
 * network (cnn.h): train-vpat-td's scheme with the net in place of vpat.
 *
 * Self-play: each game starts from the empty board with black's centre stone
 * played.  A move is uniformly random with probability --epsilon; otherwise,
 * with probability --ext-ratio, --ext's; otherwise the net's own: the legal
 * non-eye move (among featurepol's top --fp-width, 0 = all) with the best
 * 1 - V(child), PASS when there is none.  --bootstrap N plays the first N games
 * fully random.  Games are played in rounds of 2 x --threads, in parallel, with
 * the weights fixed; each game's moves depend only on its seed and the weights.
 *
 * Targets, 2-ply TD(0) as train-vpat-td: every position s_t moves toward
 * V(s_t+2), the next position with the same player to move, evaluated with the
 * current weights when the sample is drawn; a game's last two positions move
 * toward its outcome.  V is P(side to move wins), so no flip.  Loss:
 * cross-entropy against the target (no gradient through it).  Optimiser: Adam.
 *
 * Positions go into a window of the last --window; after each round the
 * trainer owes --reuse samples per new position and pays them in batches of
 * --batch, each sample drawn uniformly (with replacement) from the window with
 * a random board symmetry.
 *
 * Rows (games geometric x1.4, and the last game of a --games run): play% is
 * the interval's wall time spent in self-play; komi the scoring komi; bWin% and
 * moves black's win share and the mean game length over the interval's games
 * (bootstrap games included); trLoss and trMSE against the TD targets; teMSE
 * (with --test-evals) against those oracle labels, as train_vnet's; infer the
 * CPU time of one target evaluation (compact encoding + forward); mdMae / elo
 * the MD 1-ply test, as train_vnet's (--elo-map / --md, the same 20% wall-time
 * gate).  The checkpoint is written at every row.
 *
 * --komi auto steps komi by +/-1 every 500 non-bootstrap games while black's
 * win share sits outside [45%, 55%] (auto:<start> seeds it; a VNET1 file does
 * not carry komi).  The default is fixed at the game's komi (3.5 on 13x13), the
 * currency of the MD file's and the ptw1000 sets' labels.
 *
 * --ext random | fp | puct: g2_random_legal_move; featurepol (fpol.c, the
 * --fp-model, stone8AdjLib3 only) sampled at temperature 1; or puct.c (the
 * puct-ppat-fp search with C defaults, --fp-model priors, --ppat-model
 * playouts from phase 0.55) at --ext-playouts.
 *
 *   train_vnet_td.bin [--size 13] [--channels 32] [--blocks 6] [--hidden 32]
 *       [--load PATH] [--save PATH] [--games N] [--bootstrap N] [--epsilon 0.1]
 *       [--ext random|fp|puct --ext-ratio R] [--ext-playouts 1000]
 *       [--fp-width 5] [--fp-model FILE] [--ppat-model FILE] [--komi K]
 *       [--window 200000] [--reuse 1] [--batch 256] [--lr 0.001] [--threads 1]
 *       [--seed 1] [--test-evals FILE] [--test-pos N] [--elo-map [FILE]] [--md FILE]
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
#include "ppat.h"
#include "puct.h"
#include "rng.h"
#include "vnet_tool.h"

#define KOMI_WINDOW 500
#define PUCT_PPAT_MIN_PHASE 0.55   /* puct-ppat-fp's PPAT_MIN_PHASE */

enum { EXT_NONE, EXT_RANDOM, EXT_FP, EXT_PUCT };

/* Run-wide settings, fixed after main's parsing. */
static int N, area, fp_width, ext, ext_playouts;
static double epsilon, ext_ratio, komi;
static VnetLayout L;
static int32_t nbr9[MAX_CAP * 9];
static PuctCfg puct_cfg;

/* Per-thread state. */
typedef struct {
    VnetWork *wk;
    Game3 *g3;
    FpolState fs;
    PuctSearch *puct;
} Thread;

/* One game's positions, compacted. */
typedef struct {
    int8_t *code;          /* n * area */
    VnetMeta *meta;
    int n, cap;
    int8_t winner;
} GameRec;

static double thread_cpu(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int8_t winner_of(const Game2 *g) {
    const Score s = g2_estimate_score(g);   /* white's includes g2_komi */
    return s.black > s.white - g2_komi(g->N) + komi ? BLACK : WHITE;
}

/* The net's move: the candidate with the best 1 - V(child). */
static int32_t net_move(const Game2 *g, const float *w, Thread *T) {
    int32_t cand[MAX_CAP];
    int nc = 0;
    if (fp_width > 0) {
        fpol_eval(g, &T->fs, 1.0);
        int order[MAX_CAP];
        fpol_rank(&T->fs, order);
        for (int i = 0; i < T->fs.n && i < fp_width; i++) cand[nc++] = T->fs.moves[order[i]];
    } else {
        for (int m = 0; m < area; m++)
            if (g->cells[m] == EMPTY && g2_is_legal(g, m) && !g2_is_true_eye_at(g, m)) cand[nc++] = m;
    }
    int32_t best = PASS; float best_v = -1;
    for (int i = 0; i < nc; i++) {
        Game2 child; g2_clone(&child, g);
        if (!g2_play(&child, cand[i])) continue;
        float v;
        if (child.game_over) v = winner_of(&child) == g->current;
        else {
            vnet_encode(&child, T->g3, NULL, vnet_work_input(T->wk));
            v = 1 - vnet_forward(&L, w, nbr9, area, T->wk);   /* the opponent is to move in child */
        }
        if (v > best_v) { best_v = v; best = cand[i]; }
    }
    return best;
}

static void play_game(uint32_t seed, int random_only, const float *w, Thread *T, GameRec *R) {
    Rng r; rng_seed(&r, (long)seed);
    Game2 g; g2_new_empty(&g, N);
    g2_play(&g, (N / 2) * N + N / 2);   /* black's centre stone */
    R->n = 0;
    while (!g.game_over) {
        if (R->n == R->cap) {
            R->cap = R->cap ? 2 * R->cap : 256;
            R->code = realloc(R->code, (size_t)R->cap * area);
            R->meta = realloc(R->meta, sizeof(VnetMeta) * (size_t)R->cap);
        }
        vnet_compact(&g, T->g3, R->code + (size_t)R->n * area, &R->meta[R->n]);
        R->n++;
        int32_t mv;
        if (random_only || rng_float(&r) < epsilon) mv = g2_random_legal_move(&g, &r);
        else if (ext != EXT_NONE && rng_float(&r) < ext_ratio) {
            if (ext == EXT_RANDOM) mv = g2_random_legal_move(&g, &r);
            else if (ext == EXT_FP) { fpol_eval(&g, &T->fs, 1.0); mv = fpol_sample(&T->fs, &r); }
            else mv = puct_get_move(T->puct, &g, ext_playouts, &r).move;
        } else mv = net_move(&g, w, T);
        if (!g2_play(&g, mv)) { fprintf(stderr, "train_vnet_td: illegal move %d at ply %d\n", mv, g.move_count); exit(1); }
    }
    R->winner = winner_of(&g);
}

static int parse_ext(const char *s) {
    if (!s) return EXT_NONE;
    if (!strcmp(s, "random")) return EXT_RANDOM;
    if (!strcmp(s, "fp")) return EXT_FP;
    if (!strcmp(s, "puct")) return EXT_PUCT;
    fprintf(stderr, "train_vnet_td: --ext must be random, fp or puct (got '%s')\n", s); exit(1);
}

int main(int argc, char **argv) {
    vt_prog = "train_vnet_td";
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (arg_flag(argc, argv, "--help")) {
        fprintf(stderr, "usage: train_vnet_td.bin [--size 13] [--channels 32] [--blocks 6] [--hidden 32] [--load PATH] [--save PATH]\n"
                        "         [--games N] [--bootstrap N] [--epsilon 0.1] [--ext random|fp|puct --ext-ratio R] [--ext-playouts 1000]\n"
                        "         [--fp-width 5] [--fp-model FILE] [--ppat-model FILE] [--komi auto|auto:<start>|<number>]\n"
                        "         [--window 200000] [--reuse 1] [--batch 256] [--lr 0.001] [--threads 1] [--seed 1]\n"
                        "         [--test-evals FILE] [--test-pos N] [--elo-map [FILE]] [--md FILE]\n");
        return 1;
    }
    N = atoi(arg_str(argc, argv, "--size", "13"));
    const int C = atoi(arg_str(argc, argv, "--channels", "32")), B = atoi(arg_str(argc, argv, "--blocks", "6"));
    const int Hd = atoi(arg_str(argc, argv, "--hidden", "32")), batch = atoi(arg_str(argc, argv, "--batch", "256"));
    const float lr = (float)atof(arg_str(argc, argv, "--lr", "0.001"));
    const int threads = atoi(arg_str(argc, argv, "--threads", "1"));
    const long games_limit = atol(arg_str(argc, argv, "--games", "0")), bootstrap = atol(arg_str(argc, argv, "--bootstrap", "0"));
    epsilon = atof(arg_str(argc, argv, "--epsilon", "0.1"));
    const char *ext_s = arg_str(argc, argv, "--ext", NULL);
    ext = parse_ext(ext_s);
    ext_ratio = atof(arg_str(argc, argv, "--ext-ratio", "0"));
    ext_playouts = atoi(arg_str(argc, argv, "--ext-playouts", "1000"));
    fp_width = atoi(arg_str(argc, argv, "--fp-width", "5"));
    const char *fp_f = arg_str(argc, argv, "--fp-model", "ref/ref-fp-stone8AdjLib3.js");
    const char *ppat_f = arg_str(argc, argv, "--ppat-model", "out/ppat-data-1623136.js");
    const char *komi_s = arg_str(argc, argv, "--komi", NULL);
    const long window = atol(arg_str(argc, argv, "--window", "200000"));
    const double reuse = atof(arg_str(argc, argv, "--reuse", "1"));
    const uint32_t seed = (uint32_t)atol(arg_str(argc, argv, "--seed", "1"));
    const char *load_f = arg_str(argc, argv, "--load", NULL);
    const char *test_f = arg_str(argc, argv, "--test-evals", NULL);
    const int test_cap = atoi(arg_str(argc, argv, "--test-pos", "0"));
    const char *md_f = arg_str(argc, argv, "--md", NULL);
    const char *elo_f = arg_opt(argc, argv, "--elo-map", "out/elo-map-trunc30k-827.json");
    if (md_f && !elo_f) elo_f = "out/elo-map-trunc30k-827.json";   /* --md alone: the default map */
    char save_buf[256];
    const char *save_f = arg_str(argc, argv, "--save", NULL);
    if (!save_f) { snprintf(save_buf, sizeof save_buf, "out/vnet-td-%08x.bin", mix((uint32_t)time(NULL), seed)); save_f = save_buf; }

    if (N < 3 || N > MAX_BOARD_SIZE || C < 1 || B < 0 || B > VNET_MAX_BLOCKS || Hd < 1 || batch < 1 || threads < 1 || window < 3) {
        fprintf(stderr, "train_vnet_td: bad size/channels/blocks/batch/threads/window\n"); return 1;
    }
    if (!(epsilon >= 0 && epsilon <= 1) || !(ext_ratio >= 0 && ext_ratio <= 1) || !(reuse > 0)) {
        fprintf(stderr, "train_vnet_td: --epsilon and --ext-ratio must be in [0, 1], --reuse > 0\n"); return 1;
    }
    if ((ext_ratio > 0) != (ext != EXT_NONE)) { fprintf(stderr, "train_vnet_td: --ext and --ext-ratio > 0 go together\n"); return 1; }
    if (fp_width < 0 || (ext == EXT_PUCT && ext_playouts < 1)) { fprintf(stderr, "train_vnet_td: bad --fp-width / --ext-playouts\n"); return 1; }
    int auto_komi = 0;
    g2_init_topology(N);
    komi = g2_komi(N);
    if (komi_s) {
        if (!strncmp(komi_s, "auto", 4)) {
            auto_komi = 1;
            if (komi_s[4] == ':') komi = atof(komi_s + 5);
            else if (komi_s[4]) { fprintf(stderr, "train_vnet_td: --komi auto | auto:<start> | <number>\n"); return 1; }
        } else komi = atof(komi_s);
    }
#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    if (threads > 1) { fprintf(stderr, "train_vnet_td: built without OpenMP; --threads must be 1\n"); return 1; }
#endif

    area = N * N;
    vnet_topology(N, nbr9);
    if (fp_width > 0 || ext == EXT_FP || ext == EXT_PUCT) {
        fpol_load(fp_f);   /* exits loudly on a missing file or another spec */
        /* fpol builds its neighbour table on first use: once here, not racing in the threads */
        Game2 g; g2_new_empty(&g, N); FpolState fs; fpol_eval(&g, &fs, 1.0);
    }
    if (ext == EXT_PUCT) {
        puct_cfg = puct_default_cfg();
        puct_cfg.ppat_min_phase = PUCT_PPAT_MIN_PHASE;
        puct_cfg.ppat_w = ppat_load_model(ppat_f, &puct_cfg.early_pass, &puct_cfg.pass_weight);
        if (!puct_cfg.ppat_w) { fprintf(stderr, "train_vnet_td: cannot load --ppat-model %s\n", ppat_f); return 1; }
    }
    Set te = {0};
    if (test_f) {
        int sz = N;
        load_set(test_f, &te, &sz, test_cap);
    }
    MdTest *md = elo_f ? md_test_open(elo_f, md_f, N) : NULL;
    static int32_t sym[8 * MAX_CAP]; vnet_sym_map(N, sym);

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
    Thread *TH = calloc(threads, sizeof(Thread));
    VnetWork **WK = malloc(sizeof(VnetWork *) * threads);
    Game3 **G3 = malloc(sizeof(Game3 *) * threads);
    for (int t = 0; t < threads; t++) {
        TG[t] = calloc(np, sizeof(float));
        TH[t].wk = WK[t] = vnet_work_new(&L, area);
        TH[t].g3 = G3[t] = g3_new(N);
        if (ext == EXT_PUCT) TH[t].puct = puct_new(&puct_cfg);
    }
    const int round_games = 2 * threads;
    GameRec *rec = calloc(round_games, sizeof(GameRec));

    /* The window: position k at slot k % window; a slot's TD successor is the
     * slot two on (same game), or its outcome when out[slot] >= 0. */
    int8_t *wcode = malloc((size_t)window * area);
    VnetMeta *wmeta = malloc(sizeof(VnetMeta) * (size_t)window);
    float *wout = malloc(sizeof(float) * (size_t)window);
    long total_pos = 0;

    printf("model: %d channels, %d blocks, head %d: %d parameters%s%s  batch %d  lr %g  threads %d  seed %u\n",
           L.C, L.B, L.H, np, load_f ? "  loaded from " : "", load_f ? load_f : "", batch, (double)lr, threads, seed);
    printf("self-play: size %d  epsilon %g  ext %s", N, epsilon, ext_s ? ext_s : "(none)");
    if (ext != EXT_NONE) printf("  ext-ratio %g", ext_ratio);
    if (ext == EXT_PUCT) printf("  ext-playouts %d  ppat-model %s", ext_playouts, ppat_f);
    if (fp_width > 0 || ext == EXT_FP || ext == EXT_PUCT) printf("  fp-model %s", fp_f);
    printf("  fp-width %d  komi %s%g  bootstrap %ld  games %ld\n", fp_width, auto_komi ? "auto from " : "", komi, bootstrap, games_limit);
    printf("training: window %ld  reuse %g  save %s\n", window, reuse, save_f);
    if (test_f) printf("test: %s (%d records)\n", test_f, te.n);
    if (md) md_test_describe(md);
    printf("\n%10s %10s %9s %6s %5s %6s %6s %8s %8s%s %7s%s\n", "games", "positions", "elapsed", "play%", "komi", "bWin%", "moves",
           "trLoss", "trMSE", test_f ? "    teMSE" : "", "infer", md ? "   mdMae    elo" : "");

    long games = 0, next_row = round_games, step = 0;
    double owed = 0;
    long iv_games = 0, iv_bwins = 0, iv_moves = 0, iv_n = 0, iv_nf = 0;
    double iv_loss = 0, iv_se = 0, iv_play = 0, iv_fcpu = 0, infer = 0;
    long komi_games = 0, komi_bwins = 0;
    const double t0 = wall_s();
    double iv_t0 = t0;
    for (;;) {
        /* Self-play round. */
        const long left = games_limit > 0 ? games_limit - games : round_games;
        const int ng = left < round_games ? (int)left : round_games;
        const double p0 = wall_s();
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < ng; i++) {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            const long gi = games + i;
            play_game(mix(seed, (uint32_t)gi), gi < bootstrap, w, &TH[t], &rec[i]);
        }
        iv_play += wall_s() - p0;
        long new_pos = 0;
        for (int i = 0; i < ng; i++) {
            const GameRec *R = &rec[i];
            for (int p = 0; p < R->n; p++) {
                const long slot = total_pos % window;
                memcpy(wcode + (size_t)slot * area, R->code + (size_t)p * area, area);
                wmeta[slot] = R->meta[p];
                const int8_t mover = R->meta[p].black_to_move ? BLACK : WHITE;
                wout[slot] = p + 2 < R->n ? -1.0f : (R->winner == mover ? 1.0f : 0.0f);
                total_pos++;
            }
            new_pos += R->n;
            iv_moves += R->n; iv_bwins += R->winner == BLACK;
            if (auto_komi && games + i >= bootstrap) {
                komi_games++; komi_bwins += R->winner == BLACK;
                if (komi_games >= KOMI_WINDOW) {
                    const double bw = (double)komi_bwins / komi_games;
                    if (bw > 0.55) komi += 1; else if (bw < 0.45) komi -= 1;
                    komi_games = komi_bwins = 0;
                }
            }
        }
        games += ng; iv_games += ng;

        /* Training: --reuse samples per new position, in whole batches. */
        owed += reuse * new_pos;
        const long filled = total_pos < window ? total_pos : window;
        while (owed >= batch) {
            owed -= batch;
            double loss = 0, se = 0, fcpu = 0; long nf = 0;
            #pragma omp parallel for schedule(static) reduction(+:loss, se, fcpu, nf)
            for (int i = 0; i < batch; i++) {
                int t = 0;
#ifdef _OPENMP
                t = omp_get_thread_num();
#endif
                const uint32_t h = mix(mix(seed, 0x7d1u), (uint32_t)(step * batch + i));
                const long slot = h % (uint32_t)filled;
                float target = wout[slot];
                if (target < 0) {
                    const long j = (slot + 2) % window;
                    const double c0 = thread_cpu();
                    vnet_encode_compact(wcode + (size_t)j * area, &wmeta[j], area, NULL, vnet_work_input(WK[t]));
                    target = vnet_forward(&L, w, nbr9, area, WK[t]);
                    fcpu += thread_cpu() - c0; nf++;
                }
                const int sy = (int)(mix(h, 0x51u) & 7);
                vnet_encode_compact(wcode + (size_t)slot * area, &wmeta[slot], area, sym + (size_t)sy * area, vnet_work_input(WK[t]));
                const float v = vnet_forward(&L, w, nbr9, area, WK[t]);
                const float vc = v < 1e-7f ? 1e-7f : v > 1 - 1e-7f ? 1 - 1e-7f : v;
                loss += -(target * log(vc) + (1 - target) * log(1 - vc));
                se += (double)(v - target) * (v - target);
                vnet_backward(&L, w, nbr9, area, WK[t], v - target, TG[t]);
            }
            /* Reduce in thread order (deterministic), then Adam. */
            for (int j = 0; j < np; j++) { float g = 0; for (int t = 0; t < threads; t++) { g += TG[t][j]; TG[t][j] = 0; } G[j] = g / batch; }
            adam_step(&adam, w, G, lr);
            step++;
            iv_loss += loss; iv_se += se; iv_n += batch; iv_fcpu += fcpu; iv_nf += nf;
        }

        const int last = games_limit > 0 && games >= games_limit;
        if (games >= next_row || last) {
            if (iv_nf) infer = iv_fcpu / iv_nf;
            char tcol[16] = "";
            if (test_f) {
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
                snprintf(tcol, sizeof tcol, " %8.5f", tse / te.n);
            }
            vnet_save(save_f, &L, w);
            char mdcol[40] = "";
            if (md && infer > 0) md_test_row(md, &L, w, nbr9, area, WK, G3, threads, infer, wall_s() - t0, mdcol);
            else if (md) snprintf(mdcol, sizeof mdcol, "  %6s %6s", "-", "-");   /* no cost estimate yet */
            const double now = wall_s();
            char ibuf[16], lbuf[16], mbuf[16];
            if (iv_n) { snprintf(lbuf, sizeof lbuf, "%8.5f", iv_loss / iv_n); snprintf(mbuf, sizeof mbuf, "%8.5f", iv_se / iv_n); }
            else { strcpy(lbuf, "       -"); strcpy(mbuf, "       -"); }
            printf("%10ld %10ld %8.2fm %6.1f %5.1f %6.1f %6.1f %s %s%s %7s%s\n", games, total_pos, (now - t0) / 60,
                   100 * iv_play / (now - iv_t0), komi, 100.0 * iv_bwins / iv_games, (double)iv_moves / iv_games,
                   lbuf, mbuf, tcol, infer > 0 ? fmt_time(infer, ibuf) : "-", mdcol);
            iv_games = iv_bwins = iv_moves = iv_n = iv_nf = 0;
            iv_loss = iv_se = iv_play = iv_fcpu = 0;
            iv_t0 = now;
            while (next_row <= games) next_row = (long)(next_row * 1.4) + 1;
        }
        if (last) break;
    }
    return 0;
}
