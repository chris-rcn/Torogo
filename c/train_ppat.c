/*
 * train_ppat.c — Simulation Balancing (Huang, Coulom, Lin 2010, Algorithm 1).
 * C port of train-ppat.js.
 *
 * Input: concise format from gen_evals: "<size> <move1,move2,...> <value>"
 * Values in [0,1] mapped to [-1,1].
 *
 * Compile:
 *   cc -O2 -o train_ppat game2.c ppat.c train_ppat.c -lm
 *
 * Usage:
 *   ./train_ppat <file> [options]
 *   Options:
 *     --lr <f>              learning rate (default 10)
 *     --M <n>               rollouts for V estimate (default 100)
 *     --N <n>               rollouts for gradient (default M)
 *     --batch <n>           batch size (default 10)
 *     --test-pos <n>        test positions (default 100)
 *     --train-pos <n>       train positions (default 0 = all)
 *     --test-playouts <n>   playouts per test position (default 1000)
 *     --filter <f>          filter margin for extreme values (default 0)
 *     --iteration-limit <n> stop after n iterations (default infinite)
 *     --overfit             use same data for train and test
 */
#include "game2.h"
#include "ppat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* ── Configuration ─────────────────────────────────────────────────────────── */

static float  cfg_lr           = 10.0f;
static int    cfg_M            = 100;
static int    cfg_N            = -1;     /* -1 = use M */
static int    cfg_batch        = 10;
static int    cfg_test_pos     = 100;
static int    cfg_train_pos    = 0;
static int    cfg_test_playouts = 1000;
static float  cfg_filter       = 0.0f;
static int    cfg_iter_limit   = 0;      /* 0 = infinite */
static int    cfg_overfit      = 0;
static int    cfg_init         = 0;
static int    cfg_policy_moves = -1;  /* -1 = unlimited; else ppat for first N moves, then uniform */
static const char *cfg_file    = NULL;

/* ── Training data ─────────────────────────────────────────────────────────── */

#define MAX_HISTORY 512
#define MAX_LINES   500000

typedef struct {
    int32_t history[MAX_HISTORY];
    int     history_len;
    float   value;       /* in [-1,1] */
    int32_t best_move;   /* preferred next move from eval, or PASS if absent */
} Position;

static Position all_positions[MAX_LINES];
static int      n_all = 0;

static int     train_idx[MAX_LINES];   /* indices into all_positions */
static int     n_train = 0;
static int     test_idx[MAX_LINES];
static int     n_test = 0;

/* ── Parameter vector ──────────────────────────────────────────────────────── */

static int    TOTAL;
static float *theta;

/* Scratch buffers */
static PpatState rollout_feat_st;
static float    *rollout_grad_buf;
static float    *g_buf;
static float    *batch_buf;
static int      batch_count = 0;

static float    rollout_logits[CAP];
static float    rollout_probs[CAP];

/* ── Parse command line ────────────────────────────────────────────────────── */

static int get_int_arg(int argc, char **argv, const char *flag, int def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return atoi(argv[i+1]);
    return def;
}

static float get_float_arg(int argc, char **argv, const char *flag, float def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return (float)atof(argv[i+1]);
    return def;
}

static int has_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

/* ── Parse concise format ──────────────────────────────────────────────────── */

static int32_t parse_move(const char *s) {
    if (s[0] == 'p') return PASS;
    int x = s[0] - 'a';
    int y = atoi(s + 1) - 1;
    return y * BOARD_SIZE + x;
}

static int parse_position(const char *line, Position *pos) {
    int size;
    char moves_buf[4096];
    char best_buf[16];
    double value;
    int fields = sscanf(line, "%d %4095s %lf %15s", &size, moves_buf, &value, best_buf);
    if (fields < 3) return 0;
    if (size != BOARD_SIZE) return 0;
    pos->value = 2.0f * (float)value - 1.0f;
    pos->best_move = (fields >= 4) ? parse_move(best_buf) : PASS;
    pos->history_len = 0;
    char *tok = strtok(moves_buf, ",");
    while (tok && pos->history_len < MAX_HISTORY) {
        pos->history[pos->history_len++] = parse_move(tok);
        tok = strtok(NULL, ",");
    }
    return 1;
}

/* ── Load data ─────────────────────────────────────────────────────────────── */

static void load_data(void) {
    FILE *f = fopen(cfg_file, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", cfg_file); exit(1); }
    char buf[8192];
    int lineno = 0;
    int skipped = 0;
    while (fgets(buf, sizeof(buf), f) && n_all < MAX_LINES) {
        lineno++;
        if (buf[0] == '\n' || buf[0] == '\0') continue;
        if (parse_position(buf, &all_positions[n_all])) {
            n_all++;
        } else {
            skipped++;
            if (skipped <= 5) {
                /* Trim trailing newline for cleaner output */
                size_t len = strlen(buf);
                if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
                fprintf(stderr, "warning: skipping line %d: %s\n", lineno, buf);
            }
        }
    }
    fclose(f);
    if (skipped > 5)
        fprintf(stderr, "warning: %d more lines skipped\n", skipped - 5);
    if (n_all == 0) { fprintf(stderr, "error: no valid positions in %s (%d lines skipped)\n", cfg_file, skipped); exit(1); }
}

static void split_data(void) {
    float filter_threshold = 1.0f - 2.0f * cfg_filter;

    if (cfg_overfit) {
        int nt = cfg_train_pos > 0 ? (cfg_train_pos < n_all ? cfg_train_pos : n_all) : n_all;
        int ne = cfg_test_pos > 0 ? (cfg_test_pos < n_all ? cfg_test_pos : n_all) : n_all;
        for (int i = 0; i < nt; i++) train_idx[i] = i;
        n_train = nt;
        for (int i = 0; i < ne; i++) test_idx[i] = i;
        n_test = ne;
    } else {
        int n_test_want;
        if (cfg_train_pos && cfg_test_pos) {
            if (cfg_train_pos + cfg_test_pos <= n_all) n_test_want = cfg_test_pos;
            else n_test_want = n_all - (int)(0.5f + (float)n_all * cfg_train_pos / (cfg_train_pos + cfg_test_pos));
        } else if (cfg_test_pos) {
            n_test_want = cfg_test_pos < n_all ? cfg_test_pos : n_all;
        } else if (cfg_train_pos) {
            int nt = cfg_train_pos < n_all ? cfg_train_pos : n_all;
            n_test_want = n_all - nt;
        } else {
            n_test_want = n_all - (n_all * 2 + 2) / 3;
        }

        /* Test from front, train from back */
        for (int i = 0; i < n_test_want; i++) test_idx[i] = i;
        n_test = n_test_want;

        /* Train pool: filter extreme values */
        n_train = 0;
        int train_limit = cfg_train_pos > 0 ? cfg_train_pos : n_all;
        for (int i = n_test_want; i < n_all && n_train < train_limit; i++) {
            if (cfg_filter > 0 && fabsf(all_positions[i].value) > filter_threshold)
                continue;
            train_idx[n_train++] = i;
        }
    }
}

/* ── Shuffle train indices ─────────────────────────────────────────────────── */

static void shuffle_train(void) {
    for (int i = n_train - 1; i > 0; i--) {
        int j = g2_rand_n(i + 1);
        int tmp = train_idx[i];
        train_idx[i] = train_idx[j];
        train_idx[j] = tmp;
    }
}

/* ── Replay a position ─────────────────────────────────────────────────────── */

/* Returns: 1 = ok, 0 = game over, -1 = illegal move (sets *bad_move_idx) */
static int replay_position(const Position *pos, Game2 *g, int *bad_move_idx) {
    g2_new(g);
    for (int i = 0; i < pos->history_len; i++) {
        if (!g2_play(g, pos->history[i])) { *bad_move_idx = i; return -1; }
    }
    return !g->game_over;
}

/* ── Policy select (for gradient-tracking rollouts) ────────────────────────── */
/* Returns chosen index into rollout_feat_st, or -1 for pass.
 * Leaves rollout_feat_st and rollout_probs populated. */

static int policy_select(Game2 *g) {
    ppat_extract(g, &rollout_feat_st);
    int n = rollout_feat_st.count;
    if (n == 0) return -1;

    for (int i = 0; i < n; i++) {
        float v = theta[rollout_feat_st.pat_ids[i]];
        uint8_t m = rollout_feat_st.prev_masks[i];
        for (int b = 0; m; b++, m >>= 1)
            if (m & 1) v += theta[ppat_num_patterns + b];
        rollout_logits[i] = v;
    }

    /* Softmax */
    float mx = rollout_logits[0];
    for (int i = 1; i < n; i++) if (rollout_logits[i] > mx) mx = rollout_logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { rollout_probs[i] = expf(rollout_logits[i] - mx); sum += rollout_probs[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) rollout_probs[i] *= inv;

    /* Sample */
    float r = g2_randf();
    int chosen = n - 1;
    for (int i = 0; i < n; i++) { r -= rollout_probs[i]; if (r <= 0) { chosen = i; break; } }
    return chosen;
}

/* ── Rollout ───────────────────────────────────────────────────────────────── */
/* Returns z ∈ {-1, +1} from player's perspective.
 * If grad_acc != NULL, accumulates ψ(s,a) per step. */

static int rollout(const Game2 *game, int8_t player, float *grad_acc) {
    Game2 sim;
    g2_clone(&sim, game);
    int pm = cfg_policy_moves;

    for (int step = 0; !sim.game_over && (pm < 0 || step < pm); step++) {
        int chosen = policy_select(&sim);
        if (chosen == -1) { g2_play(&sim, PASS); continue; }

        int n = rollout_feat_st.count;

        if (grad_acc) {
            /* ψ(s,a) = φ(s,a) − Σ_b π(b|s)φ(s,b) */
            for (int i = 0; i < n; i++) {
                float p = rollout_probs[i];
                grad_acc[rollout_feat_st.pat_ids[i]] -= p;
                uint8_t m = rollout_feat_st.prev_masks[i];
                for (int b = 0; m; b++, m >>= 1)
                    if (m & 1) grad_acc[ppat_num_patterns + b] -= p;
            }
            grad_acc[rollout_feat_st.pat_ids[chosen]] += 1.0f;
            uint8_t m = rollout_feat_st.prev_masks[chosen];
            for (int b = 0; m; b++, m >>= 1)
                if (m & 1) grad_acc[ppat_num_patterns + b] += 1.0f;
        }

        g2_play(&sim, rollout_feat_st.moves[chosen]);
    }

    /* Finish game with uniform random play */
    while (!sim.game_over) g2_play(&sim, g2_random_legal_move(&sim));

    return g2_estimate_winner(&sim) == player ? 1 : -1;
}

/* ── Core update (Algorithm 1) ─────────────────────────────────────────────── */

static void update_theta(const Game2 *game, float v_star) {
    int8_t player = game->current;

    /* V: M rollouts, no gradient */
    float V = 0;
    for (int i = 0; i < cfg_M; i++) V += rollout(game, player, NULL);
    V /= cfg_M;

    /* g: N rollouts with gradient */
    int N = cfg_N;
    memset(g_buf, 0, sizeof(float) * TOTAL);
    for (int j = 0; j < N; j++) {
        memset(rollout_grad_buf, 0, sizeof(float) * TOTAL);
        int z = rollout(game, player, rollout_grad_buf);
        float scale = (float)z / N;
        for (int k = 0; k < TOTAL; k++) g_buf[k] += scale * rollout_grad_buf[k];
    }

    /* Accumulate bias*g into batch */
    float bias = v_star - V;
    for (int k = 0; k < TOTAL; k++) batch_buf[k] += bias * g_buf[k];
    batch_count++;

    /* Flush batch */
    if (batch_count >= cfg_batch) {
        float scale = cfg_lr / batch_count;
        for (int k = 0; k < TOTAL; k++) { theta[k] += scale * batch_buf[k]; batch_buf[k] = 0; }
        batch_count = 0;
    }
}

/* ── Fast uniform rollout (no feature extraction) ──────────────────────────── */

static int uniform_rollout(const Game2 *game, int8_t player) {
    Game2 sim;
    g2_clone(&sim, game);
    while (!sim.game_over) g2_play(&sim, g2_random_legal_move(&sim));
    return g2_estimate_winner(&sim) == player ? 1 : -1;
}

/* ── Measure test ──────────────────────────────────────────────────────────── */

typedef struct { float mean_abs; float rms; float move_match; } TestResult;

static TestResult measure_test(int use_uniform) {
    PpatState mm_st;
    memset(&mm_st, 0, sizeof(mm_st));
    PpatWeights mm_w = { theta, theta + ppat_num_patterns };
    float abs_sum = 0, sq_sum = 0;
    int count = 0, mm_match = 0, mm_total = 0;
    for (int ti = 0; ti < n_test; ti++) {
        Position *pos = &all_positions[test_idx[ti]];
        Game2 g;
        int bad = -1;
        int rp = replay_position(pos, &g, &bad);
        if (rp < 0) { fprintf(stderr, "warning: illegal move #%d (idx %d) in test position %d, skipping\n", bad, pos->history[bad], test_idx[ti]); continue; }
        if (rp == 0) continue;

        /* Move match */
        if (pos->best_move != PASS) {
            if (ppat_policy_move(&g, &mm_st, &mm_w) == pos->best_move) mm_match++;
            mm_total++;
        }

        /* RMS */
        int8_t player = g.current;
        float sum = 0;
        for (int i = 0; i < cfg_test_playouts; i++)
            sum += use_uniform ? uniform_rollout(&g, player) : rollout(&g, player, NULL);
        float d = pos->value - sum / cfg_test_playouts;
        abs_sum += fabsf(d);
        sq_sum += d * d;
        count++;
    }
    float mm = mm_total > 0 ? (float)mm_match / mm_total : 0;
    if (count == 0) return (TestResult){0, 0, mm};
    return (TestResult){ abs_sum / count, sqrtf(sq_sum / count), mm };
}

/* ── Policy vs random ──────────────────────────────────────────────────────── */

static float policy_vs_random(void) {
    PpatState pvr_st;
    memset(&pvr_st, 0, sizeof(pvr_st));
    PpatWeights w = { theta, theta + ppat_num_patterns };

    int wins = 0, games = 0;
    clock_t deadline = clock() + CLOCKS_PER_SEC;
    while (clock() < deadline) {
        int8_t policy_color = (games % 2 == 0) ? BLACK : WHITE;
        Game2 g;
        g2_new(&g);
        while (!g.game_over) {
            if (g.current == policy_color)
                g2_play(&g, ppat_policy_move(&g, &pvr_st, &w));
            else
                g2_play(&g, g2_random_legal_move(&g));
        }
        if (g2_estimate_winner(&g) == policy_color) wins++;
        games++;
    }
    return games > 0 ? (float)wins / games : 0.5f;
}

/* ── Save weights ──────────────────────────────────────────────────────────── */

static char weights_file[256];

static void save_weights(int iterations, int total_positions, const char *elapsed) {
    FILE *f = fopen(weights_file, "w");
    if (!f) return;
    fprintf(f, "'use strict';\n");
    fprintf(f, "// Generated by train_ppat (C) — iterations: %d, positions: %d, elapsed: %s\n",
            iterations, total_positions, elapsed);
    fprintf(f, "const _w = { pat: new Float32Array([");
    for (int i = 0; i < ppat_num_patterns; i++) {
        if (i > 0) fputc(',', f);
        fprintf(f, "%.8g", theta[i]);
    }
    fprintf(f, "]), prev: new Float32Array([");
    for (int i = 0; i < 7; i++) {
        if (i > 0) fputc(',', f);
        fprintf(f, "%.8g", theta[ppat_num_patterns + i]);
    }
    fprintf(f, "]) };\n");
    fprintf(f, "if (typeof module !== 'undefined') module.exports = _w;\n");
    fprintf(f, "else window.PPATWeights = _w;\n");
    fclose(f);
}

/* ── Print stats ───────────────────────────────────────────────────────────── */

static clock_t start_time;
static double  next_print;
static double  cumulative_test_s = 0;

static void print_stats(int iterations, int total_positions, int use_uniform, int show_weights) {
    clock_t test_t0 = clock();
    TestResult tr = measure_test(use_uniform);
    float pvr = policy_vs_random();
    cumulative_test_s += (double)(clock() - test_t0) / CLOCKS_PER_SEC;
    float mean_abs = tr.mean_abs, rms = tr.rms;
    double elapsed_s = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    char elapsed_buf[32];
    snprintf(elapsed_buf, sizeof(elapsed_buf), "%.1fs", elapsed_s);

    printf("%6d  %9d  %6.3f  %6.3f  %5.1f%%  %5.1f%%  %5.1fs  %8s",
           iterations, total_positions, mean_abs, rms, pvr * 100.0f, tr.move_match * 100.0f,
           cumulative_test_s, elapsed_buf);
    if (show_weights) {
        printf("  [");
        for (int i = 0; i < 7; i++)
            printf("%8.3f", expf(theta[ppat_num_patterns + i]));
        printf("]");
    }
    printf("\n");
    fflush(stdout);

    save_weights(iterations, total_positions, elapsed_buf);
    next_print = (double)(clock() - start_time) / CLOCKS_PER_SEC * 1.5 +
                 (double)(clock() - start_time) / CLOCKS_PER_SEC;
    /* Equivalent: next_print = elapsed_s + 0.5 * elapsed_s = 1.5 * elapsed_s
     * But using wall-clock offset from start: */
    next_print = elapsed_s + 0.5 * (elapsed_s > 0 ? elapsed_s : 1.0);
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        fprintf(stderr, "Usage: %s <file> [--lr <f>] [--M <n>] [--N <n>]\n", argv[0]);
        fprintf(stderr, "       [--batch <n>] [--test-pos <n>] [--train-pos <n>]\n");
        fprintf(stderr, "       [--test-playouts <n>] [--filter <f>] [--iteration-limit <n>]\n");
        fprintf(stderr, "       [--overfit]\n");
        return 1;
    }

    cfg_file         = argv[1];
    cfg_lr           = get_float_arg(argc, argv, "--lr", 10.0f);
    cfg_M            = get_int_arg(argc, argv, "--M", 100);
    cfg_N            = get_int_arg(argc, argv, "--N", cfg_M);
    cfg_batch        = get_int_arg(argc, argv, "--batch", 10);
    cfg_test_pos     = get_int_arg(argc, argv, "--test-pos", 100);
    cfg_train_pos    = get_int_arg(argc, argv, "--train-pos", 0);
    cfg_test_playouts = get_int_arg(argc, argv, "--test-playouts", 1000);
    cfg_filter       = get_float_arg(argc, argv, "--filter", 0.0f);
    cfg_iter_limit   = get_int_arg(argc, argv, "--iteration-limit", 0);
    cfg_overfit      = has_flag(argc, argv, "--overfit");
    cfg_init         = get_int_arg(argc, argv, "--init", 0);
    cfg_policy_moves = get_int_arg(argc, argv, "--policy-moves", -1);

    g2_seed((uint32_t)time(NULL));
    g2_init_topology();
    ppat_init();

    TOTAL = ppat_num_patterns + 7;
    theta           = calloc(TOTAL, sizeof(float));
    if (cfg_init) {
        static const float init_prev[7] = {7.43f, 151.04f, 0.53f, 23.11f, 0.02f, 6.37f, 141.80f};
        for (int i = 0; i < 7; i++) theta[ppat_num_patterns + i] = logf(init_prev[i]);
    }
    rollout_grad_buf = calloc(TOTAL, sizeof(float));
    g_buf           = calloc(TOTAL, sizeof(float));
    batch_buf       = calloc(TOTAL, sizeof(float));
    memset(&rollout_feat_st, 0, sizeof(rollout_feat_st));

    load_data();
    split_data();

    /* Output filename */
    snprintf(weights_file, sizeof(weights_file), "out/ppat-data-%08x.js", g2_rand());

    printf("train: %s (%d positions)  lr: %.1f  M: %d  N: %d  batch: %d  overfit: %s  filter: %.1f  init: %s  policy-moves: %d\n",
           cfg_file, n_train, cfg_lr, cfg_M, cfg_N, cfg_batch,
           cfg_overfit ? "true" : "false", cfg_filter,
           cfg_init ? "true" : "false", cfg_policy_moves);
    printf("test: %d positions  test-playouts: %d  output: %s\n",
           n_test, cfg_test_playouts, weights_file);
    printf("%6s  %9s  %6s  %6s  %6s  %6s  %6s  %8s\n",
           "iters", "positions", "|ΔV|", "RMS", "vsRand", "move%", "test", "elapsed");

    start_time = clock();
    int total_positions = 0;
    int iterations = 0;
    next_print = 0;

    /* Iteration 0: baseline using fast uniform rollouts */
    print_stats(iterations, total_positions, 1, 0);

    for (;;) {
        for (int li = 0; li < n_train; li++) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            if (elapsed > next_print)
                print_stats(iterations, total_positions, 0, 1);

            Position *pos = &all_positions[train_idx[li]];
            Game2 g;
            int bad = -1;
            int rp = replay_position(pos, &g, &bad);
            if (rp < 0) { fprintf(stderr, "warning: illegal move #%d (idx %d) in training position, skipping\n", bad, pos->history[bad]); continue; }
            if (rp == 0) continue;
            update_theta(&g, pos->value);
            total_positions++;
        }

        iterations++;
        shuffle_train();

        if (cfg_iter_limit > 0 && iterations >= cfg_iter_limit) {
            print_stats(iterations, total_positions, 0, 1);
            break;
        }
    }

    free(theta);
    free(rollout_grad_buf);
    free(g_buf);
    free(batch_buf);
    return 0;
}
