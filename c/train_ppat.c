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
 *     --playouts <n>        default for --M, --N, and --test-playouts (default 500)
 *     --M <n>               rollouts for V estimate (default --playouts)
 *     --N <n>               rollouts for gradient (default M)
 *     --batch <n>           batch size (default 10)
 *     --test-pos <n>        test positions (default 100)
 *     --test-file <path>    take the test set from THIS file instead of the head
 *                           of <file>; the whole of <file> is then training data.
 *                           Use a fixed high-playout set as a permanent yardstick:
 *                           teMSE then compares across runs and datasets.
 *     --train-pos <n>       train positions (default 0 = all)
 *     --test-playouts <n>   playouts per test position (default --playouts)
 *     --no-extreme <f>      drop TRAIN positions whose value is more extreme than ±(1-2f) (default 0 = keep all)
 *     --iteration-limit <n> stop after n iterations (default infinite)
 *     --overfit             use same data for train and test
 *     --phases <n>          number of phase-conditioned weight slices (default 1)
 *     --phase <p>           train only phase p; freezes the other phases.
 *                           The test head stays UNFILTERED (all phases), so
 *                           teMSE is always the end-to-end number, comparable
 *                           across runs regardless of masking.
 *     --init-phase-scale <f>  seed phase p's weights from phase p+1 (scaled by f)
 *                           before training (endgame-first warm-start; requires
 *                           --phase). γ → γ^f, so f<1 softens toward uniform,
 *                           f=0 seeds zeros. Absent = no seeding.
 */
#include "game2.h"
#include "ppat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <unistd.h>     /* usleep */
#include <sys/stat.h>   /* mkdir */

#define SYNC_POLL_US 2000   /* parameter-averaging barrier poll: 2 ms base (jittered) */
#define MAX_PRINT_CYCLE_S 3600.0   /* cap the geometric print/test gap at 1 h (inline + monitor) */

/* ── Configuration ───────────────────────────────────────────────────────────
 * Defaults live ONLY at the get_*_arg() call sites in main() (single source of
 * truth); these declarations are zero/NULL-initialised.  Comments document the
 * MEANING of special values, not the default. */

static float  cfg_lr;
static int    cfg_M;
static int    cfg_N;
static int    cfg_batch;
static int    cfg_test_pos;
static int    cfg_train_pos;
static int    cfg_test_playouts;
static float  cfg_no_extreme;
static int    cfg_iter_limit;          /* 0 = infinite */
static int    cfg_overfit;
static int    cfg_init;
static int    cfg_train_moves;         /* -1 = unlimited; else ppat for first N moves, then uniform (train + test rollouts) */
static int    cfg_phase;               /* -1 = all phases; >= 0 = train/test only this phase */

/* Parameter-averaging across processes: K workers each run batch-1 SB with their
 * own seed; every --sync-every positions they file-barrier all-reduce (average) θ. */
static int    cfg_workers;
static int    cfg_worker_id;
static int    cfg_sync_every;          /* 0 = no sync (single process) */
static const char *cfg_sync_dir;
static const char *cfg_file;
static const char *cfg_test_file;      /* NULL = carve the test set from cfg_file's head */
static int    cfg_test_pos_given;      /* was --test-pos passed explicitly? */
static const char *cfg_load;           /* path to weights file to load */
static const char *cfg_save;           /* fixed checkpoint path (else a random out/ name) */
static const char *cfg_monitor;        /* if set: run as a test-only monitor of this checkpoint */

/* ── Training data ─────────────────────────────────────────────────────────── */

#define MAX_HISTORY 512
#define MAX_LINES   500000

typedef struct {
    int32_t history[MAX_HISTORY];
    int     history_len;
    int     board_size;
    int     phase;       /* game phase at this position */
    float   value;       /* in [-1,1] */
    int32_t best_move;   /* preferred next move from eval, or PASS if absent */
} Position;

static Position all_positions[MAX_LINES];
static int      n_all = 0;

static int     train_idx[MAX_LINES];   /* indices into all_positions */
static int     n_train = 0;            /* this worker's slice of the train set */
static int     n_train_total = 0;      /* full train set across all workers' slices */
static int     test_idx[MAX_LINES];
static int     n_test = 0;

/* Per-epoch training-fit: accumulate Σ (v* − V)^2 (v*, V normalised to win-prob
 * [0,1], matching the SB paper's MSE units) over the current epoch (V is the
 * M-rollout policy value already computed for the gradient), then latch into done_*
 * at each epoch boundary.  The reported value is the LAST completed epoch over the
 * worker's fixed training set (same positions every epoch → stable, like the fixed
 * test set), aggregated across workers. */
static double  epoch_sq_sum = 0;
static long    epoch_sq_count = 0;
static double  done_sq_sum = 0;        /* last completed epoch (this worker) */
static long    done_sq_count = 0;
static double  agg_train_sq_sum = 0;   /* last completed epoch, summed over workers (worker 0) */
static long    agg_train_sq_count = 0;
static double  agg_part_sq_sum = 0;    /* current (partial) epoch, summed over workers */
static long    agg_part_sq_count = 0;

/* ── Parameter vector ──────────────────────────────────────────────────────── */

static Rng  g_rng;          /* this process's RNG (seeded in main) */
static int    TOTAL;
static float *theta;

/* Scratch buffers */
static PpatState rollout_feat_st;
static float    *rollout_grad_buf;
static float    *g_buf;
static float    *batch_buf;
static int      batch_count = 0;

static float    rollout_logits[MAX_CAP];
static float    rollout_probs[MAX_CAP];

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

static const char *get_str_arg(int argc, char **argv, const char *flag, const char *def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i+1];
    return def;
}

static int has_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) return 1;
    return 0;
}

/* ── Parse concise format ──────────────────────────────────────────────────── */

static int32_t parse_move(const char *s, int N) {
    if (s[0] == 'p') return PASS;
    int x = s[0] - 'a';
    int y = atoi(s + 1) - 1;
    return y * N + x;
}

static int parse_position(const char *line, Position *pos) {
    int size;
    char moves_buf[4096];
    char best_buf[16];
    double value;
    int fields = sscanf(line, "%d %4095s %lf %15s", &size, moves_buf, &value, best_buf);
    if (fields < 3) return 0;
    if (size > MAX_BOARD_SIZE) return 0;
    pos->board_size = size;
    pos->value = 2.0f * (float)value - 1.0f;
    pos->best_move = (fields >= 4) ? parse_move(best_buf, size) : PASS;
    pos->history_len = 0;
    char *tok = strtok(moves_buf, ",");
    while (tok && pos->history_len < MAX_HISTORY) {
        pos->history[pos->history_len++] = parse_move(tok, size);
        tok = strtok(NULL, ",");
    }
    return 1;
}

/* ── Load data ─────────────────────────────────────────────────────────────── */

static int replay_position(const Position *pos, Game2 *g, int *bad_move_idx);

/* Load records from `path` into all_positions[].  `test_head` is how many of the
 * FIRST kept records are test (filters never apply to them); with --test-file the
 * caller loads the test file first with test_head = its whole length, then the
 * train file with test_head = 0. */
static void load_positions_from(const char *path, int test_head) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    const int base = n_all;
    char buf[8192];
    int lineno = 0;
    int skipped = 0;       /* unparseable lines */
    int filtered = 0;      /* valid positions dropped by --no-extreme / --phase */
    /* Topology is global and must agree across BOTH files (--test-file), so these
     * persist between calls rather than resetting per file. */
    static int topo_init = 0;
    static int topo_size = 0;
    float extreme_threshold = 1.0f - 2.0f * cfg_no_extreme;

    while (fgets(buf, sizeof(buf), f) && n_all < MAX_LINES) {
        lineno++;
        if (buf[0] == '\n' || buf[0] == '\0') continue;
        if (buf[0] == '#') continue;   /* comment line (e.g. an agent load banner) */
        Position pos;
        if (!parse_position(buf, &pos)) {
            skipped++;
            if (skipped <= 5) {
                size_t len = strlen(buf);
                if (len > 0 && buf[len-1] == '\n') buf[len-1] = '\0';
                fprintf(stderr, "WARNING: skipping line %d: %s\n", lineno, buf);
            }
            continue;
        }

        /* Init topology on first valid position; the board size is global, so all
         * positions must share it — mixed sizes would index the wrong neighbour
         * tables (a segfault).  Throw on the first discrepancy instead. */
        if (!topo_init) { g2_init_topology(pos.board_size); topo_size = pos.board_size; topo_init = 1; }
        else if (pos.board_size != topo_size) {
            fprintf(stderr, "error: mixed board sizes in %s (line %d: size %d, expected %d; train one size per file)\n",
                    path, lineno, pos.board_size, topo_size);
            exit(1);
        }

        /* The first --test-pos kept records become the test head; filters
         * apply only past it (train pool), so the test set is identical
         * across all filter/mask configs and teMSE is always end-to-end.
         * (--overfit shares records between train and test, so it keeps the
         * filters everywhere.) */
        int in_test_head = !cfg_overfit && (n_all - base) < test_head;

        /* Value filter (train pool only) */
        if (!in_test_head && cfg_no_extreme > 0 && fabsf(pos.value) > extreme_threshold) { filtered++; continue; }

        /* Compute phase; filter by it (train pool only) */
        Game2 g;
        int bad;
        if (replay_position(&pos, &g, &bad) > 0)
            pos.phase = ppat_phase_count * (g.cap - g.empty_count) / g.cap;
        else
            pos.phase = -1;
        if (!in_test_head && cfg_phase >= 0 && pos.phase != cfg_phase) { filtered++; continue; }

        all_positions[n_all++] = pos;
    }
    fclose(f);
    if (skipped > 5)
        fprintf(stderr, "WARNING: %d more lines skipped\n", skipped - 5);
    if (n_all == base) { fprintf(stderr, "error: no valid positions in %s (%d lines skipped)\n", path, skipped); exit(1); }
    if (cfg_worker_id == 0)
        printf("records: %d kept of %d in %s (%d train-filtered, %d skipped)\n",
               n_all - base, n_all - base + filtered, path, filtered, skipped);
}

/* n_test_file: how many leading records came from --test-file (0 when unset).
 * This is the SIZE OF THE TEST BLOCK, which is where training data starts — it
 * stays separate from how many of those records we actually test on, so that
 * capping the test count with --test-pos can never leak the remainder into the
 * training pool. */
static int n_test_file = 0;

static void load_positions(void) {
    if (cfg_test_file) {
        /* Test file first: every one of its records is test, filters excluded. */
        load_positions_from(cfg_test_file, MAX_LINES);
        n_test_file = n_all;
        load_positions_from(cfg_file, 0);   /* all of it is training data */
    } else {
        load_positions_from(cfg_file, cfg_test_pos);
    }
}

static void split_data(void) {
    if (cfg_overfit) {
        int nt = cfg_train_pos > 0 ? (cfg_train_pos < n_all ? cfg_train_pos : n_all) : n_all;
        int ne = cfg_test_pos > 0 ? (cfg_test_pos < n_all ? cfg_test_pos : n_all) : n_all;
        for (int i = 0; i < nt; i++) train_idx[i] = i;
        n_train = nt;
        for (int i = 0; i < ne; i++) test_idx[i] = i;
        n_test = ne;
    } else {
        /* Test from the head of the file, train from the data after it — no
         * proportional split and no random draw, so both sets are deterministic and
         * identical across runs (head and tail of a shuffled file are each
         * representative).  Keeping the test set as a fixed block at the start means
         * new training samples can be appended to the end of the file without
         * disturbing the (known, reproducible) test set.  --train-pos unset → train
         * uses all data after the test head. */
        /* With --test-file the leading n_test_file records are the test BLOCK and
         * all of the train file follows them; --test-pos then optionally caps how
         * many of the block we test on (default: all of it), without moving where
         * training starts.  Otherwise the block is the head of the single file. */
        int block_n = cfg_test_file ? n_test_file : (cfg_test_pos > 0 ? cfg_test_pos : 0);
        if (block_n > n_all) block_n = n_all;
        int test_n  = block_n;
        if (cfg_test_file && cfg_test_pos_given && cfg_test_pos > 0 && cfg_test_pos < block_n)
            test_n = cfg_test_pos;
        int avail   = n_all - block_n;                                    /* data after the test block */
        int train_n = cfg_train_pos > 0 ? cfg_train_pos : avail;
        if (train_n > avail) {
            fprintf(stderr, "WARNING: train (%d) exceeds data after the test head (%d) in %d total; clamping\n",
                    train_n, avail, n_all);
            train_n = avail;
        }

        int train_start = block_n;                                        /* train begins after the whole test block */
        n_train_total = train_n;
        /* Partition the train tail into disjoint contiguous slices: worker w trains
         * on [w·n/K, (w+1)·n/K) within the train region, so one epoch is a single pass
         * over the whole train set (the workers cover it between them), not K passes.
         * No shuffle here — a contiguous slice of the pre-shuffled file is already a
         * random sample; each worker shuffles its own slice's order every epoch. */
        int lo = (int)((long)cfg_worker_id       * train_n / cfg_workers);
        int hi = (int)((long)(cfg_worker_id + 1)  * train_n / cfg_workers);
        n_train = hi - lo;
        for (int i = 0; i < n_train; i++) train_idx[i] = train_start + lo + i;  /* this worker's slice */
        n_test = test_n;
        for (int i = 0; i < test_n; i++) test_idx[i] = i;                /* head (shared) */
    }
}

/* ── Shuffle train indices ─────────────────────────────────────────────────── */

static void shuffle_train(void) {
    for (int i = n_train - 1; i > 0; i--) {
        int j = rng_below(&g_rng, i + 1);
        int tmp = train_idx[i];
        train_idx[i] = train_idx[j];
        train_idx[j] = tmp;
    }
}

/* ── Replay a position ─────────────────────────────────────────────────────── */

/* Returns: 1 = ok, 0 = game over, -1 = illegal move (sets *bad_move_idx) */
static int replay_position(const Position *pos, Game2 *g, int *bad_move_idx) {
    g2_new(g, pos->board_size);
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
        float v = 0;
        for (int fi = rollout_feat_st.feat_start[i]; fi < rollout_feat_st.feat_start[i + 1]; fi++)
            v += theta[rollout_feat_st.feat[fi]];
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
    float r = rng_float(&g_rng);
    int chosen = n - 1;
    for (int i = 0; i < n; i++) { r -= rollout_probs[i]; if (r <= 0) { chosen = i; break; } }
    return chosen;
}

/* ── Rollout ───────────────────────────────────────────────────────────────── */
/* Returns z ∈ {-1, +1} from player's perspective.
 * If grad_acc != NULL, accumulates ψ(s,a) per step.
 * If out_steps != NULL, reports T for the paper's 1/T gradient normalisation:
 * all policy steps when training every phase, or only the in-phase steps
 * (board phase == cfg_phase) when a single phase is masked — i.e. T_P, matching
 * the steps whose ψ survives mask_to_phase. */

static int rollout(const Game2 *game, int8_t player, float *grad_acc, int ppat_moves, int *out_steps) {
    Game2 sim;
    g2_clone(&sim, game);
    int pm = ppat_moves;
    int steps = 0;

    for (int step = 0; !sim.game_over && (pm < 0 || step < pm); step++) {
        int chosen = policy_select(&sim);
        if (chosen == -1) {
            g2_play(&sim, PASS);
            continue;
        }

        int n = rollout_feat_st.count;

        if (grad_acc) {
            /* ψ(s,a) = φ(s,a) − Σ_b π(b|s)φ(s,b) */
            for (int i = 0; i < n; i++) {
                float p = rollout_probs[i];
                for (int fi = rollout_feat_st.feat_start[i]; fi < rollout_feat_st.feat_start[i + 1]; fi++)
                    grad_acc[rollout_feat_st.feat[fi]] -= p;
            }
            for (int fi = rollout_feat_st.feat_start[chosen]; fi < rollout_feat_st.feat_start[chosen + 1]; fi++)
                grad_acc[rollout_feat_st.feat[fi]] += 1.0f;

            /* Count this step toward T (all phases) / T_P (masked single phase). */
            if (cfg_phase < 0 ||
                ppat_phase_count * (sim.cap - sim.empty_count) / sim.cap == cfg_phase)
                steps++;
        }

        int32_t mv = rollout_feat_st.moves[chosen];
        g2_play(&sim, mv);
    }

    /* Finish game with uniform random play. */
    while (!sim.game_over) g2_play(&sim, g2_random_legal_move(&sim, &g_rng));

    if (out_steps) *out_steps = steps;
    return g2_estimate_winner(&sim) == player ? 1 : -1;
}

/* ── Core update (Algorithm 1) ─────────────────────────────────────────────── */

/* When --phase P is set, zero the gradient outside phase P's pattern and
 * prev-move slices so the other phases stay frozen at their loaded/init value.
 * Rollouts still traverse later phases for move selection, but only phase P's
 * weights are updated — exact coordinate-restricted SB gradient descent. */
static void mask_to_phase(float *v) {
    const int np = ppat_num_patterns;
    const int prev_base = ppat_phase_count * np;
    const int pat_lo = cfg_phase * np,           pat_hi = pat_lo + np;
    const int prev_lo = prev_base + cfg_phase * 7, prev_hi = prev_lo + 7;
    for (int k = 0; k < TOTAL; k++)
        if (!((k >= pat_lo && k < pat_hi) || (k >= prev_lo && k < prev_hi)))
            v[k] = 0.0f;
}

static void update_theta(const Game2 *game, float v_star) {
    int8_t player = game->current;

    /* V: M rollouts, no gradient */
    float V = 0;
    for (int i = 0; i < cfg_M; i++) V += rollout(game, player, NULL, cfg_train_moves, NULL);
    V /= cfg_M;

    /* g: N rollouts with gradient.  Algorithm 1: g ← g + z/(N·T)·Σ_t ψ.  T is the
     * rollout's policy-step count (T_P, the in-phase steps, when a phase is masked).
     * T == 0 means no ψ was accumulated, so that rollout contributes nothing. */
    int N = cfg_N;
    memset(g_buf, 0, sizeof(float) * TOTAL);
    for (int j = 0; j < N; j++) {
        memset(rollout_grad_buf, 0, sizeof(float) * TOTAL);
        int T = 0;
        int z = rollout(game, player, rollout_grad_buf, cfg_train_moves, &T);
        if (T > 0) {
            float scale = (float)z / ((float)N * (float)T);
            for (int k = 0; k < TOTAL; k++) g_buf[k] += scale * rollout_grad_buf[k];
        }
    }
    if (cfg_phase >= 0) mask_to_phase(g_buf);

    /* Gradient uses the un-normalised [-1,1] bias (the SB paper's faster-learning
     * -1/1 regime).  The MSE byproduct normalises v* and V to win-probability
     * [0,1] before squaring, so trMSE/teMSE are reported in the paper's units. */
    float bias = v_star - V;
    float v01 = 0.5f * (v_star + 1.0f), V01 = 0.5f * (V + 1.0f);
    epoch_sq_sum += (double)(v01 - V01) * (v01 - V01);
    epoch_sq_count++;
    for (int k = 0; k < TOTAL; k++) batch_buf[k] += bias * g_buf[k];
    batch_count++;

    /* Flush batch */
    if (batch_count >= cfg_batch) {
        float scale = cfg_lr / batch_count;
        for (int k = 0; k < TOTAL; k++) { theta[k] += scale * batch_buf[k]; batch_buf[k] = 0; }
        batch_count = 0;
    }
}

/* Apply any pending (sub-full) batch — called before a parameter-averaging sync so
 * θ reflects every update this worker has made. */
static void flush_batch(void) {
    if (batch_count == 0) return;
    float scale = cfg_lr / batch_count;
    for (int k = 0; k < TOTAL; k++) { theta[k] += scale * batch_buf[k]; batch_buf[k] = 0; }
    batch_count = 0;
}

/* ── Parameter-averaging all-reduce (file-based barrier) ───────────────────────
 * Every worker writes θ for the current round (atomic via tmp+rename), waits for
 * all K workers' round files, then sets θ ← mean over workers.  Cleanup of round
 * r-1 is safe once all round-r files exist: a worker only writes round r after it
 * finished reading every round r-1 file, so no one is still reading r-1. */
static int sync_round = 0;
static double cumulative_sync_s = 0;   /* wall time spent in parameter-averaging (I/O + barrier wait) */

static double wall_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void barrier_sync_average(void) {
    double _sync_t0 = wall_now();
    char path[600], tmp[600];
    /* Publish this worker's cumulative train-fit (Σ bias², count) before θ, so a
     * peer's .stat is guaranteed present once its θ is visible to worker 0. */
    snprintf(tmp,  sizeof(tmp),  "%s/r%d_w%d.stat.tmp", cfg_sync_dir, sync_round, cfg_worker_id);
    snprintf(path, sizeof(path), "%s/r%d_w%d.stat",     cfg_sync_dir, sync_round, cfg_worker_id);
    { FILE *sf = fopen(tmp, "w");
      if (sf) { fprintf(sf, "%.9g %ld %.9g %ld\n", done_sq_sum, done_sq_count, epoch_sq_sum, epoch_sq_count); fclose(sf); rename(tmp, path); } }
    snprintf(tmp,  sizeof(tmp),  "%s/r%d_w%d.tmp", cfg_sync_dir, sync_round, cfg_worker_id);
    snprintf(path, sizeof(path), "%s/r%d_w%d.f32", cfg_sync_dir, sync_round, cfg_worker_id);
    FILE *f = fopen(tmp, "wb");
    if (!f) { fprintf(stderr, "sync: cannot write %s\n", tmp); exit(1); }
    fwrite(theta, sizeof(float), TOTAL, f);
    fclose(f);
    rename(tmp, path);   /* atomic publish */

    const long want = (long)TOTAL * (long)sizeof(float);
    static float *acc = NULL, *buf = NULL;
    if (!acc) { acc = malloc(want); buf = malloc(want); }
    memset(acc, 0, want);
    for (int w = 0; w < cfg_workers; w++) {
        char wp[600];
        snprintf(wp, sizeof(wp), "%s/r%d_w%d.f32", cfg_sync_dir, sync_round, w);
        FILE *wf;
        for (;;) {                                   /* sleep-poll until peer file is fully written */
            wf = fopen(wp, "rb");
            if (wf) {
                fseek(wf, 0, SEEK_END);
                if (ftell(wf) == want) { fseek(wf, 0, SEEK_SET); break; }
                fclose(wf);
            }
            /* 2 ms base poll with 100% jitter (uniform [0, 4 ms), mean 2 ms) so workers
             * don't fall into lockstep polling on the same boundaries. */
            usleep((useconds_t)(rng_next(&g_rng) % (2 * SYNC_POLL_US)));
        }
        if (fread(buf, sizeof(float), TOTAL, wf) != (size_t)TOTAL) { fprintf(stderr, "sync: short read %s\n", wp); exit(1); }
        fclose(wf);
        for (int k = 0; k < TOTAL; k++) acc[k] += buf[k];
    }
    float inv = 1.0f / cfg_workers;
    for (int k = 0; k < TOTAL; k++) theta[k] = acc[k] * inv;

    /* Worker 0 aggregates every worker's cumulative train-fit for this round.
     * Each peer's .stat was published before its θ (already read above), so all
     * are present without polling. */
    if (cfg_worker_id == 0) {
        agg_train_sq_sum = 0; agg_train_sq_count = 0; agg_part_sq_sum = 0; agg_part_sq_count = 0;
        for (int w = 0; w < cfg_workers; w++) {
            char wp[600];
            snprintf(wp, sizeof(wp), "%s/r%d_w%d.stat", cfg_sync_dir, sync_round, w);
            FILE *wf = fopen(wp, "r");
            if (wf) {
                double ds = 0, ps = 0; long dc = 0, pc = 0;
                if (fscanf(wf, "%lf %ld %lf %ld", &ds, &dc, &ps, &pc) == 4) {
                    agg_train_sq_sum += ds; agg_train_sq_count += dc;
                    agg_part_sq_sum  += ps; agg_part_sq_count  += pc;
                }
                fclose(wf);
            }
        }
    }

    if (sync_round >= 1) {
        char old[600];
        snprintf(old, sizeof(old), "%s/r%d_w%d.f32", cfg_sync_dir, sync_round - 1, cfg_worker_id);
        remove(old);
        snprintf(old, sizeof(old), "%s/r%d_w%d.stat", cfg_sync_dir, sync_round - 1, cfg_worker_id);
        remove(old);
    }
    sync_round++;
    cumulative_sync_s += wall_now() - _sync_t0;
}

/* ── Fast uniform rollout (no feature extraction) ──────────────────────────── */

static int uniform_rollout(const Game2 *game, int8_t player) {
    Game2 sim;
    g2_clone(&sim, game);
    while (!sim.game_over) g2_play(&sim, g2_random_legal_move(&sim, &g_rng));
    return g2_estimate_winner(&sim) == player ? 1 : -1;
}

/* ── Move probability ──────────────────────────────────────────────────────── */

/* Returns the softmax probability the current policy assigns to `move`. */
static float move_probability(const Game2 *g, int32_t move) {
    static PpatState st;
    static float logits[MAX_CAP];
    ppat_extract(g, &st);
    int n = st.count;
    if (n == 0) return 0;
    float mx = -1e30f;
    int target = -1;
    for (int i = 0; i < n; i++) {
        float v = 0;
        for (int fi = st.feat_start[i]; fi < st.feat_start[i + 1]; fi++)
            v += theta[st.feat[fi]];
        logits[i] = v;
        if (v > mx) mx = v;
        if (st.moves[i] == move) target = i;
    }
    if (target < 0) return 0;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - mx);
    return expf(logits[target] - mx) / sum;
}

/* ── Measure test ──────────────────────────────────────────────────────────── */

typedef struct { float mean_abs; float mse; float move_match; } TestResult;

#define TEST_RNG_SEED 0x7e57c0deL   /* fixed seed → reproducible test rollouts */

/* Test the first `n` of the test positions; n is clamped to [1, n_test]. */
static TestResult measure_test(int use_uniform, int n) {
    if (n < 1) n = 1;
    if (n > n_test) n = n_test;
    /* Deterministic test rollouts: seed a FIXED stream so teMSE is reproducible
     * run-to-run and across report lines (differences reflect the weights, not MC
     * noise).  Save/restore g_rng so training's own stream is untouched. */
    Rng saved_rng = g_rng;
    rng_seed(&g_rng, TEST_RNG_SEED);
    float abs_sum = 0, sq_sum = 0, prob_sum = 0;
    int count = 0, prob_n = 0;
    for (int ti = 0; ti < n; ti++) {
        Position *pos = &all_positions[test_idx[ti]];
        Game2 g;
        int bad = -1;
        int rp = replay_position(pos, &g, &bad);
        if (rp < 0) { fprintf(stderr, "WARNING: illegal move #%d (idx %d) in test position %d, skipping\n", bad, pos->history[bad], test_idx[ti]); continue; }
        if (rp == 0) continue;

        if (pos->best_move != PASS) {
            prob_sum += move_probability(&g, pos->best_move);
            prob_n++;
        }

        int8_t player = g.current;
        float sum = 0;
        for (int i = 0; i < cfg_test_playouts; i++)
            sum += use_uniform ? uniform_rollout(&g, player) : rollout(&g, player, NULL, cfg_train_moves, NULL);
        /* Normalise v* and the rollout mean from [-1,1] to win-probability [0,1]
         * before the error, so MSE matches the SB paper's units. */
        float v01 = 0.5f * (pos->value + 1.0f);
        float V01 = 0.5f * (sum / cfg_test_playouts + 1.0f);
        float d = v01 - V01;
        abs_sum += fabsf(d);
        sq_sum += d * d;
        count++;
    }
    float mp = prob_n > 0 ? prob_sum / prob_n : 0;
    g_rng = saved_rng;   /* restore training's RNG stream */
    if (count == 0) return (TestResult){0, 0, mp};
    return (TestResult){ abs_sum / count, sq_sum / count, mp };   /* mse = mean squared error */
}


/* ── Save weights ──────────────────────────────────────────────────────────── */

static char weights_file[256];

static clock_t start_time;
static double  wall_start;            /* wall clock at training start, for pos/s */
static double  next_print;
static double  cumulative_test_s = 0;

/* Render the trMSE column.  A freshly-completed epoch (full_count>0, mean differs
 * from *last_full) is shown once with an 'F' suffix = "full"; otherwise the current
 * epoch's partial running mean is shown with a trailing space; else "-".
 * *last_full latches the most recent full value so it prints exactly once. */
static const char *trmse_col(double full_sum, long full_count, double part_sum, long part_count,
                             double *last_full, char *buf, size_t n) {
    if (full_count > 0) {
        double fm = full_sum / (double)full_count;
        if (fm != *last_full) { *last_full = fm; snprintf(buf, n, "%.4fF", fm); return buf; }
    }
    if (part_count > 0) snprintf(buf, n, "%.4f ", part_sum / (double)part_count);
    else { buf[0] = '-'; buf[1] = 0; }
    return buf;
}

/* teMSE column: '*' when v is a new minimum (best generalization so far), else
 * ' ' for equal spacing.  *best tracks the lowest teMSE seen. */
static const char *temse_col(float v, float *best, char *buf, size_t n) {
    char mark = ' ';
    if (v < *best) { *best = v; mark = '*'; }
    snprintf(buf, n, "%.4f%c", v, mark);
    return buf;
}

static void save_weights(int iterations, int total_positions, const char *elapsed) {
    char comment[384];
    /* Training-fit accumulators for the monitor: the last completed epoch (full)
     * and the current in-progress epoch (partial), each Σ bias² + count. */
    double dsum = (cfg_workers > 1) ? agg_train_sq_sum   : done_sq_sum;
    long   dcnt = (cfg_workers > 1) ? agg_train_sq_count : done_sq_count;
    double psum = (cfg_workers > 1) ? agg_part_sq_sum    : epoch_sq_sum;
    long   pcnt = (cfg_workers > 1) ? agg_part_sq_count  : epoch_sq_count;
    snprintf(comment, sizeof(comment),
             "Generated by train_ppat (C) — iterations: %d, positions: %d, elapsed: %s, phases: %d, trainSqSum: %.9g, trainSqCount: %ld, trainPartSum: %.9g, trainPartCount: %ld",
             iterations, total_positions, elapsed, ppat_phase_count, dsum, dcnt, psum, pcnt);
    /* Atomic: write to a tmp file then rename, so a reader (the monitor) never
     * sees a half-written checkpoint. */
    char tmp[300];
    snprintf(tmp, sizeof(tmp), "%s.tmp", weights_file);
    ppat_save_weights(tmp, theta, TOTAL, comment);
    rename(tmp, weights_file);
}

/* Snapshot the currently-loaded theta to "<checkpoint>-best.js" — called whenever
 * a new teMSE low is found.  "-best" is inserted before a trailing .js so
 * out/foo.js → out/foo-best.js.  Written atomically (tmp + rename). */
static void best_path(const char *ckpt_path, char *buf, size_t n) {
    const char *dot = strrchr(ckpt_path, '.');
    if (dot && strcmp(dot, ".js") == 0)
        snprintf(buf, n, "%.*s-best.js", (int)(dot - ckpt_path), ckpt_path);
    else
        snprintf(buf, n, "%s-best.js", ckpt_path);
}

static void save_best(const char *ckpt_path, const char *comment) {
    char best[320];
    best_path(ckpt_path, best, sizeof best);
    char tmp[330];
    snprintf(tmp, sizeof tmp, "%s.tmp", best);
    ppat_save_weights(tmp, theta, TOTAL, comment);
    rename(tmp, best);
}

/* Read the `positions: N` count embedded in a checkpoint comment (-1 if absent). */
static long ckpt_positions(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char buf[1024]; long pos = -1;
    while (fgets(buf, sizeof(buf), f)) {
        char *p = strstr(buf, "positions: ");
        if (p) { pos = atol(p + 11); break; }
    }
    fclose(f);
    return pos;
}

/* Read the cumulative (Σ bias², count) from a checkpoint comment; 0 on success. */
static int ckpt_train_sq(const char *path, double *dsum, long *dcnt, double *psum, long *pcnt) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char buf[1024]; int got = 0;
    while (fgets(buf, sizeof(buf), f)) {
        char *a = strstr(buf, "trainSqSum: "),    *b = strstr(buf, "trainSqCount: ");
        char *c = strstr(buf, "trainPartSum: "),  *d = strstr(buf, "trainPartCount: ");
        if (a && b && c && d) {
            *dsum = atof(a + 12); *dcnt = atol(b + 14);
            *psum = atof(c + 14); *pcnt = atol(d + 16);
            got = 1; break;
        }
    }
    fclose(f);
    return got ? 0 : -1;
}

/* Mean magnitude of the weight vector, Σ|theta[i]| / TOTAL — a single scalar for
 * tracking how large the learned weights are growing (overfitting often shows up
 * as a steadily climbing avgW). */
static double avg_abs_weight(void) {
    if (TOTAL <= 0) return 0;
    /* With --phase P, average only phase P's slice (patterns + prev-move) so the
     * column tracks the weights actually being trained, not the frozen phases. */
    if (cfg_phase >= 0) {
        const int np = ppat_num_patterns;
        const int prev_base = ppat_phase_count * np;
        double s = 0;
        for (int i = 0; i < np; i++) s += fabs(theta[cfg_phase * np + i]);
        for (int i = 0; i < 7; i++)  s += fabs(theta[prev_base + cfg_phase * 7 + i]);
        return s / (np + 7);
    }
    double s = 0;
    for (int i = 0; i < TOTAL; i++) s += fabs(theta[i]);
    return s / TOTAL;
}

/* Short names for the 7 prev-move features (≤6 chars to align with %6.2f). */
static const char *PREV_FEAT_NAMES[7] = {
    "contig", "savcap", "cap+sa", "extend", "ext+sa", "ko", "semeai"
};

/* Print one phase's 7 exp'd prev-move feature multipliers as "[ ... ]". */
static void print_phase_weights(int p) {
    int pb = ppat_phase_count * ppat_num_patterns + p * 7;
    printf("[ ");
    for (int i = 0; i < 7; i++) printf("%6.2f ", expf(theta[pb + i]));
    printf("]");
}

/* Print the exp'd prev-move feature multipliers from theta, as a suffix.
 * Single phase (--phase P, or phaseCount 1): that phase's vector.
 * Multiple phases trained together: first and last phase, to show divergence.
 * Shared by the solo print_stats and the parallel monitor. */
static void print_weights(void) {
    printf("  ");
    if (cfg_phase >= 0) {
        print_phase_weights(cfg_phase);
    } else if (ppat_phase_count <= 1) {
        print_phase_weights(0);
    } else {
        printf("p0 ");                          print_phase_weights(0);
        printf("  p%d ", ppat_phase_count - 1);  print_phase_weights(ppat_phase_count - 1);
    }
}

/* Print the prev-move feature-name header aligned under one phase's "[ ... ]". */
static void print_phase_header(void) {
    printf("[ ");
    for (int i = 0; i < 7; i++) printf("%6s ", PREV_FEAT_NAMES[i]);
    printf("]");
}

/* Header counterpart of print_weights: same layout, feature names instead of
 * values, so the column titles sit over the right brackets. */
static void print_weights_header(void) {
    printf("  ");
    if (cfg_phase >= 0) {
        print_phase_header();
    } else if (ppat_phase_count <= 1) {
        print_phase_header();
    } else {
        printf("p0 ");                          print_phase_header();
        printf("  p%d ", ppat_phase_count - 1);  print_phase_header();
    }
}

/* Dedicated monitor: repeatedly load the latest checkpoint and test it, printing
 * the metrics — without training or touching the sync barrier, so the training
 * workers never stall on the (expensive) test. */
static void run_monitor(void) {
    printf("monitor: %d test positions, test-playouts %d, %d workers, phases %d",
           n_test, cfg_test_playouts, cfg_workers, ppat_phase_count);
    if (cfg_phase >= 0) printf(", phase %d only (others frozen)", cfg_phase);
    printf("\n");
    printf("%9s  %7s  %7s  %7s  %5s  %8s  %7s",
           "positions", "trMSE", "teMSE", "avgW", "move%", "elapsedS", "pos/s");
    print_weights_header();
    printf("\n");
    fflush(stdout);
    ppat_load_quiet = 1;   /* suppress the per-cycle "loaded N weights" noise */
    wall_start = wall_now();
    float mon_best_te = 1e30f;   /* lowest teMSE seen, for the '*' new-low marker */

    /* Baseline row.  Fresh run: the uniform no-skill reference.  --load: the
     * loaded model's actual test + its weights (theta already holds the loaded
     * weights at this point), so the table starts from the real starting point. */
    {
        int loaded = (cfg_load != NULL);
        TestResult tr = measure_test(loaded ? 0 : 1, n_test);
        double el = wall_now() - wall_start;
        char eb[32]; snprintf(eb, sizeof(eb), "%.1fs", el);
        char tebuf[16];
        if (loaded) {
            printf("%9d  %7s  %7s  %7.4f  %5.1f  %8s  %7s",
                   0, "-", temse_col(tr.mse, &mon_best_te, tebuf, sizeof tebuf),
                   avg_abs_weight(), tr.move_match * 100.0f, eb, "-");
            print_weights();
            printf("\n");
        } else {
            printf("%9d  %7s  %7s  %7s  %5.1f  %8s  %7s\n",
                   0, "-", temse_col(tr.mse, &mon_best_te, tebuf, sizeof tebuf), "-", tr.move_match * 100.0f, eb, "-");
        }
        fflush(stdout);
    }

    double mon_last_full = -1;   /* latches the last full-epoch trMSE shown */
    struct stat mon_last_st; memset(&mon_last_st, 0, sizeof mon_last_st);
    /* Geometric test cadence, same shape as single-process: after a row at
     * elapsed E the next test is due at E + clamp(0.5·E, last_test_s,
     * MAX_PRINT_CYCLE_S).  Workers save every --sync-every positions, so
     * without this the monitor would test back-to-back forever (one row per
     * test duration) and burn a core on redundant tests deep into a run. */
    double mon_next_test = 0;
    double mon_last_test_s = 0;

    for (;;) {
        double el_now = wall_now() - wall_start;
        if (el_now < mon_next_test) { usleep(500000); continue; }   /* not due yet */

        /* Sleepy loop: only test + print when the checkpoint actually changed
         * (saves are tmp+rename, so inode/mtime/size move atomically). */
        struct stat st;
        if (stat(cfg_monitor, &st) != 0) { usleep(200000); continue; }  /* wait for first checkpoint */
        if (st.st_ino == mon_last_st.st_ino &&
            st.st_mtim.tv_sec == mon_last_st.st_mtim.tv_sec &&
            st.st_mtim.tv_nsec == mon_last_st.st_mtim.tv_nsec &&
            st.st_size == mon_last_st.st_size) {
            usleep(500000);
            continue;
        }
        float *w = ppat_load_weights(cfg_monitor);
        if (!w) { usleep(200000); continue; }
        mon_last_st = st;
        free(theta); theta = w; TOTAL = ppat_total_weights();

        /* Row time = when this test STARTED (relative to wall_start): scheduling
         * from here rather than from the post-test elapsed keeps the printed rows
         * geometric.  Scheduling from the finish time would add one test duration
         * to every gap, so the observed row ratio would be 1.5 + T/E — visibly
         * above 1.5 until E >> T. */
        double test_t0 = wall_now();
        double row_el = test_t0 - wall_start;
        TestResult tr = measure_test(0, n_test);     /* policy (full) test */
        mon_last_test_s = wall_now() - test_t0;

        /* Read the position count AFTER the test so positions and wall are both
         * current — otherwise pos/s is understated by the (long) test duration. */
        long my_pos = ckpt_positions(cfg_monitor);
        long agg = my_pos < 0 ? 0 : (long)cfg_workers * my_pos;
        double el = wall_now() - wall_start;
        double posps = el > 0 ? agg / el : 0;
        char eb[32]; snprintf(eb, sizeof(eb), "%.1fs", el);
        /* Train MSE = last completed epoch over the (fixed) training set, aggregated
         * across workers — read straight from the checkpoint. */
        double dsum, psum; long dcnt, pcnt; char trbuf[24], tebuf[16];
        if (ckpt_train_sq(cfg_monitor, &dsum, &dcnt, &psum, &pcnt) == 0)
            trmse_col(dsum, dcnt, psum, pcnt, &mon_last_full, trbuf, sizeof trbuf);
        else { trbuf[0] = '-'; trbuf[1] = 0; }
        int is_best = tr.mse < mon_best_te;
        printf("%9ld  %7s  %7s  %7.4f  %5.1f  %8s  %7.0f",
               agg, trbuf, temse_col(tr.mse, &mon_best_te, tebuf, sizeof tebuf),
               avg_abs_weight(), tr.move_match * 100.0f, eb, posps);
        print_weights();
        printf("\n");
        if (is_best) {
            char bc[256];
            snprintf(bc, sizeof bc, "Best by teMSE: %.6f, positions: %ld", tr.mse, agg);
            save_best(cfg_monitor, bc);
        }
        fflush(stdout);

        /* Schedule the next test geometrically from THIS test's start, so the
         * gap between printed rows is ~1.5x regardless of test cost.  The floor
         * is still one test duration, so back-to-back testing is the worst case. */
        double cycle = 0.5 * (row_el > 0 ? row_el : 1.0);
        if (cycle > MAX_PRINT_CYCLE_S) cycle = MAX_PRINT_CYCLE_S;
        if (cycle < mon_last_test_s) cycle = mon_last_test_s;
        mon_next_test = row_el + cycle;
        usleep(200000);   /* small floor so tiny test sets don't spin */
    }
}

/* ── Print stats ───────────────────────────────────────────────────────────── */

static void print_stats(int iterations, int total_positions, int use_uniform, int show_weights, int test_cap) {
    int test_n = (test_cap > 0 && test_cap < n_test) ? test_cap : n_test;
    if (test_n < 1) test_n = 1;
    clock_t test_t0 = clock();
    TestResult tr = measure_test(use_uniform, test_n);
    double last_test_s = (double)(clock() - test_t0) / CLOCKS_PER_SEC;
    cumulative_test_s += last_test_s;
    float mse = tr.mse;
    double elapsed_s = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    char elapsed_buf[32];
    snprintf(elapsed_buf, sizeof(elapsed_buf), "%.1fs", elapsed_s);

    double train_s = elapsed_s - cumulative_test_s;
    double pos_ms = total_positions > 0 ? 1000.0 * train_s / total_positions : 0;
    /* Aggregate throughput: total work (≈ workers × this process's positions, since the
     * barrier keeps workers in lockstep) over WALL time.  = positions/wall in single-process. */
    double wall_el = wall_now() - wall_start;
    double pos_per_s = wall_el > 0 ? (double)cfg_workers * total_positions / wall_el : 0;

    /* positions column = aggregate across workers (≈ workers × this process's count,
     * barrier-locked); posMs stays per-process (worker 0's CPU time / its own count).
     * tPos = how many test positions this row used (always the full test set). */
    static double last_full = -1;
    static float best_te = 1e30f;
    char trbuf[24], tebuf[16];
    int is_best = mse < best_te;
    trmse_col(done_sq_sum, done_sq_count, epoch_sq_sum, epoch_sq_count, &last_full, trbuf, sizeof trbuf);
    printf("%9ld  %7s  %7s  %7.4f  %5.1f  %5d  %6.1f  %6.1f  %8s  %6.1f  %7.0f",
           (long)cfg_workers * total_positions, trbuf,
           temse_col(mse, &best_te, tebuf, sizeof tebuf), avg_abs_weight(), tr.move_match * 100.0f,
           test_n, cumulative_test_s, cumulative_sync_s, elapsed_buf, pos_ms, pos_per_s);
    if (show_weights) print_weights();
    printf("\n");
    fflush(stdout);

    save_weights(iterations, total_positions, elapsed_buf);
    if (is_best) {
        char bc[256];
        snprintf(bc, sizeof bc, "Best by teMSE: %.6f, positions: %d", mse, total_positions);
        save_best(weights_file, bc);
    }
    /* Exponential cadence (½·elapsed of training between tests), but never let the
     * train cycle be shorter than the test that just ran — otherwise an expensive
     * test would dominate wall time.  Capped at 1 hour so a long run keeps
     * reporting (and checkpointing) instead of drifting to multi-hour gaps;
     * the floor still wins if a single test costs more than that. */
    double cycle = 0.5 * (elapsed_s > 0 ? elapsed_s : 1.0);
    if (cycle > MAX_PRINT_CYCLE_S) cycle = MAX_PRINT_CYCLE_S;
    if (cycle < last_test_s) cycle = last_test_s;
    next_print = elapsed_s + cycle;
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        fprintf(stderr, "Usage: %s <file> [--lr <f>] [--playouts <n>] [--M <n>] [--N <n>]\n", argv[0]);
        fprintf(stderr, "       [--batch <n>] [--test-pos <n>] [--train-pos <n>] [--test-file <path>]\n");
        fprintf(stderr, "       [--test-playouts <n>] [--no-extreme <f>] [--iteration-limit <n>]\n");
        fprintf(stderr, "       [--phases <n>] [--phase <p>] [--init-phase-scale <f>] [--overfit]\n");
        return 1;
    }

    cfg_file         = argv[1];
    cfg_lr           = get_float_arg(argc, argv, "--lr", 10.0f);
    int playouts     = get_int_arg(argc, argv, "--playouts", 500);  /* default for M, N, test-playouts */
    cfg_M            = get_int_arg(argc, argv, "--M", playouts);
    cfg_N            = get_int_arg(argc, argv, "--N", cfg_M);
    cfg_batch        = get_int_arg(argc, argv, "--batch", 1);
    cfg_test_pos     = get_int_arg(argc, argv, "--test-pos", 100);
    cfg_train_pos    = get_int_arg(argc, argv, "--train-pos", 0);
    cfg_test_playouts = get_int_arg(argc, argv, "--test-playouts", playouts);
    cfg_no_extreme       = get_float_arg(argc, argv, "--no-extreme", 0.0f);
    cfg_iter_limit   = get_int_arg(argc, argv, "--iteration-limit", 0);
    cfg_overfit      = has_flag(argc, argv, "--overfit");
    int baseline_only = has_flag(argc, argv, "--baseline-only");  /* print uniform baseline, then exit */
    cfg_init         = get_int_arg(argc, argv, "--init", 0);
    cfg_train_moves = get_int_arg(argc, argv, "--train-moves", -1);
    ppat_phase_count = get_int_arg(argc, argv, "--phases", 1);
    cfg_test_file = get_str_arg(argc, argv, "--test-file", NULL);
    cfg_test_pos_given = has_flag(argc, argv, "--test-pos");
    cfg_load = get_str_arg(argc, argv, "--load", NULL);
    cfg_save = get_str_arg(argc, argv, "--save", NULL);
    cfg_monitor = get_str_arg(argc, argv, "--monitor", NULL);
    cfg_phase = get_int_arg(argc, argv, "--phase", -1);
    /* Presence of --init-phase-scale enables seeding phase P from phase P+1. */
    int cfg_init_from_next = has_flag(argc, argv, "--init-phase-scale");
    float cfg_init_phase_scale = get_float_arg(argc, argv, "--init-phase-scale", 1.0f);
    cfg_workers    = get_int_arg(argc, argv, "--workers", 1);
    cfg_worker_id  = get_int_arg(argc, argv, "--worker-id", 0);
    cfg_sync_every = get_int_arg(argc, argv, "--sync-every", 100);
    cfg_sync_dir   = get_str_arg(argc, argv, "--sync-dir", "out/ppat-sync");
    int parallel = (cfg_workers > 1 && cfg_sync_every > 0);
    if (parallel) mkdir(cfg_sync_dir, 0777);   /* idempotent; launcher should clear it first */

    /* Per-worker seed so workers explore independently before averaging. */
    rng_seed(&g_rng, (uint32_t)time(NULL) ^ (uint32_t)(cfg_worker_id * 0x9e3779b9u + 1u));
    ppat_init();

    load_positions();
    split_data();

    TOTAL = ppat_total_weights();
    theta           = calloc(TOTAL, sizeof(float));
    if (cfg_load) {
        float *loaded = ppat_load_weights(cfg_load);
        if (!loaded) { fprintf(stderr, "failed to load weights\n"); exit(1); }
        TOTAL = ppat_total_weights();
        free(theta);
        theta = loaded;
    } else if (cfg_init) {
        static const float init_prev[7] = {7.43f, 151.04f, 0.53f, 23.11f, 0.02f, 6.37f, 141.80f};
        int prev_base = ppat_phase_count * ppat_num_patterns;
        for (int p = 0; p < ppat_phase_count; p++)
            for (int i = 0; i < 7; i++)
                theta[prev_base + p * 7 + i] = logf(init_prev[i]);
    }

    /* Warm-start (--init-phase-scale present): seed phase P's weights from the
     * already-trained phase P+1, scaled by sc (the endgame-first chain). Only the
     * pattern + prev-move slices of phase P are overwritten; the rest of theta
     * (incl. the frozen later phases) is left as loaded. The gradient mask then
     * refines phase P alone. */
    if (cfg_init_from_next) {
        if (cfg_phase < 0) {
            fprintf(stderr, "error: --init-phase-scale requires --phase\n");
            exit(1);
        }
        if (cfg_phase + 1 >= ppat_phase_count) {
            fprintf(stderr, "error: --init-phase-scale: phase %d has no successor (phases %d)\n",
                    cfg_phase, ppat_phase_count);
            exit(1);
        }
        const int np = ppat_num_patterns;
        const int prev_base = ppat_phase_count * np;
        const float sc = cfg_init_phase_scale;   /* θ scale: γ → γ^sc (sc<1 softens toward uniform) */
        for (int i = 0; i < np; i++)
            theta[cfg_phase * np + i] = sc * theta[(cfg_phase + 1) * np + i];
        for (int i = 0; i < 7; i++)
            theta[prev_base + cfg_phase * 7 + i] = sc * theta[prev_base + (cfg_phase + 1) * 7 + i];
    }

    rollout_grad_buf = calloc(TOTAL, sizeof(float));
    g_buf           = calloc(TOTAL, sizeof(float));
    batch_buf       = calloc(TOTAL, sizeof(float));
    memset(&rollout_feat_st, 0, sizeof(rollout_feat_st));

    /* Monitor: test the checkpoint in a loop and exit (never trains). */
    if (cfg_monitor) { run_monitor(); return 0; }

    /* Output filename */
    if (cfg_save) snprintf(weights_file, sizeof(weights_file), "%s", cfg_save);
    else          snprintf(weights_file, sizeof(weights_file), "out/ppat-data-%08x.js", rng_next(&g_rng));

    if (cfg_worker_id == 0) {
    char scalebuf[32];
    if (cfg_init_from_next) snprintf(scalebuf, sizeof scalebuf, "%.3g", cfg_init_phase_scale);
    else                    snprintf(scalebuf, sizeof scalebuf, "off");
    printf("train: %s (%d positions; %d/worker × %d)  lr: %.1f  M: %d  N: %d  batch: %d  overfit: %s  no-extreme: %.1f  init: %s  train-moves: %d  phases: %d  phase: %d  init-phase-scale: %s\n",
           cfg_file, n_train_total, n_train, cfg_workers, cfg_lr, cfg_M, cfg_N, cfg_batch,
           cfg_overfit ? "true" : "false", cfg_no_extreme,
           cfg_init ? "true" : "false", cfg_train_moves, ppat_phase_count, cfg_phase, scalebuf);
    if (parallel)
        printf("parallel: %d workers  sync-every: %d  sync-dir: %s\n", cfg_workers, cfg_sync_every, cfg_sync_dir);
    char best_file[320]; best_path(weights_file, best_file, sizeof best_file);
    printf("test: %d positions%s%s  test-playouts: %d  output: %s  best: %s\n",
           n_test, cfg_test_file ? " from " : "", cfg_test_file ? cfg_test_file : "",
           cfg_test_playouts, weights_file, best_file);
    printf("%9s  %7s  %7s  %7s  %5s  %5s  %6s  %6s  %8s  %6s  %7s",
           "positions", "trMSE", "teMSE", "avgW", "move%", "tPos", "testS", "syncS", "elapsedS", "posMs", "pos/s");
    print_weights_header();
    printf("\n");
    }

    start_time = clock();
    wall_start = wall_now();
    int total_positions = 0;
    int iterations = 0;
    next_print = 0;

    /* Inline testing/printing only in single-process mode.  In parallel, testing is
     * done by a separate --monitor process so the training workers never stall on the
     * barrier; worker 0 just saves the averaged checkpoint after each sync. */
    const int do_inline = (cfg_workers == 1);

    /* baseline: fresh run shows the uniform no-skill reference; --load shows the
     * loaded model's actual test + weights.  --baseline-only stops here. */
    if (do_inline || baseline_only) {
        print_stats(iterations, total_positions, cfg_load ? 0 : 1, cfg_load ? 1 : 0, n_test);
        if (baseline_only) return 0;
    }

    for (;;) {
        for (int li = 0; li < n_train; li++) {
            if (do_inline) {
                double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
                if (elapsed > next_print) {
                    /* Always test the FULL --test-pos set: rows (and the best-*
                     * star) stay statistically comparable.  The exponential print
                     * cadence bounds the amortized test cost instead. */
                    print_stats(iterations, total_positions, 0, 1, 0);
                }
            }

            Position *pos = &all_positions[train_idx[li]];
            Game2 g;
            int bad = -1;
            int rp = replay_position(pos, &g, &bad);
            if (rp < 0) { fprintf(stderr, "WARNING: illegal move #%d (idx %d) in training position, skipping\n", bad, pos->history[bad]); continue; }
            if (rp == 0) continue;
            update_theta(&g, pos->value);
            total_positions++;

            if (parallel && total_positions % cfg_sync_every == 0) {
                flush_batch();
                barrier_sync_average();   /* θ ← mean over workers */
                if (cfg_worker_id == 0) save_weights(iterations, total_positions, "");
            }
        }

        /* Epoch boundary: latch this epoch's training-fit over the full fixed set
         * (published in the next sync's .stat / used by the solo column). */
        done_sq_sum = epoch_sq_sum; done_sq_count = epoch_sq_count;
        epoch_sq_sum = 0; epoch_sq_count = 0;

        iterations++;
        shuffle_train();

        if (cfg_iter_limit > 0 && iterations >= cfg_iter_limit) {
            if (do_inline) print_stats(iterations, total_positions, 0, 1, 0);
            break;
        }
    }

    free(theta);
    free(rollout_grad_buf);
    free(g_buf);
    free(batch_buf);
    return 0;
}
