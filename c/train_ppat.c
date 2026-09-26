/*
 * train_ppat.c — Simulation Balancing (Huang, Coulom, Lin 2010, Algorithm 1).
 * C port of train-ppat.js.
 *
 * Input: either eval-file format, discriminated per line (a move list starts
 * with a coordinate letter, a phase with a digit):
 *   legacy (gen_evals):        "<size> <move1,move2,...> <value> [best_move]"
 *   new (gen-agent-evals):     "<size> <phase> <move1,move2,...> <winRatio>"
 * Values in [0,1] mapped to [-1,1].
 *
 * Compile:
 *   cc -O2 -o train_ppat game2.c ppat.c train_ppat.c -lm
 *
 * Usage:
 *   ./train_ppat <file> [options]
 *   Options:
 *     --lr <f>              learning rate (default 10)
 *     --playouts <n>        default for --value-playouts and --gradient-playouts (default 100)
 *     --value-playouts <n>  rollouts for the V estimate (default --playouts)
 *     --gradient-playouts <n>  rollouts for the gradient (default --playouts)
 *     --trunc-vpat <path>   TRUNCATED training rollouts: after ceil(delta*area)
 *                           moves, if the phase there is <= --trunc-max-phase
 *                           (default 0.55, the deployed gate) the rollout stops
 *                           and z becomes this vpat evaluator's value; a cut past
 *                           the gate runs full as before.
 *                           Makes an early band mouth affordable: playout cost
 *                           becomes delta*area moves + one eval, flat in the
 *                           mouth's phase, with the evaluator confined to the
 *                           band deployment already trusts.  Applies to train
 *                           AND test rollouts (same estimator; directWR, the
 *                           primary readout, is match-based and unaffected).
 *     --trunc-delta <f>     the cut distance, in phase units (moves-method:
 *                           ceil(delta * area) moves past the start).  Defaults
 *                           to the model file's baked delta (trunc.delta) when
 *                           omitted; required if the model has none.
 *     --trunc-max-phase <f> the gate B (default 0.55)
 *     --match-phases A,B    directWR match band (default 0.6,1): inside [A, B]
 *                           each side plays its own weights; outside, BOTH
 *                           sides play reference moves, so games differ only
 *                           where the subject is trained and the out-of-band
 *                           play is fixed, competent and symmetric (fixed-seed
 *                           rows still pair).  For truncation-band training
 *                           set it to the corpus band.  Rows are comparable
 *                           only across runs at the same band
 *     --phase-compensation-buckets <n>
 *                           reweight per-step gradient credit so applied
 *                           pressure is uniform by phase (default 0 = off).
 *                           Buckets track applied WEIGHT (1/(N*T) per counted
 *                           step, pre-compensation); correction shrinks toward
 *                           1 on thin evidence and normalizes so the
 *                           pressure-weighted mean is 1 (total applied
 *                           throughput conserved), so the effective lr keeps
 *                           its meaning.  Without it, pressure by phase is an
 *                           artifact of corpus geometry (a mouth corpus ramps
 *                           across its band then plateaus to the game-end
 *                           taper; truncation reshapes it again)
 *     --batch <n>           batch size (default 1)
 *     --test-pos <n>        test positions (default 0 = no teMSE test).  The match
 *                           columns are the primary readout now; a test set costs
 *                           real time (5000 positions x 500 playouts is ~90s a row)
 *                           and teMSE has been observed flat across 5x more
 *                           training while directWR still climbed.  Ask for one
 *                           explicitly when you want it.
 *     --test-file <path>    take the test set from THIS file instead of the head
 *                           of <file>; the whole of <file> is then training data.
 *                           Use a fixed high-playout set as a permanent yardstick:
 *                           teMSE then compares across runs and datasets.
 *     --train-pos <n>       train positions (default 0 = all)
 *     --test-playouts <n>   playouts per test position (default: derived from
 *                           --test-total-playouts).  When set explicitly it
 *                           overrides the total.  The per-position playout-
 *                           variance floor is 0.25/n (worst case); teMSE_c
 *                           subtracts it, so teMSE_c stays comparable across n and
 *                           n only trades per-position compute for lower teMSE_c
 *                           variance.
 *     --test-total-playouts <n>  the default source of the per-position count:
 *                           spread this TOTAL over the test set, per-position =
 *                           round(n / #test-positions), floored at 1 (default
 *                           200000).  Fixes the cost of one teMSE evaluation
 *                           regardless of test-set size.  An explicit
 *                           --test-playouts overrides it; set 0 to disable.
 *                           teMSE_c stays comparable because it removes the
 *                           0.25/N floor.
 *     --no-extreme <f>      drop TRAIN positions whose value is more extreme than ±(1-2f) (default 0 = keep all)
 *     --iteration-limit <n> stop after n iterations (default infinite)
 *     --overfit             use same data for train and test
 *
 * EARLY PASS: always on.  The playout policy offers PASS as a candidate at
 * logit 0 and every checkpoint is stamped earlyPass, so a consumer cannot load
 * the weights without the anchor they were fitted against.  The anchor adds no
 * parameter — it IDENTIFIES the additive constant on the pattern weights, which
 * is otherwise unconstrained, because shifting every pattern weight leaves the
 * softmax unchanged.  With it pinned, the absolute logit level means "how good
 * a move must be to be worth playing".  Fine-tuning an unflagged model is how
 * one is converted.
 *     --twelvecell          extend an ALL-EMPTY ninecell with a second key for
 *                           the four distance-2 orthogonals (21 D4 orbits).  In
 *                           open areas the ninecell is one shared weight, so the
 *                           policy picks uniformly there; this differentiates.
 *                           The block is APPENDED, so --load of a model without
 *                           it fine-tunes (old weights keep their indices, the
 *                           new ones start at zero).
 *     --atari <n>           graded gives-atari feature (default 0 = off): a
 *                           gated key when the candidate reduces an adjacent
 *                           enemy chain to one liberty, one-hot on
 *                           min(largest such chain's size, n).  Invisible to
 *                           the cap-2 pattern (two enemy liberties code the
 *                           same as safe), so this is liberty resolution the
 *                           ninecell lacks.  Appended block; fine-tunes in
 *     --capture <n>         graded capture-size feature (default 0 = off): a
 *                           gated key when the candidate captures, one-hot on
 *                           min(total stones captured, n).  The pattern sees
 *                           capture EXISTENCE at cap 2; this adds the size
 *     --capture-by-self-atari <n>
 *                           ko-take / snapback interaction grid (default 0 =
 *                           off): fires when the candidate captures AND ends
 *                           in atari itself, one-hot on the (own size,
 *                           stones captured) grid capped at n each
 *     --atari-by-self-atari <n>
 *                           mutual-atari interaction grid (default 0 = off):
 *                           fires when the candidate is BOTH self-atari and
 *                           gives atari (capturing-race fills, snapbacks),
 *                           one-hot on the (own size, victim size) grid
 *                           capped at n each — n*n weights per phase
 *     --twelvecell2         the same key on a LOOSER trigger: the four ADJACENT
 *                           points empty, whatever the diagonals hold.  Fires
 *                           strictly more often.  Mutually exclusive with
 *                           --twelvecell — they share the one weight block.
 *     --adj-lib <n>         orthogonal liberty cap in the 3x3 pattern (2..4,
 *                           default 2 = the historical atari-only encoding).
 *                           Higher caps resolve more liberty levels at zero
 *                           runtime cost but a larger pattern table; the cap is
 *                           written into the weights file as adjLib.  Ignored
 *                           with --load (the file's own cap wins).
 *     --no-local            freeze the 7 previous-move ("local") features at 0 —
 *                           contiguous, save-atari by capture/extension (+self-atari
 *                           variants), ko, 2-point semeai.  Their gradient is masked
 *                           out, so they stay at their initial 0 and contribute
 *                           nothing to the logit: the model behaves exactly as if
 *                           they did not exist, while the weight layout is
 *                           unchanged (7 zeros per phase).  An ablation of what the
 *                           local features are worth, alongside --adj-lib 1, which
 *                           ablates liberty information from the 3x3 pattern.
 *     --ref-weights <path>  reference model for the directWR column (default
 *                           out/ppat-data-233162-best-ref-candidate.js, the
 *                           pat-only adj-lib-2 model; "none" disables the
 *                           column).  Each row plays
 *                           --ref-games policy-only games (no search, no tree)
 *                           between the current model and the reference and
 *                           reports the current model's win rate, so it reads
 *                           MOVE quality — teMSE reads rollout VALUE, which is
 *                           what SB actually optimises.  The reference must
 *                           share this run's adjLib and phase count.
 *                           A second column, WR, plays the same two models as
 *                           mc-ppat agents (one playout per candidate move,
 *                           then play a winner) — the way an agent actually
 *                           consumes a playout policy.
 *     --no-direct           disable the directWR match entirely and hide the
 *                           column (same effect as --ref-weights none); use when
 *                           directWR is not trusted to represent the target.
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
#include "vpat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <unistd.h>     /* usleep */
#include <sys/stat.h>   /* mkdir */

#define SYNC_POLL_US 2000   /* parameter-sync barrier poll: 2 ms base (jittered) */
#define MAX_PRINT_CYCLE_S 3600.0   /* cap the geometric print/test gap at 1 h (inline + monitor) */
#define MON_GROWTH        1.3      /* parallel monitor: row-to-row elapsed ratio */

/* ── Configuration ───────────────────────────────────────────────────────────
 * Defaults live ONLY at the get_*_arg() call sites in main() (single source of
 * truth); these declarations are zero/NULL-initialised.  Comments document the
 * MEANING of special values, not the default. */

static float  cfg_lr;
static int    cfg_value_po;
static int    cfg_gradient_po;
static int    cfg_batch;
static int    cfg_test_pos;
static int    cfg_train_pos;
static int    cfg_test_playouts;        /* explicit per-position count; 0 = derive from the total */
static int    cfg_test_total_playouts;  /* > 0: spread this total over the test set instead */
static int    cfg_test_playouts_derived; /* set when cfg_test_playouts came from the total */
static float  cfg_no_extreme;
static int    cfg_iter_limit;          /* 0 = infinite */
static int    cfg_overfit;
static int    cfg_phase;               /* -1 = all phases; >= 0 = train/test only this phase */
static int    cfg_no_local;            /* 1 = freeze the 7 previous-move ("local") features at 0 */
/* Board size of the training positions; global because topology is (g2_init_topology). */
static int    topo_size = 0;        /* board size of the TRAIN file — training's topology */
static int    test_board_size = 0;  /* board size of --test-file (== topo_size when unset) */
static int    last_loaded_size = 0; /* size of the file load_positions_from() just read */
static const char *cfg_ref_weights;    /* reference model for the directWR column (NULL = off) */

/* Match-column calibration.  Constants rather than CLI flags: these are
 * instrument settings, not per-run choices — pick good ones once.
 *
 * Scales (6000 games each, colour-swap pairing on a shared opening):
 *   directWR  50.0 = as good as the reference (27.3 = uniform was measured
 *             against the retired 287076 reference; re-measure per ref).  The neutral
 *             point is EXACT, not estimated: with identical weights both seats
 *             play the same game, so each colour-swapped pair scores one win and
 *             one loss by construction and the self-match cannot drift off 50. */
#define DIRECT_GAMES     10000   /* directWR, FIRST row: SE ~0.5pp (~3s at size 10, ~3x that at 13) */
#define MATCH_GROWTH       1.1   /* match effort grows this much per printed row */
#define MATCH_MAX_S      600.0   /* wall-clock ceiling per match; growth stops once hit */
/* directWR match band (--match-phases A,B): inside [A, B] each side plays its
 * own weights; OUTSIDE the band both sides play REFERENCE moves, so the games
 * differ only where the subject is trained and the opening/endgame are a
 * fixed, competent, symmetric policy.  Default 0.6,1. */
static float cfg_match_phase_lo = 0.6f, cfg_match_phase_hi = 1.0f;
#define DEPLOY_BOARD_SIZE   13   /* The size the policy is FIELDED at.  directWR (its own
                                  * games) measures there whatever size the training data is,
                                  * because small-board verdicts must be re-validated at 13
                                  * anyway.  The topology is swapped around the match (a
                                  * microsecond rebuild) and restored.  teMSE follows its
                                  * --test-file's size, since its records replay. */

/* Polyak-Ruppert weight averaging.  The window is in AGGREGATE POSITIONS, not in
 * sync rounds: --sync-every is a comms/round-error knob that should scale with lr
 * and worker count, and coupling the smoothing to it would mean tuning stability
 * silently retunes the averaging.  The EMA can only SAMPLE at points where theta
 * is meaningful (after a barrier, since between barriers each worker has its own),
 * but it WEIGHTS by elapsed positions, so the window is invariant to --sync-every
 * and --workers; those only change the sampling density. */
#define EMA_SAMPLE_POS     100   /* standalone sampling cadence (no barriers there).
                                  * Not a tuning knob: --ema-window weights by elapsed
                                  * positions, so sampling density cannot move the
                                  * window, only how finely it is resolved. */

/* Parameter sync across processes: K workers each run batch-1 SB with their own
 * seed; every --sync-every positions they file-barrier all-reduce the SUM of their
 * θ displacements (not the mean — see barrier_sync_average), so --lr means the same
 * thing at any worker count. */
static int    cfg_workers;

/* Truncated training rollouts (--trunc-vpat + friends): after ceil(delta *
 * area) moves, if the phase there is <= trunc-max-phase the rollout stops and
 * z becomes the vpat value — the same estimator deployment trusts, gated to the
 * same band (the evaluator is never consulted past B; a rollout whose cut
 * overshoots the gate just runs to the end as before).  --trunc-delta defaults
 * to the model file's baked delta when omitted. */
static int    cfg_trunc_on;
static float  cfg_trunc_delta;
static float  cfg_trunc_max_phase;
static int    cfg_trunc_delta_from_model;  /* delta defaulted from the model's baked trunc block, not --trunc-delta */
static const char *cfg_trunc_vpat = "";
static int    cfg_init_from_next;
static float  cfg_init_phase_scale = 1.0f;

/* --phase-comp: reweight per-step gradient credit so the applied pressure is
 * uniform by phase.  Without it, pressure is a pure artifact of corpus
 * geometry (a mouth corpus ramps across its band, then a plateau to the
 * game-end taper; truncation reshapes it again), which makes configs hard to
 * reason about.  Buckets track APPLIED weight (each gradient step adds its
 * 1/(N*T) coefficient — steps are not equal across rollouts, 3x so under
 * mixed truncated/full), pre-compensation so the estimator never chases its
 * own output.  The correction shrinks toward 1 on thin evidence:
 * c[b] = Wbar/(w[b] + Wbar/PC_SHRINK), normalized to mean 1 over occupied
 * buckets so the effective lr keeps its meaning; count-only during warmup. */
#define PC_MAX_BUCKETS 256
#define PC_STEP_CAP    (MAX_CAP * 8)
#define PC_SHRINK      8.0f
#define PC_WARMUP_POSITIONS 100
static int    cfg_pc_buckets;          /* --phase-compensation-buckets, 0 = off */
static double pc_w[PC_MAX_BUCKETS];    /* accumulated raw (pre-comp) weight per bucket */
static float  pc_c[PC_MAX_BUCKETS];    /* current correction factors */
static long   pc_positions;            /* update_theta calls seen (warmup gate) */
static int    pc_step_bucket[PC_STEP_CAP];   /* per-rollout: bucket of each counted step */

/* Refresh pc_c from pc_w: shrinkage toward 1, then mean-1 normalization. */
static void pc_refresh(void) {
    double total = 0; int occ = 0;
    for (int b = 0; b < cfg_pc_buckets; b++) if (pc_w[b] > 0) { total += pc_w[b]; occ++; }
    if (occ == 0 || pc_positions < PC_WARMUP_POSITIONS) {
        for (int b = 0; b < cfg_pc_buckets; b++) pc_c[b] = 1.0f;
        return;
    }
    const double wbar = total / occ, eps = wbar / PC_SHRINK;
    /* Normalize so the PRESSURE-WEIGHTED mean of the correction is 1
     * (sum w*c = sum w): total applied gradient throughput is conserved and
     * only its distribution changes.  An unweighted mean-1 normalization
     * deflates globally — c and w are inversely related, so the heavy
     * buckets' sub-1 factors dominate the applied product (observed live as
     * avgW growing much slower with compensation on). */
    double wcsum = 0;
    for (int b = 0; b < cfg_pc_buckets; b++) {
        pc_c[b] = (float)(wbar / (pc_w[b] + eps));
        wcsum += pc_w[b] * pc_c[b];
    }
    const float norm = (float)(total / wcsum);
    for (int b = 0; b < cfg_pc_buckets; b++) pc_c[b] *= norm;
}
static int    cfg_worker_id;
static int    cfg_sync_every;          /* 0 = no sync (single process) */
static const char *cfg_sync_dir;
static const char *cfg_file;
static const char *cfg_test_file;      /* NULL = carve the test set from cfg_file's head */
static int    cfg_test_pos_given;      /* was --test-pos passed explicitly? */
/* movedetails MAE column: mc-ppat move-selection regret vs a labelled *.md set,
 * computed on the (slow) test rows.  Default file so the column is on by default. */
static const char *cfg_md_file = "movedetails_5059.md";
static int    cfg_mae_cand_playouts = 10;   /* rollouts per candidate */
static float  cfg_mae_band_lo = 0.6f;       /* phase band (--mae-band A,B) */
static float  cfg_mae_band_hi = 1.0f;
static int    cfg_mae_pos = 0;              /* in-band positions to score (0 = all); --mae-pos caps the cost */
static const char *cfg_load;           /* path to weights file to load */
static bool ref_early_pass;            /* the REFERENCE model's own earlyPass, from its file */
static float ref_pass_weight = 0;      /* and its own learned pass logit.  Used in the
                                        * match when set; an unflagged reference borrows
                                        * the run's anchor and pass weight instead. */
/* The trained pass logit.  SB cannot fit it: the update along the shared level
 * is Cov(z, passed), which vanishes when an early mutual stop is outcome-
 * neutral — and under one-step area scoring it largely is.  So it is driven by
 * a control loop instead: each rollout that CHOSE to stop reports whether the
 * final board still had unfinished business — a chain in atari (an unresolved
 * capture) or a dame (a point that counts for neither side until someone takes
 * it).  Either means it stopped too early.
 * Either finding means it stopped too early; a clean board means only that it
 * did not, which is not evidence that it stopped as early as it could have.  So
 * the two directions carry different weight — see PASS_STEP. */

static float run_pass_weight = 0;
/* Polyak average of the pass weight, on the same window as theta_ema.  The
 * saved threshold has to be consistent with the saved board weights: it
 * competes against their log-sum-exp, so pairing averaged patterns with a raw
 * scalar caught at an excursion gives the file a stopping point neither
 * iterate had. */
static float run_pass_weight_ema = 0;
/* The two directions are not the same kind of evidence.  Unfinished business on
 * the final board is a KNOWN error — the rollout stopped with points or captures
 * still on the table — so it moves 10x.  A clean board proves nothing: it is consistent
 * with stopping at exactly the right moment or far too late, so it only nudges
 * upward by 1x to keep the weight from sinking forever.  That 10:1 ratio sets
 * the equilibrium — the weight settles where about 1 chosen termination in 11
 * leaves an unresolved capture. */
#define PASS_STEP         0.00001f

/* Fraction of rollouts in which the model's FIRST pass is REJECTED and it is
 * made to play on.  A rollout that stops early is scored on an unfinished
 * board, so its outcome is a poor estimate of the position's value and the
 * board weights are fitted against it; forcing play past the pass makes z a
 * better target.  The cost is that the gradient goes off-policy — ψ is a score
 * function assuming the action came from π — so the weights are fitted to a
 * behaviour policy that half-refuses passes while deployment does not.  The
 * mismatch scales with this rate; reduce the dose if it hurts.
 *
 * A rejected rollout is excluded from the pass-weight controller: it plays on
 * to a natural finish, which by construction has no dame and no atari, so the
 * detector would read "clean" and nudge the threshold UP in exactly the case
 * that should push it down. */
#define PASS_REJECT_RATE  0.5f
/* Deliberately small.  This is one scalar governing when EVERY playout stops,
 * so being wrong by much is expensive in a way no pattern weight is — an error
 * here mis-scores every label at once.  The dynamics make a small step cheap:
 * the signal only exists on rollouts that CHOSE to pass, so a too-eager weight
 * produces a sample almost every rollout and descends fast, while near
 * equilibrium passes are rare, samples are rare, and the weight hovers.  The
 * random walk around the equilibrium then has amplitude ~step·sqrt(samples),
 * i.e. hundredths rather than units. */
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
/* trMSE_c: the same rows' playout-variance floor, Σ V01(1-V01)/(M-1) over the
 * epoch, subtracted from trMSE to strip the value-playout noise (the training-
 * side analogue of teMSE_c).  Only the completed-epoch sum is ever reported
 * (trmse_col ignores the partial sum), so only done_floor_sum is serialized to
 * the monitor in parallel mode. */
static double  epoch_floor_sum = 0;
static double  done_floor_sum = 0;
static double  agg_train_sq_sum = 0;   /* last completed epoch, summed over workers (worker 0) */
static long    agg_train_sq_count = 0;
static double  agg_part_sq_sum = 0;    /* current (partial) epoch, summed over workers */
static long    agg_part_sq_count = 0;
static double  agg_train_floor_sum = 0;/* last completed epoch's floor, summed over workers (worker 0) */

/* ── Parameter vector ──────────────────────────────────────────────────────── */

static Rng  g_rng;          /* this process's RNG (seeded in main) */
static int    TOTAL;
static float *theta;
/* Polyak average of theta.  This is what gets SAVED and therefore what the monitor
 * loads and measures — training continues on the raw iterate.  Note --load of such
 * a checkpoint resumes from the average, not from the raw state that produced it. */
static float *theta_ema = NULL;
static long   ema_last_pos = 0;
static int    cfg_ema_window;          /* --ema-window, aggregate positions; default 2000, 0 = off */
static int    cfg_seed;                /* --seed; 0 = seed from the clock (non-reproducible) */
static long   cfg_test_from;           /* --test-from: pin the position where testing starts */

/* Scratch buffers */
static PpatState rollout_feat_st;
static float    *rollout_grad_buf;
static float    *g_buf;
static float    *batch_buf;
static int      batch_count = 0;

/* avgW, matching the JS trainers' convention: mean |weight| encountered across
 * weight UPDATES (frequency-weighted over the weights actually being trained),
 * reset at every print — NOT the mean over all stored weights, which would be
 * dominated by the ~74% of the dense table that no legal position ever reaches
 * and would shift with adjLib for reasons unrelated to training. */
static double   w_abs_sum = 0;
static long     w_update_count = 0;
/* First POLICY pass of each rollout, as board fullness (cap-empty)/cap.  A raw
 * minimum over an interval is useless — thousands of rollouts pin it to the
 * uniform-gate floor on the first row and it never moves — so this is the MEAN
 * over rollouts of when each one first chose to stop.  Forced passes (the move
 * list running out, PICK_NO_MOVES) are excluded: those are the old behaviour and
 * would drag the mean toward 1.0.  Rollouts that never pass contribute nothing.
 * Cumulative, like w_abs_sum, and differenced per printed row. */
static double   pass_phase_sum = 0;
static long     pass_phase_count = 0;

static float    rollout_logits[MAX_CAP];
static float    rollout_probs[MAX_CAP];

/* ── Parse command line ────────────────────────────────────────────────────── */

/* Every argv index consumed by a parser below is marked here; after all
 * parsing, check_unknown_args rejects anything unmarked (a typo like
 * --train-games would otherwise be silently ignored). */
#define MAX_ARGS 256
static char arg_used[MAX_ARGS];

static int get_int_arg(int argc, char **argv, const char *flag, int def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) { arg_used[i] = arg_used[i+1] = 1; return atoi(argv[i+1]); }
    return def;
}

static float get_float_arg(int argc, char **argv, const char *flag, float def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) { arg_used[i] = arg_used[i+1] = 1; return (float)atof(argv[i+1]); }
    return def;
}

static const char *get_str_arg(int argc, char **argv, const char *flag, const char *def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], flag) == 0) { arg_used[i] = arg_used[i+1] = 1; return argv[i+1]; }
    return def;
}

static int has_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], flag) == 0) { arg_used[i] = 1; return 1; }
    return 0;
}

/* Call after the last get_*_arg/has_flag: argv[1] is the <file> positional;
 * everything else must have been consumed by a parser. */
static void check_unknown_args(int argc, char **argv) {
    if (argc > MAX_ARGS) { fprintf(stderr, "error: too many arguments (%d)\n", argc); exit(1); }
    int bad = 0;
    for (int i = 2; i < argc; i++)
        if (!arg_used[i]) { fprintf(stderr, "error: unknown argument '%s'\n", argv[i]); bad = 1; }
    if (bad) exit(1);
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
    /* Two line formats (gen-agent-evals announces the new one with a
     * "# format:" header comment, but detection is per line so concatenated
     * mixed files work): the second token starts with a digit only in the
     * new format (a phase like "0.345"); a move list starts with a letter. */
    if (sscanf(line, "%d %4095s", &size, moves_buf) != 2) return 0;
    int fields;
    if (moves_buf[0] >= '0' && moves_buf[0] <= '9') {
        /* new: <size> <phase> <moves> <winRatio> — phase is redundant with the
         * replay (which recomputes it for --phase filtering), so skip it. */
        fields = sscanf(line, "%d %*s %4095s %lf", &size, moves_buf, &value);
        if (fields < 3) return 0;
        fields = 3;   /* no best-move token in the new format */
    } else {
        fields = sscanf(line, "%d %4095s %lf %15s", &size, moves_buf, &value, best_buf);
        if (fields < 3) return 0;
    }
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
    /* One size per FILE (the records are replayed here for their phase, which
     * needs this file's topology active).  The train and test files may differ:
     * measure_test swaps the topology the same way direct_match_wr does. */
    int file_size = 0;
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
        if (!file_size) { file_size = pos.board_size; g2_init_topology(file_size); }
        else if (pos.board_size != file_size) {
            fprintf(stderr, "error: mixed board sizes in %s (line %d: size %d, expected %d; one size per file)\n",
                    path, lineno, pos.board_size, file_size);
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

        /* Phase is needed ONLY for the --phase filter: the stored pos.phase is
         * read nowhere else (truncation and the phase mask recompute it from the
         * live rollout board), so without a filter this play-through is pure
         * waste — skip it. */
        if (cfg_phase >= 0) {
            Game2 g;
            int bad;
            if (replay_position(&pos, &g, &bad) > 0)
                pos.phase = ppat_phase_count * (g.cap - g.empty_count) / g.cap;
            else
                pos.phase = -1;
            if (!in_test_head && pos.phase != cfg_phase) { filtered++; continue; }
        } else {
            pos.phase = -1;   /* unused when no --phase filter */
        }

        all_positions[n_all++] = pos;
    }
    fclose(f);
    last_loaded_size = file_size;
    if (skipped > 5)
        fprintf(stderr, "WARNING: %d more lines skipped\n", skipped - 5);
    if (n_all == base) { fprintf(stderr, "error: no valid positions in %s (%d lines skipped)\n", path, skipped); exit(1); }
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
        test_board_size = last_loaded_size;
        load_positions_from(cfg_file, 0);   /* all of it is training data */
        topo_size = last_loaded_size;
    } else {
        load_positions_from(cfg_file, cfg_test_pos);
        topo_size = test_board_size = last_loaded_size;
    }
    /* The test file may have left ITS topology active; training owns the global. */
    g2_init_topology(topo_size);
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
        /* Every worker needs a non-empty slice.  With train_n < cfg_workers the
         * low-numbered workers get lo == hi and DEADLOCK the run: their inner loop
         * body never executes, so total_positions never advances and they never
         * reach the barrier, while the workers that did get positions block
         * forever waiting for them.  Three cores spin at 100% and nothing
         * progresses, with no output to say why — so fail here instead. */
        if (train_n < cfg_workers) {
            fprintf(stderr, "error: --train-pos %d is fewer than --workers %d; every worker needs at "
                            "least one position.\n       --train-pos is a TOTAL, split across workers "
                            "(%d workers => use --train-pos %d or more).\n",
                    train_n, cfg_workers, cfg_workers, cfg_workers);
            exit(1);
        }
        int lo = (int)((long)cfg_worker_id       * train_n / cfg_workers);
        int hi = (int)((long)(cfg_worker_id + 1)  * train_n / cfg_workers);
        n_train = hi - lo;
        for (int i = 0; i < n_train; i++) train_idx[i] = train_start + lo + i;  /* this worker's slice */
        n_test = test_n;
        for (int i = 0; i < test_n; i++) test_idx[i] = i;                /* head (shared) */
    }
    /* --test-total-playouts: spread a total budget evenly over the test set, so
     * one teMSE evaluation costs a fixed amount regardless of how many test
     * positions there are.  teMSE_c subtracts the 0.25/N floor, so the metric
     * stays comparable even as the per-position count N varies with test size. */
    if (cfg_test_playouts <= 0 && cfg_test_total_playouts > 0) {
        cfg_test_playouts = n_test > 0 ? (cfg_test_total_playouts + n_test / 2) / n_test
                                       : cfg_test_total_playouts;
        cfg_test_playouts_derived = 1;
    }
    if (cfg_test_playouts < 1) cfg_test_playouts = 1;
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
/* Returns the chosen index into rollout_feat_st, PICK_NO_MOVES when there are no
 * candidates at all, or PICK_PASSED when the policy CHOSE to pass.  The two are
 * distinguished because only the second carries a gradient.
 * Leaves rollout_feat_st and rollout_probs populated; with the pass anchor on,
 * rollout_probs is normalised over the board moves AND the pass, so the board
 * probabilities sum to less than 1.  That is what identifies the additive
 * constant on the pattern weights: without a pass entry, shifting every pattern
 * weight leaves this softmax unchanged and the direction has no gradient. */
#define PICK_NO_MOVES (-1)
#define PICK_PASSED   (-2)
/* The model being TRAINED always carries the anchor — that is what fits the
 * absolute logit level.  A loaded model's own flag does not override it:
 * fine-tuning an unflagged file is how one is converted.  The REFERENCE model
 * in the directWR match plays its own flag when its file has one, and falls
 * back to this anchor when it does not — see the note there.
 *
 * Every use below is guarded by this: 0 disables the anchor, the controller,
 * the pass-rejection hack and the subject's pass in the match, and stamps
 * earlyPass false on saved weights.  The REFERENCE never passes either way. */
#define RUN_EARLY_PASS 1

static int policy_select(Game2 *g) {
    ppat_extract(g, &rollout_feat_st);
    int n = rollout_feat_st.count;
    if (n == 0) return PICK_NO_MOVES;

    for (int i = 0; i < n; i++) {
        float v = 0;
        for (int fi = rollout_feat_st.feat_start[i]; fi < rollout_feat_st.feat_start[i + 1]; fi++)
            v += theta[rollout_feat_st.feat[fi]];
        rollout_logits[i] = v;
    }

    /* Softmax, including the pass at logit 0 when the run trains for it. */
    float mx = rollout_logits[0];
    for (int i = 1; i < n; i++) if (rollout_logits[i] > mx) mx = rollout_logits[i];
    if (RUN_EARLY_PASS && run_pass_weight > mx) mx = run_pass_weight;
    float sum = 0;
    for (int i = 0; i < n; i++) { rollout_probs[i] = expf(rollout_logits[i] - mx); sum += rollout_probs[i]; }
    float p_pass = RUN_EARLY_PASS ? expf(run_pass_weight - mx) : 0.0f;
    sum += p_pass;
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) rollout_probs[i] *= inv;
    p_pass *= inv;

    /* Sample */
    float r = rng_float(&g_rng);
    if (RUN_EARLY_PASS) { r -= p_pass; if (r <= 0) return PICK_PASSED; }
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

static float rollout(const Game2 *game, int8_t player, float *grad_acc, int *out_steps) {
    Game2 sim;
    g2_clone(&sim, game);
    int steps = 0;

    /* Truncation cut, in MOVES past the start (the deployed moves-method). */
    const int cut_steps = cfg_trunc_on ? (int)ceilf(cfg_trunc_delta * (float)(sim.N * sim.N)) : -1;

    int passed_yet = 0, rejected_pass = 0, reject_first = 0;
    if (RUN_EARLY_PASS && rng_float(&g_rng) < PASS_REJECT_RATE) reject_first = 1;
    for (int step = 0; !sim.game_over; step++) {
        if (step == cut_steps) {
            const float area = (float)(sim.N * sim.N);
            const float ph = 1.0f - (float)sim.empty_count / area;
            if (ph <= cfg_trunc_max_phase) {
                /* Evaluate here, exactly as deployment does (puct-ppat-fp-trunc):
                 * v = sigma(z), mapped to the rollout's [-1, 1] convention. */
                float v = (float)(1.0 / (1.0 + exp(-vpat_evaluate_z(&sim))));
                if (player != BLACK) v = 1.0f - v;
                if (out_steps) *out_steps = steps;
                return 2.0f * v - 1.0f;
            }
            /* Cut overshot the gate: run the rollout to the end as usual. */
        }
        int chosen = policy_select(&sim);
        if (chosen == PICK_NO_MOVES) {
            /* Nothing to choose between — no decision, so no gradient. */
            g2_play(&sim, PASS);
            continue;
        }

        int n = rollout_feat_st.count;

        if (grad_acc) {
            /* Phase compensation: this step's whole ψ contribution is scaled
             * by the phase bucket's correction (chosen-move and expectation
             * term alike); the bucket is recorded below, where the step is
             * counted, so the rollout's raw 1/(N*T) weight lands in pc_w. */
            float pc = 1.0f;
            int pcb = -1;
            if (cfg_pc_buckets) {
                int b = cfg_pc_buckets * (sim.cap - sim.empty_count) / sim.cap;
                if (b >= cfg_pc_buckets) b = cfg_pc_buckets - 1;
                pcb = b;
                pc = pc_c[b];
            }
            /* ψ(s,a) = φ(s,a) − Σ_b π(b|s)φ(s,b).  The pass carries no features,
             * so a chosen pass contributes only the −Σ term: a uniform downward
             * push on every board feature, which is exactly the update that
             * lowers the absolute logit level toward passing more often. */
            for (int i = 0; i < n; i++) {
                float p = pc * rollout_probs[i];
                for (int fi = rollout_feat_st.feat_start[i]; fi < rollout_feat_st.feat_start[i + 1]; fi++)
                    grad_acc[rollout_feat_st.feat[fi]] -= p;
            }
            if (chosen != PICK_PASSED)
                for (int fi = rollout_feat_st.feat_start[chosen]; fi < rollout_feat_st.feat_start[chosen + 1]; fi++)
                    grad_acc[rollout_feat_st.feat[fi]] += pc;

            /* Count this step toward T (all phases) / T_P (masked single phase). */
            if (cfg_phase < 0 ||
                ppat_phase_count * (sim.cap - sim.empty_count) / sim.cap == cfg_phase) {
                if (pcb >= 0 && steps < PC_STEP_CAP) pc_step_bucket[steps] = pcb;
                steps++;
            }
        }

        if (chosen == PICK_PASSED) {
            if (!passed_yet) {
                passed_yet = 1;
                pass_phase_sum += (double)(sim.cap - sim.empty_count) / sim.cap;
                pass_phase_count++;
                if (reject_first) {
                    /* Refuse it and play on: resample among the board moves
                     * only, so the rollout reaches a finished position and its
                     * outcome is a usable target.  The ψ term above was already
                     * accumulated for the pass the policy actually wanted —
                     * that is the off-policy part. */
                    rejected_pass = 1;
                    float r2 = rng_float(&g_rng), tot = 0;
                    for (int i = 0; i < n; i++) tot += rollout_probs[i];
                    r2 *= tot;
                    int alt = n - 1;
                    for (int i = 0; i < n; i++) { r2 -= rollout_probs[i]; if (r2 <= 0) { alt = i; break; } }
                    g2_play(&sim, rollout_feat_st.moves[alt]);
                    continue;
                }
            }
            g2_play(&sim, PASS);
            continue;
        }
        int32_t mv = rollout_feat_st.moves[chosen];
        g2_play(&sim, mv);
    }

    /* Pass-weight control loop.  Only rollouts the policy played to the end and
     * actually chose to stop are evidence. */
    if (RUN_EARLY_PASS && passed_yet && !rejected_pass) {
        /* Unfinished business on the final board, of either kind:
         *   - a chain in atari: an unresolved capture, so the score is unreliable
         *   - a DAME (empty point touching both colours): it counts for neither
         *     side under the one-step area score, so whoever plays it gains it —
         *     leaving one is giving away a point that was there for the taking.
         * The dame half is what gives the policy any reason to fill them before
         * stopping; atari alone says nothing about them. */
        int unresolved = 0;
        for (int i = 0; i < sim.cap && !unresolved; i++) {
            if (sim.cells[i] != EMPTY) {
                if (sim.ls[sim.gid[i]] == 1) unresolved = 1;
            } else {
                int b = 0, w = 0;
                for (int d = 0; d < 4; d++) {
                    int8_t c = sim.cells[g2_nbr[i * 4 + d]];
                    if (c == BLACK) b = 1; else if (c == WHITE) w = 1;
                }
                if (b && w) unresolved = 1;
            }
        }
        run_pass_weight += unresolved ? -10.0f * PASS_STEP : PASS_STEP;
    }

    if (out_steps) *out_steps = steps;
    return g2_estimate_winner(&sim) == player ? 1.0f : -1.0f;
}

/* ── Core update (Algorithm 1) ─────────────────────────────────────────────── */

/* When --phase P is set, zero the gradient outside phase P's pattern and
 * prev-move slices so the other phases stay frozen at their loaded/init value.
 * Rollouts still traverse later phases for move selection, but only phase P's
 * weights are updated — exact coordinate-restricted SB gradient descent. */
/* Zero the gradient for the 7 previous-move ("local") features of every phase,
 * so they never leave their initial 0.  A weight of 0 adds nothing to the logit,
 * so this is a true ablation, not merely a frozen parameter. */
static void mask_local(float *v) {
    /* Only the 7 prev-move slots — the twelvecell/self-atari/atari blocks
     * live past them and must keep their gradient (this used to zero
     * everything to TOTAL, a latent bug since the twelvecell block landed). */
    const int prev_base = ppat_phase_count * ppat_num_patterns;
    const int prev_end  = prev_base + ppat_phase_count * 7;
    for (int k = prev_base; k < prev_end; k++) v[k] = 0.0f;
}

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

    /* Phase compensation: correction factors refresh once per position from
     * the counters (cheap; count-only until the warmup completes). */
    if (cfg_pc_buckets) { pc_refresh(); pc_positions++; }

    /* V: M value-playouts, no gradient. */
    float V = 0;
    for (int i = 0; i < cfg_value_po; i++) V += rollout(game, player, NULL, NULL);
    V /= cfg_value_po;

    /* g: N rollouts with gradient.  Algorithm 1: g ← g + z/(N·T)·Σ_t ψ.  T is the
     * rollout's policy-step count (T_P, the in-phase steps, when a phase is masked).
     * T == 0 means no ψ was accumulated, so that rollout contributes nothing. */
    int N = cfg_gradient_po;
    memset(g_buf, 0, sizeof(float) * TOTAL);
    for (int j = 0; j < N; j++) {
        memset(rollout_grad_buf, 0, sizeof(float) * TOTAL);
        int T = 0;
        float z = rollout(game, player, rollout_grad_buf, &T);
        if (T > 0) {
            float scale = z / ((float)N * (float)T);
            for (int k = 0; k < TOTAL; k++) g_buf[k] += scale * rollout_grad_buf[k];
            /* Track the rollout's raw applied weight per bucket: each counted
             * step carried 1/(N*T), PRE-compensation, so the estimator never
             * chases its own output. */
            if (cfg_pc_buckets) {
                const double wstep = 1.0 / ((double)N * (double)T);
                const int lim = T < PC_STEP_CAP ? T : PC_STEP_CAP;
                for (int k = 0; k < lim; k++) pc_w[pc_step_bucket[k]] += wstep;
            }
        }
    }
    if (cfg_phase >= 0) mask_to_phase(g_buf);
    if (cfg_no_local)   mask_local(g_buf);

    /* Gradient uses the un-normalised [-1,1] bias (the SB paper's faster-learning
     * -1/1 regime).  The MSE byproduct normalises v* and V to win-probability
     * [0,1] before squaring, so trMSE/teMSE are reported in the paper's units. */
    float bias = v_star - V;
    float v01 = 0.5f * (v_star + 1.0f);
    float V01 = 0.5f * (V + 1.0f);
    epoch_sq_sum += (double)(v01 - V01) * (v01 - V01);
    epoch_sq_count++;
    /* trMSE_c floor: variance of V01 as a mean of M=cfg_value_po playout outcomes,
     * estimated by V01(1-V01)/(M-1).  M<=1 gives no estimate, so contribute 0. */
    epoch_floor_sum += (cfg_value_po > 1)
        ? (double)V01 * (1.0 - V01) / (cfg_value_po - 1) : 0.0;
    for (int k = 0; k < TOTAL; k++) batch_buf[k] += bias * g_buf[k];
    batch_count++;

    /* Flush batch */
    if (batch_count >= cfg_batch) {
        float scale = cfg_lr / batch_count;
        for (int k = 0; k < TOTAL; k++) {
            if (batch_buf[k] != 0.0f) {            /* this weight was updated */
                theta[k] += scale * batch_buf[k];
                w_abs_sum += fabs(theta[k]);
                w_update_count++;
                batch_buf[k] = 0;
            }
        }
        batch_count = 0;
    }
}

/* Apply any pending (sub-full) batch — called before a parameter-sync so
 * θ reflects every update this worker has made. */
static void flush_batch(void) {
    if (batch_count == 0) return;
    float scale = cfg_lr / batch_count;
    for (int k = 0; k < TOTAL; k++) {
        if (batch_buf[k] != 0.0f) {
            theta[k] += scale * batch_buf[k];
            w_abs_sum += fabs(theta[k]);
            w_update_count++;
            batch_buf[k] = 0;
        }
    }
    batch_count = 0;
}

/* ── Parameter-sync all-reduce (file-based barrier) ────────────────────────────
 * Every worker writes θ for the current round (atomic via tmp+rename), waits for
 * all K workers' round files, then sets θ ← θ₀ + Σ (θ_worker − θ₀): the SUM of
 * every worker's displacement this round, NOT the mean (see barrier_sync_average
 * for why summing keeps --lr invariant to worker count).  Cleanup of round
 * r-1 is safe once all round-r files exist: a worker only writes round r after it
 * finished reading every round r-1 file, so no one is still reading r-1. */
static int sync_round = 0;
static double cumulative_sync_s = 0;   /* wall time spent in parameter-sync (I/O + barrier wait) */

static double wall_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* θ at the start of the current sync round — identical across workers, since it
 * is whatever the previous barrier produced.  Needed to turn each worker's
 * absolute θ back into the displacement it contributed. */
static float *round_start = NULL;

static void capture_round_start(void) {
    if (!round_start) round_start = malloc((size_t)TOTAL * sizeof(float));
    memcpy(round_start, theta, (size_t)TOTAL * sizeof(float));
}

static void barrier_sync_average(void) {
    double _sync_t0 = wall_now();
    char path[600], tmp[600];
    /* Publish this worker's cumulative train-fit (Σ bias², count) before θ, so a
     * peer's .stat is guaranteed present once its θ is visible to worker 0. */
    snprintf(tmp,  sizeof(tmp),  "%s/r%d_w%d.stat.tmp", cfg_sync_dir, sync_round, cfg_worker_id);
    snprintf(path, sizeof(path), "%s/r%d_w%d.stat",     cfg_sync_dir, sync_round, cfg_worker_id);
    { FILE *sf = fopen(tmp, "w");
      if (sf) { fprintf(sf, "%.9g %ld %.9g %ld %.9g\n", done_sq_sum, done_sq_count, epoch_sq_sum, epoch_sq_count, done_floor_sum); fclose(sf); rename(tmp, path); } }
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
        /* Accumulate DISPLACEMENTS, not parameters.  Averaging θ also averages the
         * displacements, so K workers' worth of gradient yields one worker's worth
         * of step and the effective learning rate per aggregate position becomes
         * lr/K — measured directly: at K=2 the run processed 2.24x the positions
         * for identical avgW and an identical directWR.  Summing the displacements
         * makes the round's step equal the sum of all gradients, which is what one
         * worker over the same positions would have taken, so --lr means the same
         * thing at any worker count.
         *
         * Subtract per worker before summing rather than summing θ and correcting
         * afterwards: acc - (K-1)*θ₀ would cancel large near-equal numbers and lose
         * most of the delta's precision in float, since |θ| reaches ~50 while a
         * round's displacement is ~1e-4. */
        for (int k = 0; k < TOTAL; k++) acc[k] += buf[k] - round_start[k];
    }
    for (int k = 0; k < TOTAL; k++) theta[k] = round_start[k] + acc[k];
    capture_round_start();                 /* θ₀ for the next round */

    /* Worker 0 aggregates every worker's cumulative train-fit for this round.
     * Each peer's .stat was published before its θ (already read above), so all
     * are present without polling. */
    if (cfg_worker_id == 0) {
        agg_train_sq_sum = 0; agg_train_sq_count = 0; agg_part_sq_sum = 0; agg_part_sq_count = 0;
        agg_train_floor_sum = 0;
        for (int w = 0; w < cfg_workers; w++) {
            char wp[600];
            snprintf(wp, sizeof(wp), "%s/r%d_w%d.stat", cfg_sync_dir, sync_round, w);
            FILE *wf = fopen(wp, "r");
            if (wf) {
                double ds = 0, ps = 0, fs = 0; long dc = 0, pc = 0;
                /* 5th field (done_floor_sum) is optional: pre-field .stat files
                 * still aggregate, contributing 0 floor (trMSE_c falls back to trMSE). */
                if (fscanf(wf, "%lf %ld %lf %ld %lf", &ds, &dc, &ps, &pc, &fs) >= 4) {
                    agg_train_sq_sum += ds; agg_train_sq_count += dc;
                    agg_part_sq_sum  += ps; agg_part_sq_count  += pc;
                    agg_train_floor_sum += fs;
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
/* ── Measure test ──────────────────────────────────────────────────────────── */

typedef struct { float mean_abs; float mse; float mse_c; float mae; } TestResult;

#define TEST_RNG_SEED 0x7e57c0deL   /* fixed seed → reproducible test rollouts */

/* ── movedetails MAE: mc-ppat move-selection regret vs a labelled *.md set ───── */
typedef struct { int move; float wr; } MdCand;   /* wr = winRatio in [0,1]; <0 = terminal/unrated */
typedef struct { int size; float phase; int32_t *hist; int n_hist; MdCand *cand; int n_cand; } MdPos;
static MdPos *md_pos = NULL;
static int    n_md = 0;
static int    md_size = 0;

/* coord "g7" -> board index (matches game2.js parseMove); "pass" -> PASS. */
static int md_parse_coord(const char *s, int N) {
    if (s[0] == 'p') return PASS;
    return (atoi(s + 1) - 1) * N + (s[0] - 'a');
}

/* Load the *.md file (movedetails-format.js).  Returns positions loaded (0 on
 * failure — the mae column is then simply absent). */
static int load_md_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "WARNING: --md-file %s not found; mae column disabled\n", path); return 0; }
    size_t cap = 1024; md_pos = malloc(cap * sizeof *md_pos);
    static char line[1 << 16];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') continue;
        char *save;
        char *id     = strtok_r(line, " ",   &save);
        char *sizeS  = strtok_r(NULL, " ",   &save);
        char *phaseS = strtok_r(NULL, " ",   &save);
        char *histS  = strtok_r(NULL, " ",   &save);
        char *candS  = strtok_r(NULL, " \n", &save);
        if (!id || strcmp(id, "f1") != 0 || !sizeS || !phaseS || !histS || !candS) continue;
        const int N = atoi(sizeS);
        if (n == (int)cap) { cap *= 2; md_pos = realloc(md_pos, cap * sizeof *md_pos); }
        MdPos *p = &md_pos[n];
        p->size = N; p->phase = (float)atof(phaseS);
        p->hist = NULL; p->n_hist = 0; p->cand = NULL; p->n_cand = 0;
        if (strcmp(histS, "-") != 0) {
            int hc = 1; for (char *c = histS; *c; c++) if (*c == ',') hc++;
            p->hist = malloc(hc * sizeof(int32_t));
            char *hs; for (char *t = strtok_r(histS, ",", &hs); t; t = strtok_r(NULL, ",", &hs))
                p->hist[p->n_hist++] = md_parse_coord(t, N);
        }
        if (strcmp(candS, "-") != 0) {
            int cc = 1; for (char *c = candS; *c; c++) if (*c == ',') cc++;
            p->cand = malloc(cc * sizeof(MdCand));
            char *cs; for (char *t = strtok_r(candS, ",", &cs); t; t = strtok_r(NULL, ",", &cs)) {
                char *colon = strchr(t, ':');
                MdCand mc;
                if (colon) { *colon = '\0'; mc.move = md_parse_coord(t, N); mc.wr = colon[1] == '\0' ? -1.0f : atoi(colon + 1) / 1000.0f; }
                else       { mc.move = md_parse_coord(t, N); mc.wr = -1.0f; }
                p->cand[p->n_cand++] = mc;
            }
        }
        n++;
    }
    fclose(f);
    md_size = n ? md_pos[0].size : 0;
    return n;
}

/* One mc-ppat-style playout from `start` (the candidate move already played),
 * returning +1 if `mover` wins else -1.  Mirrors ai/mc-ppat.js's playout(): the
 * deployment policy ppat_policy_move (NOT the training rollout), no early-pass
 * rejection, capped at 3*empty+20 moves, then estimateWinner.  The
 * ppat_uniform_below_phase gate (--uniform-below-phase) applies inside it. */
static float md_playout(const Game2 *start, int8_t mover, PpatState *st, Rng *rng) {
    Game2 g; g2_clone(&g, start);
    const int move_limit = 3 * g.empty_count + 20;
    int moves = 0;
    while (!g.game_over && moves < move_limit) {
        g2_play(&g, ppat_policy_move(&g, st, theta, false, run_pass_weight, rng));
        moves++;
    }
    return g2_estimate_winner(&g) == mover ? 1.0f : -1.0f;
}

/* Mean win-ratio gap between the file's best move and the mc-ppat-selected move,
 * over the band-filtered md positions, using the CURRENT model.  A fixed RNG so
 * the column is reproducible run-to-run. */
static float md_mae(void) {
    if (n_md == 0) return 0.0f;
    static PpatState st;
    Rng saved = g_rng; rng_seed(&g_rng, TEST_RNG_SEED ^ 0x5a5aL);
    const int swap = md_size != topo_size; if (swap) g2_init_topology(md_size);
    double gap_sum = 0; int count = 0;
    for (int pi = 0; pi < n_md; pi++) {
        if (cfg_mae_pos > 0 && count >= cfg_mae_pos) break;
        MdPos *mp = &md_pos[pi];
        if (mp->phase < cfg_mae_band_lo || mp->phase > cfg_mae_band_hi) continue;
        Game2 g; g2_new_empty(&g, mp->size);
        int ok = 1;
        for (int i = 0; i < mp->n_hist; i++) if (!g2_play(&g, mp->hist[i])) { ok = 0; break; }
        if (!ok || g.game_over) continue;
        const int8_t mover = g.current;
        float best_wr = -1e30f, worst_wr = 1e30f;
        for (int c = 0; c < mp->n_cand; c++) { float w = mp->cand[c].wr;
            if (w >= 0.0f) { if (w > best_wr) best_wr = w; if (w < worst_wr) worst_wr = w; } }
        if (best_wr < -1e29f) continue;   /* no rated candidate */
        double best_val = -1e30; int picked = -1;
        for (int c = 0; c < mp->n_cand; c++) {
            const int mv = mp->cand[c].move;
            if (mv == PASS || g2_is_true_eye_at(&g, mv)) continue;   /* legal non-eye, no pass — as mc-ppat */
            Game2 clone; g2_clone(&clone, &g);
            if (!g2_play(&clone, mv)) continue;
            double val;
            if (clone.game_over) val = g2_estimate_winner(&clone) == mover ? 1.0 : -1.0;
            else { float s = 0; for (int k = 0; k < cfg_mae_cand_playouts; k++) s += md_playout(&clone, mover, &st, &g_rng);
                   val = (double)s / cfg_mae_cand_playouts; }
            /* Dither in DOUBLE: a float32 val loses a 1e-9 nudge to rounding near
             * ±1, and a strict > would then break ties by file order (best label
             * first), leaking the label into the pick. */
            val += rng_float(&g_rng) * 1e-9;
            if (val > best_val) { best_val = val; picked = c; }
        }
        if (picked < 0) continue;
        float pw = mp->cand[picked].wr; if (pw < 0.0f) pw = worst_wr;   /* terminal pick → charged worst */
        gap_sum += best_wr - pw; count++;
    }
    if (swap) g2_init_topology(topo_size);
    g_rng = saved;
    return count ? (float)(gap_sum / count) : 0.0f;
}

/* Test the first `n` of the test positions; n is clamped to [1, n_test]. */
static TestResult measure_test(int use_uniform, int n) {
    if (n < 1) n = 1;
    if (n > n_test) n = n_test;
    /* Deterministic test rollouts: seed a FIXED stream so teMSE is reproducible
     * run-to-run and across report lines (differences reflect the weights, not MC
     * noise).  Save/restore g_rng so training's own stream is untouched. */
    Rng saved_rng = g_rng;
    rng_seed(&g_rng, TEST_RNG_SEED);
    /* A test set of a different size than the training data scores at ITS size:
     * the records carry their own board_size for g2_new, and this swaps the
     * global neighbour tables to match (restored before every return). */
    const int swap_topo = test_board_size != topo_size;
    if (swap_topo) g2_init_topology(test_board_size);
    float abs_sum = 0, sq_sum = 0, cq_sum = 0;
    int count = 0;
    for (int ti = 0; ti < n; ti++) {
        Position *pos = &all_positions[test_idx[ti]];
        Game2 g;
        int bad = -1;
        int rp = replay_position(pos, &g, &bad);
        if (rp < 0) { fprintf(stderr, "WARNING: illegal move #%d (idx %d) in test position %d, skipping\n", bad, pos->history[bad], test_idx[ti]); continue; }
        if (rp == 0) continue;

        int8_t player = g.current;
        float sum = 0;
        for (int i = 0; i < cfg_test_playouts; i++)
            sum += use_uniform ? uniform_rollout(&g, player) : rollout(&g, player, NULL, NULL);
        /* Normalise v* and the rollout mean from [-1,1] to win-probability [0,1]
         * before the error, so MSE matches the SB paper's units. */
        float v01 = 0.5f * (pos->value + 1.0f);
        float V01 = 0.5f * (sum / cfg_test_playouts + 1.0f);
        float d = v01 - V01;
        abs_sum += fabsf(d);
        sq_sum += d * d;
        /* Floor-corrected: subtract the per-position playout-variance floor
         * V01(1-V01)/(N-1) (unbiased estimate of Var(V01)), so teMSE_c estimates
         * the model's true error with the 0.25/N measurement noise removed and is
         * comparable across --test-playouts.  Per-position terms can go slightly
         * negative; the mean over the test set is stable. */
        cq_sum += d * d - (cfg_test_playouts > 1
                           ? V01 * (1.0f - V01) / (cfg_test_playouts - 1) : 0.0f);
        count++;
    }
    g_rng = saved_rng;   /* restore training's RNG stream */
    if (swap_topo) g2_init_topology(topo_size);
    const float mae = n_md ? md_mae() : 0.0f;
    if (count == 0) return (TestResult){0, 0, 0, mae};
    return (TestResult){ abs_sum / count, sq_sum / count, cq_sum / count, mae };   /* mse = mean squared error */
}


/* ── Save weights ──────────────────────────────────────────────────────────── */

static char weights_file[256];

static clock_t start_time;
static double  wall_start;            /* wall clock at training start, for pos/s */
/* Standalone print schedule, in POSITIONS rather than seconds, so a --seed run
 * reports at reproducible points.  Cycles grow geometrically; the expensive
 * columns switch on only once a cycle's TRAINING time exceeds the time a test
 * costs, so the early phase — when tests would otherwise dominate wall clock —
 * spends all of it on training and still prints the free columns. */
#define PRINT_POS_FIRST    200
#define PRINT_POS_GROWTH   1.5
static long    next_print_pos;
static double  last_print_test_s;      /* duration of the most recent tested row */
static double  cumulative_test_s = 0;

/* Render the trMSE column: only FULL epochs are shown — a freshly-completed
 * epoch (full_count>0, mean differs from *last_full) prints once; every other
 * row prints "...".  Partial running means are not shown: they mix a shrinking
 * epoch fraction with position order and read as noise.  *last_full latches
 * the most recent full value so it prints exactly once. */
static const char *trmse_col(double full_sum, long full_count, double part_sum, long part_count,
                             double *last_full, char *buf, size_t n) {
    (void)part_sum;
    if (full_count > 0) {
        double fm = full_sum / (double)full_count;
        if (fm != *last_full) { *last_full = fm; snprintf(buf, n, "%.4f", fm); return buf; }
    }
    /* "..." = an epoch is in progress; "-" = nothing trained yet (row 0). */
    snprintf(buf, n, (part_count > 0 || full_count > 0) ? "..." : "-");
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


/* Seed the average with the starting weights (after --load / warm-start). */
static void ema_init(void) {
    if (cfg_ema_window <= 0) return;        /* leave theta_ema NULL: save/report raw */
    theta_ema = malloc((size_t)TOTAL * sizeof(float));
    memcpy(theta_ema, theta, (size_t)TOTAL * sizeof(float));
    run_pass_weight_ema = run_pass_weight;
    ema_last_pos = 0;
}

/* Time-weighted EMA sampled at agg_pos aggregate positions.  exp(-dpos/WINDOW)
 * makes the decay depend only on how much training elapsed, so irregular or
 * changing sampling intervals give the same window. */
static void ema_update(long agg_pos) {
    if (!theta_ema) return;                 /* EMA_WINDOW 0: averaging disabled */
    long dpos = agg_pos - ema_last_pos;
    if (dpos <= 0) return;
    const float w = expf(-(float)dpos / (float)cfg_ema_window);
    for (int k = 0; k < TOTAL; k++) theta_ema[k] = w * theta_ema[k] + (1.0f - w) * theta[k];
    run_pass_weight_ema = w * run_pass_weight_ema + (1.0f - w) * run_pass_weight;
    ema_last_pos = agg_pos;
}

static void save_weights(int iterations, int total_positions, const char *elapsed) {
    char comment[384];
    /* Training-fit accumulators for the monitor: the last completed epoch (full)
     * and the current in-progress epoch (partial), each Σ bias² + count. */
    double dsum = (cfg_workers > 1) ? agg_train_sq_sum   : done_sq_sum;
    long   dcnt = (cfg_workers > 1) ? agg_train_sq_count : done_sq_count;
    double psum = (cfg_workers > 1) ? agg_part_sq_sum    : epoch_sq_sum;
    long   pcnt = (cfg_workers > 1) ? agg_part_sq_count  : epoch_sq_count;
    /* trMSE_c floor for the completed epoch (only the full sum is reported). */
    double fsum = (cfg_workers > 1) ? agg_train_floor_sum : done_floor_sum;
    /* avgW accumulators too: in parallel mode the workers never print, so these
     * stay CUMULATIVE and the monitor differences consecutive checkpoints to get
     * a per-interval mean (worker 0's own updates — a representative sample). */
    snprintf(comment, sizeof(comment),
             "Generated by train_ppat (C) — iterations: %d, positions: %d, elapsed: %s, phases: %d, trainSqSum: %.9g, trainSqCount: %ld, trainPartSum: %.9g, trainPartCount: %ld, trainFloorSum: %.9g, wAbsSum: %.9g, wUpdateCount: %ld, passPhaseSum: %.9g, passPhaseCount: %ld",
             iterations, total_positions, elapsed, ppat_phase_count, dsum, dcnt, psum, pcnt, fsum,
             w_abs_sum, w_update_count, pass_phase_sum, pass_phase_count);
    /* Atomic: write to a tmp file then rename, so a reader (the monitor) never
     * sees a half-written checkpoint. */
    char tmp[300];
    snprintf(tmp, sizeof(tmp), "%s.tmp", weights_file);
    ppat_save_weights(tmp, theta_ema ? theta_ema : theta, TOTAL, RUN_EARLY_PASS,
                      theta_ema ? run_pass_weight_ema : run_pass_weight, comment);
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
    ppat_save_weights(tmp, theta, TOTAL, RUN_EARLY_PASS, run_pass_weight, comment);
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
/* Cumulative avgW accumulators from a checkpoint comment (see save_weights). */
static int ckpt_wsum(const char *path, double *wsum, long *wcnt) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char buf[1024]; int got = 0;
    while (fgets(buf, sizeof(buf), f)) {
        char *a = strstr(buf, "wAbsSum: "), *b = strstr(buf, "wUpdateCount: ");
        if (a && b) { *wsum = atof(a + 9); *wcnt = atol(b + 14); got = 1; break; }
    }
    fclose(f);
    return got ? 0 : -1;
}

/* Cumulative first-pass-phase accumulators from a checkpoint comment. */
static int ckpt_passphase(const char *path, double *psum, long *pcnt) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char buf[1024]; int got = 0;
    while (fgets(buf, sizeof(buf), f)) {
        char *a = strstr(buf, "passPhaseSum: "), *b = strstr(buf, "passPhaseCount: ");
        if (a && b) { *psum = atof(a + 14); *pcnt = atol(b + 16); got = 1; break; }
    }
    fclose(f);
    return got ? 0 : -1;
}

static int ckpt_train_sq(const char *path, double *dsum, long *dcnt, double *psum, long *pcnt,
                         double *fsum) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char buf[1024]; int got = 0;
    while (fgets(buf, sizeof(buf), f)) {
        char *a = strstr(buf, "trainSqSum: "),    *b = strstr(buf, "trainSqCount: ");
        char *c = strstr(buf, "trainPartSum: "),  *d = strstr(buf, "trainPartCount: ");
        if (a && b && c && d) {
            *dsum = atof(a + 12); *dcnt = atol(b + 14);
            *psum = atof(c + 14); *pcnt = atol(d + 16);
            /* trainFloorSum is optional: absent in pre-field checkpoints, so
             * default 0 (trMSE_c then equals trMSE) rather than failing the parse. */
            char *e = strstr(buf, "trainFloorSum: ");
            *fsum = e ? atof(e + 15) : 0.0;
            got = 1; break;
        }
    }
    fclose(f);
    return got ? 0 : -1;
}

/* Mean magnitude of the weight vector, Σ|theta[i]| / TOTAL — a single scalar for
 * tracking how large the learned weights are growing (overfitting often shows up
 * as a steadily climbing avgW). */
/* Per-interval, update-frequency-weighted mean |weight| (see w_abs_sum above).
 * Frozen phases contribute nothing automatically: they receive no updates, so
 * the old --phase special-case is unnecessary.  Resets the accumulators, so it
 * must be called exactly once per printed row. */
static double avg_abs_weight(void) {
    double v = w_update_count > 0 ? w_abs_sum / (double)w_update_count : 0;
    w_abs_sum = 0; w_update_count = 0;
    return v;
}

/* Per-interval mean phase of each rollout's FIRST policy pass (see
 * pass_phase_sum).  Bounded below by the uniform gate, since below it the
 * playout goes through g2_random_legal_move, which has no pass.  Resets the
 * accumulators, so it must be called exactly once per printed row. */
static double avg_first_pass_phase(void) {
    double v = pass_phase_count > 0 ? pass_phase_sum / (double)pass_phase_count : 0;
    pass_phase_sum = 0; pass_phase_count = 0;
    return v;
}

/* Weights that have actually been touched (non-zero).  The table is allocated
 * densely over the whole canonical pattern space, but most patterns never occur
 * at a legal non-eye move — only ~1.7k of the 6.8k cap-2 patterns ever fire — so
 * TOTAL badly overstates the model.  This is the size that matters for
 * data-per-parameter.  Same phase scope as avg_abs_weight. */
static int live_weights(void) {
    if (TOTAL <= 0) return 0;
    int n = 0;
    if (cfg_phase >= 0) {
        const int np = ppat_num_patterns;
        const int prev_base = ppat_phase_count * np;
        for (int i = 0; i < np; i++) if (theta[cfg_phase * np + i] != 0.0f) n++;
        for (int i = 0; i < 7; i++)  if (theta[prev_base + cfg_phase * 7 + i] != 0.0f) n++;
        return n;
    }
    for (int i = 0; i < TOTAL; i++) if (theta[i] != 0.0f) n++;
    return n;
}

/* ── directWR: current model vs a fixed reference, policy-vs-policy ─────────────
 * Both sides play by ppat policy sampling only — no search, no playouts — so
 * the column measures MOVE quality, which is NOT what SB optimises (it fits
 * rollout VALUE).  It is cheap enough to carry alongside teMSE (a few hundred
 * games in ~1s) and gives an independent read when teMSE stops discriminating.
 *
 * The reference must share this run's adjLib and phase count: the canonical
 * pattern table is a global (one active cap at a time), so a mismatched
 * reference cannot be evaluated without rebuilding it every move.  On mismatch
 * the column is disabled at startup rather than silently comparing nonsense. */
static float *ref_theta = NULL;        /* reference weights, NULL = column off */
static int    match_truncated;         /* set when a match stopped at MATCH_MAX_S */

/* The reference may be built at a different adjLib / phase count than the run.
 * The canon table is cached per cap (ppat.h), so a match just swaps the active
 * encoding between moves — the two models never need to agree. */
static int    ref_adj_lib, ref_phases;
static int    run_adj_lib, run_phases;

static void use_run_model(void) { ppat_init(run_adj_lib); ppat_phase_count = run_phases; }
static void use_ref_model(void) { ppat_init(ref_adj_lib); ppat_phase_count = ref_phases; }

/* Play `games` policy-vs-policy games, alternating colours, and return the
 * CURRENT model's win rate.  A fixed seed each call, so a change in the column
 * reflects a change in the weights rather than match luck. */
static float direct_match_wr(int games) {
    /* Both sides play REFERENCE moves outside [--match-phases A, B] and their
     * own weights inside, so the contest is confined to the band being
     * trained while openings and endgames stay symmetric and competent.
     * Out-of-band play reads only the fixed reference weights, so with the
     * match seed fixed every row replays the SAME sub-band openings and
     * consecutive rows pair.  Rows are comparable only across runs at the
     * same band (the historical uniform-below-0.6 rows are a different
     * universe). */
    const float saved_ubp = ppat_uniform_below_phase;
    ppat_uniform_below_phase = 0.0f;    /* banding replaces the uniform gate */
    /* Deployment-size board for the match; the training size's topology is
     * restored on exit for the other monitor instruments. */
    g2_init_topology(DEPLOY_BOARD_SIZE);
    static PpatState st;
    Rng rng;
    const double t0 = wall_now();
    int wins = 0, played = 0;
    for (int g = 0; g < games; g++) {
        if (wall_now() - t0 >= MATCH_MAX_S) { match_truncated = 1; break; }
        /* Reseed PER PAIR (g >> 1), not per game and not once per match.
         *
         * Per game, rather than once per match: below the uniform threshold the
         * play reads no weights, so a per-game seed makes every game's opening
         * identical from row to row.  With one stream for the whole match only
         * game 1 would pair — game 2 starts from the state game 1's endgame
         * left, and that endgame does read theta.
         *
         * Per PAIR, rather than per game: games 2i and 2i+1 then share one
         * opening and swap seats, so whatever advantage that opening carries is
         * handed to each model once and cancels.  Seeding per game alternates
         * seats across DIFFERENT openings, which leaves the asymmetry in as
         * variance. */
        rng_seed(&rng, 0x5eed1234L + (g >> 1));
        const int cur_is_black = (g & 1) == 0;   /* the pair's two colour assignments */
        Game2 game;
        g2_new(&game, DEPLOY_BOARD_SIZE);
        while (!game.game_over) {
            const int black_to_move = (game.current == BLACK);
            const float ph = (float)(game.cap - game.empty_count) / (float)game.cap;
            const float *w;
            if (ph < cfg_match_phase_lo || ph > cfg_match_phase_hi) {
                use_ref_model(); w = ref_theta;   /* out of band: reference moves, both sides */
            }
            else if (black_to_move == cur_is_black) { use_run_model(); w = theta; }
            else                                    { use_ref_model(); w = ref_theta; }
            /* A reference that carries its own pass plays it — that is its
             * policy and the match should measure it.  An UNFLAGGED reference
             * borrows the run's anchor and current pass weight instead, rather
             * than playing on to the bitter end: never stopping made the match
             * asymmetric in the one dimension being trained, since the
             * reference then collected every point the subject left behind and
             * filled shared eyes at 1.7x the subject's ratio (measured
             * 2026-09-10: 1.417% vs 0.826% of provably-alive groups).  Those
             * two biases run in opposite directions and neither is what the
             * column is meant to measure. */
            /* The reference never passes early, whatever its own file says;
             * RUN_EARLY_PASS is the subject's switch, not a global one. */
            const bool  ep = (w == theta) && RUN_EARLY_PASS;
            const float pw = (w == theta) ? run_pass_weight : 0.0f;
            g2_play(&game, ppat_policy_move(&game, &st, w, ep, pw, &rng));
        }
        if ((g2_estimate_winner(&game) == BLACK) == cur_is_black) wins++;
        played++;
    }
    use_run_model();
    ppat_uniform_below_phase = saved_ubp;
    g2_init_topology(topo_size);
    return played > 0 ? (float)wins / (float)played : 0.0f;
}

/* The match steps its game count up MATCH_GROWTH per printed row.  Rows are
 * already geometric (1.3 in the monitor, 1.5 inline), so a slower 1.1 keeps the
 * match a SHRINKING fraction of each row interval while the columns get
 * quieter exactly when the differences being judged get smaller.  The early
 * rows stay cheap, which is when they are closest together. */
static double match_scale = 1.0;

static int match_games(int base) { return (int)(base * match_scale + 0.5); }

/* One column: the value, with '*' on a new high (the counterpart of temse_col's
 * new-low star) and ' ' otherwise so the columns stay aligned. */
static int peak_col(double v, double *best, const char *fmt, char *buf, size_t n) {
    const int is_peak = v > *best;
    if (is_peak) *best = v;
    char tmp[32];
    snprintf(tmp, sizeof tmp, fmt, v);
    snprintf(buf, n, "%s%c", tmp, is_peak ? '*' : ' ');
    return is_peak;
}

/* Last row's directWR and whether it set a new high — what -best keys on.
 * directWR is the one indicator that cannot be gamed by anything except playing
 * better (no oracle, no estimator layer, out-of-sample by construction) and it
 * sided with live play every time another metric disagreed.  Its retired
 * companions measured the estimator/calibration axis, which saturates almost
 * immediately under band training: teMSE went flat for 5x more training while
 * directWR climbed, and mcWR spent a million positions inside a 0.4pp band
 * while directWR gained 5pp. */
static double match_score = 0;
static int    match_score_peak = 0;

/* Fill directWR for one row, then step the effort up for the next.  Single
 * entry point so the scale advances exactly once per row, whichever print path
 * produced it. */
static void match_cols(char *dw, size_t dwn) {
    static double best_d = -1e9;
    match_truncated = 0;
    if (!ref_theta) {
        snprintf(dw, dwn, "-");
        match_score = 0; match_score_peak = 0;
        return;
    }
    const double d = 100.0 * direct_match_wr(match_games(DIRECT_GAMES));
    match_score      = d;
    match_score_peak = peak_col(d, &best_d, "%.1f", dw, dwn);
    /* Once the match is being cut short at MATCH_MAX_S, raising the effort only
     * grows a game count that will never be reached — so stop growing. */
    if (!match_truncated) match_scale *= MATCH_GROWTH;
}




/* Optional-feature summary for the model banner line (empty when all default;
 * self-atari is always on, so it is not listed). */
static void banner_features(char *buf, size_t n) {
    buf[0] = 0;
    size_t o = 0;
    #define ADD(...) do { o += snprintf(buf + o, o < n ? n - o : 0, __VA_ARGS__); } while (0)
    if (ppat_twelvecell == 1) ADD("%stwelvecell", o ? ", " : "");
    if (ppat_twelvecell == 2) ADD("%stwelvecell2", o ? ", " : "");
    if (ppat_atari_n)   ADD("%satari %d", o ? ", " : "", ppat_atari_n);
    if (ppat_xa_n)      ADD("%satari-by-self-atari %d", o ? ", " : "", ppat_xa_n);
    if (ppat_capture_n) ADD("%scapture %d", o ? ", " : "", ppat_capture_n);
    if (ppat_cs_n)      ADD("%scapture-by-self-atari %d", o ? ", " : "", ppat_cs_n);
    #undef ADD
}

/* One banner for both modes, so a solo run and a parallel monitor read the
 * same.  Shared rows (data / model / match) are identical; each mode adds only
 * the rows it owns (train params + seed for solo, worker/checkpoint for the
 * monitor).  Label column is 8 wide. */
static void print_banner(bool monitor, const char *ckpt, const char *best) {
    char feats[256]; banner_features(feats, sizeof feats);

    if (cfg_test_file)
        printf("data      %s (%d train), %s (%d test)\n",
               cfg_file, n_train_total, cfg_test_file, n_test);
    else
        printf("data      %s  (%d train, %d test)\n",
               cfg_file, n_train_total, n_test);
    if (cfg_test_playouts_derived && n_test > 0)
        printf("          test-total-playouts %d over %d positions => %d per position\n",
               cfg_test_total_playouts, n_test, cfg_test_playouts);
    printf("model     adjLib %d%s%s%s\n",
           ppat_adj_lib,
           cfg_no_local ? ", no-local" : "",
           feats[0] ? " | " : "", feats);

    /* The train line describes the run regardless of who prints it, so the
     * monitor shows it too (it parsed the same args). */
    printf("train     lr %.3g, value-playouts %d, gradient-playouts %d, batch %d, %d phase(s)",
           (double)cfg_lr, cfg_value_po, cfg_gradient_po, cfg_batch, ppat_phase_count);
    if (cfg_phase >= 0)        printf(", phase %d only", cfg_phase);
    if (cfg_ema_window > 0)    printf(", ema %d", cfg_ema_window);
    if (cfg_overfit)           printf(", overfit");
    if (cfg_no_extreme > 0)    printf(", no-extreme %.1f", cfg_no_extreme);
    if (cfg_init_from_next)    printf(", init-scale %.3g", cfg_init_phase_scale);
    printf("\n");
    if (cfg_pc_buckets)
        printf("          phase-comp %d buckets (shrink %g, warmup %d)\n",
               cfg_pc_buckets, (double)PC_SHRINK, PC_WARMUP_POSITIONS);
    if (cfg_trunc_on)
        printf("          trunc vpat %s, delta %g%s, max-phase %g\n",
               cfg_trunc_vpat, (double)cfg_trunc_delta,
               cfg_trunc_delta_from_model ? " (model)" : "",
               (double)cfg_trunc_max_phase);

    /* The run line carries the only mode-specific facts: worker count, plus the
     * seed (solo, replayable) or a monitor tag (parallel). */
    if (monitor)
        printf("run       %d workers + monitor\n", cfg_workers);
    else if (cfg_workers > 1)
        printf("run       %d workers, sync-every %d, seed %d\n", cfg_workers, cfg_sync_every, cfg_seed);
    else
        printf("run       1 worker, seed %d\n", cfg_seed);

    char best_derived[320];
    if (!best) { best_path(ckpt, best_derived, sizeof best_derived); best = best_derived; }
    printf("out       %s  (best %s)\n", ckpt, best);
}

/* Dedicated monitor: repeatedly load the latest checkpoint and test it, printing
 * the metrics — without training or touching the sync barrier, so the training
 * workers never stall on the (expensive) test. */
static void run_monitor(void) {
    print_banner(true, cfg_monitor, NULL);
    printf("%9s  %7s  %7s", "positions", "trMSE", "trMSE_c");
    printf("  %6s  %7s  %6s", "nWts", "avgW", "pass1");
    if (ref_theta) printf("  %8s", "directWR");
    if (n_test > 0) printf("  %7s", "teMSE_c");
    if (n_md > 0) printf("  %7s", "mdMae");
    if (n_test > 0) printf("  %6s", "testM");
    printf("  %8s  %7s", "elapsedM", "pos/s");
    printf("\n");
    fflush(stdout);
    wall_start = wall_now();
    float mon_best_te_c = 1e30f; /* lowest teMSE_c seen, for the '*' new-low marker */
    float mon_best_mae = 1e30f;  /* lowest mae seen, for the mae '*' new-low marker */
    double mon_cumulative_test_s = 0; /* testM column: running total of eval+match cost in seconds (printed /60), like solo */

    /* Baseline row.  Fresh run: the uniform no-skill reference.  --load: the
     * loaded model's actual test + its weights (theta already holds the loaded
     * weights at this point), so the table starts from the real starting point. */
    {
        int loaded = (cfg_load != NULL);
        double bl_t0 = wall_now();
        TestResult tr = measure_test(loaded ? 0 : 1, n_test);
        char tecbuf[16];
        char maebuf[16];
        char dwbuf[16];
        match_cols(dwbuf, sizeof dwbuf);
        mon_cumulative_test_s += wall_now() - bl_t0;   /* baseline testM: eval + match cost */
        /* elapsed AFTER the match columns, as every later row does — otherwise
         * the baseline row under-reports its own cost by the match time. */
        double el = wall_now() - wall_start;
        char eb[32]; snprintf(eb, sizeof(eb), "%.1fm", el / 60.0);
        if (loaded) {
            printf("%9d  %7s  %7s", 0, "-", "-");
            printf("  %6d  %7s  %6s", live_weights(), "-", "-");
            if (ref_theta) printf("  %8s", dwbuf);
            if (n_test > 0) printf("  %7s", temse_col(tr.mse_c, &mon_best_te_c, tecbuf, sizeof tecbuf));
            if (n_md > 0) printf("  %7s", temse_col(tr.mae, &mon_best_mae, maebuf, sizeof maebuf));
            if (n_test > 0) printf("  %6.1f", mon_cumulative_test_s / 60.0);
            printf("  %8s  %7s", eb, "-");
            printf("\n");
        } else {
            printf("%9d  %7s  %7s", 0, "-", "-");
            printf("  %6s  %7s  %6s", "-", "-", "-");
            if (ref_theta) printf("  %8s", dwbuf);
            if (n_test > 0) printf("  %7s", temse_col(tr.mse_c, &mon_best_te_c, tecbuf, sizeof tecbuf));
            if (n_md > 0) printf("  %7s", temse_col(tr.mae, &mon_best_mae, maebuf, sizeof maebuf));
            if (n_test > 0) printf("  %6.1f", mon_cumulative_test_s / 60.0);
            printf("  %8s  %7s\n", eb, "-");
        }
        fflush(stdout);
    }

    double mon_last_full = -1;   /* latches the last full-epoch trMSE shown */
    double mon_last_full_c = -1; /* same, for trMSE_c */
    struct stat mon_last_st; memset(&mon_last_st, 0, sizeof mon_last_st);
    /* Geometric test cadence, same shape as single-process: after a row at
     * elapsed E the next test is due at E + clamp(0.5·E, last_test_s,
     * MAX_PRINT_CYCLE_S).  Workers save every --sync-every positions, so
     * without this the monitor would test back-to-back forever (one row per
     * test duration) and burn a core on redundant tests deep into a run. */
    double mon_next_test = 0;
    double mon_last_test_s = 0;
    /* Previous checkpoint's cumulative avgW accumulators, for per-interval means. */
    double mon_prev_wsum = 0; long mon_prev_wcnt = 0;
    double mon_prev_psum = 0; long mon_prev_pcnt = 0;

    /* Per-interval avgW from two consecutive checkpoints (0 when nothing new). */
    #define MON_AVGW(path) ({ \
        double _ws; long _wc; double _v = 0; \
        if (ckpt_wsum((path), &_ws, &_wc) == 0) { \
            long _dc = _wc - mon_prev_wcnt; \
            if (_dc > 0) _v = (_ws - mon_prev_wsum) / (double)_dc; \
            mon_prev_wsum = _ws; mon_prev_wcnt = _wc; \
        } \
        _v; })

    /* Per-interval mean first-pass phase, same differencing. */
    #define MON_PASS1(path) ({ \
        double _ps; long _pc; double _v = 0; \
        if (ckpt_passphase((path), &_ps, &_pc) == 0) { \
            long _dc = _pc - mon_prev_pcnt; \
            if (_dc > 0) _v = (_ps - mon_prev_psum) / (double)_dc; \
            mon_prev_psum = _ps; mon_prev_pcnt = _pc; \
        } \
        _v; })

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
        /* Take the checkpoint's pass weight too: the monitor is a separate
         * process with its own run_pass_weight, so without this it would score
         * every row as though the model passed at logit 0 — the most eager
         * setting there is — regardless of what the worker trained it to. */
        float *w = ppat_load_weights(cfg_monitor, NULL, &run_pass_weight);
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
        char dwbuf[16];
        match_cols(dwbuf, sizeof dwbuf);
        mon_last_test_s = wall_now() - test_t0;
        mon_cumulative_test_s += mon_last_test_s;

        /* Read the position count AFTER the test so positions and wall are both
         * current — otherwise pos/s is understated by the (long) test duration. */
        long my_pos = ckpt_positions(cfg_monitor);
        long agg = my_pos < 0 ? 0 : (long)cfg_workers * my_pos;
        double el = wall_now() - wall_start;
        double posps = el > 0 ? agg / el : 0;
        char eb[32]; snprintf(eb, sizeof(eb), "%.1fm", el / 60.0);
        /* Train MSE = last completed epoch over the (fixed) training set, aggregated
         * across workers — read straight from the checkpoint. */
        double dsum, psum, fsum; long dcnt, pcnt; char trbuf[24], trcbuf[24], tecbuf[16], maebuf[16];
        if (ckpt_train_sq(cfg_monitor, &dsum, &dcnt, &psum, &pcnt, &fsum) == 0) {
            trmse_col(dsum, dcnt, psum, pcnt, &mon_last_full, trbuf, sizeof trbuf);
            /* trMSE_c: floor-corrected, same completed-epoch blend as trMSE. */
            trmse_col(dsum - fsum, dcnt, psum, pcnt, &mon_last_full_c, trcbuf, sizeof trcbuf);
        } else { trbuf[0] = '-'; trbuf[1] = 0; trcbuf[0] = '-'; trcbuf[1] = 0; }
        int is_best = (n_test > 0) ? (tr.mse_c < mon_best_te_c)
                                   : (ref_theta ? match_score_peak : 0);
        printf("%9ld  %7s  %7s", agg, trbuf, trcbuf);
        printf("  %6d  %7.4f  %6.3f", live_weights(), MON_AVGW(cfg_monitor),
               MON_PASS1(cfg_monitor));
        if (ref_theta) printf("  %8s", dwbuf);
        if (n_test > 0) printf("  %7s", temse_col(tr.mse_c, &mon_best_te_c, tecbuf, sizeof tecbuf));
        if (n_md > 0) printf("  %7s", temse_col(tr.mae, &mon_best_mae, maebuf, sizeof maebuf));
        if (n_test > 0) printf("  %6.1f", mon_cumulative_test_s / 60.0);   /* testM: cumulative teMSE-eval + match cost (minutes) */
        printf("  %8s  %7.1f", eb, posps);
        printf("\n");
        if (is_best) {
            char bc[256];
            if (ref_theta) snprintf(bc, sizeof bc, "Best by directWR: %.2f, positions: %ld", match_score, agg);
            else           snprintf(bc, sizeof bc, "Best by teMSE_c: %.6f, positions: %ld", tr.mse_c, agg);
            save_best(cfg_monitor, bc);
        }
        fflush(stdout);

        /* Schedule the next test geometrically from THIS test's start, so the
         * gap between printed rows is ~MON_GROWTH regardless of test cost.  The
         * floor is still one test duration, so back-to-back testing is the worst
         * case.  1.3 rather than the inline path's 1.5: the monitor is a dedicated
         * process, and denser sampling matters because -best is only ever recorded
         * at a test point — too sparse a schedule walks past better models without
         * seeing them. */
        double cycle = (MON_GROWTH - 1.0) * (row_el > 0 ? row_el : 1.0);
        if (cycle > MAX_PRINT_CYCLE_S) cycle = MAX_PRINT_CYCLE_S;
        if (cycle < mon_last_test_s) cycle = mon_last_test_s;
        mon_next_test = row_el + cycle;
        usleep(200000);   /* small floor so tiny test sets don't spin */
    }
}

/* ── Print stats ───────────────────────────────────────────────────────────── */

static void print_stats(int iterations, int total_positions, int use_uniform,
                        int test_cap, int run_tests) {
    /* Everything below reads the global theta — the test rollouts, both match
     * columns, live_weights(), and the printed feature vector.  Point it at the
     * Polyak average for the duration so the row describes the model that
     * save_weights writes, not the raw iterate that training continues from. */
    float *theta_raw = theta;
    const float pass_raw = run_pass_weight;
    if (theta_ema) { theta = theta_ema; run_pass_weight = run_pass_weight_ema; }
    int test_n = (test_cap > 0 && test_cap < n_test) ? test_cap : n_test;
    if (test_n < 0) test_n = 0;   /* NOT clamped up to 1: with --test-pos 0 nothing
                                   * is tested, and the column must say so */
    /* run_tests == 0: emit a row from the free columns only.  The expensive ones
     * print "-" rather than being omitted, so the layout is identical across the
     * whole run and the switch-on point is visible. */
    clock_t test_t0 = clock();
    TestResult tr = (TestResult){0, 0, 0, 0};
    char dwbuf[16] = "-";
    if (run_tests) {
        tr = measure_test(use_uniform, test_n);
        /* Inside the test window: the match is part of the per-row cost, so testM
         * and the switch-on threshold both account for it. */
        match_cols(dwbuf, sizeof dwbuf);
        last_print_test_s = (double)(clock() - test_t0) / CLOCKS_PER_SEC;
        cumulative_test_s += last_print_test_s;
    }
    float mse_c = tr.mse_c;
    double elapsed_s = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    char elapsed_buf[32];
    snprintf(elapsed_buf, sizeof(elapsed_buf), "%.1fm", elapsed_s / 60.0);

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
    static double last_full_c = -1;
    static float best_te_c = 1e30f;
    static float best_mae = 1e30f;
    char trbuf[24], trcbuf[24], tecbuf[16], maebuf[16];
    /* A test set decides -best by teMSE_c (floor-corrected, a held-out
     * yardstick); without one, fall back to the directWR peak.  With neither,
     * write no -best at all. */
    int is_best = run_tests && (n_test > 0 ? (mse_c < best_te_c)
                                           : (ref_theta ? match_score_peak : 0));
    trmse_col(done_sq_sum, done_sq_count, epoch_sq_sum, epoch_sq_count, &last_full, trbuf, sizeof trbuf);
    /* trMSE_c: floor-corrected, same completed-epoch blend as trMSE (solo only). */
    trmse_col(done_sq_sum - done_floor_sum, done_sq_count, epoch_sq_sum, epoch_sq_count,
              &last_full_c, trcbuf, sizeof trcbuf);
    printf("%9ld  %7s  %7s", (long)cfg_workers * total_positions, trbuf, trcbuf);
    printf("  %6d  %7.4f  %6.3f", live_weights(), avg_abs_weight(),
           avg_first_pass_phase());
    if (ref_theta) printf("  %8s", dwbuf);
    if (n_test > 0) printf("  %7s", run_tests ? temse_col(mse_c, &best_te_c, tecbuf, sizeof tecbuf) : "-");
    if (n_md > 0) printf("  %7s", run_tests ? temse_col(tr.mae, &best_mae, maebuf, sizeof maebuf) : "-");
    if (n_test > 0) printf("  %6.1f", cumulative_test_s / 60.0);
    printf("  %6.1f  %8s  %6.1f  %7.1f",
           cumulative_sync_s, elapsed_buf, pos_ms, pos_per_s);
    printf("\n");
    fflush(stdout);

    theta = theta_raw;                 /* training resumes on the raw iterate */
    run_pass_weight = pass_raw;        /* and on the raw pass weight */
    save_weights(iterations, total_positions, elapsed_buf);
    /* Without a test set teMSE is identically 0, so "a new minimum" fires once on
     * the baseline row and never again — -best would be frozen at the UNTRAINED
     * weights for the whole run.  Write no -best at all rather than a misleading
     * one. */
    if (is_best) {
        char bc[256];
        if (ref_theta) snprintf(bc, sizeof bc, "Best by directWR: %.2f, positions: %d", match_score, total_positions);
        else           snprintf(bc, sizeof bc, "Best by teMSE_c: %.6f, positions: %d", mse_c, total_positions);
        save_best(weights_file, bc);
    }
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

static void print_help(FILE *out, const char *prog) {
    fprintf(out,
"train_ppat — Simulation Balancing trainer for the ppat playout policy\n"
"(Huang, Coulom, Lin 2010, Algorithm 1).  C port of train-ppat.js.\n"
"\n"
"Usage: %s <file> [options]\n"
"\n"
"<file>  training corpus, one position per line, either format (discriminated\n"
"        per line — a move list starts with a coordinate letter, a phase with a\n"
"        digit):\n"
"          \"<size> <move1,move2,...> <value> [best_move]\"   (gen_evals)\n"
"          \"<size> <phase> <move1,move2,...> <winRatio>\"     (gen-agent-evals)\n"
"        Values in [0,1] map to [-1,1].  One board size per file.\n"
"\n", prog);
    fputs(
"Playouts\n"
"  --lr F                     learning rate (default 10)\n"
"  --playouts N               default for --value/--gradient-playouts (default 100)\n"
"  --value-playouts N         rollouts for the V estimate (default: --playouts)\n"
"  --gradient-playouts N      rollouts for the gradient (default: --playouts)\n"
"  --batch N                  positions per weight update (default 1)\n"
"  --no-extreme F             drop TRAIN positions with |value| > 1-2F (default 0 = keep all)\n"
"\n"
"Truncated rollouts (affordable early-band training)\n"
"  --trunc-vpat PATH          after ceil(delta*area) moves, if phase there is <=\n"
"                             the gate B the rollout stops and z becomes this vpat\n"
"                             evaluator's value; a cut past B runs full.  Applies\n"
"                             to train AND test rollouts.\n"
"  --trunc-delta F            cut distance in phase units (default: the model's baked delta)\n"
"  --trunc-max-phase F        the gate B (default 0.55)\n"
"\n"
"Pattern features (all APPENDED blocks; --load of a model without them fine-tunes)\n"
"  --adj-lib N                orthogonal liberty cap in the 3x3 pattern, 2..4\n"
"                             (default 2 = atari-only encoding).  Ignored with --load.\n"
"  --twelvecell               2nd key on an all-empty ninecell (dist-2 orthogonals)\n"
"  --twelvecell2              same key, looser trigger (adjacent points empty);\n"
"                             mutually exclusive with --twelvecell\n"
"  --atari N                  graded gives-atari feature, one-hot on chain size (default 0 = off)\n"
"  --capture N                graded capture-size feature (default 0 = off)\n"
"  --atari-by-self-atari N    mutual-atari interaction grid, NxN (default 0 = off)\n"
"  --capture-by-self-atari N  ko-take / snapback interaction grid (default 0 = off)\n"
"  --no-local                 freeze the 7 previous-move local features at 0 (ablation)\n"
"\n"
"Phases\n"
"  --phases N                 phase-conditioned weight slices (default 1)\n"
"  --phase P                  train only phase P; freeze the rest (test set stays unfiltered)\n"
"  --init-phase-scale F       seed phase P from phase P+1 scaled by F (endgame-first\n"
"                             warm-start; requires --phase)\n"
"\n"
"Test set / teMSE (off by default; the match columns are the primary readout)\n"
"  --test-pos N               test positions from the head of <file> (default 0 = no teMSE)\n"
"  --test-file PATH           take the test set from PATH; all of <file> is then training\n"
"  --test-playouts N          playouts per test position (default: from --test-total-playouts)\n"
"  --test-total-playouts N    spread this TOTAL over the test set (default 200000; 0 disables)\n"
"  --train-pos N              cap training positions (default 0 = all)\n"
"  --overfit                  use the same data for train and test\n"
"\n"
"mdMae (mc-ppat move-selection regret vs a labelled movedetails file, on test rows)\n"
"  --md-file PATH             movedetails .md file (default movedetails_5059.md; '' disables)\n"
"  --mae-band LO,HI           keep only positions with phase in [LO,HI] (default 0.6,1.0)\n"
"  --mae-cand-playouts N      playouts per candidate move (default 10)\n"
"  --mae-pos N                cap kept positions (default 0 = all)\n"
"  --uniform-below-phase F    playouts play uniform-random below fullness F (default 0.6)\n"
"\n"
"directWR match (move-quality readout: current model vs a fixed reference)\n"
"  --ref-weights PATH|none    reference model for the directWR/WR columns (default\n"
"                             out/ppat-data-233162-best-ref-candidate.js; \"none\" off;\n"
"                             must share this run's adjLib and phase count)\n"
"  --no-direct                disable the directWR match and hide the column\n"
"  --match-phases A,B         in-band each side plays its own weights, out-of-band\n"
"                             both play the reference (default 0.6,1)\n"
"\n"
"Gradient shaping\n"
"  --phase-compensation-buckets N\n"
"                             make applied gradient pressure uniform by phase\n"
"                             (default 0 = off; else 2..max)\n"
"\n"
"Parallelism (barrier-synced multi-worker; a monitor handles testing)\n"
"  --workers N                worker count (default 1)\n"
"  --worker-id N              this worker's id, 0-based (default 0)\n"
"  --sync-every N             positions between sync barriers (default 30)\n"
"  --sync-dir PATH            shared sync directory (default out/ppat-sync)\n"
"  --monitor PATH             run as a test-only monitor of the checkpoint at PATH\n"
"  --ema-window N             Polyak averaging window in aggregate positions\n"
"                             (default 2000; 0 = off, save the raw iterate)\n"
"\n"
"Checkpoints / reproducibility\n"
"  --load PATH                initial weights (fine-tune from an existing model)\n"
"  --save PATH                checkpoint path (default: a random out/ name)\n"
"  --seed N                   RNG seed; makes a SINGLE-worker run reproducible (default 0)\n"
"  --test-from N              pin the position where the expensive columns switch on (default 0)\n"
"\n"
"Control\n"
"  --iteration-limit N        stop after N iterations (default 0 = infinite)\n"
"  --baseline-only            print the uniform-policy baseline row, then exit\n"
"  -h, --help                 show this help and exit\n"
"\n"
"EARLY PASS is always on: PASS is offered as a candidate at logit 0 and every\n"
"checkpoint is stamped earlyPass, pinning the otherwise-free additive constant on\n"
"the pattern weights.  There is no flag for it.\n", out);
}

int main(int argc, char **argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_help(stdout, argv[0]);
        return 0;
    }
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file> [options]   (run with --help for the full list)\n", argv[0]);
        return 1;
    }

    ppat_load_quiet = 1;   /* the banner reports what loaded; the per-file notices are noise */

    cfg_file         = argv[1];
    cfg_lr           = get_float_arg(argc, argv, "--lr", 10.0f);
    int playouts     = get_int_arg(argc, argv, "--playouts", 100);  /* default for M, N */
    cfg_value_po     = get_int_arg(argc, argv, "--value-playouts", playouts);
    cfg_gradient_po  = get_int_arg(argc, argv, "--gradient-playouts", playouts);
    cfg_batch        = get_int_arg(argc, argv, "--batch", 1);
    cfg_test_pos     = get_int_arg(argc, argv, "--test-pos", 0);
    cfg_train_pos    = get_int_arg(argc, argv, "--train-pos", 0);
    cfg_test_playouts = get_int_arg(argc, argv, "--test-playouts", 0);
    cfg_test_total_playouts = get_int_arg(argc, argv, "--test-total-playouts", 200000);
    cfg_no_extreme       = get_float_arg(argc, argv, "--no-extreme", 0.0f);
    cfg_iter_limit   = get_int_arg(argc, argv, "--iteration-limit", 0);
    cfg_overfit      = has_flag(argc, argv, "--overfit");
    int baseline_only = has_flag(argc, argv, "--baseline-only");  /* print uniform baseline, then exit */
    ppat_phase_count = get_int_arg(argc, argv, "--phases", 1);
    cfg_test_file = get_str_arg(argc, argv, "--test-file", NULL);
    cfg_test_pos_given = has_flag(argc, argv, "--test-pos");
    cfg_md_file = get_str_arg(argc, argv, "--md-file", cfg_md_file);
    cfg_mae_cand_playouts = get_int_arg(argc, argv, "--mae-cand-playouts", cfg_mae_cand_playouts);
    { const char *mb = get_str_arg(argc, argv, "--mae-band", NULL);
      if (mb) { float a, b; if (sscanf(mb, "%f,%f", &a, &b) == 2) { cfg_mae_band_lo = a; cfg_mae_band_hi = b; } } }
    cfg_mae_pos = get_int_arg(argc, argv, "--mae-pos", cfg_mae_pos);
    /* Deployment playout gate (mc-ppat's PPAT_MIN_PHASE): ppat_policy_move plays
     * uniform-random below this board fullness.  Default 0.6, matching the fielded
     * playout and the mc-ppat agent the mae column mirrors. */
    ppat_uniform_below_phase = get_float_arg(argc, argv, "--uniform-below-phase", 0.6f);
    cfg_no_local       = has_flag(argc, argv, "--no-local");
    cfg_ref_weights    = get_str_arg(argc, argv, "--ref-weights", "out/ppat-data-233162-best-ref-candidate.js");
    if (strcmp(cfg_ref_weights, "none") == 0) cfg_ref_weights = NULL;
    /* --no-direct: disable the directWR match entirely (and hide its column).
     * Forcing ref_theta off is enough — every directWR path is gated on it. */
    if (has_flag(argc, argv, "--no-direct")) cfg_ref_weights = NULL;
    cfg_load = get_str_arg(argc, argv, "--load", NULL);
    cfg_save = get_str_arg(argc, argv, "--save", NULL);
    cfg_monitor = get_str_arg(argc, argv, "--monitor", NULL);
    cfg_phase = get_int_arg(argc, argv, "--phase", -1);
    /* Presence of --init-phase-scale enables seeding phase P from phase P+1. */
    cfg_init_from_next = has_flag(argc, argv, "--init-phase-scale");
    cfg_init_phase_scale = get_float_arg(argc, argv, "--init-phase-scale", 1.0f);
    {
        const char *tv = get_str_arg(argc, argv, "--trunc-vpat", NULL);
        int delta_given     = has_flag(argc, argv, "--trunc-delta");
        cfg_trunc_delta     = get_float_arg(argc, argv, "--trunc-delta", 0.0f);
        cfg_trunc_max_phase = get_float_arg(argc, argv, "--trunc-max-phase", 0.55f);
        if (tv) {
            vpat_load(tv);              /* load first so the model's baked delta is available */
            cfg_trunc_vpat = tv;
            cfg_trunc_on = 1;

            /* Delta: an explicit --trunc-delta wins; otherwise default to the
             * model's baked delta (as the JS consumers do). */
            double bd;
            if (!delta_given && vpat_trunc(&bd)) {
                cfg_trunc_delta = (float)bd;
                cfg_trunc_delta_from_model = 1;
            }
            if (cfg_trunc_delta <= 0.0f) {
                fprintf(stderr, "error: --trunc-vpat requires --trunc-delta > 0 "
                                "(%s has no baked trunc.delta to default from)\n", tv);
                exit(1);
            }
        } else if (delta_given || has_flag(argc, argv, "--trunc-max-phase")) {
            fprintf(stderr, "error: --trunc-delta/--trunc-max-phase need --trunc-vpat\n");
            exit(1);
        }
    }
    {
        const char *mp = get_str_arg(argc, argv, "--match-phases", NULL);
        if (mp) {
            if (sscanf(mp, "%f,%f", &cfg_match_phase_lo, &cfg_match_phase_hi) != 2 ||
                !(cfg_match_phase_lo >= 0.0f && cfg_match_phase_lo < cfg_match_phase_hi &&
                  cfg_match_phase_hi <= 1.0f)) {
                fprintf(stderr, "error: --match-phases needs A,B with 0 <= A < B <= 1\n");
                exit(1);
            }
        }
    }
    cfg_pc_buckets = get_int_arg(argc, argv, "--phase-compensation-buckets", 0);
    if (cfg_pc_buckets != 0 && (cfg_pc_buckets < 2 || cfg_pc_buckets > PC_MAX_BUCKETS)) {
        fprintf(stderr, "error: --phase-compensation-buckets must be 0 (off) or 2..%d\n", PC_MAX_BUCKETS);
        exit(1);
    }
    cfg_workers    = get_int_arg(argc, argv, "--workers", 1);
    cfg_worker_id  = get_int_arg(argc, argv, "--worker-id", 0);
    /* 30, not 100: rounds are pure approximation error (each worker's later
     * gradients are evaluated away from the common theta0, and the barrier sums
     * K of those displacements at once), so shorter is strictly more correct and
     * cost is the only reason not to.  Measured under identical load at K=3:
     * 100 -> 36 pos/s, 30 -> 37, 25 -> 34, 20 -> 35, 5 -> 28.  Down to ~25 the
     * barrier is nearly free; it only bites below that. */
    cfg_sync_every = get_int_arg(argc, argv, "--sync-every", 30);
    /* Polyak averaging window in AGGREGATE POSITIONS (default 2000; 0 = off,
     * save the raw iterate).  A window is a real hyperparameter, and an untuned
     * one silently lags every column for its first window's worth of training;
     * 2000 is small (well under a report row).  ~30000 is where to start a sweep
     * if raising it. */
    cfg_ema_window = get_int_arg(argc, argv, "--ema-window", 2000);
    cfg_seed       = get_int_arg(argc, argv, "--seed", 0);
    /* Pin the position where the expensive columns switch on.  Without it the
     * switch is timing-derived (first cycle whose TRAINING time exceeds a test),
     * which is the right default but is not reproducible — two --seed runs can
     * start testing on different rows.  Set it to align rows across a sweep. */
    cfg_test_from  = (long)get_int_arg(argc, argv, "--test-from", 0);
    cfg_sync_dir   = get_str_arg(argc, argv, "--sync-dir", "out/ppat-sync");
    /* Barrier sync is only meaningful with peers to combine displacements with;
     * with a single worker there is no peer, but it must still SAVE on the same cadence
     * so the monitor sees fresh checkpoints (see save below). */
    int parallel = (cfg_workers > 1 && cfg_sync_every > 0);
    int wrapper_run = has_flag(argc, argv, "--sync-dir");
    if (parallel) mkdir(cfg_sync_dir, 0777);   /* idempotent; launcher should clear it first */

    /* Per-worker seed so workers explore independently before averaging.
     *
     * --seed makes a SINGLE-THREADED run bit-reproducible: the epoch shuffle, the
     * rollouts and the updates all draw from g_rng, and measure_test re-seeds to a
     * fixed constant and restores the stream, so nothing else perturbs it.  Two
     * runs with the same seed then follow identical trajectories — which is what
     * makes an A/B of anything that does NOT feed back into training (--ema-window,
     * the match columns) an exactly paired comparison rather than one swamped by
     * run-to-run noise.
     *
     * It does NOT make a parallel run reproducible: the barrier's poll loop draws
     * from this same stream, and how many times it spins depends on wall-clock
     * timing and machine load, so the training stream advances by a different
     * amount every run.  Warn rather than pretend. */
    if (cfg_seed != 0 && parallel)
        fprintf(stderr, "WARNING: --seed does not make a parallel run reproducible — the sync\n"
                        "         barrier's poll loop draws from the same RNG, so wall-clock\n"
                        "         timing perturbs the training stream.  Use --workers 1.\n");
    /* One derivation for both paths, so the seed reported below is exactly what
     * --seed needs to reproduce the run.  Masked to fit a positive int, since
     * --seed is parsed with atoi. */
    if (cfg_seed == 0) cfg_seed = (int)((uint32_t)time(NULL) & 0x7fffffff);
    rng_seed(&g_rng, (long)cfg_seed + (long)cfg_worker_id * 0x9e3779b9L);
    /* --load resolves the cap from the file (ppat_load_weights calls ppat_init);
     * a fresh run takes it from --adj-lib. */
    int cfg_adj_lib = get_int_arg(argc, argv, "--adj-lib", 2);   /* default: historical encoding */
    /* Twelvecell extension: a second key for the four distance-2 orthogonals
     * when the ninecell is all-empty.  Its block is APPENDED, so --load of a
     * model without it is a fine-tune: the old weights keep their indices and
     * the 21 new ones start at zero. */
    const int t12_1 = has_flag(argc, argv, "--twelvecell");
    const int t12_2 = has_flag(argc, argv, "--twelvecell2");
    if (t12_1 && t12_2) {
        fprintf(stderr, "error: --twelvecell and --twelvecell2 are mutually exclusive"
                        " (they share one weight block; a model uses one trigger or the other)\n");
        exit(1);
    }
    ppat_twelvecell = t12_2 ? 2 : t12_1 ? 1 : 0;
    ppat_atari_n = get_int_arg(argc, argv, "--atari", 0);
    if (ppat_atari_n < 0 || ppat_atari_n > PPAT_ATARI_MAX) {
        fprintf(stderr, "error: --atari must be 0..%d\n", PPAT_ATARI_MAX);
        exit(1);
    }
    ppat_xa_n = get_int_arg(argc, argv, "--atari-by-self-atari", 0);
    if (ppat_xa_n < 0 || ppat_xa_n > PPAT_XA_MAX) {
        fprintf(stderr, "error: --atari-by-self-atari must be 0..%d\n", PPAT_XA_MAX);
        exit(1);
    }
    ppat_capture_n = get_int_arg(argc, argv, "--capture", 0);
    if (ppat_capture_n < 0 || ppat_capture_n > PPAT_CAP_MAX) {
        fprintf(stderr, "error: --capture must be 0..%d\n", PPAT_CAP_MAX);
        exit(1);
    }
    ppat_cs_n = get_int_arg(argc, argv, "--capture-by-self-atari", 0);
    if (ppat_cs_n < 0 || ppat_cs_n > PPAT_CS_MAX) {
        fprintf(stderr, "error: --capture-by-self-atari must be 0..%d\n", PPAT_CS_MAX);
        exit(1);
    }
    check_unknown_args(argc, argv);
    ppat_init(cfg_adj_lib);

    load_positions();
    if (cfg_md_file && cfg_md_file[0]) n_md = load_md_file(cfg_md_file);
    split_data();

    TOTAL = ppat_total_weights();
    theta           = calloc(TOTAL, sizeof(float));
    if (cfg_load) {
        /* A loaded model's own pass weight carries over — fine-tuning continues
         * the controller rather than restarting it.  An unflagged file has none,
         * so conversion starts from 0. */
        float *loaded = ppat_load_weights(cfg_load, NULL, &run_pass_weight);
        if (!loaded) { fprintf(stderr, "failed to load weights\n"); exit(1); }
        TOTAL = ppat_total_weights();
        free(theta);
        theta = loaded;
    }


    /* Reference model for the directWR column.  Loaded AFTER the run's own weights,
     * because ppat_load_weights rebuilds the global canon table for the file's
     * adjLib — so we capture the run's cap/phases first, load the reference,
     * then verify nothing moved.  A mismatch disables the column loudly rather
     * than comparing models built on different tables. */
    run_adj_lib = ppat_adj_lib;
    run_phases  = ppat_phase_count;
    if (cfg_ref_weights) {
        const int run_total = TOTAL;
        ref_theta = ppat_load_weights(cfg_ref_weights, &ref_early_pass, &ref_pass_weight);
        if (!ref_theta) {
            fprintf(stderr, "WARNING: --ref-weights %s could not be loaded"
                            " — directWR and WR columns disabled\n", cfg_ref_weights);
        } else {
            ref_adj_lib = ppat_adj_lib;
            ref_phases  = ppat_phase_count;
        }
        use_run_model();                        /* put the run's encoding back */
        TOTAL = run_total;
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
    else {
        /* Its own stream: drawing from g_rng here would mean that passing --save
         * or not changed the training trajectory under a fixed --seed. */
        Rng nrng; rng_seed_entropy(&nrng);
        snprintf(weights_file, sizeof(weights_file), "out/ppat-data-%08x.js", rng_next(&nrng));
    }

    if (cfg_worker_id == 0) {
    char best_file[320]; best_path(weights_file, best_file, sizeof best_file);
    print_banner(false, weights_file, best_file);
    printf("%9s  %7s  %7s", "positions", "trMSE", "trMSE_c");
    printf("  %6s  %7s  %6s", "nWts", "avgW", "pass1");
    if (ref_theta) printf("  %8s", "directWR");
    if (n_test > 0) printf("  %7s", "teMSE_c");
    if (n_md > 0) printf("  %7s", "mdMae");
    if (n_test > 0) printf("  %6s", "testM");
    printf("  %6s  %8s  %6s  %7s", "syncS", "elapsedM", "posMs", "pos/s");
    printf("\n");
    }

    /* θ₀ for the first sync round: every worker starts here, so displacements are
     * measured against it. */
    if (parallel) capture_round_start();
    ema_init();                        /* after --load / warm-start: seed from the real start */

    start_time = clock();
    wall_start = wall_now();
    int total_positions = 0;
    int iterations = 0;
    next_print_pos = PRINT_POS_FIRST;
    last_print_test_s = 0;
    int testing_started = 0;
    double cycle_t0 = wall_now();

    /* Inline testing/printing only in single-process mode.  In parallel, testing is
     * done by a separate --monitor process so the training workers never stall on the
     * barrier; worker 0 just saves the combined checkpoint after each sync. */
    /* Who owns testing?  A monitor does it whenever one exists, and one exists
     * exactly when we were launched by train-ppat-parallel — which always passes
     * --sync-dir.  Keying this off the worker count instead used to make
     * --workers 1 a special case: the trainer believed it was standalone and ran
     * its own (expensive, silenced) test pass while the monitor tested the same
     * checkpoints, duplicating the work. */
    const int do_inline = !has_flag(argc, argv, "--sync-dir");

    /* baseline: fresh run shows the uniform no-skill reference; --load shows the
     * loaded model's actual test + weights.  --baseline-only stops here. */
    if (do_inline || baseline_only) {
        /* The baseline always tests: it is where the cost of a test is measured. */
        print_stats(iterations, total_positions, cfg_load ? 0 : 1, n_test, 1);
        cycle_t0 = wall_now();
        if (baseline_only) return 0;
    }

    for (;;) {
        for (int li = 0; li < n_train; li++) {
            if (do_inline && total_positions >= next_print_pos) {
                /* Switch the expensive columns on once a cycle's training time
                 * exceeds what a test costs — i.e. once testing is at most half
                 * the wall clock.  --test-from pins that point by position
                 * instead, which is what makes a --seed run's rows reproducible.
                 * Once on, it stays on. */
                const double cycle_train_s = wall_now() - cycle_t0;
                int run_tests = cfg_test_from > 0
                              ? (long)total_positions >= cfg_test_from
                              : (testing_started || cycle_train_s > last_print_test_s);
                if (run_tests) testing_started = 1;
                /* Always test the FULL --test-pos set: rows (and the best-* star)
                 * stay statistically comparable. */
                print_stats(iterations, total_positions, 0, 0, run_tests);
                next_print_pos = (long)(next_print_pos * PRINT_POS_GROWTH) + 1;
                cycle_t0 = wall_now();
            }

            /* Standalone runs have no barrier, so sample on a fixed position
             * cadence instead; the weighting makes the window identical either way. */
            if (!wrapper_run && total_positions > 0 && total_positions % EMA_SAMPLE_POS == 0)
                ema_update(total_positions);

            Position *pos = &all_positions[train_idx[li]];
            Game2 g;
            int bad = -1;
            int rp = replay_position(pos, &g, &bad);
            if (rp < 0) { fprintf(stderr, "WARNING: illegal move #%d (idx %d) in training position, skipping\n", bad, pos->history[bad]); continue; }
            if (rp == 0) continue;
            update_theta(&g, pos->value);
            total_positions++;

            /* Under the wrapper, checkpoint on the sync cadence regardless of
             * worker count — the monitor is the only thing that reads it, and it
             * needs fresh weights.  With >1 worker this is also the parameter-sync
             * barrier; with 1 there is no peer to combine with, so just save. */
            if (wrapper_run && total_positions % cfg_sync_every == 0) {
                flush_batch();
                if (parallel) barrier_sync_average();   /* θ ← θ₀ + Σ displacements */
                /* After the barrier theta is the consensus, so this is the only
                 * point in a parallel run where sampling it is meaningful. */
                ema_update((long)cfg_workers * total_positions);
                if (cfg_worker_id == 0) save_weights(iterations, total_positions, "");
            }
        }

        /* Epoch boundary: latch this epoch's training-fit over the full fixed set
         * (published in the next sync's .stat / used by the solo column). */
        done_sq_sum = epoch_sq_sum; done_sq_count = epoch_sq_count;
        done_floor_sum = epoch_floor_sum;
        epoch_sq_sum = 0; epoch_sq_count = 0; epoch_floor_sum = 0;

        iterations++;
        shuffle_train();

        if (cfg_iter_limit > 0 && iterations >= cfg_iter_limit) {
            if (do_inline) print_stats(iterations, total_positions, 0, 0, 1);
            break;
        }
    }

    free(theta);
    free(rollout_grad_buf);
    free(g_buf);
    free(batch_buf);
    return 0;
}
