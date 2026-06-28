/*
 * gen_evals.c — Play full games with RAVE, sample random positions,
 * re-evaluate with more playouts, and emit training data.
 *
 * Output format (one line per sample):
 *   <size> <move1,move2,...> <winRatio>
 *
 * Compile: see build.sh (game2.c rave.c gen_evals.c).
 *
 * Usage:
 *   ./gen_evals.bin <size> <playouts> <eval-playouts>
 * All three are required (no silent defaults — the params end up encoded in the
 * output filename, so they must be chosen explicitly).
 */
#include "game2.h"
#include "rave.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>   /* getpid */

/* Convert flat index to coordinate string (e.g. 10 on 9x9 → "b2") */
static int board_size;

static void coord_str(int32_t move, char *buf) {
    if (move == PASS) { buf[0]='p'; buf[1]='a'; buf[2]='s'; buf[3]='s'; buf[4]=0; return; }
    int x = move % board_size;
    int y = move / board_size;
    buf[0] = 'a' + x;
    int n = y + 1;
    if (n >= 10) { buf[1] = '0' + n/10; buf[2] = '0' + n%10; buf[3] = 0; }
    else         { buf[1] = '0' + n; buf[2] = 0; }
}

/* One uniform-random rollout from g to game end; returns the estimated winner. */
static int8_t uniform_playout(const Game2 *g, Rng *rng) {
    Game2 sim; g2_clone(&sim, g);
    int move_limit = 3 * sim.empty_count + 20, moves = 0;
    while (!sim.game_over && moves < move_limit) {
        g2_play(&sim, g2_random_legal_move(&sim, rng));
        moves++;
    }
    return g2_estimate_winner(&sim);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --eval-playouts <n> [--size <n>] [--pos-playouts <n>] [--pos-seed <n>] [--limit <n>] [--uniform-eval]\n"
        "  --size           board size (default: 9)\n"
        "  --eval-playouts  playouts for evaluation of the position (RAVE, or uniform rollouts with --uniform-eval)\n"
        "  --pos-playouts   RAVE playouts used to generate the position (default: eval-playouts/10)\n"
        "  --pos-seed       RNG seed for position generation (default: non-deterministic)\n"
        "  --limit          number of positions to evaluate, then stop (default: 0 = unlimited)\n"
        "  --uniform-eval   label each position with the mean of N pure uniform rollouts (no tree/UCB)\n"
        "Emits one line per sample: <size> <move1,move2,...> <winRatio> <bestMove>\n",
        prog);
}

int main(int argc, char **argv) {
    int size_arg = 9, pos_playouts = -1, eval_playouts = -1, limit = 0;
    int have_pos_seed = 0, uniform_eval = 0;
    unsigned long pos_seed_arg = 0;
    for (int i = 1; i < argc; i++) {
        if      (i + 1 < argc && !strcmp(argv[i], "--size"))          size_arg      = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--pos-playouts"))  pos_playouts  = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--eval-playouts")) eval_playouts = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--limit"))         limit         = atoi(argv[++i]);
        else if (i + 1 < argc && !strcmp(argv[i], "--pos-seed"))    { pos_seed_arg  = strtoul(argv[++i], NULL, 10); have_pos_seed = 1; }
        else if (!strcmp(argv[i], "--uniform-eval"))                  uniform_eval  = 1;
        else { usage(argv[0]); return 1; }   /* unknown / missing value / --help */
    }
    if (size_arg < 1 || eval_playouts < 1) {
        usage(argv[0]);
        return 1;
    }
    if (pos_playouts < 1) {                 /* default: a tenth of the eval budget */
        pos_playouts = eval_playouts / 10;
        if (pos_playouts < 1) pos_playouts = 1;
    }
    board_size = size_arg;
    uint32_t pos_seed = have_pos_seed ? (uint32_t)pos_seed_arg : (uint32_t)time(NULL);

    /* Two independent streams.  pos_rng drives game-play + sampling and is fully
     * determined by pos-seed, so the position SET is reproducible.  eval_rng
     * drives the re-evaluation and is ALWAYS non-deterministic (time+pid, never
     * tied to pos-seed) — so relabeling the same positions yields independent
     * label samples instead of identical ones. */
    Rng pos_rng, eval_rng;
    rng_seed(&pos_rng, pos_seed);
    rng_seed_entropy(&eval_rng);
    g2_init_topology(board_size);

    RaveState *s = rave_create();

    int32_t moves[4 * MAX_CAP];
    char cbuf[8];
    int emitted = 0;

    for (;;) {
        /* Play a full game recording every move */
        Game2 g;
        g2_new(&g, board_size);
        int nmoves = 0;
        while (!g.game_over) {
            RaveResult r = rave_search(s, &g, pos_playouts, 0, &pos_rng);
            moves[nmoves++] = r.move;
            g2_play(&g, r.move);
        }

        /* Find the index of the first pass */
        int first_pass = -1;
        for (int i = 0; i < nmoves; i++) {
            if (moves[i] == PASS) { first_pass = i; break; }
        }

        /* Valid sample range: at least 5 moves in, at least 10 before first pass */
        int lo = 5;
        int hi = (first_pass == -1 ? nmoves : first_pass) - 10;
        if (hi < lo) continue;

        int pos = lo + (int)rng_below(&pos_rng, hi - lo + 1);

        /* Replay the game up to that position */
        Game2 replay;
        g2_new(&replay, board_size);
        for (int i = 0; i < pos; i++) g2_play(&replay, moves[i]);

        /* Re-evaluate with the eval budget.  --uniform-eval: label = mean of
         * eval-playouts pure uniform rollouts (no tree/UCB), P(side-to-move wins);
         * else: RAVE search win-ratio. */
        float win_ratio;
        int32_t best = PASS;
        if (uniform_eval) {
            int8_t player = replay.current;
            int wins = 0;
            for (int i = 0; i < eval_playouts; i++)
                if (uniform_playout(&replay, &eval_rng) == player) wins++;
            win_ratio = (float)wins / eval_playouts;
        } else {
            RaveResult r = rave_search(s, &replay, eval_playouts, 0, &eval_rng);
            win_ratio = r.win_ratio;
            best = r.move;
        }

        /* Emit: size move1,move2,... winRatio bestMove */
        fprintf(stdout, "%d ", board_size);
        for (int i = 0; i < pos; i++) {
            if (i > 0) fputc(',', stdout);
            coord_str(moves[i], cbuf);
            fputs(cbuf, stdout);
        }
        char best_buf[8];
        coord_str(best, best_buf);
        fprintf(stdout, " %.15g %s\n", (double)win_ratio, best_buf);
        fflush(stdout);

        if (limit > 0 && ++emitted >= limit) break;
    }

    rave_destroy(s);
    return 0;
}
