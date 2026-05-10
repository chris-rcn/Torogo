/*
 * gen_evals.c — Play full games with RAVE, sample random positions,
 * re-evaluate with more playouts, and emit training data.
 *
 * Output format (one line per sample):
 *   <size> <move1,move2,...> <winRatio>
 *
 * Compile:
 *   cc -O2 -o gen_evals game2.c rave.c gen_evals.c -lm
 *
 * Usage:
 *   ./gen_evals [playouts] [eval-playouts]
 *   Default: playouts=100, eval-playouts=1000
 */
#include "game2.h"
#include "rave.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

int main(int argc, char **argv) {
    board_size        = argc > 1 ? atoi(argv[1]) : 9;
    int playouts      = argc > 2 ? atoi(argv[2]) : 100;
    int eval_playouts = argc > 3 ? atoi(argv[3]) : 1000;

    g2_seed((uint32_t)time(NULL));
    g2_init_topology(board_size);

    RaveState *s = rave_create();

    int32_t moves[4 * MAX_CAP];
    char cbuf[8];

    for (;;) {
        /* Play a full game recording every move */
        Game2 g;
        g2_new(&g, board_size);
        int nmoves = 0;
        while (!g.game_over) {
            RaveResult r = rave_search(s, &g, playouts, 0);
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

        int pos = lo + (int)g2_rand_n(hi - lo + 1);

        /* Replay the game up to that position */
        Game2 replay;
        g2_new(&replay, board_size);
        for (int i = 0; i < pos; i++) g2_play(&replay, moves[i]);

        /* Re-evaluate with the eval budget */
        RaveResult r = rave_search(s, &replay, eval_playouts, 0);
        float win_ratio = r.win_ratio;

        /* Emit: size move1,move2,... winRatio bestMove */
        fprintf(stdout, "%d ", board_size);
        for (int i = 0; i < pos; i++) {
            if (i > 0) fputc(',', stdout);
            coord_str(moves[i], cbuf);
            fputs(cbuf, stdout);
        }
        char best_buf[8];
        coord_str(r.move, best_buf);
        fprintf(stdout, " %.15g %s\n", (double)win_ratio, best_buf);
        fflush(stdout);
    }

    rave_destroy(s);
    return 0;
}
