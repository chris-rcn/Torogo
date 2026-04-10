/*
 * selfplay.c — Play RAVE vs RAVE at different playout counts.
 * Alternates colors each game. Prints summary on exponential time intervals.
 *
 * Compile: cc -O2 -o selfplay game2.c rave.c selfplay.c -lm
 * Usage:   ./selfplay [playouts_a] [playouts_b]
 *          Default: 1000 vs 2000
 */
#include "game2.h"
#include "rave.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int po_a = argc > 1 ? atoi(argv[1]) : 1000;
    int po_b = argc > 2 ? atoi(argv[2]) : 2000;

    g2_seed((uint32_t)time(NULL));
    g2_init_topology();

    RaveState *sa = rave_create();
    RaveState *sb = rave_create();

    int wins_a = 0, wins_b = 0, draws = 0, games = 0;
    clock_t t0 = clock();
    double next_print = 1.0;

    printf("A: %d playouts  B: %d playouts  board: %dx%d\n", po_a, po_b, BOARD_SIZE, BOARD_SIZE);
    printf("%6s  %8s  %7s\n", "games", "elapsed", "B win%");

    for (;;) {
        /* Alternate: even games A=BLACK, odd games A=WHITE */
        int8_t a_color = (games % 2 == 0) ? BLACK : WHITE;

        Game2 g;
        g2_new(&g);
        while (!g.game_over) {
            RaveState *s;
            int po;
            if (g.current == a_color) { s = sa; po = po_a; }
            else                      { s = sb; po = po_b; }
            RaveResult r = rave_search(s, &g, po, 0);
            g2_play(&g, r.move);
        }

        int8_t winner = g2_estimate_winner(&g);
        if      (winner == a_color)  wins_a++;
        else if (winner == -a_color) wins_b++;
        else                         draws++;
        games++;

        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
        if (elapsed >= next_print) {
            float b_pct = games > 0 ? 100.0f * wins_b / games : 0;
            printf("%6d  %7.1fs  %6.1f%%\n", games, elapsed, b_pct);
            fflush(stdout);
            next_print = elapsed * 1.5;
        }
    }

    rave_destroy(sa);
    rave_destroy(sb);
    return 0;
}
