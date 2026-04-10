/*
 * find_ppat_features.c — Count ppat feature bit frequencies over random games.
 * Useful for comparing feature distributions across code changes.
 *
 * Compile: cc -O2 -o find_ppat_features game2.c ppat.c find_ppat_features.c -lm
 * Usage:   ./find_ppat_features [games] [moves_per_game]
 */
#include "game2.h"
#include "ppat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    int games = argc > 1 ? atoi(argv[1]) : 50;
    int moves = argc > 2 ? atoi(argv[2]) : 60;

    g2_seed((uint32_t)time(NULL));
    g2_init_topology();
    ppat_init();

    int bits[7] = {0};
    int total = 0;
    PpatState st;
    memset(&st, 0, sizeof(st));

    for (int t = 0; t < games; t++) {
        Game2 g;
        g2_new(&g);
        for (int m = 0; m < moves && !g.game_over; m++) {
            ppat_extract(&g, &st);
            for (int i = 0; i < st.count; i++) {
                for (int b = 0; b < 7; b++)
                    if (st.prev_masks[i] & (1 << b)) bits[b]++;
                total++;
            }
            g2_play(&g, g2_random_legal_move(&g));
        }
    }

    printf("games: %d  moves/game: %d  candidates: %d\n", games, moves, total);
    printf("  bit 0 (contiguous):       %6d  (%.2f%%)\n", bits[0], 100.0 * bits[0] / total);
    printf("  bit 1 (save by capture):  %6d  (%.2f%%)\n", bits[1], 100.0 * bits[1] / total);
    printf("  bit 2 (capture+sa):       %6d  (%.2f%%)\n", bits[2], 100.0 * bits[2] / total);
    printf("  bit 3 (extend):           %6d  (%.2f%%)\n", bits[3], 100.0 * bits[3] / total);
    printf("  bit 4 (extend+sa):        %6d  (%.2f%%)\n", bits[4], 100.0 * bits[4] / total);
    printf("  bit 5 (ko-solve):         %6d  (%.2f%%)\n", bits[5], 100.0 * bits[5] / total);
    printf("  bit 6 (2pt semeai):       %6d  (%.2f%%)\n", bits[6], 100.0 * bits[6] / total);
    return 0;
}
