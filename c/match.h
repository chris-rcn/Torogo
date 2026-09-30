#ifndef MATCH_H
#define MATCH_H

#include <stdint.h>
#include "puct.h"

/* The ppat match metric's games, as selfplay.js plays
 *   --p1 puct-ppat-fp --min-phase F --fallback rff --rand-mirror-pairs M:
 * an opening of M random mirror pairs from an empty board (exactly balanced:
 * invariant under 180-degree rotation + colour swap), then rff (featurepol,
 * fpol.h, sampled at temperature 1) for both sides until phase >= F, then P1
 * and P2 (PUCT searches of `playouts` simulations) to the end.  Each opening
 * is played twice, P1 black then P2 black.
 *
 * Randomness: a pair is fully determined by its seed.  The opening draws from
 * the pair stream; each SEAT (black, white) then gets its own stream, reseeded
 * identically for the colour-swapped game (common random numbers, as
 * selfplay's seat seeds).  fpol must be loaded and, when a side has ppat
 * playouts, ppat.h's globals must be that model's. */

typedef struct {
    int     size;           /* 13 */
    int     mirror_pairs;   /* selfplay --rand-mirror-pairs, 3 (0 = free centre stone, no pairs) */
    double  min_phase;      /* selfplay --min-phase: rff plays both sides below it */
    int     playouts;       /* per search move, both sides */
    PuctCfg side[2];        /* [0] = P1, [1] = P2 */
} MatchCfg;

typedef struct {
    int     games, pairs, p2_wins, black_wins;
    double  pair_sum, pair_sq;       /* P2's per-pair score (wins / 2) */
    long    moves[2];                /* search moves by P1, P2 */
    double  ms[2];                   /* their search time */
    long    opening_moves, total_moves;
} MatchStats;

typedef struct Match Match;

Match *match_new(const MatchCfg *cfg);
void   match_free(Match *m);

/* Play one opening from `seed` twice with colours swapped, adding to *st.
 * Returns P2's wins in the pair (0..2). */
int    match_play_pair(Match *m, uint64_t seed, MatchStats *st);

/* P2's win ratio and its standard error from the pair scores. */
double match_p2_ratio(const MatchStats *st);
double match_p2_se(const MatchStats *st);

#endif
