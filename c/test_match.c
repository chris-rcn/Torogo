/* test_match — checks on match.c's games (it includes match.c for its statics).
 *
 *   ./test_match.bin ref/ref-fp-fast.js
 *
 *   1. mirror-pair openings are antisymmetric, 2 stones per pair, and rff
 *      carries every opening to min-phase;
 *   2. a pair is a pure function of its seed;
 *   3. identical sides mirror exactly across a pair (common seat streams),
 *      so P2 wins exactly one game of every pair.
 * Tiny playout counts: these test the game machinery, not strength. */

#include "match.c"

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <fpol-model.js>\n", argv[0]); return 1; }
    fpol_load(argv[1]);
    int bad = 0;
    MatchCfg cfg = { .size = 13, .mirror_pairs = 3, .min_phase = 0, .playouts = 8 };
    cfg.side[0] = cfg.side[1] = puct_default_cfg();
    Match *m = match_new(&cfg);
    g2_init_topology(13);

    /* 1 */
    int full = 0; double open_len = 0;
    for (int s = 1; s <= 2000; s++) {
        MatchStats st = {0};
        Rng rng; rng_seed(&rng, s);
        m->cfg.min_phase = 0;
        build_opening(m, &rng, &st);
        if (!is_antisymmetric(&m->opening)) { fprintf(stderr, "seed %d: opening not antisymmetric\n", s); bad++; }
        if (m->opening.move_count % 2) { fprintf(stderr, "seed %d: odd mirror opening\n", s); bad++; }
        full += m->opening.move_count == 6;
        rng_seed(&rng, s);
        m->cfg.min_phase = 0.5;
        build_opening(m, &rng, &st);
        if (!m->opening.game_over && phase_of(&m->opening) < 0.5) { fprintf(stderr, "seed %d: opening stopped below min-phase\n", s); bad++; }
        open_len += m->opening.move_count;
    }
    printf("openings: %d of 2000 got all 3 mirror pairs; mean length to phase 0.5: %.2f moves\n", full, open_len / 2000);

    /* 2 and 3 */
    m->cfg.min_phase = 0.5;
    MatchStats a = {0}, b = {0};
    int mirrored = 0;
    for (int s = 1; s <= 6; s++) {
        const int w = match_play_pair(m, (uint64_t)s, &a);
        mirrored += w == 1;
        match_play_pair(m, (uint64_t)s, &b);
    }
    if (a.p2_wins != b.p2_wins || a.black_wins != b.black_wins || a.total_moves != b.total_moves) {
        fprintf(stderr, "same seeds, different pairs\n"); bad++;
    }
    if (mirrored != 6) { fprintf(stderr, "identical sides: %d of 6 pairs split 1-1\n", mirrored); bad++; }
    printf("identical sides: %d of 6 pairs split 1-1; replay identical: %s\n", mirrored,
           a.total_moves == b.total_moves ? "yes" : "no");
    match_free(m);
    printf("%s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
