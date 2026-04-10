/*
 * test_ppat.c — Tests for the C ppat library.
 * Compile: cc -O2 -o test_ppat game2.c ppat.c test_ppat.c -lm && ./test_ppat
 */
#include "game2.h"
#include "ppat.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static int passed = 0, failed = 0;

static void check(const char *label, int ok) {
    if (ok) { passed++; }
    else { failed++; fprintf(stderr, "FAIL: %s\n", label); }
}

/* ── Helpers ───────────────────────────────────────────────────────────────── */

static bool play_seq(Game2 *g, const int32_t *moves, int n) {
    for (int i = 0; i < n; i++)
        if (!g2_play(g, moves[i])) return false;
    return true;
}

/* Get prevMask for a specific candidate cell. Returns -1 if not found. */
static int get_mask(const Game2 *g, int32_t cell) {
    PpatState st;
    memset(&st, 0, sizeof(st));
    ppat_extract(g, &st);
    for (int i = 0; i < st.count; i++)
        if (st.moves[i] == cell) return st.prev_masks[i];
    return -1;
}

/* ── Tests ─────────────────────────────────────────────────────────────────── */

static void test_pattern_count(void) {
    check("NUM_PATTERNS is 6810", ppat_num_patterns == 6810);
}

static void test_extract_basic(void) {
    Game2 g;
    g2_new(&g);
    PpatState st;
    memset(&st, 0, sizeof(st));
    ppat_extract(&g, &st);
    check("extract: count > 0", st.count > 0);
    /* All patIds in valid range */
    int valid = 1;
    for (int i = 0; i < st.count; i++) {
        if (st.pat_ids[i] < 0 || st.pat_ids[i] >= ppat_num_patterns) { valid = 0; break; }
    }
    check("extract: all patIds valid", valid);
}

static void test_d4_symmetry(void) {
    /* FRIEND at N vs E vs S vs W of center should all have same patId */
    int N = BOARD_SIZE;
    int center = (N/2)*N + (N/2);
    int cells[4];
    cells[0] = g2_nbr[center*4+0]; /* N */
    cells[1] = g2_nbr[center*4+3]; /* E */
    cells[2] = g2_nbr[center*4+1]; /* S */
    cells[3] = g2_nbr[center*4+2]; /* W */

    int32_t ids[4];
    for (int d = 0; d < 4; d++) {
        Game2 g;
        g2_new_empty(&g);
        g2_play(&g, cells[d]); /* B at neighbor */
        g2_play(&g, PASS);     /* W pass → B's turn */
        PpatState st;
        memset(&st, 0, sizeof(st));
        ppat_extract(&g, &st);
        ids[d] = -1;
        for (int i = 0; i < st.count; i++) {
            if (st.moves[i] == center) { ids[d] = st.pat_ids[i]; break; }
        }
    }
    check("D4: N and E same patId", ids[0] >= 0 && ids[0] == ids[1]);
    check("D4: N and S same patId", ids[0] >= 0 && ids[0] == ids[2]);
    check("D4: N and W same patId", ids[0] >= 0 && ids[0] == ids[3]);
}

static void test_mover_relative(void) {
    /* FRIEND at N vs FOE at N should have different patIds */
    int N = BOARD_SIZE;
    int center = (N/2)*N + (N/2);
    int north  = g2_nbr[center*4+0];

    Game2 gA, gB;
    g2_new_empty(&gA);
    g2_play(&gA, north); g2_play(&gA, PASS); /* B at north, B's turn */
    g2_new_empty(&gB);
    g2_play(&gB, PASS); g2_play(&gB, north); /* W at north, B's turn */

    PpatState st;
    memset(&st, 0, sizeof(st));
    int32_t idA = -1, idB = -1;
    ppat_extract(&gA, &st);
    for (int i = 0; i < st.count; i++) if (st.moves[i] == center) { idA = st.pat_ids[i]; break; }
    ppat_extract(&gB, &st);
    for (int i = 0; i < st.count; i++) if (st.moves[i] == center) { idB = st.pat_ids[i]; break; }
    check("mover-relative: FRIEND vs FOE different patIds", idA >= 0 && idB >= 0 && idA != idB);
}

static void test_atari_encoding(void) {
    /* B@31 in atari (surround with W except one lib). Pattern should differ from non-atari. */
    Game2 g1, g2;
    g2_new_empty(&g1);
    int N = BOARD_SIZE;
    int c = (N/2)*N + (N/2); /* 40 on 9x9 */
    int atari_cell = g2_nbr[c*4+0]; /* N of center = 31 on 9x9 */
    int n_n = g2_nbr[atari_cell*4+0]; /* N of 31 = 22 */
    int n_w = g2_nbr[atari_cell*4+2]; /* W of 31 = 30 */
    int n_e = g2_nbr[atari_cell*4+3]; /* E of 31 = 32 */

    /* Atari: B@31 with W@22, W@30, W@32. Lib = S = center. */
    int32_t seq1[] = {atari_cell, n_n, 0, n_w, 1, n_e, 2, PASS};
    play_seq(&g1, seq1, 8);
    check("atari setup: in atari", g1.gid[atari_cell] >= 0 && g1.ls[g1.gid[atari_cell]] == 1);

    /* Non-atari: B@31 with W@22, W@30 only. 2 libs. */
    g2_new_empty(&g2);
    int32_t seq2[] = {atari_cell, n_n, 0, n_w, 1, PASS};
    play_seq(&g2, seq2, 6);
    check("non-atari setup: 2 libs", g2.gid[atari_cell] >= 0 && g2.ls[g2.gid[atari_cell]] == 2);

    PpatState st;
    memset(&st, 0, sizeof(st));
    int32_t id1 = -1, id2 = -1;
    ppat_extract(&g1, &st);
    for (int i = 0; i < st.count; i++) if (st.moves[i] == c) { id1 = st.pat_ids[i]; break; }
    ppat_extract(&g2, &st);
    for (int i = 0; i < st.count; i++) if (st.moves[i] == c) { id2 = st.pat_ids[i]; break; }
    check("atari vs non-atari: different patId", id1 >= 0 && id2 >= 0 && id1 != id2);
}

static void test_feature1_contiguity(void) {
    /* Feature 1: bit 0 set for 8-neighbors of lastMove, not for others */
    Game2 g;
    g2_new_empty(&g);
    int N = BOARD_SIZE;
    int c = (N/2)*N + (N/2);
    int other = g2_nbr[g2_nbr[c*4+0]*4+0]; /* 2 steps N of center */
    g2_play(&g, c);     /* B@center */
    g2_play(&g, other); /* W@other, lastMove=other */

    PpatState st;
    memset(&st, 0, sizeof(st));
    ppat_extract(&g, &st);

    /* Build set of 8-neighbors of other */
    uint8_t is_nbr[CAP];
    memset(is_nbr, 0, sizeof(is_nbr));
    for (int d = 0; d < 4; d++) {
        is_nbr[g2_nbr[other*4+d]]  = 1;
        is_nbr[g2_dnbr[other*4+d]] = 1;
    }

    int all_nbr_ok = 1, all_non_ok = 1;
    for (int i = 0; i < st.count; i++) {
        int m = st.moves[i];
        if (is_nbr[m]) { if (!(st.prev_masks[i] & 1)) all_nbr_ok = 0; }
        else           { if (st.prev_masks[i] & 1) all_non_ok = 0; }
    }
    check("F1: 8-neighbors have bit 0", all_nbr_ok);
    check("F1: non-neighbors lack bit 0", all_non_ok);
}

#if BOARD_SIZE == 9
/* The following tests use specific 9x9 coordinates */

static void test_feature2_save_by_capture(void) {
    /* 2a: B@31 in atari (lib=22). W@32 in atari (lib=41). lastMove=W@40.
     * Candidate 41 captures W@32, saving B@31. Not self-atari. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {31, 32, 23, 30, 33, 40};
        play_seq(&g, seq, 6);
        check("F2a setup: B@31 atari", g.ls[g.gid[31]] == 1);
        check("F2a setup: W@32 atari", g.ls[g.gid[32]] == 1);
        int mask = get_mask(&g, 41);
        check("F2a: bit 0 set", mask != -1 && (mask & 1));
        check("F2a: bit 1 set", mask != -1 && (mask & 2));
        check("F2a: bit 2 not set", mask != -1 && !(mask & 4));
    }
    /* 2b: B@40 in atari (lib=31). W@39 in atari (lib=48). lastMove=W@49. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {40, 39, 30, 41, 38, 49};
        play_seq(&g, seq, 6);
        check("F2b setup: B@40 atari", g.ls[g.gid[40]] == 1);
        check("F2b setup: W@39 atari", g.ls[g.gid[39]] == 1);
        int mask = get_mask(&g, 48);
        check("F2b: bit 0 set", mask != -1 && (mask & 1));
        check("F2b: bit 1 set", mask != -1 && (mask & 2));
        check("F2b: bit 2 not set", mask != -1 && !(mask & 4));
    }
}

static void test_feature3_capture_self_atari(void) {
    /* 3a: Same as 2a but W@50,W@42 make capture self-atari. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {31, 32, 23, 30, 33, 50, 0, 42, 1, 40};
        play_seq(&g, seq, 10);
        int mask = get_mask(&g, 41);
        check("F3a: bit 0 set", mask != -1 && (mask & 1));
        check("F3a: bit 2 set", mask != -1 && (mask & 4));
        check("F3a: bit 1 not set", mask != -1 && !(mask & 2));
    }
    /* 3b: Same as 2b but W@57,W@47 make capture self-atari. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {40, 39, 30, 41, 38, 57, 0, 47, 1, 49};
        play_seq(&g, seq, 10);
        int mask = get_mask(&g, 48);
        check("F3b: bit 0 set", mask != -1 && (mask & 1));
        check("F3b: bit 2 set", mask != -1 && (mask & 4));
        check("F3b: bit 1 not set", mask != -1 && !(mask & 2));
    }
}

static void test_feature23_realistic(void) {
    /* Realistic F2/F3: W@F6 creates new atari on E6 group.
     * G6 saves by capture (not self-atari) = Feature 2.
     * C6 saves by capture (self-atari) but outside 8-nbr of F6. */
    Game2 g;
    g2_parse_board(&g,
        "9 . . . . . . . . .  \n"
        "8 . O O O O . . . .  \n"
        "7 . O X X O X . . .  \n"
        "6 . O . O X . . . .  \n"
        "5 . O X X X O X . .  \n"
        "4 . . O O O X . . .  \n"
        "3 . . . . . . . . .  \n"
        "2 . . . . . . . . .  \n"
        "1 . . . . . . . . .  \n",
        WHITE);
    g2_play(&g, 50); /* W@F6, puts E6 group in atari */

    check("F2/3 setup: E6 group in atari", g.ls[g.gid[49]] == 1);

    int maskG6 = get_mask(&g, 51); /* G6 */
    check("F2c: G6 bit 0 set", maskG6 != -1 && (maskG6 & 1));
    check("F2c: G6 bit 1 set (save by capture)", maskG6 != -1 && (maskG6 & 2));
    check("F2c: G6 bit 2 not set", maskG6 != -1 && !(maskG6 & 4));

    /* C6 captures W@D3 (saving E6 group) but is self-atari = Feature 3. */
    int maskC6 = get_mask(&g, 47);
    check("F3c: C6 bit 0 set", maskC6 != -1 && (maskC6 & 1));
    check("F3c: C6 bit 2 set (save by capture, self-atari)", maskC6 != -1 && (maskC6 & 4));
    check("F3c: C6 bit 1 not set", maskC6 != -1 && !(maskC6 & 2));
}

static void test_feature4_extend(void) {
    /* 4a: B@31 in atari (lib=40). lastMove=W@30 (W of 31, orthogonal).
     * Candidate 40 (SE of 30, in 8-nbr) extends B@31. Not self-atari. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {31, 22, 0, 32, 1, 30};
        play_seq(&g, seq, 6);
        check("F4a setup: B@31 atari", g.ls[g.gid[31]] == 1);
        int mask = get_mask(&g, 40);
        check("F4a: bit 0 set", mask != -1 && (mask & 1));
        check("F4a: bit 3 set", mask != -1 && (mask & 8));
        check("F4a: bit 4 not set", mask != -1 && !(mask & 16));
    }
    /* 4b: Multi-stone {40,41} in atari (lib=42). lastMove=W@50. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {40, 31, 41, 32, 0, 39, 1, 49, 2, 50};
        play_seq(&g, seq, 10);
        check("F4b setup: {40,41} atari", g.ls[g.gid[40]] == 1);
        int mask = get_mask(&g, 42);
        check("F4b: bit 0 set", mask != -1 && (mask & 1));
        check("F4b: bit 3 set", mask != -1 && (mask & 8));
        check("F4b: bit 4 not set", mask != -1 && !(mask & 16));
    }
}

static void test_feature5_extend_self_atari(void) {
    /* 5a: B@31 in atari (lib=40). lastMove=W@30 (W of 31, orthogonal). W@49,W@39 block.
     * Extend at 40 = self-atari (only lib=41 after). */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {31, 22, 0, 32, 1, 49, 2, 39, 3, 30};
        play_seq(&g, seq, 10);
        check("F5a setup: B@31 atari", g.ls[g.gid[31]] == 1);
        int mask = get_mask(&g, 40);
        check("F5a: bit 0 set", mask != -1 && (mask & 1));
        check("F5a: bit 4 set", mask != -1 && (mask & 16));
        check("F5a: bit 3 not set", mask != -1 && !(mask & 8));
    }
    /* 5b: B@39 in atari (lib=40). W@31,W@49 block. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {39, 30, 0, 38, 1, 31, 2, 49, 3, 48};
        play_seq(&g, seq, 10);
        check("F5b setup: B@39 atari", g.ls[g.gid[39]] == 1);
        int mask = get_mask(&g, 40);
        check("F5b: bit 0 set", mask != -1 && (mask & 1));
        check("F5b: bit 4 set", mask != -1 && (mask & 16));
        check("F5b: bit 3 not set", mask != -1 && !(mask & 8));
    }
}

static void test_feature6_ko_solve(void) {
    /* Realistic ko-solving scenario.
     * B@C3 captures W@C4, creating ko at C4. W@F9 is a ko threat.
     * B@B2 captures B3(O) adjacent to the ko area — solves the ko.
     * Uses ko_stone tracking to detect the active ko fight. */
    Game2 g;
    g2_parse_board(&g,
        "9 . . . . . . . . .  \n"
        "8 . . . . . X O O .  \n"
        "7 . . . . X . X O .  \n"
        "6 . . . . . . X O .  \n"
        "5 . . X O . . . X O  \n"
        "4 . X O X X . . X O  \n"
        "3 X O . O . . . X .  \n"
        "2 . . O . . . . . .  \n"
        "1 . . . . . . . . .  \n",
        BLACK);

    /* C3=20 captures W@C4=29 (ko); then W@F9=77 (ko threat) */
    check("F6c setup: C4 is WHITE", g.cells[29] == WHITE);
    check("F6c setup: C3 is empty", g.cells[20] == EMPTY);
    g2_play(&g, 20);  /* B@C3: captures W@C4, ko at C4 */
    check("F6c setup: ko at C4", g.ko == 29);
    g2_play(&g, 77);  /* W@F9: ko threat */
    check("F6c setup: ko cleared by F9", g.ko == PASS);
    check("F6c setup: B@B2 is a capture", g2_is_capture(&g, 10));
    check("F6c setup: B3 has 1 lib", g.ls[g.gid[19]] == 1);

    int mask = get_mask(&g, 10);
    check("F6c: bit 0 set (contiguous)", mask != -1 && (mask & 1));
    check("F6c: bit 5 set (ko-solve capture)", mask != -1 && (mask & 32));
}

static void test_feature7_semeai(void) {
    /* 7a: B@31 2 libs. W@30 2 libs. lastMove=W@40. c=39 gives W@30 atari. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {31, 30, 21, 40};
        play_seq(&g, seq, 4);
        check("F7a setup: B@31 2 libs", g.ls[g.gid[31]] == 2);
        check("F7a setup: W@30 2 libs", g.ls[g.gid[30]] == 2);
        int mask = get_mask(&g, 39);
        check("F7a: bit 0 set", mask != -1 && (mask & 1));
        check("F7a: bit 6 set", mask != -1 && (mask & 64));
    }
    /* 7b: B@39 2 libs. W@30 2 libs. lastMove=W@40. c=31 gives atari. */
    {
        Game2 g; g2_new_empty(&g);
        int32_t seq[] = {39, 30, 21, 40};
        play_seq(&g, seq, 4);
        check("F7b setup: B@39 2 libs", g.ls[g.gid[39]] == 2);
        check("F7b setup: W@30 2 libs", g.ls[g.gid[30]] == 2);
        int mask = get_mask(&g, 31);
        check("F7b: bit 0 set", mask != -1 && (mask & 1));
        check("F7b: bit 6 set", mask != -1 && (mask & 64));
    }
}
static void test_feature7_realistic(void) {
    /* Realistic F7: B@H2 leaves WHITE H3 group with 2 libs.
     * E4 BLACK group has 2 libs (D5, E5). W@E5 kills, W@D5 doesn't.
     * Neither in 8-nbr of H2.
     * KNOWN LIMITATION: Feature 7 gated by 8-neighborhood of prev.
     * Also: code doesn't distinguish killing vs non-killing atari. */
    Game2 g;
    g2_parse_board(&g,
        "9 . . . . . . . . .  \n"
        "8 . . . . . X . . .  \n"
        "7 . . O O O O X . .  \n"
        "6 . O X X X O X . .  \n"
        "5 . O X . . X X X .  \n"
        "4 . O X X X O O X .  \n"
        "3 X . O O O X O O O  \n"
        "2 . . . . O X X . .  \n"
        "1 . . . . . . . . .  \n",
        BLACK);
    g2_play(&g, 16); /* B@H2 */

    check("F7c setup: H3 group 2 libs", g.ls[g.gid[25]] == 2);
    check("F7c setup: E4 group 2 libs", g.ls[g.gid[31]] == 2);

    int maskE5 = get_mask(&g, 40);
    int maskD5 = get_mask(&g, 39);
    check("F7c: E5 bit 0 set", maskE5 != -1 && (maskE5 & 1));
    check("F7c: E5 bit 6 set (kills)", maskE5 != -1 && (maskE5 & 64));
    check("F7c: D5 bit 6 NOT set (does not kill)", !(maskD5 & 64));
}
#endif /* BOARD_SIZE == 9 */

static void test_policy_move(void) {
    Game2 g;
    g2_new(&g);
    /* Zero weights → uniform policy, should still return a legal move */
    float pat[6810];
    float prev[7];
    memset(pat, 0, sizeof(pat));
    memset(prev, 0, sizeof(prev));
    PpatWeights w = { pat, prev };
    PpatState st;
    memset(&st, 0, sizeof(st));
    int32_t m = ppat_policy_move(&g, &st, &w);
    check("policy: returns legal move", m == PASS || g2_is_legal(&g, m));
    check("policy: not a true eye", m == PASS || !g2_is_true_eye(&g, m));
}

static void test_consistency_with_js(void) {
    /* Run many random games, extract features at each position,
     * verify all patIds are in range and prevMasks have valid bits. */
    int ok = 1;
    PpatState st;
    memset(&st, 0, sizeof(st));
    for (int trial = 0; trial < 50 && ok; trial++) {
        Game2 g;
        g2_new(&g);
        while (!g.game_over) {
            ppat_extract(&g, &st);
            for (int i = 0; i < st.count; i++) {
                if (st.pat_ids[i] < 0 || st.pat_ids[i] >= ppat_num_patterns) { ok = 0; break; }
                if (st.prev_masks[i] & 0x80) { ok = 0; break; } /* only bits 0-6 valid */
            }
            if (!ok) break;
            int32_t m = g2_random_legal_move(&g);
            g2_play(&g, m >= 0 ? m : PASS);
        }
    }
    check("consistency: 50 random games, all features valid", ok);
}

/* ── Performance benchmark ─────────────────────────────────────────────────── */

static void bench_extract(void) {
    /* Generate 200 positions, benchmark extractFeatures */
    Game2 positions[200];
    for (int p = 0; p < 200; p++) {
        g2_new(&positions[p]);
        int moves = 15 + ((p * 7) % 40);
        for (int i = 0; i < moves && !positions[p].game_over; i++) {
            int32_t m = g2_random_legal_move(&positions[p]);
            g2_play(&positions[p], m >= 0 ? m : PASS);
        }
    }

    PpatState st;
    memset(&st, 0, sizeof(st));
    int total_moves = 0, calls = 0;
    clock_t t0 = clock();
    double target = 2.0;
    do {
        for (int p = 0; p < 200; p++) {
            ppat_extract(&positions[p], &st);
            total_moves += st.count;
            calls++;
        }
    } while ((double)(clock() - t0) / CLOCKS_PER_SEC < target);
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    double us_per_call = elapsed / calls * 1e6;
    printf("bench: %.1f µs/call avg (%.1f moves/pos avg, %d calls in %.1fs)\n",
           us_per_call, (double)total_moves / calls, calls, elapsed);
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(void) {
    g2_seed((uint32_t)time(NULL));
    g2_init_topology();
    ppat_init();

    test_pattern_count();
    test_extract_basic();
    test_d4_symmetry();
    test_mover_relative();
    test_atari_encoding();
    test_feature1_contiguity();

#if BOARD_SIZE == 9
    test_feature2_save_by_capture();
    test_feature3_capture_self_atari();
    test_feature23_realistic();
    test_feature4_extend();
    test_feature5_extend_self_atari();
    test_feature6_ko_solve();
    test_feature7_semeai();
    test_feature7_realistic();
#endif

    test_policy_move();
    test_consistency_with_js();

    printf("\n%d passed, %d failed\n\n", passed, failed);

    bench_extract();

    return failed > 0 ? 1 : 0;
}
