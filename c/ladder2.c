/*
 * ladder2.c — C port of ladder2.js.
 *
 * Mirrors the JS branch-for-branch.  All exploration uses g3_play / g3_undo
 * so no Game3 clones are needed.
 */
#include "ladder2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── defender capture moves ────────────────────────────────────────────────── */

/* Moves the defender can play to capture an adjacent enemy chain in atari: the
 * single remaining liberty of each opponent group touching the chased group at
 * `idx`.  Capturing frees liberties for the chased group.  Iterates the chased
 * group's own stones via its bitset (O(chain size), not O(board)).  Fills `out`
 * (deduped) with up to `max` capture moves and returns the count.  Mirrors
 * _defenderCaptureMoves in ladder2.js. */
static int32_t ladder2_defender_capture_moves(Game3 *g, int32_t idx,
                                              int32_t *out, int32_t max) {
    const int8_t    enemy  = (int8_t)(-g->cells[idx]);
    const int32_t   gid    = g->gid[idx];
    const int32_t   W      = g->W;
    const int32_t  *nbr    = g->nbr;
    const int8_t   *cells  = g->cells;
    const int32_t  *gidArr = g->gid;
    const int32_t  *ls     = g->ls;
    const uint32_t *sw     = g->sw;
    const int32_t   base   = gid * W;
    int32_t count = 0;
    for (int32_t w = 0; w < W; w++) {
        uint32_t bits = sw[base + w];
        while (bits) {
            const int32_t stone = (w << 5) + __builtin_ctz(bits);
            bits &= bits - 1;
            const int32_t nb = stone * 4;
            for (int d = 0; d < 4; d++) {
                const int32_t ni = nbr[nb + d];
                if (cells[ni] != enemy) continue;          /* adjacent enemy stone */
                if (ls[gidArr[ni]] != 1) continue;          /* enemy not in atari */
                const int32_t lib = g3_group_libs2(g, ni).lib0;
                bool dup = false;
                for (int32_t k = 0; k < count; k++) if (out[k] == lib) { dup = true; break; }
                if (!dup && count < max) out[count++] = lib;
            }
        }
    }
    return count;
}

/* ── _canReach3Libs ────────────────────────────────────────────────────────── */

bool ladder2_can_reach_3libs(Game3 *g, int32_t idx) {
    G3GroupLibs2 gl = g3_group_libs2(g, idx);
    const int32_t lc = gl.count;
    if (lc >= 3) return true;
    if (lc == 0) return false;

    const int8_t def_color = g->cells[idx];
    const int32_t libs[2]  = { gl.lib0, gl.lib1 };
    const int     n_libs   = (lc == 1) ? 1 : 2;

    if (g->current == def_color) {
        /* Defender's turn: extend onto a liberty, or capture an adjacent enemy
         * chain in atari; succeed if any reaches safety. */
        int32_t moves[LADDER2_MAX_LIBS + LADDER2_MAX_CAPTURES];
        int32_t nmoves = 0;
        for (int i = 0; i < n_libs; i++) moves[nmoves++] = libs[i];
        int32_t caps[LADDER2_MAX_CAPTURES];
        const int32_t ncaps = ladder2_defender_capture_moves(g, idx, caps, LADDER2_MAX_CAPTURES);
        for (int32_t c = 0; c < ncaps; c++) {
            bool dup = false;
            for (int32_t k = 0; k < nmoves; k++) if (moves[k] == caps[c]) { dup = true; break; }
            if (!dup) moves[nmoves++] = caps[c];
        }
        for (int32_t i = 0; i < nmoves; i++) {
            const int32_t move_idx = moves[i];
            if (!g3_play(g, move_idx)) continue;          /* suicide/illegal → skip */
            bool captured = (g->cells[idx] == G3_EMPTY);
            bool result   = !captured && ladder2_can_reach_3libs(g, idx);
            g3_undo(g);
            if (result) return true;
        }
        return false;
    }

    /* Attacker's turn (1 or 2 libs): try each liberty; succeed if any kills. */
    for (int i = 0; i < n_libs; i++) {
        const int32_t lib_idx = libs[i];
        if (!g3_play(g, lib_idx)) continue;              /* illegal → skip */

        if (g->cells[idx] == G3_EMPTY) {
            /* Attacker captured the defender outright. */
            g3_undo(g);
            return false;
        }

        const int32_t after_lc = g3_group_libs2(g, idx).count;
        if (after_lc == 0) {
            /* Shouldn't really happen (0 libs == captured) but matches JS. */
            g3_undo(g);
            return false;
        }
        if (after_lc == 1) {
            bool result = !ladder2_can_reach_3libs(g, idx);
            g3_undo(g);
            if (result) return false;
        } else {
            g3_undo(g);
        }
    }
    return true;
}

/* ── getLadderStatus ───────────────────────────────────────────────────────── */

Ladder2Status ladder2_get_status(Game3 *g, int32_t stone_idx) {
    Ladder2Status r;
    memset(&r, 0, sizeof(r));

    G3GroupLibs2 gl = g3_group_libs2(g, stone_idx);
    const int32_t lc = gl.count;
    if (lc < 1 || lc > 2) {
        const int N = g->N;
        fprintf(stderr,
                "ladder2_get_status: group at %d,%d has %d liberties (expected <= 2)\n",
                stone_idx % N, stone_idx / N, lc);
        r.valid = false;
        return r;
    }
    r.valid     = true;
    r.lib_count = lc;
    r.libs[0]   = gl.lib0;
    if (lc == 2) r.libs[1] = gl.lib1;

    const bool    atari     = (lc == 1);
    const int8_t  g_color   = g->cells[stone_idx];
    const int8_t  mover     = g->current;
    const bool    defending = (g_color == mover);

    /* Try opponent playing first (i.e., mover passes). */
    bool escape;
    if (defending && atari) {
        /* Defender in atari with attacker to move next: defender cannot
         * escape via passing.  Skip the play(PASS) and short-circuit. */
        escape = false;
    } else {
        g3_play(g, G3_PASS);
        escape = ladder2_can_reach_3libs(g, stone_idx);
        g3_undo(g);
    }
    if (defending == escape) {
        /* Group is not urgent — mover can leave it alone and still be OK. */
        r.mover_succeeds = true;
        r.urgent_count   = 0;
        return r;
    }

    /* Try mover playing each candidate first.  When defending, the saving moves
     * also include captures of adjacent atari'd enemy chains (not just libs). */
    int32_t moves[LADDER2_MAX_LIBS + LADDER2_MAX_CAPTURES];
    int32_t nmoves = 0;
    for (int i = 0; i < lc; i++) moves[nmoves++] = r.libs[i];
    if (defending) {
        int32_t caps[LADDER2_MAX_CAPTURES];
        const int32_t ncaps = ladder2_defender_capture_moves(g, stone_idx, caps, LADDER2_MAX_CAPTURES);
        for (int32_t c = 0; c < ncaps; c++) {
            bool dup = false;
            for (int32_t k = 0; k < nmoves; k++) if (moves[k] == caps[c]) { dup = true; break; }
            if (!dup) moves[nmoves++] = caps[c];
        }
    }
    for (int32_t i = 0; i < nmoves; i++) {
        const int32_t move_idx = moves[i];
        if (!defending && atari) {
            /* Mover is attacker and group is in atari: playing the lib
             * captures, so defender doesn't escape.  Skip play-undo. */
            escape = false;
        } else {
            if (!g3_play(g, move_idx)) continue;
            escape = ladder2_can_reach_3libs(g, stone_idx);
            g3_undo(g);
        }
        if (defending == escape) {
            r.mover_succeeds = true;
            r.urgent_libs[r.urgent_count++] = move_idx;
        }
    }
    return r;
}

/* ── getAllLadderStatuses ──────────────────────────────────────────────────── */

int32_t ladder2_get_all_statuses(Game3 *g, int32_t min_chain_size,
                                  Ladder2Result *out, int32_t max) {
    const int32_t cap = g->cap;
    /* visited[gid] — sized to next_gid up front (monotonic) */
    bool *visited = (bool *)calloc(g->max_g, sizeof(bool));
    int32_t total = 0;

    for (int32_t i = 0; i < cap; i++) {
        if (g->cells[i] == G3_EMPTY) continue;
        const int32_t gid = g->gid[i];
        if (gid < 0 || visited[gid]) continue;
        visited[gid] = true;
        if (g3_group_size(g, gid) < min_chain_size) continue;
        const int32_t lc = g3_group_libs2(g, i).count;
        if (lc == 0 || lc > 2) continue;

        Ladder2Status st = ladder2_get_status(g, i);
        if (total < max) {
            out[total].gid    = gid;
            out[total].color  = g->cells[i];
            out[total].status = st;
        }
        total++;
    }

    free(visited);
    return total;
}
