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
        /* Defender's turn: try each liberty; succeed if any reaches safety. */
        for (int i = 0; i < n_libs; i++) {
            const int32_t lib_idx = libs[i];
            if (!g3_play(g, lib_idx)) continue;          /* suicide → skip */
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

    /* Try mover playing each liberty first. */
    for (int i = 0; i < lc; i++) {
        const int32_t lib_idx = r.libs[i];
        if (!defending && atari) {
            /* Mover is attacker and group is in atari: playing the lib
             * captures, so defender doesn't escape.  Skip play-undo. */
            escape = false;
        } else {
            if (!g3_play(g, lib_idx)) continue;
            escape = ladder2_can_reach_3libs(g, stone_idx);
            g3_undo(g);
        }
        if (defending == escape) {
            r.mover_succeeds = true;
            r.urgent_libs[r.urgent_count++] = lib_idx;
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
