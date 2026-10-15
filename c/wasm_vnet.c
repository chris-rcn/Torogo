/*
 * wasm_vnet.c — the WebAssembly export layer for the value network (cnn.h) in
 * the browser (ai/fp-vnet.js, ai/puct-vnet-wasm.js).  Built by c/build-wasm.sh, loaded by
 * vnet-wasm.js.
 *
 * The model comes from JS: the bytes of a VNET1 file (vnet_save's format)
 * copied into this module's memory (wasm_alloc) and adopted by
 * wasm_vnet_set_model.  The position comes as in wasm_ppat.c: the cells plus
 * the game-state scalars, rebuilt once per decision (wasm_vnet_set_root).
 *
 * wasm_vnet_eval_moves values each candidate the agent's 1-ply search
 * considers: the move is played on a copy of the root and the child encoded
 * (vnet_encode: liberty classes, ladder status, ko, last move, colour) and
 * run forward, giving 1 - V(child) = P(the mover wins after the move).
 */
#include <stdlib.h>
#include <string.h>
#include "game2.h"
#include "game3.h"
#include "cnn.h"

#define EXPORT(name) __attribute__((export_name(name)))

EXPORT("wasm_alloc") void *wasm_alloc(int bytes) { return malloc((size_t)bytes); }
EXPORT("wasm_free")  void  wasm_free(void *p)    { free(p); }

static int      board_n = 0;
static int32_t  nbr9[MAX_CAP * 9];
static Game3   *g3 = NULL;                 /* ladder-read scratch for vnet_encode */
static VnetLayout L;
static float   *params = NULL;
static VnetWork *work = NULL;

static int8_t   cells_buf[MAX_CAP];        /* JS writes the root's cells here */
static int32_t  moves_buf[MAX_CAP];        /* ... and the candidate moves here */
static float    values_buf[MAX_CAP];       /* the module writes their values here */
static Game2    root, child;

EXPORT("wasm_vnet_cells_buf")  int8_t  *wasm_vnet_cells_buf(void)  { return cells_buf; }
EXPORT("wasm_vnet_moves_buf")  int32_t *wasm_vnet_moves_buf(void)  { return moves_buf; }
EXPORT("wasm_vnet_values_buf") float   *wasm_vnet_values_buf(void) { return values_buf; }
EXPORT("wasm_vnet_max_moves")  int      wasm_vnet_max_moves(void)  { return MAX_CAP; }

/* Board size for everything after: neighbour tables, the ladder scratch and
 * (once a model is set) the network's work buffers. */
EXPORT("wasm_vnet_init") void wasm_vnet_init(int N) {
    g2_init_topology(N);
    vnet_topology(N, nbr9);
    board_n = N;
    if (g3) g3_free(g3);
    g3 = g3_new(N);
    if (params) { vnet_work_free(work); work = vnet_work_new(&L, N * N); }
}

/* Adopt a VNET1 file's bytes ("VNET1", int32 C, B, H, planes, n_params, then
 * n_params floats).  Returns 0; -1 bad magic or size, -2 a plane count other
 * than this build's, -3 a parameter count that does not match C, B, H. */
EXPORT("wasm_vnet_set_model") int wasm_vnet_set_model(const uint8_t *bytes, int n) {
    int32_t hdr[5];
    if (n < 5 + (int)sizeof hdr || memcmp(bytes, "VNET1", 5) != 0) return -1;
    memcpy(hdr, bytes + 5, sizeof hdr);
    if (hdr[3] != VNET_PLANES) return -2;
    if (hdr[1] < 0 || hdr[1] > VNET_MAX_BLOCKS) return -3;
    const VnetLayout nl = vnet_layout(hdr[0], hdr[1], hdr[2]);
    if (nl.n_params != hdr[4]) return -3;
    if (n != 5 + (int)sizeof hdr + 4 * nl.n_params) return -1;
    free(params);
    params = malloc(sizeof(float) * (size_t)nl.n_params);
    memcpy(params, bytes + 5 + sizeof hdr, sizeof(float) * (size_t)nl.n_params);
    L = nl;
    if (work) vnet_work_free(work);
    work = board_n ? vnet_work_new(&L, board_n * board_n) : NULL;
    return 0;
}
EXPORT("wasm_vnet_param_count") int wasm_vnet_param_count(void) { return params ? L.n_params : 0; }

/* The root, from cells_buf and JS Game2's state, as wasm_ppat.c's
 * wasm_set_root: stones replayed in index order with current set to each
 * one's colour (legal and capture-free for any position reachable by legal
 * play), then the state restored.  Returns 0, or -(i+1) when stone i is
 * rejected or captures (an unreachable position). */
EXPORT("wasm_vnet_set_root")
int wasm_vnet_set_root(int current, int ko, int last_move, int ko_stone_white, int ko_stone_black,
                       int consecutive_passes, int move_count) {
    g2_new_empty(&root, board_n);
    for (int i = 0; i < root.cap; i++) {
        if (cells_buf[i] == EMPTY) continue;
        root.current = cells_buf[i];
        if (!g2_play(&root, i) || root.last_capture_count != 0) return -(i + 1);
    }
    root.current = (int8_t)current;
    root.ko = (int16_t)ko;
    root.last_move = (int16_t)last_move;
    root.ko_stone[0] = (int16_t)ko_stone_white;   /* indexed colour + 1 */
    root.ko_stone[2] = (int16_t)ko_stone_black;
    root.consecutive_passes = (int16_t)consecutive_passes;
    root.game_over = consecutive_passes >= 2 || move_count >= 4 * root.cap;
    root.move_count = (int16_t)move_count;
    root.last_capture_count = 0;
    return 0;
}

/* V(root): P(the side to move wins).  -1 with no model. */
EXPORT("wasm_vnet_eval_root") float wasm_vnet_eval_root(void) {
    if (!params || !work) return -1;
    vnet_encode(&root, g3, NULL, vnet_work_input(work));
    return vnet_forward(&L, params, nbr9, root.cap, work);
}

/* values_buf[i] = P(the root's mover wins after moves_buf[i]): 1 - V(child),
 * or the exact result when the move ends the game; -1 for an illegal move.
 * Returns 0, or -1 with no model or a bad count. */
EXPORT("wasm_vnet_eval_moves") int wasm_vnet_eval_moves(int n) {
    if (!params || !work || n < 0 || n > MAX_CAP) return -1;
    const int8_t mover = root.current;
    for (int i = 0; i < n; i++) {
        g2_clone(&child, &root);
        if (!g2_play(&child, moves_buf[i])) { values_buf[i] = -1; continue; }
        if (child.game_over) { values_buf[i] = g2_estimate_winner(&child) == mover ? 1.0f : 0.0f; continue; }
        vnet_encode(&child, g3, NULL, vnet_work_input(work));
        values_buf[i] = 1 - vnet_forward(&L, params, nbr9, child.cap, work);   /* the opponent is to move in child */
    }
    return 0;
}
