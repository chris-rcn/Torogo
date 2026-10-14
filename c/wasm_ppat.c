/*
 * wasm_ppat.c — the WebAssembly export layer for ppat playouts in the browser
 * (puct-trunc-wasm).  Built by c/build-wasm.sh, loaded by ppat-wasm.js.
 *
 * The model comes from JS, not a file: ppat-lib.js's loadWeights object is
 * copied into this module's memory (wasm_alloc) and declared with
 * wasm_set_model, which adopts its encoding and checks the weight count.
 * Fatal errors (bad board size, bad adjLib) still print and exit; ppat-wasm.js
 * turns those into JS exceptions.
 *
 * Positions: the search's root is sent once per decision (wasm_set_root: the
 * cells plus the game-state scalars, rebuilt here), and each playout's leaf is
 * reached by replaying the tree-descent path from it (wasm_descend), so a
 * playout never pays for a full rebuild.
 *
 * Playouts: wasm_playout runs puct-trunc.js's untruncated playout from the
 * leaf — 3*empty + 20 moves at most, uniform below the agent's ppat gate,
 * incremental ppat from there — and leaves the colour-signed first-occupancy
 * RAVE trace in a dense float array JS reads as its `played` (wasm_trace_buf),
 * and the ppat / uniform move counts (expansion work) in wasm_counts_buf.
 *
 * -DWASM_PPAT_TEST adds wasm_test_bench, bench_ppat.c's playout loop, so a
 * hand-off module can be checked against bench_ppat.wasm (the file-loader
 * build) for an identical move checksum, and readers of the descended board.
 */
#include <stdlib.h>
#include <string.h>
#include "game2.h"
#include "ppat.h"

#define EXPORT(name) __attribute__((export_name(name)))

static const float *model_w = NULL;
static bool  model_early_pass = false;
static float model_pass_weight = 0;

EXPORT("wasm_alloc") void *wasm_alloc(int bytes) { return malloc((size_t)bytes); }
EXPORT("wasm_free")  void  wasm_free(void *p)    { free(p); }

/* ── Positions ──────────────────────────────────────────────────────────── */

#define WASM_MAX_PATH 1024
static int     board_n = 0;
static int8_t  cells_buf[MAX_CAP];          /* JS writes the root's cells here */
static int32_t path_buf[WASM_MAX_PATH];     /* ... and each playout's descent path here */
static Game2   root, sim;

/* Board size for everything after: neighbour tables. */
EXPORT("wasm_init") void wasm_init(int N) { g2_init_topology(N); board_n = N; }

EXPORT("wasm_cells_buf") int8_t  *wasm_cells_buf(void) { return cells_buf; }
EXPORT("wasm_path_buf")  int32_t *wasm_path_buf(void)  { return path_buf; }
EXPORT("wasm_path_cap")  int      wasm_path_cap(void)  { return WASM_MAX_PATH; }

/* The root, from cells_buf and JS Game2's state: the stones are replayed in
 * index order with current set to each stone's colour — legal and
 * capture-free for any position reachable by legal play (CLAUDE.md's replay
 * invariant) — then the state restored.  Returns 0, or -(i+1) when stone i is
 * rejected or captures (an unreachable position). */
EXPORT("wasm_set_root")
int wasm_set_root(int current, int ko, int last_move, int ko_stone_white, int ko_stone_black,
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

/* The leaf: the root with path_buf[0..n) played.  Returns 0, or -(i+1) when
 * move i is illegal. */
EXPORT("wasm_descend")
int wasm_descend(int n) {
    if (n < 0 || n > WASM_MAX_PATH) return -(WASM_MAX_PATH + 1);
    g2_clone(&sim, &root);
    for (int i = 0; i < n; i++) if (!g2_play(&sim, path_buf[i])) return -(i + 1);
    return 0;
}

/* Adopt a model: its encoding (ppat-lib.js's loadWeights fields) and its n
 * weights, already copied to w.  Returns 0, or the weight count this encoding
 * expects when n differs (the module keeps no model then). */
EXPORT("wasm_set_model")
int wasm_set_model(const float *w, int n, int adj_lib, int phases, int twelvecell, int self_atari,
                   int atari, int xa, int capture, int cs, int early_pass, float pass_weight) {
    const PpatEncoding e = { adj_lib, phases, twelvecell, self_atari, atari, xa, capture, cs };
    ppat_set_encoding(&e);
    const int total = ppat_total_weights();
    if (n != total) { model_w = NULL; return total; }
    model_w = w;
    model_early_pass = early_pass != 0;
    model_pass_weight = pass_weight;
    ppat_min_phase = 0;   /* the uniform gate is the agent's, applied outside */
    return 0;
}

/* ── Playouts ───────────────────────────────────────────────────────────── */

static float   trace[MAX_CAP];
static int32_t counts[2];                   /* [0] ppat moves, [1] uniform moves */
static PpatInc playout_inc;

EXPORT("wasm_trace_buf")  float   *wasm_trace_buf(void)  { return trace; }
EXPORT("wasm_counts_buf") int32_t *wasm_counts_buf(void) { return counts; }

/* splitmix32's finaliser: a per-playout seed from JS becomes a well-mixed,
 * nonzero xorshift state, so nearby seeds start uncorrelated streams. */
static uint32_t mix32(uint32_t z) {
    z += 0x9e3779b9u;
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return (z ^ (z >> 16)) | 1u;
}

/* One playout from the leaf (wasm_descend) with the agent's ppat gate
 * min_phase (uniform moves while fullness < it).  Returns P(BLACK wins) in
 * {0, 1}, or -1 with no model. */
EXPORT("wasm_playout")
int wasm_playout(uint32_t seed, float min_phase) {
    if (!model_w) return -1;
    Rng rng; rng.lo = mix32(seed); rng.hi = mix32(seed ^ 0x5bd1e995u);
    ppat_min_phase = min_phase;
    const int cap = sim.cap;
    memset(trace, 0, sizeof(float) * (size_t)cap);
    const int move_limit = 3 * sim.empty_count + 20;
    const double step = 1.0 / cap;
    double weight = 1.0;
    int moves = 0, uniform = 0;
    ppat_inc_reset(&playout_inc);
    while (!sim.game_over && moves < move_limit) {
        const int8_t cur = sim.current;
        /* Classify BEFORE the move, as ppat_inc_policy_move's gate does. */
        if (min_phase > 0 && (float)(cap - sim.empty_count) / cap < min_phase) uniform++;
        const int32_t mv = ppat_inc_policy_move(&sim, &playout_inc, model_w, model_early_pass, model_pass_weight, &rng);
        if (mv != PASS && weight > 0 && trace[mv] == 0)
            trace[mv] = (float)(cur == BLACK ? weight : -weight);
        ppat_inc_play(&sim, &playout_inc, mv);
        moves++;
        weight -= step;
    }
    counts[0] = moves - uniform;
    counts[1] = uniform;
    return g2_estimate_winner(&sim) == BLACK ? 1 : 0;
}

#ifdef WASM_PPAT_TEST
/* The descended board, for checking it against JS. */
EXPORT("wasm_sim_cells") const int8_t *wasm_sim_cells(void) { return sim.cells; }
/* Liberties of the chain at idx, -1 when empty. */
EXPORT("wasm_sim_libs") int wasm_sim_libs(int idx) { return sim.cells[idx] == EMPTY ? -1 : sim.ls[sim.gid[idx]]; }
/* 0 current, 1 ko, 2 empty count, 3 consecutive passes, 4 game over, 5 last move,
 * 6 white's ko stone, 7 black's ko stone, 8 move count. */
EXPORT("wasm_sim_info") int wasm_sim_info(int k) {
    switch (k) {
        case 0: return sim.current;            case 1: return sim.ko;
        case 2: return sim.empty_count;        case 3: return sim.consecutive_passes;
        case 4: return sim.game_over;          case 5: return sim.last_move;
        case 6: return sim.ko_stone[0];        case 7: return sim.ko_stone[2];
        case 8: return sim.move_count;
    }
    return -999;
}

/* bench_ppat.c's loop: n_starts positions, start i replayed from an empty board
 * by moves[offsets[i] .. offsets[i+1]), each given `playouts` incremental
 * playouts from one rng seeded seed + 1.  Returns the FNV-1a move checksum. */
EXPORT("wasm_test_bench")
uint32_t wasm_test_bench(const int32_t *moves, const int32_t *offsets, int n_starts, int size,
                         int playouts, int seed) {
    static PpatInc inc;
    Rng rng; rng_seed(&rng, seed + 1);
    uint32_t h = 0x811c9dc5u;
    Game2 start, g;
    for (int i = 0; i < n_starts; i++) {
        g2_new_empty(&start, size);
        for (int k = offsets[i]; k < offsets[i + 1]; k++) if (!g2_play(&start, moves[k])) return 0;
        for (int p = 0; p < playouts; p++) {
            g2_clone(&g, &start);
            const int limit = 3 * g.empty_count + 20;
            ppat_inc_reset(&inc);
            for (int m = 0; !g.game_over && m < limit; m++) {
                const int32_t mv = ppat_inc_policy_move(&g, &inc, model_w, model_early_pass, model_pass_weight, &rng);
                ppat_inc_play(&g, &inc, mv);
                h = (h ^ (uint32_t)(mv + 1)) * 0x01000193u;
            }
        }
    }
    return h;
}
#endif
