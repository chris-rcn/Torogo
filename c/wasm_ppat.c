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
 * -DWASM_PPAT_TEST adds wasm_test_bench, bench_ppat.c's playout loop, so a
 * hand-off module can be checked against bench_ppat.wasm (the file-loader
 * build) for an identical move checksum.
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

/* Board size for everything after: neighbour tables. */
EXPORT("wasm_init") void wasm_init(int N) { g2_init_topology(N); }

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

#ifdef WASM_PPAT_TEST
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
