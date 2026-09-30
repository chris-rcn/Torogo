#ifndef PUCT_H
#define PUCT_H

#include <stdbool.h>
#include "game2.h"
#include "rng.h"
#include "ppat.h"

/* PUCT search of ai/puct-ppat-fp.js, in C: featurepol (fpol.h, loaded by the
 * caller) supplies the priors and the per-node top-K, RAVE blends Q, edges
 * expand lazily after n_expand visits, and each simulation ends in a playout
 * or an exact terminal score.  Fixed playout count only.
 *
 * Playouts are uniform while board fullness < ppat_min_phase, ppat after (the
 * JS PPAT_MIN_PHASE gate; ppat_w NULL = uniform throughout).  The ppat model
 * reads ppat.h's globals (adjLib, phases, feature flags), so they must be the
 * ones its file set.  The uniform sampler draws as game2.js randomLegalMove,
 * and wins/visits/priors are float as the JS Float32Arrays, so with uniform
 * playouts a search is bit-identical to the JS agent on the same seed.  C ppat
 * sampling is not bit-identical to ppat-lib.js (float softmax, rng_float). */

typedef struct {
    double c_puct;          /* C_PUCT  0.5 */
    double rave_k;          /* RAVE_K  400 (0 = no RAVE) */
    int    top_k_a;         /* TOP_K_A 10: top-K at phase 0 */
    int    top_k_b;         /* TOP_K_B 40: top-K at phase 1 (both 0 = full width) */
    int    n_expand;        /* N_EXPAND 2 */
    const float *ppat_w;    /* playout policy, NULL = uniform */
    bool   early_pass;      /* travel with ppat_w's file */
    float  pass_weight;
    double ppat_min_phase;  /* PPAT_MIN_PHASE 0.6 */
} PuctCfg;

/* The JS agent's defaults, uniform playouts. */
PuctCfg puct_default_cfg(void);

typedef struct PuctSearch PuctSearch;

PuctSearch *puct_new(const PuctCfg *cfg);
void        puct_free(PuctSearch *s);

typedef struct {
    int32_t move;           /* PASS for pass/resign */
    int     playouts;       /* 0 when no search ran (game over, obvious pass) */
    double  root_win_ratio; /* P(side to move wins) */
} PuctResult;

/* getMove: search `playouts` simulations from g and pick the most-visited root
 * edge (ties by node score).  g is not modified. */
PuctResult puct_get_move(PuctSearch *s, const Game2 *g, int playouts, Rng *rng);

/* Root edges of the last search (valid until the next call). */
int puct_root_stats(const PuctSearch *s, const int32_t **moves, const float **visits, const float **wins);

#endif
