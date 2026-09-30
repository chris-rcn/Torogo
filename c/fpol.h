#ifndef FPOL_H
#define FPOL_H

#include <stdbool.h>
#include "game2.h"
#include "rng.h"

/* featurepol policy, stone8AdjLib3 only — the C twin of featurepol-lib.js for
 * that one spec (the fp-fast model, ref/ref-fp-fast.js).
 *
 * Each legal non-true-eye move gets one key: the 8 nearest cells, the 4
 * orthogonals carrying liberty counts capped at 3 (0 empty, 1..3 own, 4..6
 * enemy) and the 4 diagonals shape only (0 empty, 1 own, 2 enemy), encoded in
 * mixed radix (7 then 3) and minimised over the 8 D4 symmetries, then hashed
 * with the space and term salts exactly as featurepol-lib.js does.  The move's
 * logit is that key's weight (0 for a key the model does not hold).  Weights
 * are the file's int16 values over its scale, held as float like the JS
 * Float32Array, so logits match the JS twin bit for bit.
 *
 * Any other spec in the file is refused loudly at load.
 */

typedef struct {
    int      n;                 /* candidate moves (legal, non-true-eye), empty-cell order */
    int32_t  moves[MAX_CAP];
    double   logits[MAX_CAP];
    double   probs[MAX_CAP];    /* softmax at the temperature given to fpol_eval */
} FpolState;

/* Parse the model JS file; exits loudly unless its spec is stone8AdjLib3. */
bool    fpol_load(const char *path);

/* Fill st with every candidate's logit and softmax probability at the given
 * temperature (0 = one-hot on the first maximum).  Returns the candidate count. */
int     fpol_eval(const Game2 *g, FpolState *st, double temperature);

/* Draw a move from st's probabilities; PASS when there are no candidates. */
int32_t fpol_sample(const FpolState *st, Rng *rng);

/* Candidate indices into st, ordered by logit, highest first (ties keep
 * candidate order).  Returns the count written, st->n. */
int     fpol_rank(const FpolState *st, int *order);

#endif
