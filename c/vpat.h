#ifndef VPAT_H
#define VPAT_H

#include <stdbool.h>
#include "game2.h"

/* Minimal vpatterns evaluator — the C twin of vpatterns.js's evaluate(), for
 * truncated playouts inside the trainers.  Supports the deployed pattern-only
 * spec shape:
 *
 *   - the TURN feature ('t', optionally phase-binned 'tp<N>'): one
 *     antisymmetric feature, +1 when BLACK is to move, keyed by phase bucket;
 *   - 2x2 and 3x3 window families over the capped-liberty alphabet
 *     ('2:<ml>', '3:<ml>' with ml in 1..15), hashed with the prime-leaf
 *     X-hash recursion, one feature per window with polarity from the
 *     normal-vs-colour-inverted hash comparison.
 *
 * That covers 'tp9,2:3,3:3' and the 2:4,3:4 champion family.  Anything else
 * in a model's specs (sizes 1/4/34/23, ladder 'L', health 'H', phase-binned
 * pattern specs) is refused loudly at load — silently dropping a family
 * would score wrong.
 *
 * V = sigmoid(sum_i pol_i * w[key_i]) = P(BLACK wins), matching the JS twin
 * bit-for-bit in the integer key pipeline (weights are kept as doubles, the
 * same values the JS Map holds).
 */

bool   vpat_load(const char *path);       /* parses the model JS file; exits loudly on unsupported specs */
double vpat_evaluate(const Game2 *g);     /* P(BLACK wins) for the current position */
double vpat_evaluate_z(const Game2 *g);   /* the raw logit, for callers applying a logit-space offset */

/* Baked truncation default (trunc: {delta}) if the model file has it; returns
 * false when absent.  The output pointer may be NULL. */
bool   vpat_trunc(double *delta);

#endif
