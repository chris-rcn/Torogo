/*
 * featurepol.h — C port of featurepol-lib.js (hash-keyed linear policy over
 * D4-canonical local-pattern features).  Mirrors the JS engine bit-for-bit:
 * same 32-bit hashing, same D4 canonicalisation, same spec parsing, same
 * softmax and REINFORCE update, same serialized model format.
 *
 * Coverage (incremental port): cells-only feature kinds — stones{4,8,12,20},
 * local, localAlways, dist<n>, stoneExpand<n>.  Group/ladder kinds are added
 * as the underlying game2/game3/ladder2 helpers are wired in.
 */
#ifndef FEATUREPOL_H
#define FEATUREPOL_H

#include <stdint.h>
#include <stdbool.h>
#include "game2.h"
#include "game3.h"

#define FP_MAX_SLOTS         16
#define FP_MAX_SPACES        16
#define FP_MAX_TERMS         8        /* terms per space */
#define FP_NEAR_MAX          20
#define FP_MAX_KEYS_PER_MOVE 64

/* ── 32-bit hashing (must match featurepol-lib.js exactly) ─────────────────── */
static inline uint32_t fp_mix32(uint32_t x) {
    x = (x ^ (x >> 16)) * 0x45d9f3bu;
    x = (x ^ (x >> 16)) * 0x45d9f3bu;
    return x ^ (x >> 16);
}
static inline uint32_t fp_hash_combine(uint32_t h, uint32_t v) {
    return fp_mix32(h ^ fp_mix32(v));
}
uint32_t fp_hash_str(const char *s);

/* ── Feature kinds ─────────────────────────────────────────────────────────── */
typedef enum {
    FP_STONES, FP_LOCAL, FP_LOCAL_ALWAYS, FP_DIST, FP_STONE_EXPAND,
    /* milestone 2: group-helper kinds */
    FP_ADJLIB, FP_STONE8ADJLIB, FP_CAPTURE, FP_ATARI, FP_SELFATARI, FP_LIB,
    FP_JOINS, FP_FLAGS, FP_KO, FP_ANYKO,
    /* milestone 3: ladder kinds (read per-cell ladder_sizes) */
    FP_LADDER_STATUS, FP_URGENT_KILL, FP_URGENT_SAVE, FP_WASTED_EXTEND, FP_WASTED_ATTACK
} FpKind;

/* role: how the term's value enters a key */
typedef enum { FP_DESCRIPTOR = 0, FP_BINARY = 1, FP_CUMULATIVE = 2 } FpRole;

typedef struct {
    FpKind   kind;
    int      param;       /* trailing digits, or 0 */
    uint32_t salt;        /* hash_str(token) */
    FpRole   role;
    int      max_level;   /* for cumulative */
    int      max_near;
    int      needs_ladder;
    int      slot;        /* memo slot (shared by identical tokens across spaces) */
} FpTerm;

typedef struct {
    uint32_t salt;                 /* hash_str("space:" + spaceStr) */
    int      gate[FP_MAX_TERMS];   int n_gate;
    /* baseTerms: {slot, salt, bin} */
    int      bt_slot[FP_MAX_TERMS]; uint32_t bt_salt[FP_MAX_TERMS]; int bt_bin[FP_MAX_TERMS]; int n_bt;
    /* cumTerms: {slot, salt, maxLevel} */
    int      ct_slot[FP_MAX_TERMS]; uint32_t ct_salt[FP_MAX_TERMS]; int ct_max[FP_MAX_TERMS]; int n_ct;
} FpSpace;

typedef struct {
    char     str[256];
    int      n_spaces;
    FpSpace  spaces[FP_MAX_SPACES];
    int      num_slots;
    FpKind   slot_kind[FP_MAX_SLOTS];   /* computer for each slot */
    int      slot_param[FP_MAX_SLOTS];
    int      max_near;
    bool     needs_ladder;
} FpSpec;

/* ── Weights: hash -> dense idx (insertion-ordered) + value arrays ─────────── */
typedef struct {
    FpSpec   spec;
    /* open-addressing lookup hash -> idx+1 (0 = empty) */
    uint32_t *lk_hash;     /* probe table: stored hash */
    int32_t  *lk_idx;      /* probe table: idx (-1 empty) */
    int       lk_mask;     /* table size - 1 (power of two) */
    /* dense arrays indexed by idx (insertion order) */
    uint32_t *key_hash;    /* hash for each idx */
    double   *vals;
    double   *delta;
    int32_t  *count;
    int       size;
    int       cap;
} FpWeights;

/* ── Per-board-size extraction state ───────────────────────────────────────── */
typedef struct {
    int       N, cap, near_stride;
    int32_t  *near_nbr;    /* cap * near_stride */
    int32_t  *moves;       /* cap */
    int32_t  *keys;        /* cap * FP_MAX_KEYS_PER_MOVE (dense idxs) */
    int32_t  *key_off;     /* cap + 1 */
    double   *logits;      /* cap */
    double   *probs;       /* cap */
    int32_t  *touched;     /* scratch */
    int32_t  *ladder_sizes; /* cap * 4: per-cell ladder flag sizes (built once per position) */
    int       count;
} FpState;

/* ── API ───────────────────────────────────────────────────────────────────── */
bool      fp_parse_spec(const char *spec_str, FpSpec *out, char *err, int errlen);
FpWeights *fp_create_weights(const FpSpec *spec);
void      fp_free_weights(FpWeights *w);
int32_t   fp_intern(FpWeights *w, uint32_t hash);   /* hash -> dense idx (creates) */

FpState  *fp_create_state(int N, const FpSpec *spec);
void      fp_free_state(FpState *st);

/* Extract features for every legal non-true-eye move into st (dense key idxs).
 * g3 is the lockstep Game3 mirror (needed only for ladder specs; may be NULL otherwise). */
void      fp_extract(const Game2 *g, Game3 *g3, FpState *st, FpWeights *w);
/* Softmax over current keys/weights at temperature t; fills st->probs. */
void      fp_softmax(FpState *st, const FpWeights *w, double temperature);

/* REINFORCE update for one step (advantage·(1{i=chosen} − π_i), per-move /K, decoupled
 * L2 decay).  wabs_sum/wcount accumulate frequency-weighted |weight| (avgW); may be NULL.
 * Returns the number of touched-weight pushes. */
int       fp_reinforce_update(FpState *st, int chosen, double advantage, FpWeights *w,
                              double lr, double weight_decay, double *wabs_sum, long *wcount);

int32_t   fp_lookup(const FpWeights *w, uint32_t hash);   /* dense idx or -1 (no insert) */

/* Model file I/O (same .js format as featurepol-lib.js: base64 [Int32 keys][Int16 qvals]). */
bool      fp_serialize(const FpWeights *w, const char *path, double ema, long total_updates);
FpWeights *fp_load_model(const char *path, double *ema_out, long *tu_out, char *err, int errlen);

#endif /* FEATUREPOL_H */
