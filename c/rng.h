/*
 * rng.h — the single PRNG used everywhere: xorshift128/64 matching xorshift.js
 * (makeRng), so C and JS produce identical streams from the same seed.
 *
 * General-purpose, explicit state (no globals): a caller owns an Rng and passes
 * it to any function that consumes randomness, so independent concerns get
 * independent streams.
 */
#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <time.h>
#include <unistd.h>

typedef struct { uint32_t lo, hi; } Rng;

static inline void rng_seed(Rng *r, long seed) {
    r->lo = (uint32_t)(seed) ? (uint32_t)seed : 1u;
    r->hi = (uint32_t)((uint64_t)seed >> 32) ? (uint32_t)((uint64_t)seed >> 32)
                                             : 0x9e3779b9u;
}

/* Non-deterministic per-process seed: mixes wall-clock time with the pid so
 * sibling processes launched in the same second still get independent streams
 * (e.g. parallel match/eval shards). */
static inline void rng_seed_entropy(Rng *r) {
    rng_seed(r, (long)time(NULL) ^ ((long)getpid() << 16));
}

/* Next 32-bit value. */
static inline uint32_t rng_next(Rng *r) {
    uint32_t tlo = r->lo, thi = r->hi;
    tlo ^= tlo << 13; tlo ^= tlo >> 7; tlo ^= tlo << 17;
    uint32_t carry = r->hi << 13;
    thi ^= thi >> 17; thi ^= thi << 5;
    r->lo = tlo ^ carry; r->hi = thi;
    return r->lo ^ r->hi;
}

/* Uniform integer in [0, n). */
static inline uint32_t rng_below(Rng *r, uint32_t n) {
    return rng_next(r) % n;
}

/* Uniform float in [0, 1). */
static inline float rng_float(Rng *r) {
    return (float)(rng_next(r) >> 1) / (float)(0x7FFFFFFFu);
}

/* Uniform double in [0, 1) — matches xorshift.js random(). */
static inline double rng_random(Rng *r) {
    return (double)rng_next(r) / 4294967296.0;
}

#endif /* RNG_H */
