/*
 * fatal.h — the fatal-error hook: FATAL(fmt, ...) with up to three int
 * arguments, for errors that end the program (bad board size, bad adjLib,
 * a failed table invariant, out of memory).
 *
 * Native builds print the message to stderr and exit(1).  The browser
 * WebAssembly build (-DTOROGO_WASM_FATAL, c/build-wasm.sh) instead calls the
 * JS import env.torogo_fatal(fmt, a, b, c), which fills in the %d's and
 * throws — so the module needs no stdio (printf alone was ~10 KB of it).
 */
#ifndef FATAL_H
#define FATAL_H

#ifdef TOROGO_WASM_FATAL
__attribute__((import_module("env"), import_name("torogo_fatal"), noreturn))
void torogo_fatal(const char *fmt, int a, int b, int c);
#else
#include <stdio.h>
#include <stdlib.h>
__attribute__((noreturn))
static inline void torogo_fatal(const char *fmt, int a, int b, int c) {
    fprintf(stderr, fmt, a, b, c);   /* unused trailing arguments are ignored */
    exit(1);
}
#endif

/* FATAL("fmt with up to three %d\n", x, y, z): unused arguments become 0. */
#define FATAL(...) FATAL_(__VA_ARGS__, 0, 0, 0)
#define FATAL_(fmt, a, b, c, ...) torogo_fatal(fmt, (int)(a), (int)(b), (int)(c))

#endif /* FATAL_H */
