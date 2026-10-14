/*
 * bench_ppat.c — ppat playout cost on a fixed, reproducible workload (the C
 * counterpart of bench-ppat.js).
 *
 * Start positions: every position with phase >= --min-phase of --games games,
 * each played from an empty board by the ppat policy itself (uniform below
 * fullness 0.55, as deployed) from a seeded rng; or, with --starts FILE, the
 * positions listed there, one per line as "<size> <idx>,<idx>,..." (board
 * indices from an empty board; "-" = pass).  From each start position
 * --playouts ppat playouts run to the end (or 3*empty + 20 moves), every move
 * by the ppat policy (ppat_min_phase 0), from one seeded rng.
 *
 * Only the playouts are timed (process CPU time; wall time under WASI or -DBENCH_WALL).  The checksum (FNV-1a over every playout move)
 * is identical for two implementations of the same policy given the same
 * arguments — the exactness check for an optimisation.  --incremental runs the
 * playouts on ppat's incremental path (ppat_inc_policy_move / ppat_inc_play),
 * which must reproduce the full path's checksum.
 *
 *   bench_ppat.bin [--model F] [--games 20] [--playouts 10] [--min-phase 0.55]
 *                  [--seed 1] [--size 13] [--starts FILE] [--incremental]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "game2.h"
#include "ppat.h"

static const char *arg_str(int argc, char **argv, const char *flag, const char *def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}
static int arg_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], flag)) return 1;
    return 0;
}
/* Process CPU time, robust to sharing the machine with other jobs — except
 * under WASI, which has no process clock (wasi-libc's clock() emulation reads
 * the wall clock), and with -DBENCH_WALL, which times wall-clock natively too
 * for a like-for-like comparison with a WASI build. */
#if defined(__wasi__) || defined(BENCH_WALL)
#define BENCH_CLOCK CLOCK_MONOTONIC
#define BENCH_TIMER "wall"
#else
#define BENCH_CLOCK CLOCK_PROCESS_CPUTIME_ID
#define BENCH_TIMER "process CPU"
#endif
static double now_ms(void) { struct timespec t; clock_gettime(BENCH_CLOCK, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

/* Grown by doubling: a fixed worst-case array would be ~1 GB, which native
 * builds never touch but a WebAssembly memory would have to commit. */
static Game2 *starts = NULL;
static int n_starts = 0, cap_starts = 0;

static void add_start(const Game2 *g) {
    if (n_starts == cap_starts) {
        cap_starts = cap_starts ? 2 * cap_starts : 1024;
        starts = realloc(starts, sizeof(Game2) * (size_t)cap_starts);
        if (!starts) { fprintf(stderr, "bench_ppat: out of memory at %d start positions\n", n_starts); exit(1); }
    }
    g2_clone(&starts[n_starts++], g);
}

static void load_starts(const char *path, int size) {
    static char line[16384];
    /* Two passes, allocating between them: count, reserve, then parse with the
     * file open and no allocation.  Under Node 22's WASI, a realloc of the
     * ~13 MB array between fgets calls crashed the process (a 20-line repro
     * needs only that; reading first or allocating first is fine). */
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "bench_ppat: cannot open %s\n", path); exit(1); }
    int lines = 0;
    while (fgets(line, sizeof line, f)) if (line[0] != '#' && line[0] != '\n') lines++;
    fclose(f);
    if (lines > cap_starts) {
        cap_starts = lines;
        starts = realloc(starts, sizeof(Game2) * (size_t)cap_starts);
        if (!starts) { fprintf(stderr, "bench_ppat: out of memory for %d start positions\n", lines); exit(1); }
    }
    f = fopen(path, "r");
    if (!f) { fprintf(stderr, "bench_ppat: cannot open %s\n", path); exit(1); }
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *p = line;
        const int n = (int)strtol(p, &p, 10);
        if (n != size) { fprintf(stderr, "bench_ppat: start of size %d, --size is %d\n", n, size); exit(1); }
        Game2 g; g2_new_empty(&g, size);
        while (*p == ' ') p++;
        while (*p && *p != '\n') {
            int32_t m;
            if (*p == '-') { m = PASS; p++; } else m = (int32_t)strtol(p, &p, 10);
            if (!g2_play(&g, m)) { fprintf(stderr, "bench_ppat: illegal start move %d in %s\n", m, path); exit(1); }
            if (*p == ',') p++;
        }
        add_start(&g);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    if (arg_flag(argc, argv, "--help")) {
        printf("usage: bench_ppat.bin [--model F] [--games 20] [--playouts 10] [--min-phase 0.55] [--seed 1] [--size 13] [--starts FILE] [--incremental]\n");
        return 0;
    }
    const char *model_path = arg_str(argc, argv, "--model", "../out/ppat-data-1623136.js");
    const int games      = atoi(arg_str(argc, argv, "--games", "20"));
    const int playouts   = atoi(arg_str(argc, argv, "--playouts", "10"));
    const double min_ph  = atof(arg_str(argc, argv, "--min-phase", "0.55"));
    const long seed      = atol(arg_str(argc, argv, "--seed", "1"));
    const int size       = atoi(arg_str(argc, argv, "--size", "13"));
    const char *starts_f = arg_str(argc, argv, "--starts", NULL);
    const int incremental = arg_flag(argc, argv, "--incremental");
    if (games < 1 || playouts < 1) { fprintf(stderr, "--games and --playouts must be >= 1\n"); return 1; }
    if (size > MAX_BOARD_SIZE) { fprintf(stderr, "--size %d exceeds MAX_BOARD_SIZE %d\n", size, MAX_BOARD_SIZE); return 1; }

    g2_init_topology(size);
    ppat_load_quiet = 1;
    bool early_pass = false; float pass_w = 0;
    float *w = ppat_load_model(model_path, &early_pass, &pass_w);
    if (!w) return 1;

    static PpatState st;
    if (starts_f) load_starts(starts_f, size);
    else {
        Rng gen; rng_seed(&gen, seed);
        ppat_min_phase = 0.55f;
        for (int gi = 0; gi < games; gi++) {
            Game2 g; g2_new_empty(&g, size);
            for (int ply = 0; !g.game_over && ply < 4 * size * size; ply++) {
                if ((double)(g.cap - g.empty_count) / g.cap >= min_ph) add_start(&g);
                g2_play(&g, ppat_policy_move(&g, &st, w, early_pass, pass_w, &gen));
            }
        }
    }
    if (n_starts == 0) { fprintf(stderr, "bench_ppat: no start positions\n"); return 1; }
    ppat_min_phase = 0;   /* every playout move by the ppat policy */

    uint32_t h = 0x811c9dc5u;
    long n_moves = 0;
    Rng rng; rng_seed(&rng, seed + 1);
    Game2 g;
    static PpatInc inc;
    const double t0 = now_ms();
    for (int i = 0; i < n_starts; i++) {
        for (int k = 0; k < playouts; k++) {
            g2_clone(&g, &starts[i]);
            const int limit = 3 * g.empty_count + 20;
            if (incremental) ppat_inc_reset(&inc);
            for (int moves = 0; !g.game_over && moves < limit; moves++) {
                int32_t m;
                if (incremental) { m = ppat_inc_policy_move(&g, &inc, w, early_pass, pass_w, &rng); ppat_inc_play(&g, &inc, m); }
                else             { m = ppat_policy_move(&g, &st, w, early_pass, pass_w, &rng); g2_play(&g, m); }
                n_moves++;
                h = (h ^ (uint32_t)(m + 1)) * 0x01000193u;
            }
        }
    }
    const double ms = now_ms() - t0;
    printf("model: %s, start positions: %d%s, playouts per position: %d, seed: %ld, path: %s, timer: %s\n",
           model_path, n_starts, starts_f ? " (from file)" : "", playouts, seed, incremental ? "incremental" : "full", BENCH_TIMER);
    printf("us/move: %.3f, moves/s: %.0f, checksum: %08x\n", 1000.0 * ms / n_moves, n_moves / (ms / 1000.0), h);
    free(w); free(starts);
    return 0;
}
