/* vnet_tool.h — what the value-net trainers (train_vnet, train_vnet_td)
 * share: command-line helpers, gen-agent-evals record sets, Adam, and the MD
 * 1-ply test with its Elo estimate. */
#ifndef VNET_TOOL_H
#define VNET_TOOL_H

#include <stdint.h>
#include "game2.h"
#include "game3.h"
#include "cnn.h"

extern const char *vt_prog;   /* the program name error messages start with */

/* The value after `flag`, or def when the flag is absent.  A flag given
 * without a value (last, or followed by another --flag) is an error, never a
 * silent default. */
const char *arg_str(int argc, char **argv, const char *flag, const char *def);
/* A flag whose value is optional: def when given bare, NULL when absent. */
const char *arg_opt(int argc, char **argv, const char *flag, const char *def);
int         arg_flag(int argc, char **argv, const char *flag);
/* After every arg_* call: exits on a flag none of them asked about, or a
 * stray argument that is no flag's value. */
void        args_check(int argc, char **argv);
double      wall_s(void);
double      cpu_s(void);               /* process CPU time */
const char *fmt_time(double s, char *buf);   /* "850us" / "2.2ms" / "1.3s" (buf: 16 bytes) */
uint32_t    mix(uint32_t a, uint32_t b);     /* a 32-bit hash of two words */

/* gen-agent-evals records "<size> <phase> <moves> <winRatio>", compacted. */
typedef struct { int8_t code[MAX_CAP]; VnetMeta meta; float label; } Record;
typedef struct { Record *r; int n, cap; } Set;
/* One board size per run (*size 0 = take the file's, and init the topology);
 * a record that does not replay is reported, counted and dropped.  limit 0 =
 * every record.  Replays and encodes on `threads` threads, in file order. */
void load_set(const char *path, Set *S, int *size, int limit, int threads);

/* Adam (beta1 0.9, beta2 0.999, eps 1e-8) over the flat parameter array. */
typedef struct { float *m, *v; long step; int n; } Adam;
Adam adam_new(int n);
void adam_step(Adam *a, float *w, const float *G, float lr);

/* The MD 1-ply test: at every MD position (movedetails, *.md) each legal move
 * except filling one's own true eye (or, fp_width > 0, featurepol's top
 * fp_width of them: fpol_load first) is played and valued 1 - V(child), and the
 * best is charged its gap to the file's top move (the worst rated candidate
 * when it has no rating, as evalmovedetails does).  mdMae is the mean gap; elo
 * maps the per-band gaps to a CGOS Elo estimate with the elo map's curves
 * (elo = intercept - A * sum_b exp(k*mid_b) * mae_b), only when the MD file is
 * the map's own (sha256 fingerprint of its sorted rows). */
typedef struct MdTest MdTest;
/* md_f NULL = the map's own mdFile. */
MdTest *md_test_open(const char *elo_f, const char *md_f, int N, int fp_width);
void    md_test_describe(const MdTest *T);   /* the "md: ..." header line */
/* One row's "  mdMae    elo" cell into col (40 bytes): the test runs only
 * when it keeps its share of the run's wall time (now, seconds) within 20%,
 * its cost estimated as its evaluations x infer (CPU seconds per forward) /
 * threads, scaled by how far the last run's actual time was off its estimate;
 * otherwise '-'.  WK and G3: one per thread. */
void    md_test_row(MdTest *T, const VnetLayout *L, const float *w, const int32_t *nbr9, int area,
                    VnetWork **WK, Game3 **G3, int threads, double infer, double now, char *col);

#endif
