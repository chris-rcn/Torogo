/* test_featurepol.c — verify C feature extraction + softmax match the JS oracle.
 *
 *   node ../fp-oracle.js --spec <S> --size <N> --positions <P> --temp <T> | ./test_featurepol.bin <S> <T>
 *
 * Compares, per candidate move: the sorted set of 32-bit key hashes (must be
 * identical) and the softmax probability (within 1e-9).  Exit 0 iff all match.
 */
#include "featurepol.h"
#include "game2.h"
#include "game3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double w_for_hash(uint32_t h) {
    return ((double)fp_mix32(h ^ 0xABCD1234u) / 4294967296.0 - 0.5) * 0.2;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

#define MAXLINE (1 << 20)
#define MAXCAND 512
#define MAXK    64

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <spec> [temp]\n", argv[0]); return 2; }
    const char *spec_str = argv[1];
    double temp = argc > 2 ? atof(argv[2]) : 1.0;

    FpSpec spec; char err[256];
    if (!fp_parse_spec(spec_str, &spec, err, sizeof(err))) { fprintf(stderr, "spec parse: %s\n", err); return 2; }

    FpWeights *w = fp_create_weights(&spec);
    FpState *st = NULL;
    int cur_N = -1;

    char *line = malloc(MAXLINE);
    int positions = 0, keymism = 0, probmism = 0, candmism = 0;
    double maxprobdiff = 0;

    /* oracle per-move arrays */
    static int   o_m[MAXCAND];
    static uint32_t o_k[MAXCAND][MAXK]; static int o_nk[MAXCAND];
    static double o_p[MAXCAND];

    while (fgets(line, MAXLINE, stdin)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        /* --- parse size --- */
        char *q = strstr(line, "\"size\":"); if (!q) continue;
        int size = atoi(q + 7);
        if (size != cur_N) {
            g2_init_topology(size);
            if (st) fp_free_state(st);
            st = fp_create_state(size, &spec);
            cur_N = size;
        }

        /* --- replay pos into a board --- */
        Game2 g; g2_new(&g, size);
        q = strstr(line, "\"pos\":\""); char *pe = NULL;
        if (q) { q += 7; pe = strchr(q, '"'); }
        if (q && pe && pe > q) {
            char tmp[4096]; int L = (int)(pe - q); if (L > 4095) L = 4095; memcpy(tmp, q, L); tmp[L] = 0;
            char *sv = NULL;
            for (char *m = strtok_r(tmp, ",", &sv); m; m = strtok_r(NULL, ",", &sv))
                g2_play(&g, atoi(m));
            q = pe;
        }

        /* --- parse cands --- */
        int nc = 0;
        char *p = strstr(line, "\"cands\":[");
        if (p) p += 9;
        while (p && (p = strstr(p, "\"m\":"))) {
            if (nc >= MAXCAND) break;
            o_m[nc] = atoi(p + 4);
            char *kp = strstr(p, "\"k\":["); if (!kp) break; kp += 5;
            int nk = 0;
            char *end = strchr(kp, ']');
            char *scan = kp;
            while (scan < end) {
                while (scan < end && (*scan == ',' || *scan == ' ')) scan++;
                if (scan >= end) break;
                o_k[nc][nk++] = (uint32_t)strtoul(scan, &scan, 10);
            }
            o_nk[nc] = nk;
            char *pp = strstr(end, "\"p\":"); if (!pp) break; pp += 4;
            o_p[nc] = strtod(pp, NULL);
            qsort(o_k[nc], nk, sizeof(uint32_t), cmp_u32);
            p = pp;
            nc++;
        }

        /* --- C extraction + softmax --- */
        Game3 *g3 = NULL;
        if (spec.needs_ladder) { g3 = g3_new(size); g3_from_game2(g3, &g); }
        fp_extract(&g, g3, st, w);
        if (g3) g3_free(g3);
        for (int i = 0; i < w->size; i++) w->vals[i] = w_for_hash(w->key_hash[i]);
        fp_softmax(st, w, temp);

        if (st->count != nc) { candmism++; positions++; continue; }

        for (int i = 0; i < st->count; i++) {
            int mv = st->moves[i];
            /* find oracle cand with same move */
            int oj = -1;
            for (int j = 0; j < nc; j++) if (o_m[j] == mv) { oj = j; break; }
            if (oj < 0) { candmism++; continue; }

            uint32_t ck[MAXK]; int cnk = st->key_off[i + 1] - st->key_off[i];
            for (int k = 0; k < cnk; k++) ck[k] = w->key_hash[st->keys[st->key_off[i] + k]];
            qsort(ck, cnk, sizeof(uint32_t), cmp_u32);

            int km = (cnk != o_nk[oj]);
            for (int k = 0; !km && k < cnk; k++) if (ck[k] != o_k[oj][k]) km = 1;
            if (km) keymism++;

            /* Keys are bit-exact; probs carry only exp/summation ULP noise (JS Math.exp
             * vs libm exp, and candidate-order-dependent sum), so allow 1e-8. */
            double pd = fabs(st->probs[i] - o_p[oj]);
            if (pd > maxprobdiff) maxprobdiff = pd;
            if (pd > 1e-8) probmism++;
        }
        positions++;
    }

    printf("positions: %d  key-mismatches: %d  cand-mismatches: %d  prob-mismatches(>1e-9): %d  maxprobdiff: %.3e\n",
           positions, keymism, candmism, probmism, maxprobdiff);
    free(line);
    if (st) fp_free_state(st);
    fp_free_weights(w);
    return (keymism || candmism || probmism) ? 1 : 0;
}
