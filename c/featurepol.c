/* featurepol.c — see featurepol.h.  Mirrors featurepol-lib.js bit-for-bit. */
#include "featurepol.h"
#include "ladder2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>

uint32_t fp_hash_str(const char *s) {
    uint32_t h = 0x811c9dc5u;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 0x01000193u;
    return h;
}

/* ── Nearest-cell offsets and D4 permutation (match featurepol-lib.js) ─────── */
static const int NEAR_OFF[FP_NEAR_MAX][2] = {
    {-1, 0}, {0, 1}, {1, 0}, {0, -1},
    {-1, 1}, {1, 1}, {1, -1}, {-1, -1},
    {-2, 0}, {0, 2}, {2, 0}, {0, -2},
    {-2, -1}, {-2, 1}, {-1, 2}, {1, 2},
    {2, 1}, {2, -1}, {1, -2}, {-1, -2},
};
static int NEAR_PERM[8][FP_NEAR_MAX];
static int g_tables_ready = 0;

static void fp_init_tables(void) {
    if (g_tables_ready) return;
    /* d4[s](r,c) — must match the JS d4 list */
    for (int s = 0; s < 8; s++) {
        for (int i = 0; i < FP_NEAR_MAX; i++) {
            int r = NEAR_OFF[i][0], c = NEAR_OFF[i][1], tr, tc;
            switch (s) {
                case 0: tr =  r; tc =  c; break;
                case 1: tr =  c; tc = -r; break;
                case 2: tr = -r; tc = -c; break;
                case 3: tr = -c; tc =  r; break;
                case 4: tr =  r; tc = -c; break;
                case 5: tr = -r; tc =  c; break;
                case 6: tr =  c; tc =  r; break;
                default: tr = -c; tc = -r; break;
            }
            int j = -1;
            for (int k = 0; k < FP_NEAR_MAX; k++)
                if (NEAR_OFF[k][0] == tr && NEAR_OFF[k][1] == tc) { j = k; break; }
            NEAR_PERM[s][i] = j;   /* offsets are D4-closed, so j >= 0 */
        }
    }
    g_tables_ready = 1;
}

static uint32_t fp_canon_radix(const int8_t *cv, int n, int radix) {
    uint32_t best = 0xFFFFFFFFu;
    for (int s = 0; s < 8; s++) {
        const int *perm = NEAR_PERM[s];
        uint64_t raw = 0;
        for (int i = 0; i < n; i++) raw = raw * radix + cv[perm[i]];
        if ((uint32_t)raw < best) best = (uint32_t)raw;
    }
    return best;
}

/* Mixed-radix D4 canon for stone8AdjLib: 4 orthogonals radix R, 4 diagonals radix 3. */
static uint32_t fp_canon_mixed8(const int8_t *cv, int R) {
    uint32_t best = 0xFFFFFFFFu;
    for (int s = 0; s < 8; s++) {
        const int *perm = NEAR_PERM[s];
        uint64_t raw = 0;
        for (int i = 0; i < 4; i++) raw = raw * R + cv[perm[i]];
        for (int i = 4; i < 8; i++) raw = raw * 3 + cv[perm[i]];
        if ((uint32_t)raw < best) best = (uint32_t)raw;
    }
    return best;
}

/* ── Group-liberty helpers (mirror game2.js / featurepol-lib.js) ───────────── */
static int fp_capture_count(const Game2 *g, int idx) {
    int foe = -g->current, b = idx * 4, total = 0, s0 = -1, s1 = -1, s2 = -1;
    for (int d = 0; d < 4; d++) {
        int ni = g2_nbr[b + d];
        if (g->cells[ni] != foe) continue;
        int gid = g->gid[ni];
        if (g->ls[gid] != 1) continue;
        if (gid == s0 || gid == s1 || gid == s2) continue;
        if (s0 < 0) s0 = gid; else if (s1 < 0) s1 = gid; else s2 = gid;
        total += g->ss[gid];
    }
    return total;
}

static int fp_atari_stones(const Game2 *g, int idx) {
    int foe = -g->current, b = idx * 4, total = 0, s0 = -1, s1 = -1, s2 = -1;
    for (int d = 0; d < 4; d++) {
        int ni = g2_nbr[b + d];
        if (g->cells[ni] != foe) continue;
        int gid = g->gid[ni];
        if (g->ls[gid] != 2) continue;
        if (gid == s0 || gid == s1 || gid == s2) continue;
        if (s0 < 0) s0 = gid; else if (s1 < 0) s1 = gid; else s2 = gid;
        total += g->ss[gid];
    }
    return total;
}

static int fp_adj_friendly_chains(const Game2 *g, int idx) {
    int me = g->current, b = idx * 4, n = 0, s0 = -1, s1 = -1, s2 = -1;
    for (int d = 0; d < 4; d++) {
        int ni = g2_nbr[b + d];
        if (g->cells[ni] != me) continue;
        int gid = g->gid[ni];
        if (gid == s0 || gid == s1 || gid == s2) continue;
        if (s0 < 0) s0 = gid; else if (s1 < 0) s1 = gid; else s2 = gid;
        n++;
    }
    return n;
}

/* Liberty bitset of the would-be merged group; returns merged stone count + 1. */
static int fp_build_result_libs(const Game2 *g, int idx, uint32_t *L) {
    int color = g->current, W = g->W;
    const int8_t *cells = g->cells;
    const int16_t *gidA = g->gid, *ls = g->ls, *ss = g->ss;
    const uint32_t *lw = g->lw, *sw = g->sw;
    for (int w = 0; w < W; w++) L[w] = 0;
    int base = idx * 4;
    int f[4] = {-1, -1, -1, -1}, nf = 0, e[4] = {-1, -1, -1, -1}, ne = 0, friendStones = 0;
    for (int d = 0; d < 4; d++) {
        int ni = g2_nbr[base + d], c = cells[ni];
        if (c == EMPTY) { L[ni >> 5] |= (1u << (ni & 31)); continue; }
        int gid = gidA[ni];
        if (c == color) {
            int dup = 0; for (int k = 0; k < nf; k++) if (f[k] == gid) { dup = 1; break; }
            if (dup) continue;
            f[nf++] = gid; friendStones += ss[gid];
            const uint32_t *gb = lw + (size_t)gid * W;
            for (int w = 0; w < W; w++) L[w] |= gb[w];
        } else {
            if (ls[gid] != 1) continue;
            if (((lw[(size_t)gid * W + (idx >> 5)] >> (idx & 31)) & 1u) == 0) continue;
            int dup = 0; for (int k = 0; k < ne; k++) if (e[k] == gid) { dup = 1; break; }
            if (dup) continue;
            e[ne++] = gid;
        }
    }
    L[idx >> 5] &= ~(1u << (idx & 31));
    for (int k = 0; k < ne; k++) {
        int gid = e[k];
        const uint32_t *gb = sw + (size_t)gid * W;
        for (int wi = 0; wi < W; wi++) {
            uint32_t bits = gb[wi];
            while (bits) {
                int s = wi * 32 + __builtin_ctz(bits);
                bits &= bits - 1;
                int sb = s * 4;
                for (int d = 0; d < 4; d++) {
                    int tcell = g2_nbr[sb + d];
                    if (tcell == idx) { L[s >> 5] |= (1u << (s & 31)); break; }
                    if (cells[tcell] == color) {
                        int tg = gidA[tcell], hit = 0;
                        for (int kk = 0; kk < nf; kk++) if (f[kk] == tg) { hit = 1; break; }
                        if (hit) { L[s >> 5] |= (1u << (s & 31)); break; }
                    }
                }
            }
        }
    }
    return friendStones + 1;
}

static int fp_resulting_liberty_count(const Game2 *g, int idx) {
    uint32_t L[MAX_BW]; fp_build_result_libs(g, idx, L);
    int count = 0; for (int w = 0; w < g->W; w++) count += g2_popcount(L[w]);
    return count;
}

static int fp_self_atari_size(const Game2 *g, int idx) {
    uint32_t L[MAX_BW]; int size = fp_build_result_libs(g, idx, L);
    int count = 0;
    for (int w = 0; w < g->W; w++) {
        uint32_t bits = L[w];
        while (bits) { if (++count > 1) return 0; bits &= bits - 1; }
    }
    return count == 1 ? size : 0;
}

static bool fp_creates_ko(const Game2 *g, int idx) {
    int foe = -g->current, base = idx * 4, captures = 0, s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    for (int d = 0; d < 4; d++) {
        int ni = g2_nbr[base + d];
        if (g->cells[ni] != foe) return false;
        int gid = g->gid[ni];
        if (gid == s0 || gid == s1 || gid == s2 || gid == s3) continue;
        if (s0 < 0) s0 = gid; else if (s1 < 0) s1 = gid; else if (s2 < 0) s2 = gid; else s3 = gid;
        if (g->ls[gid] == 1) { if (g->ss[gid] != 1) return false; captures++; }
    }
    return captures == 1;
}

/* Per-cell ladder flag sizes (mirror featurepol-lib.js _buildLadderSizes).
 * out is cap*4: out[cell*4 + flag] = summed stone count of chains for which cell
 * is that flag's liberty (0 kill, 1 save, 2 extend, 3 attack). */
enum { FP_LADF_KILL = 0, FP_LADF_SAVE = 1, FP_LADF_EXTEND = 2, FP_LADF_ATTACK = 3 };

static void fp_build_ladder_sizes(const Game2 *g2, Game3 *g3, int32_t *out) {
    int cap = g2->cap;
    memset(out, 0, (size_t)cap * 4 * sizeof(int32_t));
    if (g2->empty_count == cap) return;
    static Ladder2Result results[MAX_CAP];
    int n = ladder2_get_all_statuses(g3, 1, results, cap);
    if (n > cap) n = cap;
    int cur = g2->current;
    for (int i = 0; i < n; i++) {
        Ladder2Result *r = &results[i];
        if (!r->status.valid) continue;
        bool defending = (r->color == cur);
        int size = g3_group_size(g3, r->gid);
        int flag; const int32_t *targets; int nt;
        if (r->status.urgent_count > 0) {
            flag = defending ? FP_LADF_SAVE : FP_LADF_KILL;
            targets = r->status.urgent_libs; nt = r->status.urgent_count;
        } else if (!r->status.mover_succeeds) {
            flag = defending ? FP_LADF_EXTEND : FP_LADF_ATTACK;
            targets = r->status.libs; nt = r->status.lib_count;
        } else continue;
        for (int k = 0; k < nt; k++) out[targets[k] * 4 + flag] += size;
    }
}

static double fp_distance(int a, int b, int N) {
    int ax = a % N, ay = a / N, bx = b % N, by = b / N;
    int dxr = ax - bx; if (dxr < 0) dxr = -dxr;
    int dyr = ay - by; if (dyr < 0) dyr = -dyr;
    int dx = dxr < N - dxr ? dxr : N - dxr;
    int dy = dyr < N - dyr ? dyr : N - dyr;
    return 0.4 * (dx + dy) + 0.6 * sqrt((double)(dx * dx + dy * dy));
}

/* ── Spec parsing ──────────────────────────────────────────────────────────── */
static const int STONES_N[] = {4, 8, 12, 20};
static int stones_ok(int n) { for (int i = 0; i < 4; i++) if (STONES_N[i] == n) return 1; return 0; }

/* Build one term from a token; returns false (with err) on bad token. */
static bool make_term(const char *tok, FpTerm *t, char *err, int errlen) {
    /* kind = token minus its TRAILING digit run; param = that trailing run (lazy
     * leading / greedy trailing, matching the JS regex — interior digits stay in kind). */
    int len = (int)strlen(tok), e = len;
    while (e > 0 && isdigit((unsigned char)tok[e - 1])) e--;
    char kind[64]; int kl = e; if (kl < 0) kl = 0; if (kl > 63) kl = 63;
    memcpy(kind, tok, (size_t)kl); kind[kl] = 0;
    int has_param = (e < len);
    int param = has_param ? atoi(tok + e) : 0;
    t->salt = fp_hash_str(tok);
    t->param = param;
    t->role = FP_DESCRIPTOR; t->max_level = param; t->max_near = 0; t->needs_ladder = 0;

    if (strcmp(kind, "stones") == 0) {
        if (!has_param || !stones_ok(param)) { snprintf(err, errlen, "stones<n> needs n in {4,8,12,20}: '%s'", tok); return false; }
        t->kind = FP_STONES; t->role = FP_DESCRIPTOR; t->max_near = param;
    } else if (strcmp(kind, "local") == 0) {
        t->kind = FP_LOCAL; t->role = FP_BINARY; t->max_near = 8;
    } else if (strcmp(kind, "localAlways") == 0) {
        t->kind = FP_LOCAL_ALWAYS; t->role = FP_DESCRIPTOR; t->max_near = 8;
    } else if (strcmp(kind, "dist") == 0) {
        if (!has_param) { snprintf(err, errlen, "dist<n> needs a cap: '%s'", tok); return false; }
        t->kind = FP_DIST; t->role = FP_CUMULATIVE; t->max_level = param; t->max_near = 0;
    } else if (strcmp(kind, "stoneExpand") == 0) {
        if (!has_param || param < 1) { snprintf(err, errlen, "stoneExpand<N> needs N>=1: '%s'", tok); return false; }
        t->kind = FP_STONE_EXPAND; t->role = FP_DESCRIPTOR; t->max_near = FP_NEAR_MAX;
    } else if (strcmp(kind, "adjLib") == 0) {
        if (!has_param) { snprintf(err, errlen, "adjLib<n> needs a liberty cap: '%s'", tok); return false; }
        t->kind = FP_ADJLIB; t->role = FP_DESCRIPTOR; t->max_near = 4;
    } else if (strcmp(kind, "stone8AdjLib") == 0) {
        if (!has_param) { snprintf(err, errlen, "stone8AdjLib<n> needs a liberty cap: '%s'", tok); return false; }
        t->kind = FP_STONE8ADJLIB; t->role = FP_DESCRIPTOR; t->max_near = 8;
    } else if (strcmp(kind, "capture") == 0) {
        if (!has_param) { snprintf(err, errlen, "capture<n> needs a size: '%s'", tok); return false; }
        t->kind = FP_CAPTURE; t->role = FP_CUMULATIVE; t->max_level = param;
    } else if (strcmp(kind, "atari") == 0) {
        if (!has_param) { snprintf(err, errlen, "atari<n> needs a size: '%s'", tok); return false; }
        t->kind = FP_ATARI; t->role = FP_CUMULATIVE; t->max_level = param;
    } else if (strcmp(kind, "selfAtari") == 0) {
        if (!has_param) { snprintf(err, errlen, "selfAtari<n> needs a size: '%s'", tok); return false; }
        t->kind = FP_SELFATARI; t->role = FP_CUMULATIVE; t->max_level = param;
    } else if (strcmp(kind, "lib") == 0) {
        if (!has_param) { snprintf(err, errlen, "lib<n> needs a size: '%s'", tok); return false; }
        t->kind = FP_LIB; t->role = FP_CUMULATIVE; t->max_level = param;
    } else if (strcmp(kind, "joins") == 0) {
        if (has_param) { snprintf(err, errlen, "joins takes no parameter: '%s'", tok); return false; }
        t->kind = FP_JOINS; t->role = FP_DESCRIPTOR; t->max_near = 0;
    } else if (strcmp(kind, "flags") == 0) {
        if (has_param) { snprintf(err, errlen, "flags takes no parameter: '%s'", tok); return false; }
        t->kind = FP_FLAGS; t->role = FP_DESCRIPTOR; t->max_near = 8;
    } else if (strcmp(kind, "ko") == 0) {
        t->kind = FP_KO; t->role = FP_BINARY; t->max_near = 0;
    } else if (strcmp(kind, "anyKo") == 0) {
        t->kind = FP_ANYKO; t->role = FP_BINARY; t->max_near = 0;
    } else if (strcmp(kind, "ladderStatus") == 0) {
        t->kind = FP_LADDER_STATUS; t->role = FP_DESCRIPTOR; t->max_near = 0; t->needs_ladder = 1;
    } else if (strcmp(kind, "urgentKill") == 0 || strcmp(kind, "urgentSave") == 0 ||
               strcmp(kind, "wastedExtend") == 0 || strcmp(kind, "wastedAttack") == 0) {
        if (!has_param) { snprintf(err, errlen, "%s<n> needs a size: '%s'", kind, tok); return false; }
        t->kind = strcmp(kind, "urgentKill") == 0 ? FP_URGENT_KILL :
                  strcmp(kind, "urgentSave") == 0 ? FP_URGENT_SAVE :
                  strcmp(kind, "wastedExtend") == 0 ? FP_WASTED_EXTEND : FP_WASTED_ATTACK;
        t->role = FP_CUMULATIVE; t->max_level = param; t->needs_ladder = 1;
    } else {
        snprintf(err, errlen, "unknown/unsupported feature kind '%s' in '%s'", kind, tok);
        return false;
    }
    return true;
}

bool fp_parse_spec(const char *spec_str, FpSpec *out, char *err, int errlen) {
    fp_init_tables();
    memset(out, 0, sizeof(*out));
    snprintf(out->str, sizeof(out->str), "%s", spec_str);
    out->max_near = 0; out->needs_ladder = false; out->num_slots = 0;

    /* slot assignment: distinct token salt -> slot */
    uint32_t slot_salt[FP_MAX_SLOTS]; int n_slot = 0;

    char buf[256]; snprintf(buf, sizeof(buf), "%s", spec_str);
    int ns = 0;
    char *sp_save = NULL;
    for (char *space_tok = strtok_r(buf, ",", &sp_save); space_tok; space_tok = strtok_r(NULL, ",", &sp_save)) {
        if (ns >= FP_MAX_SPACES) { snprintf(err, errlen, "too many spaces"); return false; }
        FpSpace *sp = &out->spaces[ns];
        memset(sp, 0, sizeof(*sp));
        char spacestr[128]; snprintf(spacestr, sizeof(spacestr), "%s", space_tok);
        char saltbuf[160]; snprintf(saltbuf, sizeof(saltbuf), "space:%s", spacestr);
        sp->salt = fp_hash_str(saltbuf);

        char tbuf[128]; snprintf(tbuf, sizeof(tbuf), "%s", space_tok);
        char *t_save = NULL;
        for (char *term_tok = strtok_r(tbuf, "+", &t_save); term_tok; term_tok = strtok_r(NULL, "+", &t_save)) {
            FpTerm t;
            if (!make_term(term_tok, &t, err, errlen)) return false;
            if (t.max_near > out->max_near) out->max_near = t.max_near;
            if (t.needs_ladder) out->needs_ladder = true;
            /* find or assign slot by salt */
            int slot = -1;
            for (int k = 0; k < n_slot; k++) if (slot_salt[k] == t.salt) { slot = k; break; }
            if (slot < 0) {
                if (n_slot >= FP_MAX_SLOTS) { snprintf(err, errlen, "too many distinct terms"); return false; }
                slot = n_slot++;
                slot_salt[slot] = t.salt;
                out->slot_kind[slot] = t.kind; out->slot_param[slot] = t.param;
            }
            t.slot = slot;
            if (t.role == FP_CUMULATIVE) {
                sp->gate[sp->n_gate++] = slot;
                sp->ct_slot[sp->n_ct] = slot; sp->ct_salt[sp->n_ct] = t.salt; sp->ct_max[sp->n_ct] = t.max_level; sp->n_ct++;
            } else if (t.role == FP_BINARY) {
                sp->gate[sp->n_gate++] = slot;
                sp->bt_slot[sp->n_bt] = slot; sp->bt_salt[sp->n_bt] = t.salt; sp->bt_bin[sp->n_bt] = 1; sp->n_bt++;
            } else {
                sp->bt_slot[sp->n_bt] = slot; sp->bt_salt[sp->n_bt] = t.salt; sp->bt_bin[sp->n_bt] = 0; sp->n_bt++;
            }
        }
        ns++;
    }
    out->n_spaces = ns; out->num_slots = n_slot;
    if (ns == 0) { snprintf(err, errlen, "empty spec"); return false; }
    return true;
}

/* ── Term value (memo) ─────────────────────────────────────────────────────── */
static uint32_t fp_term_value(FpKind kind, int param, const Game2 *g, const FpState *st, int idx) {
    const int32_t *nn = st->near_nbr + (size_t)idx * st->near_stride;
    int cur = g->current;
    switch (kind) {
        case FP_STONES: {
            int8_t cv[FP_NEAR_MAX];
            for (int i = 0; i < param; i++) { int8_t c = g->cells[nn[i]]; cv[i] = c == 0 ? 0 : (c == cur ? 1 : 2); }
            return fp_canon_radix(cv, param, 3);
        }
        case FP_LOCAL:
        case FP_LOCAL_ALWAYS: {
            int prev = g->last_move; if (prev < 0) return 0;
            for (int i = 0; i < 8; i++) if (nn[i] == prev) return 1;
            return 0;
        }
        case FP_DIST: {
            int prev = g->last_move; if (prev < 0) return 0;
            int lv = (int)floor(fp_distance(idx, prev, g->N) * 2.0 - 1.0);
            return (uint32_t)(lv < 0 ? 0 : lv);
        }
        case FP_STONE_EXPAND: {
            int8_t cv[FP_NEAR_MAX];
            static const int SH[3] = {8, 12, FP_NEAR_MAX};
            int n = 0, stones = 0, si = 0;
            for (; n < 8; n++) { int8_t c = g->cells[nn[n]]; cv[n] = c == 0 ? 0 : (c == cur ? 1 : 2); if (c != 0) stones++; }
            while (stones < param && n < FP_NEAR_MAX) {
                int nx = SH[++si];
                for (; n < nx; n++) { int8_t c = g->cells[nn[n]]; cv[n] = c == 0 ? 0 : (c == cur ? 1 : 2); if (c != 0) stones++; }
            }
            return fp_hash_combine(fp_canon_radix(cv, n, 3), (uint32_t)n);
        }
        case FP_ADJLIB: {
            int n = param, R = 2 * n + 1; int8_t cv[8];
            for (int i = 0; i < 4; i++) {
                int ni = nn[i]; int8_t c = g->cells[ni];
                if (c == 0) { cv[i] = 0; continue; }
                int lib = g->ls[g->gid[ni]]; if (lib > n) lib = n;
                cv[i] = (c == cur) ? lib : n + lib;
            }
            return fp_canon_radix(cv, 4, R);
        }
        case FP_STONE8ADJLIB: {
            int n = param, R = 2 * n + 1; int8_t cv[8];
            for (int i = 0; i < 4; i++) {
                int ni = nn[i]; int8_t c = g->cells[ni];
                if (c == 0) { cv[i] = 0; continue; }
                int lib = g->ls[g->gid[ni]]; if (lib > n) lib = n;
                cv[i] = (c == cur) ? lib : n + lib;
            }
            for (int i = 4; i < 8; i++) { int8_t c = g->cells[nn[i]]; cv[i] = c == 0 ? 0 : (c == cur ? 1 : 2); }
            return fp_canon_mixed8(cv, R);
        }
        case FP_CAPTURE:   return (uint32_t)fp_capture_count(g, idx);
        case FP_ATARI:     return (uint32_t)fp_atari_stones(g, idx);
        case FP_SELFATARI: return (uint32_t)fp_self_atari_size(g, idx);
        case FP_LIB:       return (uint32_t)fp_resulting_liberty_count(g, idx);
        case FP_JOINS:     return (uint32_t)fp_adj_friendly_chains(g, idx);
        case FP_FLAGS: {
            uint32_t m = 0;
            if (fp_self_atari_size(g, idx) > 0)      m |= 1;
            if (fp_capture_count(g, idx) > 0)        m |= 2;
            if (fp_atari_stones(g, idx) > 0)         m |= 4;
            if (fp_creates_ko(g, idx))               m |= 8;
            if (fp_adj_friendly_chains(g, idx) >= 2) m |= 16;
            int prev = g->last_move;
            if (prev >= 0) { for (int i = 0; i < 8; i++) if (nn[i] == prev) { m |= 32; break; } }
            return m;
        }
        case FP_KO:    return fp_creates_ko(g, idx) ? 1u : 0u;
        case FP_ANYKO: return (g->ko != PASS) ? 1u : 0u;
        case FP_LADDER_STATUS: {
            const int32_t *s = st->ladder_sizes + (size_t)idx * 4;
            return (uint32_t)((s[0] ? 1 : 0) | (s[1] ? 2 : 0) | (s[2] ? 4 : 0) | (s[3] ? 8 : 0));
        }
        case FP_URGENT_KILL:    return (uint32_t)st->ladder_sizes[idx * 4 + FP_LADF_KILL];
        case FP_URGENT_SAVE:    return (uint32_t)st->ladder_sizes[idx * 4 + FP_LADF_SAVE];
        case FP_WASTED_EXTEND:  return (uint32_t)st->ladder_sizes[idx * 4 + FP_LADF_EXTEND];
        case FP_WASTED_ATTACK:  return (uint32_t)st->ladder_sizes[idx * 4 + FP_LADF_ATTACK];
    }
    return 0;
}

/* ── Weights (intern map) ──────────────────────────────────────────────────── */
static void lk_alloc(FpWeights *w, int bits) {
    int sz = 1 << bits;
    w->lk_mask = sz - 1;
    w->lk_hash = calloc(sz, sizeof(uint32_t));
    w->lk_idx = malloc(sz * sizeof(int32_t));
    for (int i = 0; i < sz; i++) w->lk_idx[i] = -1;
}
static void lk_insert(FpWeights *w, uint32_t hash, int32_t idx) {
    int slot = hash & w->lk_mask;
    while (w->lk_idx[slot] != -1) slot = (slot + 1) & w->lk_mask;
    w->lk_hash[slot] = hash; w->lk_idx[slot] = idx;
}
static void lk_grow(FpWeights *w) {
    uint32_t *oh = w->lk_hash; int32_t *oi = w->lk_idx; int osz = w->lk_mask + 1;
    int bits = 1; while ((1 << bits) < osz * 2) bits++;
    lk_alloc(w, bits);
    for (int i = 0; i < osz; i++) if (oi[i] != -1) lk_insert(w, oh[i], oi[i]);
    free(oh); free(oi);
}

FpWeights *fp_create_weights(const FpSpec *spec) {
    FpWeights *w = calloc(1, sizeof(FpWeights));
    w->spec = *spec;
    w->cap = 1024;
    w->key_hash = malloc(w->cap * sizeof(uint32_t));
    w->vals = calloc(w->cap, sizeof(double));
    w->delta = calloc(w->cap, sizeof(double));
    w->count = calloc(w->cap, sizeof(int32_t));
    w->size = 0;
    lk_alloc(w, 12);   /* 4096 slots */
    return w;
}

void fp_free_weights(FpWeights *w) {
    if (!w) return;
    free(w->lk_hash); free(w->lk_idx); free(w->key_hash);
    free(w->vals); free(w->delta); free(w->count); free(w);
}

int32_t fp_intern(FpWeights *w, uint32_t hash) {
    int slot = hash & w->lk_mask;
    while (w->lk_idx[slot] != -1) {
        if (w->lk_hash[slot] == hash) return w->lk_idx[slot];
        slot = (slot + 1) & w->lk_mask;
    }
    int idx = w->size;
    if (idx >= w->cap) {
        w->cap *= 2;
        w->key_hash = realloc(w->key_hash, w->cap * sizeof(uint32_t));
        w->vals = realloc(w->vals, w->cap * sizeof(double));
        w->delta = realloc(w->delta, w->cap * sizeof(double));
        w->count = realloc(w->count, w->cap * sizeof(int32_t));
        for (int i = idx; i < w->cap; i++) { w->vals[i] = 0; w->delta[i] = 0; w->count[i] = 0; }
    }
    w->key_hash[idx] = hash; w->vals[idx] = 0; w->delta[idx] = 0; w->count[idx] = 0;
    w->size = idx + 1;
    /* keep load factor < 0.5 */
    if ((w->size * 2) > (w->lk_mask + 1)) lk_grow(w);
    lk_insert(w, hash, idx);
    return idx;
}

/* ── State ─────────────────────────────────────────────────────────────────── */
static int wrap(int x, int N) { x %= N; return x < 0 ? x + N : x; }

FpState *fp_create_state(int N, const FpSpec *spec) {
    fp_init_tables();
    FpState *st = calloc(1, sizeof(FpState));
    st->N = N; st->cap = N * N;
    st->near_stride = spec->max_near > 0 ? spec->max_near : 1;
    st->near_nbr = malloc((size_t)st->cap * st->near_stride * sizeof(int32_t));
    for (int idx = 0; idx < st->cap; idx++) {
        int r = idx / N, c = idx % N;
        int32_t *base = st->near_nbr + (size_t)idx * st->near_stride;
        for (int k = 0; k < st->near_stride; k++)
            base[k] = wrap(r + NEAR_OFF[k][0], N) * N + wrap(c + NEAR_OFF[k][1], N);
    }
    st->moves = malloc(st->cap * sizeof(int32_t));
    st->keys = malloc((size_t)st->cap * FP_MAX_KEYS_PER_MOVE * sizeof(int32_t));
    st->key_off = malloc((st->cap + 1) * sizeof(int32_t));
    st->logits = malloc(st->cap * sizeof(double));
    st->probs = malloc(st->cap * sizeof(double));
    /* touched can hold chosen-move keys + every candidate's keys (one push each) */
    st->touched = malloc((size_t)(st->cap + 2) * FP_MAX_KEYS_PER_MOVE * sizeof(int32_t));
    st->ladder_sizes = malloc((size_t)st->cap * 4 * sizeof(int32_t));
    return st;
}

void fp_free_state(FpState *st) {
    if (!st) return;
    free(st->near_nbr); free(st->moves); free(st->keys); free(st->key_off);
    free(st->logits); free(st->probs); free(st->touched); free(st->ladder_sizes); free(st);
}

/* ── Extraction ────────────────────────────────────────────────────────────── */
void fp_extract(const Game2 *g, Game3 *g3, FpState *st, FpWeights *w) {
    const FpSpec *spec = &w->spec;
    uint32_t memo[FP_MAX_SLOTS];
    int count = 0, pos = 0;
    if (spec->needs_ladder) fp_build_ladder_sizes(g, g3, st->ladder_sizes);
    st->key_off[0] = 0;
    for (int ei = 0; ei < g->empty_count; ei++) {
        int idx = g->empty_cells[ei];
        if (!g2_is_legal(g, idx) || g2_is_true_eye(g, idx)) continue;
        for (int sl = 0; sl < spec->num_slots; sl++)
            memo[sl] = fp_term_value(spec->slot_kind[sl], spec->slot_param[sl], g, st, idx);
        for (int s = 0; s < spec->n_spaces; s++) {
            const FpSpace *sp = &spec->spaces[s];
            bool gated = false;
            for (int gi = 0; gi < sp->n_gate; gi++) if (memo[sp->gate[gi]] < 1u) { gated = true; break; }
            if (gated) continue;
            uint32_t base = sp->salt;
            for (int bi = 0; bi < sp->n_bt; bi++) {
                uint32_t tv = sp->bt_bin[bi] ? 1u : memo[sp->bt_slot[bi]];
                base = fp_hash_combine(base, fp_hash_combine(sp->bt_salt[bi], tv));
            }
            if (sp->n_ct == 0) {
                st->keys[pos++] = fp_intern(w, base);
            } else if (sp->n_ct == 1) {
                uint32_t sz = memo[sp->ct_slot[0]]; if ((int)sz > sp->ct_max[0]) sz = sp->ct_max[0];
                for (uint32_t k = 1; k <= sz; k++)
                    st->keys[pos++] = fp_intern(w, fp_hash_combine(base, fp_hash_combine(sp->ct_salt[0], k)));
            } else {
                uint32_t acc[FP_MAX_KEYS_PER_MOVE], nxt[FP_MAX_KEYS_PER_MOVE];
                int nAcc = 1; acc[0] = base;
                for (int ci = 0; ci < sp->n_ct; ci++) {
                    uint32_t sz = memo[sp->ct_slot[ci]]; if ((int)sz > sp->ct_max[ci]) sz = sp->ct_max[ci];
                    int on = 0;
                    for (int a = 0; a < nAcc; a++)
                        for (uint32_t k = 1; k <= sz; k++)
                            nxt[on++] = fp_hash_combine(acc[a], fp_hash_combine(sp->ct_salt[ci], k));
                    memcpy(acc, nxt, on * sizeof(uint32_t)); nAcc = on;
                }
                for (int a = 0; a < nAcc; a++) st->keys[pos++] = fp_intern(w, acc[a]);
            }
        }
        st->moves[count] = idx; count++; st->key_off[count] = pos;
    }
    st->count = count;
}

/* ── Softmax ───────────────────────────────────────────────────────────────── */
void fp_softmax(FpState *st, const FpWeights *w, double temperature) {
    int n = st->count; if (n == 0) return;
    double maxL = -INFINITY; int maxI = 0;
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int k = st->key_off[i]; k < st->key_off[i + 1]; k++) s += w->vals[st->keys[k]];
        st->logits[i] = s;
        if (s > maxL) { maxL = s; maxI = i; }
    }
    if (temperature == 0) {
        for (int i = 0; i < n; i++) st->probs[i] = 0;
        st->probs[maxI] = 1;
        return;
    }
    double invT = 1.0 / temperature, sum = 0;
    for (int i = 0; i < n; i++) { double p = exp((st->logits[i] - maxL) * invT); st->probs[i] = p; sum += p; }
    double inv = 1.0 / sum;
    for (int i = 0; i < n; i++) st->probs[i] *= inv;
}

/* ── REINFORCE update ──────────────────────────────────────────────────────── */
int fp_reinforce_update(FpState *st, int chosen, double advantage, FpWeights *w,
                        double lr, double weight_decay, double *wabs_sum, long *wcount) {
    int n = st->count;
    if (n == 0 || chosen < 0) return 0;
    double step = lr * advantage;
    if (step == 0) return 0;
    double *delta = w->delta, *vals = w->vals;
    int32_t *keys = st->keys, *key_off = st->key_off, *touched = st->touched;
    double *probs = st->probs;
    int tc = 0;
    {
        int k0 = key_off[chosen], e = key_off[chosen + 1], K = e - k0;
        if (K > 0) { double add = step / K; for (int k = k0; k < e; k++) { int idx = keys[k]; touched[tc++] = idx; delta[idx] += add; } }
    }
    for (int i = 0; i < n; i++) {
        double pi = probs[i];
        if (pi == 0) continue;
        int k0 = key_off[i], e = key_off[i + 1], K = e - k0;
        if (K == 0) continue;
        double sub = step * pi / K;
        for (int k = k0; k < e; k++) { int idx = keys[k]; touched[tc++] = idx; delta[idx] -= sub; }
    }
    /* _applyTouchedDelta: net delta once per key (d!=0 guard), decoupled L2 shrink, NO counts */
    double decayStep = lr * weight_decay;
    for (int i = 0; i < tc; i++) {
        int idx = touched[i]; double d = delta[idx];
        if (d != 0) {
            vals[idx] += d - decayStep * vals[idx];
            delta[idx] = 0;
            if (wabs_sum) { *wabs_sum += fabs(vals[idx]); (*wcount)++; }
        }
    }
    return tc;
}

int32_t fp_lookup(const FpWeights *w, uint32_t hash) {
    int slot = hash & w->lk_mask;
    while (w->lk_idx[slot] != -1) {
        if (w->lk_hash[slot] == hash) return w->lk_idx[slot];
        slot = (slot + 1) & w->lk_mask;
    }
    return -1;
}

/* ── Base64 + model serialization (same .js format as featurepol-lib.js) ───── */
static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char *b64_encode(const uint8_t *data, int len) {
    int olen = 4 * ((len + 2) / 3);
    char *out = malloc(olen + 1);
    int o = 0;
    for (int i = 0; i < len;) {
        uint32_t a = i < len ? data[i++] : 0;
        uint32_t b = i < len ? data[i++] : 0;
        uint32_t c = i < len ? data[i++] : 0;
        uint32_t t = (a << 16) | (b << 8) | c;
        out[o++] = B64[(t >> 18) & 63];
        out[o++] = B64[(t >> 12) & 63];
        out[o++] = B64[(t >> 6) & 63];
        out[o++] = B64[t & 63];
    }
    int mod = len % 3;
    if (mod == 1) { out[olen - 1] = '='; out[olen - 2] = '='; }
    else if (mod == 2) { out[olen - 1] = '='; }
    out[olen] = 0;
    return out;
}

static int b64_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static uint8_t *b64_decode(const char *s, int *out_len) {
    int slen = (int)strlen(s);
    uint8_t *out = malloc(slen / 4 * 3 + 3);
    int o = 0, acc = 0, nb = 0;
    for (int i = 0; i < slen; i++) {
        if (s[i] == '=') break;
        int v = b64_val(s[i]);
        if (v < 0) continue;
        acc = (acc << 6) | v; nb += 6;
        if (nb >= 8) { nb -= 8; out[o++] = (acc >> nb) & 0xff; }
    }
    *out_len = o;
    return out;
}

bool fp_serialize(const FpWeights *w, const char *path, double ema, long total_updates) {
    int count = w->size;
    double maxAbs = 0;
    for (int i = 0; i < count; i++) { double a = fabs(w->vals[i]); if (a > maxAbs) maxAbs = a; }
    double scale = maxAbs > 0 ? 32767.0 / maxAbs : 1.0;
    uint8_t *buf = malloc((size_t)count * 6 + 6);
    for (int i = 0; i < count; i++) {
        uint32_t k = w->key_hash[i];
        buf[i * 4 + 0] = k & 0xff; buf[i * 4 + 1] = (k >> 8) & 0xff;
        buf[i * 4 + 2] = (k >> 16) & 0xff; buf[i * 4 + 3] = (k >> 24) & 0xff;
        long q = (long)floor(w->vals[i] * scale + 0.5);
        if (q > 32767) q = 32767; else if (q < -32768) q = -32768;
        uint16_t u = (uint16_t)(int16_t)q;
        buf[count * 4 + i * 2 + 0] = u & 0xff;
        buf[count * 4 + i * 2 + 1] = (u >> 8) & 0xff;
    }
    char *b64 = b64_encode(buf, count * 6);
    free(buf);

    FILE *f = fopen(path, "w");
    if (!f) { free(b64); return false; }
    fprintf(f, "'use strict';\n");
    fprintf(f, "// Auto-generated by the C featurepol trainer — do not edit by hand.\n");
    fprintf(f, "const featurepolModel = (() => {\n");
    fprintf(f, "  const count = %d;\n", count);
    fprintf(f, "  const scale = %.17g;\n", scale);
    fprintf(f, "  const spec = \"%s\";\n", w->spec.str);
    fprintf(f, "  const ema = %.10g;\n", ema);
    fprintf(f, "  const totalUpdates = %ld;\n", total_updates);
    fprintf(f, "  const b64 = '%s';\n", b64);
    fprintf(f, "  const bytes = typeof Buffer !== 'undefined'\n");
    fprintf(f, "    ? Buffer.from(b64, 'base64')\n");
    fprintf(f, "    : Uint8Array.from(atob(b64), c => c.charCodeAt(0));\n");
    fprintf(f, "  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 6);\n");
    fprintf(f, "  const keys  = new Int32Array(buf, 0, count);\n");
    fprintf(f, "  const qvals = new Int16Array(buf, count * 4, count);\n");
    fprintf(f, "  return { spec, ema, totalUpdates, count, scale, keys, qvals };\n");
    fprintf(f, "})();\n");
    fprintf(f, "if (typeof module !== 'undefined') module.exports = featurepolModel;\n");
    fprintf(f, "else window.featurepolModel = featurepolModel;\n");
    fclose(f);
    free(b64);
    return true;
}

/* find `key` then the next char `delim` after it; copy between into out (NUL-term). */
static bool extract_quoted(const char *text, const char *key, char delim, char *out, int outlen) {
    const char *p = strstr(text, key);
    if (!p) return false;
    p += strlen(key);
    const char *e = strchr(p, delim);
    if (!e) return false;
    int L = (int)(e - p); if (L >= outlen) L = outlen - 1;
    memcpy(out, p, L); out[L] = 0;
    return true;
}

FpWeights *fp_load_model(const char *path, double *ema_out, long *tu_out, char *err, int errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = malloc(fsz + 1);
    if (fread(text, 1, fsz, f) != (size_t)fsz) { fclose(f); free(text); snprintf(err, errlen, "read failed"); return NULL; }
    text[fsz] = 0; fclose(f);

    char specbuf[256], b64buf[1 << 20];
    const char *sp = strstr(text, "const scale = ");
    const char *cp = strstr(text, "const count = ");
    if (!sp || !cp || !extract_quoted(text, "const spec = \"", '"', specbuf, sizeof(specbuf))) {
        free(text); snprintf(err, errlen, "malformed model (spec/scale/count)"); return NULL;
    }
    double scale = strtod(sp + strlen("const scale = "), NULL);
    int count = atoi(cp + strlen("const count = "));
    const char *ep = strstr(text, "const ema = ");
    const char *tp = strstr(text, "const totalUpdates = ");
    if (ema_out) *ema_out = ep ? strtod(ep + strlen("const ema = "), NULL) : 0;
    if (tu_out) *tu_out = tp ? strtol(tp + strlen("const totalUpdates = "), NULL, 10) : 0;
    if (!extract_quoted(text, "const b64 = '", '\'', b64buf, sizeof(b64buf))) {
        free(text); snprintf(err, errlen, "malformed model (b64)"); return NULL;
    }
    free(text);

    int blen = 0; uint8_t *bytes = b64_decode(b64buf, &blen);
    if (blen < count * 6) { free(bytes); snprintf(err, errlen, "b64 too short"); return NULL; }

    FpSpec spec;
    if (!fp_parse_spec(specbuf, &spec, err, errlen)) { free(bytes); return NULL; }
    FpWeights *w = fp_create_weights(&spec);
    for (int i = 0; i < count; i++) {
        uint32_t k = (uint32_t)bytes[i * 4] | ((uint32_t)bytes[i * 4 + 1] << 8)
                   | ((uint32_t)bytes[i * 4 + 2] << 16) | ((uint32_t)bytes[i * 4 + 3] << 24);
        uint16_t u = (uint16_t)bytes[count * 4 + i * 2] | ((uint16_t)bytes[count * 4 + i * 2 + 1] << 8);
        int16_t q = (int16_t)u;
        int32_t idx = fp_intern(w, k);
        w->vals[idx] = (double)q / scale;
    }
    free(bytes);
    return w;
}
