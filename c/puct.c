/* puct.c — PUCT search of ai/puct-ppat-fp.js (see puct.h).  Function names
 * follow the JS: make_node, node_score, select_and_expand, backpropagate,
 * playout, puct_get_move. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "puct.h"
#include "fpol.h"

#define PRIOR_WINS          0.001
#define PRIOR_VISITS        (2 * PRIOR_WINS)
#define RESIGN_MIN_PLAYOUTS 20000

typedef struct {
    int     parent, ci;              /* parent node index (-1 root), edge index in it */
    int8_t  to_move;                 /* chooser of this node's edges (JS -mover) */
    int     sel;                     /* JS selectedChild */
    int     m;                       /* edge count */
    double  total_visits;
    int32_t moves[MAX_CAP + 1];
    int32_t children[MAX_CAP + 1];   /* node index, -1 = unexpanded */
    float   priors[MAX_CAP + 1];
    float   wins[MAX_CAP + 1];
    float   visits[MAX_CAP + 1];
    float   rave_wins[MAX_CAP];
    float   rave_visits[MAX_CAP];
} Node;

struct PuctSearch {
    PuctCfg   cfg;
    Node     *nodes;
    int       n_nodes, cap_nodes;
    int32_t  *path;
    int       path_len, cap_path;
    Game2     sim;
    FpolState fp;
    PpatState pp;
    float     played[MAX_CAP];
};

PuctCfg puct_default_cfg(void) {
    return (PuctCfg){ .c_puct = 0.5, .rave_k = 400, .top_k_a = 10, .top_k_b = 40, .n_expand = 2,
                      .ppat_w = NULL, .early_pass = false, .pass_weight = 0, .ppat_min_phase = 0.6 };
}

PuctSearch *puct_new(const PuctCfg *cfg) {
    PuctSearch *s = calloc(1, sizeof *s);
    s->cfg = *cfg;
    s->cap_nodes = 64;  s->nodes = malloc((size_t)s->cap_nodes * sizeof *s->nodes);
    s->cap_path = 256;  s->path  = malloc((size_t)s->cap_path * sizeof *s->path);
    return s;
}

void puct_free(PuctSearch *s) {
    if (!s) return;
    free(s->nodes); free(s->path); free(s);
}

/* game2.js randomLegalMove: floor(random() * (end + 1)), not rng_below's modulo. */
static int32_t js_random_legal_move(Game2 *g, Rng *rng) {
    for (int end = g->empty_count - 1; end >= 0; end--) {
        const int ri = (int)(rng_random(rng) * (end + 1));
        const int32_t idx = g->empty_cells[ri];
        if (!g2_is_true_eye_at(g, idx) && g2_is_legal(g, idx)) return idx;
        const int32_t t = g->empty_cells[end];
        g->empty_cells[ri] = t;   g->empty_cells[end] = idx;
        g->empty_slot[t] = ri;    g->empty_slot[idx] = end;
    }
    return PASS;
}

static double phase_of(const Game2 *g) { return 1.0 - (double)g->empty_count / (g->N * g->N); }

/* ppatMove with the agent's PPAT_MIN_PHASE gate; ppat.h's own gate held off. */
static int32_t playout_move(PuctSearch *s, Game2 *g, Rng *rng) {
    const PuctCfg *c = &s->cfg;
    if (!c->ppat_w || (c->ppat_min_phase > 0 && phase_of(g) < c->ppat_min_phase))
        return js_random_legal_move(g, rng);
    return ppat_policy_move(g, &s->pp, c->ppat_w, c->early_pass, c->pass_weight, rng);
}

/* Returns P(BLACK wins) in {0,1}; fills s->played with the colour-signed
 * first-occupancy RAVE trace. */
static double playout(PuctSearch *s, Game2 *g, Rng *rng) {
    const int cap = g->N * g->N;
    const int move_limit = 3 * g->empty_count + 20;
    const double step = 1.0 / cap;
    double weight = 1.0;
    memset(s->played, 0, sizeof s->played);
    const float saved_gate = ppat_uniform_below_phase;
    ppat_uniform_below_phase = 0;
    for (int moves = 0; !g->game_over && moves < move_limit; moves++) {
        const int8_t cur = g->current;
        const int32_t idx = playout_move(s, g, rng);
        if (idx != PASS && weight > 0 && s->played[idx] == 0)
            s->played[idx] = cur == BLACK ? (float)weight : (float)-weight;
        g2_play(g, idx);
        weight -= step;
    }
    ppat_uniform_below_phase = saved_gate;
    return g2_estimate_winner(g) == BLACK ? 1 : 0;
}

/* _pruneToTopK's stable descending sort by featurepol probability. */
static const double *sort_prob;
static int by_prob_desc(const void *a, const void *b) {
    const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    const double px = sort_prob[x & 0xffff], py = sort_prob[y & 0xffff];
    if (px != py) return px > py ? -1 : 1;
    return (x >> 16) - (y >> 16);             /* original order */
}

/* makeNode: returns the new node's index (the arena may move). */
static int make_node(PuctSearch *s, int parent, int ci, const Game2 *g) {
    if (s->n_nodes == s->cap_nodes) {
        s->cap_nodes *= 2;
        s->nodes = realloc(s->nodes, (size_t)s->cap_nodes * sizeof *s->nodes);
    }
    const int id = s->n_nodes++;
    Node *n = &s->nodes[id];
    const PuctCfg *c = &s->cfg;
    const int N = g->N, area = N * N;

    /* _runFp: null when the game is over or there is no candidate */
    const bool have_fp = !g->game_over && fpol_eval(g, &s->fp, 1.0) > 0;

    /* getLegalMoves: index order, then PASS when few moves remain or after a pass */
    int m = 0;
    for (int i = 0; i < area; i++)
        if (g->cells[i] == EMPTY && !g2_is_true_eye_at(g, i) && g2_is_legal(g, i)) n->moves[m++] = i;
    const bool has_pass = m < area / 3.0 || g->consecutive_passes > 0;
    if (has_pass) n->moves[m++] = PASS;

    static double prob_by_move[MAX_CAP];
    if (have_fp) {
        memset(prob_by_move, 0, sizeof(double) * area);
        for (int i = 0; i < s->fp.n; i++) prob_by_move[s->fp.moves[i]] = s->fp.probs[i];
    }
    const int k = (int)floor(c->top_k_a + (c->top_k_b - c->top_k_a) * phase_of(g) + 0.5);  /* Math.round */
    if (k > 0 && have_fp) {
        static int32_t keyed[MAX_CAP];
        int np = 0;
        for (int i = 0; i < m; i++) if (n->moves[i] != PASS) { keyed[np] = (np << 16) | n->moves[i]; np++; }
        sort_prob = prob_by_move;
        qsort(keyed, (size_t)np, sizeof *keyed, by_prob_desc);
        m = 0;
        for (int i = 0; i < np && i < k; i++) n->moves[m++] = keyed[i] & 0xffff;
        if (has_pass) n->moves[m++] = PASS;
    }
    n->m = m;

    if (have_fp) {
        const double floor_p = 1.0 / area;
        double sum = 0;
        for (int i = 0; i < m; i++) {
            const int32_t mv = n->moves[i];
            double p = mv == PASS ? floor_p : prob_by_move[mv];
            if (p == 0) p = floor_p;              /* JS: probByMove[m] || floor */
            n->priors[i] = (float)p;
            sum += p;
        }
        if (sum > 0) {
            const double inv = 1 / sum;
            for (int i = 0; i < m; i++) n->priors[i] = (float)(n->priors[i] * inv);
        }
    } else {
        const double u = 1.0 / m;
        for (int i = 0; i < m; i++) n->priors[i] = (float)u;
    }

    for (int i = 0; i < m; i++) {
        n->children[i] = -1;
        n->wins[i] = (float)PRIOR_WINS;
        n->visits[i] = (float)PRIOR_VISITS;
    }
    if (c->rave_k > 0)
        for (int i = 0; i < area; i++) { n->rave_wins[i] = (float)PRIOR_WINS; n->rave_visits[i] = (float)PRIOR_VISITS; }
    n->parent = parent; n->ci = ci;
    n->to_move = g->current;
    n->total_visits = 0.1;
    n->sel = -1;
    return id;
}

static double node_score(const PuctSearch *s, const Node *n, int i, Rng *rng) {
    const PuctCfg *c = &s->cfg;
    double q = (double)n->wins[i] / n->visits[i];
    if (c->rave_k > 0) {
        const int32_t mv = n->moves[i];
        const double rave = mv == PASS ? 0 : (double)n->rave_wins[mv] / n->rave_visits[mv];
        const double beta = c->rave_k / (c->rave_k + n->visits[i]);
        q = (1 - beta) * q + beta * rave;
    }
    const double u = c->c_puct * n->priors[i] * sqrt(n->total_visits) / (1.0 + n->visits[i]);   /* in double, as JS */
    return q + u + 0.001 * rng_random(rng);
}

static void path_push(PuctSearch *s, int32_t mv) {
    if (s->path_len == s->cap_path) {
        s->cap_path *= 2;
        s->path = realloc(s->path, (size_t)s->cap_path * sizeof *s->path);
    }
    s->path[s->path_len++] = mv;
}

/* selectAndExpand: plays into s->sim; returns the final node, sets *do_playout. */
static int select_and_expand(PuctSearch *s, const Game2 *root_g, Rng *rng, bool *do_playout) {
    Game2 *g = &s->sim;
    g2_clone(g, root_g);
    s->path_len = 0;
    *do_playout = false;
    int node = 0;
    while (!g->game_over) {
        Node *n = &s->nodes[node];
        if (n->m == 0) break;
        int best = 0; double best_score = -DBL_MAX;
        for (int i = 0; i < n->m; i++) {
            const double sc = node_score(s, n, i, rng);
            if (sc > best_score) { best_score = sc; best = i; }
        }
        const int32_t mv = n->moves[best];
        path_push(s, mv);
        g2_play(g, mv);
        if (!g->game_over && g->consecutive_passes > 0) {
            g2_play(g, PASS);
            n->sel = best;
            break;
        }
        if (n->children[best] < 0) {
            if (n->visits[best] >= s->cfg.n_expand - 1 + PRIOR_VISITS - 1e-9) {
                const int child = make_node(s, node, best, g);   /* may move the arena */
                s->nodes[node].children[best] = child;
                node = child;
            } else {
                n->sel = best;
            }
            *do_playout = true;
            break;
        }
        node = n->children[best];
        s->nodes[node].sel = -1;
    }
    return node;
}

static void update_rave(PuctSearch *s, Node *n, int d, double won, int8_t chooser, bool have_trace) {
    float *rw = n->rave_wins, *rv = n->rave_visits;
    for (int j = d; j < s->path_len; j += 2) {
        const int32_t mv = s->path[j];
        if (mv == PASS) continue;
        rv[mv] += 1;
        rw[mv] += won;
    }
    if (!have_trace) return;
    const int area = s->sim.N * s->sim.N;
    const float *pl = s->played;
    if (chooser == BLACK) {
        for (int k = 0; k < area; k++) { const float w = pl[k]; if (w > 0) { rv[k] += w; rw[k] += won * w; } }
    } else {
        for (int k = 0; k < area; k++) { const float w = pl[k]; if (w < 0) { rv[k] -= w; rw[k] -= won * w; } }
    }
}

static void backpropagate(PuctSearch *s, int node, double value, bool have_trace) {
    const bool rave = s->cfg.rave_k > 0;
    int d = s->path_len - 1;
    Node *n = &s->nodes[node];
    if (n->sel != -1) {
        const int8_t chooser = n->to_move;
        const double won = chooser == BLACK ? value : 1 - value;
        n->visits[n->sel] += 1;
        n->wins[n->sel] += won;
        n->total_visits += 1;
        if (rave) update_rave(s, n, d, won, chooser, have_trace);
    } else {
        d = s->path_len;
    }
    while (n->parent >= 0) {
        d--;
        const int ci = n->ci;
        Node *p = &s->nodes[n->parent];
        const int8_t chooser = p->to_move;
        const double won = chooser == BLACK ? value : 1 - value;
        p->visits[ci] += 1;
        p->wins[ci] += won;
        p->total_visits += 1;
        if (rave) update_rave(s, p, d, won, chooser, have_trace);
        n = p;
    }
}

PuctResult puct_get_move(PuctSearch *s, const Game2 *g, int playouts, Rng *rng) {
    PuctResult r = { PASS, 0, 0 };
    s->n_nodes = 0;
    if (g->game_over) return r;
    if (g->consecutive_passes > 0 && g2_estimate_winner(g) == g->current) { r.root_win_ratio = 1; return r; }

    make_node(s, -1, -1, g);
    do {
        r.playouts++;
        bool do_playout;
        const int node = select_and_expand(s, g, rng, &do_playout);
        double value; bool trace = false;
        if (do_playout && !s->sim.game_over) { value = playout(s, &s->sim, rng); trace = true; }
        else value = g2_estimate_winner(&s->sim) == BLACK ? 1 : 0;   /* terminal: calcWinner */
        backpropagate(s, node, value, trace);
    } while (r.playouts < playouts);

    const Node *root = &s->nodes[0];
    int best = 0; double best_visits = -1, best_score = -DBL_MAX;
    for (int i = 0; i < root->m; i++) {
        const double cv = root->visits[i];
        if (cv > best_visits || (cv == best_visits && node_score(s, root, i, rng) > best_score)) {
            best_visits = cv;
            best_score = node_score(s, root, i, rng);
            best = i;
        }
    }
    double total_wins = 0;
    for (int i = 0; i < root->m; i++) total_wins += root->wins[i];
    r.root_win_ratio = total_wins / root->total_visits;
    /* As the JS, whose float wins never fall to the double PRIOR_WINS, so this never fires. */
    if (r.playouts >= RESIGN_MIN_PLAYOUTS && g->empty_count <= g->N * g->N / 2.0 && root->wins[best] <= PRIOR_WINS)
        return r;
    r.move = root->moves[best];
    return r;
}

int puct_root_stats(const PuctSearch *s, const int32_t **moves, const float **visits, const float **wins) {
    if (s->n_nodes == 0) return 0;
    const Node *root = &s->nodes[0];
    *moves = root->moves; *visits = root->visits; *wins = root->wins;
    return root->m;
}
