/*
 * rave.c — RAVE-MCTS engine (C port of ai/rave.js).
 */
#include "rave.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

/* ── Node allocation (chunked) ─────────────────────────────────────────────── */

static RaveNode *alloc_node(RaveState *s) {
    if (s->cur_used >= RAVE_CHUNK_SIZE) {
        /* Need a new chunk — reuse next in chain or allocate */
        if (s->cur->next == NULL) {
            RaveChunk *c = malloc(sizeof(RaveChunk));
            if (!c) { fprintf(stderr, "rave: out of memory\n"); exit(1); }
            c->next = NULL;
            s->cur->next = c;
        }
        s->cur = s->cur->next;
        s->cur_used = 0;
    }
    RaveNode *n = &s->cur->nodes[s->cur_used++];
    s->total_used++;
    memset(n, 0, sizeof(RaveNode));
    return n;
}

/* ── Get legal moves ───────────────────────────────────────────────────────── */

static int get_legal_moves(const Game2 *g, int32_t *out) {
    int n = 0;
    for (int i = 0; i < CAP; i++) {
        if (g->cells[i] != EMPTY) continue;
        if (g2_is_true_eye(g, i)) continue;
        if (g2_is_legal(g, i)) out[n++] = i;
    }
    if (n < CAP / 3 || g->consecutive_passes > 0)
        out[n++] = PASS;
    return n;
}

/* ── Make node ─────────────────────────────────────────────────────────────── */

static RaveNode *make_node(RaveState *s, int32_t move, RaveNode *parent,
                           int32_t ci, const Game2 *game) {
    RaveNode *n = alloc_node(s);

    n->move   = move;
    n->parent = parent;
    n->ci     = ci;
    n->mover  = -game->current;  /* player who made `move` */
    n->total_visits = 0.1f;
    n->selected_child = -1;

    n->num_moves = get_legal_moves(game, n->legal_moves);
    int M = n->num_moves;

    for (int i = 0; i < M; i++) {
        n->children[i] = NULL;
        n->wins[i]     = RAVE_PRIOR_WINS;
        n->visits[i]   = RAVE_PRIOR_VISITS;
    }

    if (parent == NULL || parent->parent == NULL) {
        for (int i = 0; i < CAP; i++) {
            n->rave_wins[i]   = RAVE_PRIOR_WINS;
            n->rave_visits[i] = RAVE_PRIOR_VISITS;
        }
    } else {
        const RaveNode *gp = parent->parent;
        for (int i = 0; i < CAP; i++) {
            n->rave_wins[i]   = RAVE_INHERIT * gp->rave_wins[i];
            n->rave_visits[i] = RAVE_INHERIT * gp->rave_visits[i];
        }
    }

    return n;
}

/* ── UCB score ─────────────────────────────────────────────────────────────── */

static float ucb_score(int move_idx, const RaveNode *node) {
    int32_t move = node->legal_moves[move_idx];

    float rave_wr = (move == PASS) ? 0.0f
        : (node->rave_wins[move] / node->rave_visits[move]);

    float real_w  = node->wins[move_idx];
    float real_v  = node->visits[move_idx];
    float real_wr = real_w / real_v;

    float rave_weight = RAVE_K / (RAVE_K + real_v);
    float wr = (1.0f - rave_weight) * real_wr + rave_weight * rave_wr;

    return wr + 0.001f * g2_randf()
         + RAVE_EXPLORATION_C * sqrtf(logf(node->total_visits) / real_v);
}

/* ── Playout (tracked) ─────────────────────────────────────────────────────── */

static int8_t play_tracked(Game2 *g, float *played) {
    memset(played, 0, sizeof(float) * CAP);

    int move_limit = 3 * g->empty_count + 20;
    float weight_step = 1.0f / CAP;
    int moves = 0;
    float weight = 1.0f;

    while (!g->game_over && moves < move_limit) {
        int8_t current = g->current;
        int32_t idx = g2_random_legal_move(g);
        if (idx == PASS) { g2_play(g, PASS); moves++; continue; }
        if (played[idx] == 0.0f)
            played[idx] = (current == BLACK) ? weight : -weight;
        g2_play(g, idx);
        moves++;
        weight -= weight_step;
    }

    return g2_estimate_winner(g);
}

/* ── Select and expand ─────────────────────────────────────────────────────── */

typedef struct {
    RaveNode *node;
    Game2     game;
} SelectResult;

static SelectResult select_and_expand(RaveState *s, RaveNode *root, const Game2 *root_game) {
    RaveNode *node = root;
    Game2 game;
    g2_clone(&game, root_game);

    while (!game.game_over) {
        int M = node->num_moves;
        if (M == 0) break;

        /* Select best child */
        int best = 0;
        float best_score = -FLT_MAX;
        for (int i = 0; i < M; i++) {
            float sc = ucb_score(i, node);
            if (sc > best_score) { best_score = sc; best = i; }
        }

        g2_play(&game, node->legal_moves[best]);

        /* Promote if enough visits */
        if (node->children[best] == NULL && node->visits[best] >= RAVE_N_EXPAND)
            node->children[best] = make_node(s, node->legal_moves[best], node, best, &game);

        /* Force second pass to end game */
        if (!game.game_over && game.consecutive_passes > 0) {
            g2_play(&game, PASS);
            node->selected_child = best;
            break;
        }

        /* Descend into promoted child */
        if (node->children[best] != NULL) {
            node = node->children[best];
            node->selected_child = -1;
            continue;
        }

        /* Unpromoted leaf */
        node->selected_child = best;
        break;
    }

    return (SelectResult){ node, game };
}

/* ── Backpropagate ─────────────────────────────────────────────────────────── */

static void update_rave(RaveNode *node, float won, const float *played, int8_t chooser) {
    if (chooser == BLACK) {
        for (int k = 0; k < CAP; k++) {
            float w = played[k];
            if (w > 0.0f) {
                node->rave_visits[k] += w;
                node->rave_wins[k]   += won * w;
            }
        }
    } else {
        for (int k = 0; k < CAP; k++) {
            float w = played[k];
            if (w < 0.0f) {
                node->rave_visits[k] -= w;
                node->rave_wins[k]   -= won * w;
            }
        }
    }
}

static void backpropagate(RaveNode *node, int8_t winner, const float *played) {
    /* Update leaf child stats */
    int leaf = node->selected_child;
    if (leaf != -1) {
        int8_t chooser = -node->mover;
        float won = (winner == chooser) ? 1.0f : 0.0f;
        node->visits[leaf]++;
        node->wins[leaf] += won;
        node->total_visits++;
        update_rave(node, won, played, chooser);
    }

    /* Walk up the tree */
    while (node->parent != NULL) {
        int ci = node->ci;
        int8_t chooser = -node->parent->mover;
        float won = (winner == chooser) ? 1.0f : 0.0f;
        node->parent->visits[ci]++;
        node->parent->wins[ci] += won;
        node->parent->total_visits++;
        update_rave(node->parent, won, played, chooser);
        node = node->parent;
    }
}

/* ── Public API ────────────────────────────────────────────────────────────── */

RaveState *rave_create(void) {
    RaveState *s = malloc(sizeof(RaveState));
    if (!s) { fprintf(stderr, "rave: out of memory\n"); exit(1); }
    RaveChunk *c = malloc(sizeof(RaveChunk));
    if (!c) { fprintf(stderr, "rave: out of memory\n"); exit(1); }
    c->next = NULL;
    s->head = c;
    s->cur = c;
    s->cur_used = 0;
    s->total_used = 0;
    return s;
}

void rave_destroy(RaveState *s) {
    if (!s) return;
    RaveChunk *c = s->head;
    while (c) { RaveChunk *next = c->next; free(c); c = next; }
    free(s);
}

RaveResult rave_search(RaveState *s, const Game2 *root_game,
                       int playout_limit, int time_ms) {
    /* Reset pool — reuse existing chunks, no free/alloc */
    s->cur = s->head;
    s->cur_used = 0;
    s->total_used = 0;

    if (root_game->game_over)
        return (RaveResult){ PASS, 0.5f, 0 };

    /* Obvious pass: opponent just passed and we're winning */
    if (root_game->consecutive_passes > 0 &&
        g2_estimate_winner(root_game) == root_game->current)
        return (RaveResult){ PASS, 1.0f, 0 };

    RaveNode *root = make_node(s, -1, NULL, -1, root_game);

    int playouts = 0;
    if (playout_limit > 0) {
        for (int p = 0; p < playout_limit; p++) {
            SelectResult sr = select_and_expand(s, root, root_game);
            int8_t winner = play_tracked(&sr.game, s->played);
            backpropagate(sr.node, winner, s->played);
            playouts++;
        }
    } else {
        clock_t deadline = clock() + (clock_t)((double)time_ms / 1000.0 * CLOCKS_PER_SEC);
        do {
            SelectResult sr = select_and_expand(s, root, root_game);
            int8_t winner = play_tracked(&sr.game, s->played);
            backpropagate(sr.node, winner, s->played);
            playouts++;
        } while (clock() < deadline);
    }

    /* Best child: most visits, ties broken by UCB score */
    int M = root->num_moves;
    int best_idx = 0;
    float best_visits = -1.0f, best_score = -FLT_MAX;
    for (int i = 0; i < M; i++) {
        float cv = root->visits[i];
        if (cv > best_visits || (cv == best_visits && ucb_score(i, root) > best_score)) {
            best_visits = cv;
            best_score  = ucb_score(i, root);
            best_idx    = i;
        }
    }

    float total_wins = 0;
    for (int i = 0; i < M; i++) total_wins += root->wins[i];
    float win_ratio = total_wins / root->total_visits;

    /* Give up if no winning line in late game. Only reliable with enough playouts. */
    if (playouts >= RAVE_RESIGN_MIN_PLAYOUTS &&
        root_game->empty_count <= CAP / 2 &&
        root->wins[best_idx] <= RAVE_PRIOR_WINS)
        return (RaveResult){ PASS, win_ratio, playouts };

    return (RaveResult){ root->legal_moves[best_idx], win_ratio, playouts };
}
