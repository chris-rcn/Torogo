'use strict';

/**
 * Always-pass policy.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - ignored (always fast)
 */
module.exports = function getMove(_game, _timeBudgetMs) {
  return { type: 'pass' };
};
