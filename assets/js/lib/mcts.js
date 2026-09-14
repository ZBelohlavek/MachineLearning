/* ==========================================================================
   mcts.js — Monte Carlo tree search with a network in the loop (PUCT).

   This is the piece that makes AlphaZero different from everything else on
   this site. Nowhere else does a model *think* before acting: the Rocket
   League agent reads 24 numbers and reacts, but this one plays out hundreds
   of imagined continuations before it commits to a move.

   The network never learns to play by itself. It learns to guess what the
   search would have concluded, and the search then uses those guesses to
   look further ahead than it otherwise could. Each makes the other better.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /** A tree node. Statistics live on the parent, indexed by action. */
  function makeNode(nActions) {
    return {
      N: 0,                              // times this node was visited
      P: null,                           // prior from the network, over actions
      childN: new Float32Array(nActions),
      childW: new Float32Array(nActions),
      children: new Array(nActions).fill(null),
      expanded: false,
    };
  }

  /**
   * @param game  { nActions, legal(state), apply(state, a, player) -> token,
   *                undo(state, a, token), justWon(state, a, token),
   *                isDraw(state), other(player) }
   * @param evaluate  (state, player) -> { P: Float32Array, v: number }
   *                  Both from `player`'s point of view; v in [-1, 1].
   */
  class MCTS {
    constructor(game, evaluate, opts = {}) {
      this.game = game;
      this.evaluate = evaluate;
      this.cPuct = opts.cPuct ?? 1.4;
      this.rand = opts.rand || Math.random;
      this.root = null;
    }

    /** Mask the network's priors down to legal moves and renormalise. */
    maskPriors(P, legal) {
      const out = new Float32Array(P.length);
      let sum = 0;
      for (const a of legal) { out[a] = P[a]; sum += P[a]; }
      if (sum > 1e-8) { for (const a of legal) out[a] /= sum; }
      else { for (const a of legal) out[a] = 1 / legal.length; }
      return out;
    }

    /**
     * One simulation from `node`, with `board` already at that position.
     * Returns the value of the position from `player`'s point of view, and
     * leaves the board exactly as it found it.
     */
    simulate(node, board, player) {
      const g = this.game;

      if (!node.expanded) {
        const { P, v } = this.evaluate(board, player);
        node.P = this.maskPriors(P, g.legal(board));
        node.expanded = true;
        return v;
      }

      const legal = g.legal(board);
      if (!legal.length) return 0;

      // PUCT: trust the prior while a move is unexplored, trust the recorded
      // results once it has been tried. sqrt(N) is what makes it shift over.
      const sqrtN = Math.sqrt(Math.max(1, node.N));
      let best = legal[0], bestScore = -Infinity;
      for (const a of legal) {
        const n = node.childN[a];
        const q = n > 0 ? node.childW[a] / n : 0;
        const u = this.cPuct * node.P[a] * sqrtN / (1 + n);
        const score = q + u;
        if (score > bestScore) { bestScore = score; best = a; }
      }

      const token = g.apply(board, best, player);
      let value;
      if (g.justWon(board, best, token)) {
        value = 1;                        // the move we just made ended it
      } else if (g.isDraw(board)) {
        value = 0;
      } else {
        let child = node.children[best];
        if (!child) { child = makeNode(g.nActions); node.children[best] = child; }
        // The child's value is from the opponent's side, so it flips.
        value = -this.simulate(child, board, g.other(player));
      }
      g.undo(board, best, token);

      node.childN[best] += 1;
      node.childW[best] += value;
      node.N += 1;
      return value;
    }

    /**
     * Think about `board` for `sims` simulations.
     * Returns visit counts, the search policy, and the root value estimate.
     */
    search(board, player, sims, opts = {}) {
      const g = this.game;
      const root = makeNode(g.nActions);
      this.simulate(root, board, player);          // expand the root first

      if (opts.dirichlet) this.addNoise(root, g.legal(board), opts.dirichlet, opts.noiseFrac ?? 0.25);

      for (let i = 0; i < sims; i++) this.simulate(root, board, player);

      const visits = Float32Array.from(root.childN);
      let total = 0;
      for (let a = 0; a < g.nActions; a++) total += visits[a];
      const policy = new Float32Array(g.nActions);
      if (total > 0) for (let a = 0; a < g.nActions; a++) policy[a] = visits[a] / total;

      const value = root.N > 0 ? root.childW.reduce((s, w, a) =>
        s + (root.childN[a] > 0 ? w : 0), 0) / Math.max(1, total) : 0;

      this.root = root;
      return { visits, policy, value, root };
    }

    /**
     * Dirichlet noise on the root priors. Without it, self-play games all look
     * the same and the network only ever sees one opening. It is the only
     * exploration this algorithm has.
     */
    addNoise(node, legal, alpha, frac) {
      const noise = legal.map(() => this.gamma(alpha));
      const sum = noise.reduce((a, b) => a + b, 0) || 1;
      legal.forEach((a, i) => {
        node.P[a] = (1 - frac) * node.P[a] + frac * (noise[i] / sum);
      });
    }

    /** Marsaglia-Tsang gamma sampler, enough for Dirichlet noise. */
    gamma(alpha) {
      if (alpha < 1) return this.gamma(alpha + 1) * Math.pow(this.rand(), 1 / alpha);
      const d = alpha - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do { x = this.normal(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        const u = this.rand();
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    }

    normal() {
      let u = 0, v = 0;
      while (u === 0) u = this.rand();
      while (v === 0) v = this.rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
  }

  /**
   * Turn visit counts into a move. Temperature 1 samples in proportion to
   * visits, which keeps early self-play varied; temperature 0 always takes the
   * most-visited move, which is how you play a real game.
   */
  function pickMove(visits, temperature, rand = Math.random) {
    const n = visits.length;
    if (temperature <= 1e-3) {
      let best = -1, bestN = -1;
      for (let a = 0; a < n; a++) if (visits[a] > bestN) { bestN = visits[a]; best = a; }
      return best;
    }
    const w = new Float64Array(n);
    let sum = 0;
    for (let a = 0; a < n; a++) {
      w[a] = visits[a] > 0 ? Math.pow(visits[a], 1 / temperature) : 0;
      sum += w[a];
    }
    if (sum <= 0) return -1;
    let r = rand() * sum;
    for (let a = 0; a < n; a++) { r -= w[a]; if (r <= 0) return a; }
    return n - 1;
  }

  return { MCTS, pickMove, makeNode };
});
