/* ==========================================================================
   gametheory.js — regret matching, best response, and exploitability.

   Everything else on this site learns to be good. This learns to be
   *unpredictable*, which is a different goal and needs different machinery.
   In a game where your opponent adapts, there is no single best move — only a
   best mixture, and the algorithm that finds it is regret matching, the engine
   inside the poker programs that beat professionals.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /* The mixup, as a payoff matrix. Rows are what you do, columns are what they
     do, and the number is what you gain. It is antisymmetric — what you gain
     they lose — which makes it zero-sum and gives it exactly one value.

               strikeH strikeL throw  blockH blockL   */
  const ACTIONS = [
    { id: 'strikeH', label: 'Strike high', hint: 'beaten by a high block' },
    { id: 'strikeL', label: 'Strike low',  hint: 'beaten by a low block' },
    { id: 'throw',   label: 'Throw',       hint: 'beats any block, loses to any strike' },
    { id: 'blockH',  label: 'Block high',  hint: 'stops a high strike, loses to a throw' },
    { id: 'blockL',  label: 'Block low',   hint: 'stops a low strike, loses to a throw' },
  ];

  /* Balanced so that all five options are worth playing. An earlier version
     made a blocked strike cost only 1, which made striking almost free: the
     equilibrium collapsed to 50/50 high-low strikes and nobody ever blocked or
     threw. Punishing a blocked strike properly is what brings the other three
     back into the mix, which is also why real fighting games do it. */
  const PAYOFF = [
    [ 0,  0,  2, -3,  2],
    [ 0,  0,  2,  2, -3],
    [-2, -2,  0,  2,  2],
    [ 3, -2, -2,  0,  0],
    [-2,  3, -2,  0,  0],
  ];

  const N = ACTIONS.length;

  /** Expected payoff to the row player when two mixtures meet. */
  function value(rowMix, colMix) {
    let v = 0;
    for (let i = 0; i < N; i++) {
      if (!rowMix[i]) continue;
      for (let j = 0; j < N; j++) v += rowMix[i] * colMix[j] * PAYOFF[i][j];
    }
    return v;
  }

  /** What each of your actions is worth against a given opponent mixture. */
  function actionValues(colMix, out) {
    const v = out && out.length === N ? out : new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let j = 0; j < N; j++) s += colMix[j] * PAYOFF[i][j];
      v[i] = s;
    }
    return v;
  }

  /** The single best reply to a mixture, and what it earns. */
  function bestResponse(colMix) {
    const v = actionValues(colMix);
    let best = 0;
    for (let i = 1; i < N; i++) if (v[i] > v[best]) best = i;
    return { action: best, value: v[best], values: v };
  }

  /**
   * How much a perfect reader would earn against this mixture, per round.
   *
   * Zero means unexploitable: whatever they do, they break even. This is the
   * number that matters in a game against an adapting opponent, and it is not
   * the same as "am I winning right now".
   */
  function exploitability(mix) {
    return bestResponse(mix).value;
  }

  /**
   * Regret matching. After every round, ask of each action: how much better
   * off would I have been had I always played that instead? Those differences
   * accumulate as regret, and the next mixture is the regrets, normalised.
   *
   * The subtle and important part: the CURRENT mixture bounces around and is
   * often quite exploitable. It is the AVERAGE over all rounds that converges
   * to the unexploitable strategy, which is why the agent plays from its
   * running average rather than from what it most recently felt like doing.
   */
  class RegretMatcher {
    constructor(opts = {}) {
      this.regret = new Float64Array(N);
      this.sumStrategy = new Float64Array(N);
      this.rounds = 0;
      this.rand = opts.rand || Math.random;
      // Discounting lets it track an opponent who changes, at the cost of
      // never quite settling. Zero means remember everything, which is what
      // converges to equilibrium.
      this.discount = opts.discount ?? 0;
    }

    /** The mixture implied by current regrets: positive regret, normalised. */
    strategy(out) {
      const s = out && out.length === N ? out : new Float64Array(N);
      let total = 0;
      for (let i = 0; i < N; i++) { s[i] = Math.max(0, this.regret[i]); total += s[i]; }
      if (total > 1e-12) { for (let i = 0; i < N; i++) s[i] /= total; }
      else { for (let i = 0; i < N; i++) s[i] = 1 / N; }
      return s;
    }

    /** The running average of every mixture played. This is what converges. */
    average(out) {
      const s = out && out.length === N ? out : new Float64Array(N);
      let total = 0;
      for (let i = 0; i < N; i++) total += this.sumStrategy[i];
      if (total > 1e-12) { for (let i = 0; i < N; i++) s[i] = this.sumStrategy[i] / total; }
      else { for (let i = 0; i < N; i++) s[i] = 1 / N; }
      return s;
    }

    sample(mix) {
      let r = this.rand();
      for (let i = 0; i < N; i++) { r -= mix[i]; if (r <= 0) return i; }
      return N - 1;
    }

    /** Play one round: draw from the current mixture, recording it. */
    act() {
      const s = this.strategy();
      for (let i = 0; i < N; i++) this.sumStrategy[i] += s[i];
      this.rounds++;
      return { action: this.sample(s), strategy: s };
    }

    /** Learn from what the opponent did. `mine` is the action actually taken. */
    observe(mine, theirs) {
      const got = PAYOFF[mine][theirs];
      if (this.discount > 0) {
        for (let i = 0; i < N; i++) this.regret[i] *= (1 - this.discount);
      }
      for (let i = 0; i < N; i++) this.regret[i] += PAYOFF[i][theirs] - got;
    }

    reset() {
      this.regret.fill(0);
      this.sumStrategy.fill(0);
      this.rounds = 0;
    }
  }

  /**
   * The other kind of opponent: one that simply counts what you have done and
   * plays the single best answer to it. Brutal against anyone with a habit,
   * and completely exploitable itself, because a pure strategy always is.
   */
  class Reader {
    constructor(opts = {}) {
      this.counts = new Float64Array(N).fill(opts.prior ?? 1);
      this.decay = opts.decay ?? 0.97;      // recent rounds matter more
      this.rand = opts.rand || Math.random;
      this.noise = opts.noise ?? 0.1;       // a little randomness, or it is trivially baited
    }

    model(out) {
      const s = out && out.length === N ? out : new Float64Array(N);
      let total = 0;
      for (let i = 0; i < N; i++) total += this.counts[i];
      for (let i = 0; i < N; i++) s[i] = this.counts[i] / total;
      return s;
    }

    act() {
      const m = this.model();
      const br = bestResponse(m);
      if (this.rand() < this.noise) return { action: (this.rand() * N) | 0, model: m, target: br.action };
      return { action: br.action, model: m, target: br.action };
    }

    observe(_mine, theirs) {
      for (let i = 0; i < N; i++) this.counts[i] *= this.decay;
      this.counts[theirs] += 1;
    }

    reset() { this.counts.fill(1); }
  }

  /**
   * Find the equilibrium by letting two regret matchers argue for a while.
   * Returns the average strategy and how exploitable it still is.
   */
  function solve(iterations = 20000, seedRand) {
    const a = new RegretMatcher({ rand: seedRand || Math.random });
    const b = new RegretMatcher({ rand: seedRand || Math.random });
    for (let t = 0; t < iterations; t++) {
      const x = a.act(), y = b.act();
      a.observe(x.action, y.action);
      b.observe(y.action, x.action);
    }
    const mix = a.average();
    return { mix, exploitability: exploitability(mix), rounds: iterations };
  }

  return { GT: { ACTIONS, PAYOFF, N, value, actionValues, bestResponse,
                 exploitability, RegretMatcher, Reader, solve } };
});
