/* ==========================================================================
   draft-train.js — learning who wins from ten picks and a result.

   The model never sees a synergy, a counter or a champion's strength. It sees
   a vector of +1 for allies, -1 for enemies, and one bit saying who won. Any
   structure it ends up with, it worked out.

   Two models on purpose. A linear one can only learn "this champion is good",
   because with one weight per champion there is nowhere to put "these two are
   good together". A network with a hidden layer can represent interactions.
   Since the data really is built out of interactions, the gap between them is
   the whole argument for depth, on a problem small enough to check.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const req = () => (typeof module === 'object' && module.exports
    ? { nn: require('../lib/nn.js'), env: require('./draft-env.js') }
    : { nn: root.ML, env: root.ML });
  const { nn, env } = req();
  const { MLP, mulberry32 } = nn;
  const { Roster, DRAFT } = env;

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  /* Settings found by sweeping, and each one earned its place. On six thousand
     matches, forty thousand steps, held-out loss where the run ended:
       finished drafts only, wd 1e-4   2.443   (training loss 0.101)
       finished drafts only, wd 1e-3   0.797
       unfinished drafts too, wd 1e-4  0.882
       unfinished drafts too, wd 1e-3  0.598
     Twenty thousand matches with both takes it to 0.530, against 0.607 for a
     model with one weight per champion. Depth is what makes an interaction
     representable; data is what makes it findable. */
  const DEFAULT_HP = {
    hidden: 32,
    lr: 0.005,
    batch: 64,
    wd: 1e-3,
    trainMatches: 20000,
    testMatches: 2000,
    prefixShare: 0.5,
  };

  class DraftModel {
    constructor(opts = {}) {
      this.roster = opts.roster || new Roster({ seed: opts.rosterSeed ?? 99 });
      this.hp = Object.assign({}, DEFAULT_HP, opts.hp);
      this.seed = opts.seed ?? 5;
      this.kind = opts.kind || 'deep';          // 'deep' | 'linear'
      this.reset(true);
    }

    reset(hard = false) {
      this.rand = mulberry32(this.seed);
      const n = this.roster.n;
      if (hard) {
        this.train = this.roster.matches(this.hp.trainMatches, this.rand);
        this.test = this.roster.matches(this.hp.testMatches, this.rand);
      }
      this.net = this.kind === 'linear'
        ? new MLP([n, 1], { out: 'linear', rand: this.rand, outScale: 0.1 })
        : new MLP([n, this.hp.hidden, this.hp.hidden, 1],
                  { hidden: 'relu', out: 'linear', rand: this.rand, outScale: 0.1 });
      this.steps = 0;
      this.epochs = 0;
      this.history = [];
      this.buf = new Float32Array(n);
    }

    /** Predicted win probability for team A. */
    predict(teamA, teamB) {
      this.roster.encode(teamA, teamB, this.buf);
      return sigmoid(this.net.forward(this.buf)[0]);
    }

    /**
     * One minibatch of logistic regression.
     *
     * Half the batch is finished drafts and half is positions part-way through
     * one, labelled with how that match ended. Without this the model only ever
     * sees five against five, and the page asks it to rate a pick at three
     * against three — a shape it has never been shown, where it will happily
     * report 98%. Training on the prefixes costs nothing and is what a real
     * draft-assist tool does, because the whole point is to help before the
     * draft is over.
     */
    trainBatch() {
      const d = new Float32Array(1);
      this.net.zeroGrad();
      let loss = 0;
      const n = this.hp.batch;
      for (let i = 0; i < n; i++) {
        const m = this.train[(this.rand() * this.train.length) | 0];
        let a = m.a, b = m.b;
        if (this.rand() < (this.hp.prefixShare ?? 0.5)) {
          // Somewhere between the first pick and the last.
          const k = 1 + ((this.rand() * DRAFT.TEAM_SIZE) | 0);
          const j = this.rand() < 0.5 ? k : k - 1;    // either side may be on the clock
          a = m.a.slice(0, k);
          b = m.b.slice(0, Math.max(1, j));
        }
        this.roster.encode(a, b, this.buf);
        const z = this.net.forward(this.buf)[0];
        const p = sigmoid(z);
        // binary cross-entropy: the gradient at the logit is simply p - y
        d[0] = (p - m.win) / n;
        this.net.backward(d);
        loss -= m.win ? Math.log(Math.max(1e-9, p)) : Math.log(Math.max(1e-9, 1 - p));
      }
      this.net.clipGradients(5);
      this.net.step(this.hp.lr, 1, this.hp.wd);
      this.steps++;
      this.lastLoss = loss / n;
      return this.lastLoss;
    }

    trainSlice(ms = 80) {
      const t0 = Date.now();
      let n = 0;
      while (Date.now() - t0 < ms) { this.trainBatch(); n++; }
      this.epochs = (this.steps * this.hp.batch) / this.train.length;
      return n;
    }

    /**
     * Score the model on held-out matches, and — because the truth is known
     * here and nowhere else on this site — on how closely its probabilities
     * track the real ones.
     */
    evaluate() {
      let loss = 0, correct = 0;
      const preds = [], truths = [];
      for (const m of this.test) {
        const p = this.predict(m.a, m.b);
        loss -= m.win ? Math.log(Math.max(1e-9, p)) : Math.log(Math.max(1e-9, 1 - p));
        if ((p >= 0.5) === (m.win === 1)) correct++;
        preds.push(p);
        truths.push(m.p);
      }
      const n = this.test.length;
      return {
        loss: loss / n,
        accuracy: correct / n,
        correlation: pearson(preds, truths),
        // How often the coin-flip baseline would be right, for comparison.
        baseline: Math.max(
          this.test.filter((m) => m.win === 1).length / n,
          this.test.filter((m) => m.win === 0).length / n),
      };
    }

    /**
     * Did it find the synergies? Compare what the model thinks a known
     * synergy pair is worth against what it thinks a random pair is worth.
     * The model was never told any pair exists.
     */
    synergyRecall(samples = 60) {
      const R = this.roster, n = R.n;
      const strong = [], plain = [];
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          (R.synergy[i * n + j] > 0.4 ? strong : plain).push([i, j]);
        }
      }
      if (!strong.length) return null;
      const measure = (pairs) => {
        let total = 0, count = 0;
        for (let k = 0; k < Math.min(samples, pairs.length); k++) {
          const [i, j] = pairs[(this.rand() * pairs.length) | 0];
          // a fixed nondescript opposition, so only the pair changes
          const enemy = [];
          for (let e = 0; e < DRAFT.TEAM_SIZE && enemy.length < DRAFT.TEAM_SIZE; e++) {
            const c = (e * 7 + 3) % n;
            if (c !== i && c !== j) enemy.push(c);
          }
          const filler = [];
          for (let f = 0; filler.length < DRAFT.TEAM_SIZE - 2; f++) {
            const c = (f * 5 + 11) % n;
            if (c !== i && c !== j && !enemy.includes(c)) filler.push(c);
          }
          total += this.predict([i, j].concat(filler), enemy);
          count++;
        }
        return total / Math.max(1, count);
      };
      return { withSynergy: measure(strong), withoutSynergy: measure(plain), pairs: strong.length };
    }

    /** The raw logit, before the squash. Interactions are additive here. */
    logit(teamA, teamB) {
      this.roster.encode(teamA, teamB, this.buf);
      return this.net.forward(this.buf)[0];
    }

    /**
     * The interaction term for a pair, measured the way you would measure one
     * in an experiment: hold everything else fixed, and take the difference of
     * differences.
     *
     *   f(both) - f(only i) - f(only j) + f(neither)
     *
     * Everything that depends on i alone cancels, and so does everything that
     * depends on j alone. What survives is exactly the part of the prediction
     * that needs both champions at once. For a model with one weight per
     * champion this is zero — not small, not noisy, algebraically zero — which
     * is the cleanest way to see what a linear model gives up.
     */
    interaction(i, j, stage) {
      const [u, v] = stage.neutral;
      const z = (a, b) => this.logit([a, b].concat(stage.filler), stage.enemy);
      return z(i, j) - z(i, v) - z(u, j) + z(u, v);
    }

    /** The same quantity from the hidden truth, for comparison. */
    trueInteraction(i, j, stage) {
      const R = this.roster;
      const [u, v] = stage.neutral;
      const z = (a, b) => R.advantage([a, b].concat(stage.filler), stage.enemy);
      return z(i, j) - z(i, v) - z(u, j) + z(u, v);
    }

    /**
     * The champions least entangled with anything else, least first. These make
     * the stand-ins and the supporting cast for a measurement, because they
     * smuggle in no interactions of their own.
     */
    _quiet() {
      if (this._quietCache) return this._quietCache;
      const R = this.roster, n = R.n;
      const ties = [];
      for (let c = 0; c < n; c++) {
        let t = 0;
        for (let k = 0; k < n; k++) {
          t += Math.abs(R.synergy[c * n + k]) + Math.abs(R.synergy[k * n + c])
             + Math.abs(R.counter[c * n + k]) + Math.abs(R.counter[k * n + c]);
        }
        ties.push([c, t]);
      }
      ties.sort((a, b) => a[1] - b[1]);
      this._quietCache = ties.slice(0, 12).map((t) => t[0]);
      return this._quietCache;
    }

    /**
     * A fixed, boring stage for one pair: two stand-ins, three team-mates and
     * five opponents, all drawn from the quiet champions and none of them the
     * pair itself. Built per pair so that every pair in the roster can be
     * measured, rather than only those outside one fixed cast.
     */
    _stageFor(i, j) {
      const pool = this._quiet().filter((c) => c !== i && c !== j);
      return { neutral: [pool[0], pool[1]], filler: pool.slice(2, 5), enemy: pool.slice(5, 10) };
    }

    /**
     * Measure the interaction the model reports for pairs that really do work
     * together, against pairs that do not. Nothing ever told it a pair exists.
     */
    interactions() {
      const R = this.roster, n = R.n;
      const strong = [], plain = [];
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const stage = this._stageFor(i, j);
          if (stage.enemy.length < DRAFT.TEAM_SIZE) continue;
          const truth = this.trueInteraction(i, j, stage);
          const model = this.interaction(i, j, stage);
          (R.synergy[i * n + j] > 0.4 ? strong : plain).push({ i, j, truth, model });
        }
      }
      const mean = (xs) => xs.reduce((t, x) => t + x.model, 0) / Math.max(1, xs.length);
      const all = strong.concat(plain);
      return {
        strong: mean(strong),
        plain: mean(plain),
        pairs: strong.length,
        compared: all.length,
        // Across every pair, does the size of the model's interaction track the
        // size of the real one?
        correlation: pearson(all.map((x) => x.model), all.map((x) => x.truth)),
        // The largest interactions it reports, with what they are really worth.
        top: all.slice().sort((a, b) => b.model - a.model).slice(0, 6)
              .map((x) => ({ i: x.i, j: x.j, model: x.model, truth: x.truth,
                             real: R.synergy[x.i * R.n + x.j] > 0.4 })),
      };
    }

    /** The model's favourite remaining pick, for drafting against a human. */
    bestPick(teamA, teamB, taken) {
      let best = -1, bestP = -Infinity;
      for (let i = 0; i < this.roster.n; i++) {
        if (taken.has(i)) continue;
        const p = this.predict(teamA.concat([i]), teamB);
        if (p > bestP) { bestP = p; best = i; }
      }
      return { pick: best, prob: bestP };
    }
  }

  function pearson(a, b) {
    const n = a.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
      const x = a[i] - ma, y = b[i] - mb;
      num += x * y; da += x * x; db += y * y;
    }
    return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
  }

  return { DraftModel, DRAFT_HP: DEFAULT_HP, pearson };
});
