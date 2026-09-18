/* ==========================================================================
   kitchen-train.js — two cooks, one brain, one score.

   Both cooks run the same network. That is the cheap and standard way to train
   a cooperative pair, and it works extremely well right up until you put the
   result next to somebody it has never met.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const req = () => (typeof module === 'object' && module.exports
    ? { nn: require('../lib/nn.js'), env: require('./kitchen-env.js') }
    : { nn: root.ML, env: root.ML });
  const { nn, env } = req();
  const { MLP, mulberry32, softmax, sampleFrom, argmax } = nn;
  const { Kitchen, KITCHEN: K } = env;

  const DEFAULT_HP = {
    hidden: 64,
    lr: 0.008,
    ticks: 200,             // one shift
    batch: 8,               // episodes per update
    gamma: 0.97,
    // Measured: with the entropy bonus off entirely, this policy commits early
    // and never serves a single soup in five thousand episodes. With it, the
    // same settings reach the scripted pair's score. Exploration is not a
    // nicety here — the reward chain is six steps long.
    entropy: 0.004,
    shapedWeight: 1,
  };

  class KitchenTrainer {
    constructor(opts = {}) {
      this.hp = Object.assign({}, DEFAULT_HP, opts.hp);
      this.seed = opts.seed ?? 1;
      this.reset(true);
    }

    reset(hard = false) {
      if (hard) {
        this.rand = mulberry32(this.seed);
        this.policy = new MLP([K.OBS, this.hp.hidden, this.hp.hidden, K.NUM_ACTIONS],
                              { hidden: 'relu', out: 'linear', rand: this.rand, outScale: 0.1 });
      }
      this.kitchen = new Kitchen({ maxTicks: this.hp.ticks });
      this.episodes = 0;
      this.updates = 0;
      this.baseline = 0;
      this.history = [];
      this.obs = new Float32Array(K.OBS);
      this.lastServed = 0;
    }

    /** Sample an action for cook `i` from a given policy. */
    act(policy, kitchen, i, greedy = false) {
      kitchen.observe(i, this.obs);
      const logits = policy.forward(this.obs);
      const p = softmax(logits);
      return greedy ? argmax(p) : sampleFrom(p, this.rand);
    }

    /**
     * One shift, recording everything needed for a policy-gradient update.
     * Both cooks contribute to the same set of gradients, because both are the
     * same network — which is exactly the assumption that causes trouble later.
     */
    rollout() {
      const k = this.kitchen.reset();
      const steps = [];
      let served = 0;
      for (let t = 0; t < this.hp.ticks; t++) {
        const actions = [];
        const recs = [];
        for (let i = 0; i < 2; i++) {
          k.observe(i, this.obs);
          const x = Float32Array.from(this.obs);
          const logits = this.policy.forward(x);
          const p = softmax(logits);
          const a = sampleFrom(p, this.rand);
          actions.push(a);
          recs.push({ x, a, p: Float32Array.from(p) });
        }
        const res = k.step(actions, this.rand);
        served += res.served;
        const r = (k.shaped[0] + k.shaped[1]) * this.hp.shapedWeight;
        for (const rec of recs) { rec.r = r / 2; steps.push(rec); }
      }
      this.episodes++;
      this.lastServed = served;
      return { steps, served, collisions: k.collisions };
    }

    /** REINFORCE with a moving-average baseline and an entropy bonus. */
    trainBatch() {
      this.policy.zeroGrad();
      let totalServed = 0, totalReturn = 0, n = 0;

      for (let b = 0; b < this.hp.batch; b++) {
        const { steps, served } = this.rollout();
        totalServed += served;

        // discounted return from each step to the end of the shift
        let running = 0;
        const returns = new Float64Array(steps.length);
        for (let i = steps.length - 1; i >= 0; i--) {
          running = steps[i].r + this.hp.gamma * running;
          returns[i] = running;
        }
        for (let i = 0; i < steps.length; i++) { totalReturn += returns[i]; n++; }

        const d = new Float32Array(K.NUM_ACTIONS);
        const scale = 1 / (steps.length * this.hp.batch);
        for (let i = 0; i < steps.length; i++) {
          const s = steps[i];
          const adv = returns[i] - this.baseline;
          this.policy.forward(s.x);

          // Entropy of this step's distribution, needed for its own gradient.
          let H = 0;
          for (let a = 0; a < K.NUM_ACTIONS; a++) {
            if (s.p[a] > 1e-8) H -= s.p[a] * Math.log(s.p[a]);
          }
          for (let a = 0; a < K.NUM_ACTIONS; a++) {
            // policy gradient, written for descent: (p - onehot) * advantage
            const grad = (s.p[a] - (a === s.a ? 1 : 0)) * adv;
            // dH/dlogit_a = -p_a (log p_a + H); we are maximising H, so the
            // loss gradient is the negative of that. An earlier version used
            // (log p + 1) and no H, which pushed the wrong way and measurably
            // slowed learning down.
            const ent = this.hp.entropy * s.p[a] * (Math.log(Math.max(1e-8, s.p[a])) + H);
            d[a] = (grad + ent) * scale;
          }
          this.policy.backward(d);
        }
      }

      this.baseline = 0.9 * this.baseline + 0.1 * (totalReturn / Math.max(1, n));
      this.policy.clipGradients(5);
      this.policy.step(this.hp.lr);
      this.updates++;
      const avgServed = totalServed / this.hp.batch;
      this.history.push({ episodes: this.episodes, served: avgServed });
      return avgServed;
    }

    trainSlice(ms = 120) {
      const t0 = Date.now();
      let n = 0;
      while (Date.now() - t0 < ms) { this.trainBatch(); n++; }
      return n;
    }

    /**
     * Run a shift with an arbitrary pair of cooks and report the score.
     * `cook` entries are either a policy network or a function(kitchen, i).
     */
    playWith(cookA, cookB, opts = {}) {
      const k = new Kitchen({ maxTicks: opts.ticks ?? this.hp.ticks });
      k.reset();
      const pick = (who, i) => {
        if (typeof who === 'function') return who(k, i, this.rand);
        k.observe(i, this.obs);
        const p = softmax(who.forward(this.obs));
        return opts.greedy ? argmax(p) : sampleFrom(p, this.rand);
      };
      for (let t = 0; t < k.maxTicks; t++) k.step([pick(cookA, 0), pick(cookB, 1)], this.rand);
      return { served: k.served, collisions: k.collisions };
    }

    /** Average score over several shifts, because one shift is noisy. */
    evaluate(cookA, cookB, runs = 6, opts = {}) {
      let served = 0, collisions = 0;
      for (let r = 0; r < runs; r++) {
        const res = this.playWith(cookA, cookB, opts);
        served += res.served; collisions += res.collisions;
      }
      return { served: served / runs, collisions: collisions / runs, runs };
    }

    toJSON() { return { hp: this.hp, episodes: this.episodes, net: this.policy.toJSON() }; }

    loadJSON(obj) {
      this.policy = MLP.fromJSON(obj.net);
      this.episodes = obj.episodes || 0;
      return this;
    }
  }

  return { KitchenTrainer, KITCHEN_HP: DEFAULT_HP };
});
