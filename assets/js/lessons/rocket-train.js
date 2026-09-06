/* ==========================================================================
   rocket-train.js — the learning half of the Rocket League lesson.

   Algorithm: PPO (Proximal Policy Optimisation), the workhorse of modern
   game-playing RL, written out longhand:

     1. SELF-PLAY   Both cars are driven by the same network. Play a batch of
                    episodes, sampling actions from the policy and recording
                    (state, action, reward, log-probability).
     2. ADVANTAGE   Ask the value network what it expected from each state and
                    compute GAE(lambda) — a smoothed "was that better or worse
                    than expected?" signal for every single decision.
     3. IMPROVE     Take several small gradient steps that make good actions
                    more likely, but *clip* the update so the policy can never
                    lurch too far from the one that collected the data.

   Everything is plain JS. You can read the entire algorithm below.
   ========================================================================== */

/* Loaded either as a plain <script> (exports land on window.ML) or via
   require() in Node for the test suite — so the pages work from file:// too. */
;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';
  const DEPS = (typeof module === 'object' && module.exports)
    ? Object.assign({}, require('../lib/nn.js'), require('./rocket-env.js')) : root.ML;
  const { MLP, softmaxInto, sampleFrom, argmax, entropy, mulberry32 } = DEPS;
  const { Arena, ACTIONS, OBS_SIZE, DEFAULT_REWARDS, scriptedAction } = DEPS;

  const DEFAULT_HP = {
    hidden: 64,           // units per hidden layer in the policy
    lr: 0.004,            // policy + value learning rate (Adam)
    gamma: 0.99,          // discount: how much the future is worth
    lambda: 0.95,         // GAE smoothing
    clip: 0.2,            // PPO trust region
    entropyBonus: 0.008,  // pressure to keep exploring
    epochs: 3,            // passes over each batch of experience
    minibatch: 256,
    batchEpisodes: 8,
    actionRepeat: 2,      // decisions at 15 Hz over 30 Hz physics
    targetKL: 0.035,      // stop early if the policy moved too far
    freezeEvery: 20,      // refresh rate of the "past self" opponent
    curriculum: true,     // start easy (ball nearby), widen as the agent improves
  };

  class Trainer {
    constructor(opts = {}) {
      this.hp = { ...DEFAULT_HP, ...(opts.hp || {}) };
      this.rewards = { ...DEFAULT_REWARDS, ...(opts.rewards || {}) };
      this.opponent = opts.opponent || 'self';   // 'self' | 'past' | 'scripted' | 'none'
      this.rand = mulberry32(opts.seed ?? 20240115);
      this.pbuf = new Float32Array(ACTIONS.length);
      this.dLogits = new Float32Array(ACTIONS.length);
      this.buildNets();
      this.arena = new Arena({ rewards: this.rewards, seed: 12345 });
      this.reset(false);
    }

    buildNets() {
      const h = this.hp.hidden;
      // outScale keeps initial logits near zero => a near-uniform starting policy.
      // The agent begins genuinely undecided rather than stuck on one action.
      this.policy = new MLP([OBS_SIZE, h, h, ACTIONS.length],
        { hidden: 'tanh', out: 'linear', outScale: 0.05 });
      this.value = new MLP([OBS_SIZE, 48, 48, 1], { hidden: 'tanh', out: 'linear' });
      this.frozen = MLP.fromJSON(this.policy.toJSON());
    }

    reset(hard = true) {
      if (hard) this.buildNets();
      this.episodes = 0;
      this.updates = 0;
      this.steps = 0;
      this.history = [];
      this.recent = { reward: [], goals: [], scoredAny: [], touches: [], length: [] };
      this.spread = this.hp.curriculum ? 0.3 : 1;
      this.lastTermSums = null;
      this.lastStats = null;
      this.clearBuffer();
    }

    clearBuffer() {
      this.buf = { obs: [], act: [], rew: [], logp: [], trajs: [] };
    }

    setRewards(r) { this.rewards = { ...r }; this.arena.rewards = this.rewards; }

    act(obs, greedy = false, net = this.policy) {
      const probs = softmaxInto(net.forward(obs), this.pbuf);
      return { action: greedy ? argmax(probs) : sampleFrom(probs, () => this.rand()), probs };
    }

    /* ----------------------------------------------------------------------
       1. SELF-PLAY — roll out one episode into the buffer.
       ---------------------------------------------------------------------- */
    runEpisode() {
      const a = this.arena;
      const hp = this.hp;
      a.solo = this.opponent === 'none';
      a.rewards = this.rewards;
      a.spread = this.spread;
      a.reset();

      const learners = this.opponent === 'self' ? [0, 1] : [0];
      const segs = {};
      for (const i of learners) segs[i] = [];
      const obsBuf = [new Float32Array(OBS_SIZE), new Float32Array(OBS_SIZE)];

      let epReward = 0, decisions = 0;

      while (!a.done) {
        const actions = [4, 4];
        for (let i = 0; i < 2; i++) {
          if (i === 1 && a.solo) continue;
          if (i === 1 && this.opponent === 'scripted') { actions[1] = scriptedAction(a, 1); continue; }
          const net = (i === 1 && this.opponent === 'past') ? this.frozen : this.policy;
          a.observe(i, obsBuf[i]);
          const probs = softmaxInto(net.forward(obsBuf[i]), this.pbuf);
          const act = sampleFrom(probs, () => this.rand());
          actions[i] = act;
          if (segs[i]) {
            segs[i].push(this.buf.obs.length);
            this.buf.obs.push(Float32Array.from(obsBuf[i]));
            this.buf.act.push(act);
            this.buf.logp.push(Math.log(Math.max(probs[act], 1e-9)));
            this.buf.rew.push(0);
          }
        }

        // Action repeat: hold the chosen action for a few physics ticks. Fewer,
        // chunkier decisions make the credit assignment problem much easier.
        let r0 = 0, r1 = 0;
        for (let k = 0; k < hp.actionRepeat && !a.done; k++) {
          const res = a.step(actions);
          r0 += res.rewards[0]; r1 += res.rewards[1];
        }
        const stepRew = [r0, r1];
        for (const i of learners) this.buf.rew[segs[i][segs[i].length - 1]] = stepRew[i];
        epReward += r0;
        decisions++;
      }

      // Remember the final state so we can bootstrap V(s_T) when an episode
      // merely ran out of clock instead of ending in a goal.
      const terminal = a.scorer >= 0;
      for (const i of learners) {
        this.buf.trajs.push({
          idx: segs[i],
          final: terminal ? null : Float32Array.from(a.observe(i, obsBuf[i])),
          terminal,
        });
      }

      this.episodes++;
      this.steps += a.step_;
      const goal = a.scorer === 0 ? 1 : a.scorer === 1 ? -1 : 0;
      push(this.recent.reward, epReward, 150);
      push(this.recent.goals, goal, 150);
      push(this.recent.scoredAny, a.scorer >= 0 ? 1 : 0, 150);
      push(this.recent.touches, a.touches[0], 150);
      push(this.recent.length, a.step_, 150);
      this.lastTermSums = a.termSums[0];
      return { epReward, goal, decisions };
    }

    /* ----------------------------------------------------------------------
       2 + 3. ADVANTAGE and IMPROVE — one PPO update over the whole batch.
       ---------------------------------------------------------------------- */
    update() {
      const { obs, act, rew, logp, trajs } = this.buf;
      const N = obs.length;
      if (N < 2) return null;
      const hp = this.hp;

      // --- value of every visited state, under the current critic ---
      const V = new Float64Array(N);
      for (let i = 0; i < N; i++) V[i] = this.value.forward(obs[i])[0];

      // --- GAE(lambda): a low-variance estimate of "how much better than
      //     expected did things turn out after this action?" ---
      const adv = new Float64Array(N);
      const ret = new Float64Array(N);
      for (const tr of trajs) {
        // Timeouts are not real endings: ask the critic what the rest is worth.
        let nextV = tr.terminal ? 0 : this.value.forward(tr.final)[0];
        let gae = 0;
        for (let k = tr.idx.length - 1; k >= 0; k--) {
          const i = tr.idx[k];
          const delta = rew[i] + hp.gamma * nextV - V[i];
          gae = delta + hp.gamma * hp.lambda * gae;
          adv[i] = gae;
          ret[i] = gae + V[i];
          nextV = V[i];
        }
      }

      // Normalise advantages so the step size doesn't depend on the reward scale.
      let mean = 0;
      for (let i = 0; i < N; i++) mean += adv[i];
      mean /= N;
      let vsum = 0;
      for (let i = 0; i < N; i++) vsum += (adv[i] - mean) ** 2;
      const std = Math.sqrt(vsum / N) + 1e-8;
      for (let i = 0; i < N; i++) adv[i] = (adv[i] - mean) / std;

      // --- several small, clipped gradient steps over shuffled minibatches ---
      const order = new Int32Array(N);
      for (let i = 0; i < N; i++) order[i] = i;

      let meanEnt = 0, meanKL = 0, clipFrac = 0, vLoss = 0, nSeen = 0, epochsRun = 0;
      const dL = this.dLogits;

      outer:
      for (let epoch = 0; epoch < hp.epochs; epoch++) {
        shuffle(order, this.rand);
        let epKL = 0, epSeen = 0;
        for (let mb = 0; mb < N; mb += hp.minibatch) {
          const end = Math.min(N, mb + hp.minibatch);
          const size = end - mb;
          this.policy.zeroGrad();
          this.value.zeroGrad();

          for (let s = mb; s < end; s++) {
            const i = order[s];
            const probs = softmaxInto(this.policy.forward(obs[i]), this.pbuf);
            const a = act[i];
            const lp = Math.log(Math.max(probs[a], 1e-9));
            const ratio = Math.exp(lp - logp[i]);          // pi_new / pi_old
            const A = adv[i];
            const H = entropy(probs);

            // PPO's clipped objective: if this action already became much more
            // (or much less) likely than when we collected the data, stop pushing.
            const clipped = (A > 0 && ratio > 1 + hp.clip) || (A < 0 && ratio < 1 - hp.clip);
            if (clipped) clipFrac++;
            const coef = clipped ? 0 : ratio * A;

            for (let j = 0; j < dL.length; j++) {
              // d/dlogits of -coef*log pi(a) is coef*(p_j - onehot_j)
              const pg = coef * (probs[j] - (j === a ? 1 : 0));
              // entropy bonus (maximised, hence the sign)
              const eg = hp.entropyBonus * probs[j] * (Math.log(Math.max(probs[j], 1e-9)) + H);
              dL[j] = pg + eg;
            }
            this.policy.backward(dL);

            const v = this.value.forward(obs[i])[0];
            const dv = v - ret[i];
            this.value.backward(Float32Array.of(dv));

            meanEnt += H; vLoss += 0.5 * dv * dv; nSeen++;
            const kl = logp[i] - lp;                        // approximate KL
            epKL += kl; epSeen++;
          }

          this.policy.clipGradients(size * 0.5);
          this.policy.step(hp.lr, 1 / size);
          this.value.clipGradients(size * 2);
          this.value.step(hp.lr * 1.5, 1 / size);
        }
        epochsRun++;
        meanKL = Math.abs(epKL / Math.max(1, epSeen));
        // Early stop: a policy that has already moved a long way should not keep
        // training on stale data.
        if (meanKL > hp.targetKL) break outer;
      }

      this.updates++;

      /* Curriculum: once the agent reliably finds the ball, spawn it further
         away. Growing the problem with the learner is far more reliable than
         throwing it at the hardest version on step one. */
      if (this.hp.curriculum && this.spread < 1) {
        const t = avg(this.recent.touches);
        if (t > 3.5) this.spread = Math.min(1, this.spread + 0.02);
      } else if (!this.hp.curriculum) this.spread = 1;

      if (this.opponent === 'past' && this.updates % hp.freezeEvery === 0) {
        this.frozen.loadJSON(this.policy.toJSON());
      }

      const row = {
        update: this.updates,
        episodes: this.episodes,
        reward: avg(this.recent.reward),
        goalRate: avg(this.recent.goals),
        scoreRate: avg(this.recent.scoredAny),
        spread: this.spread,
        touches: avg(this.recent.touches),
        length: avg(this.recent.length),
        entropy: meanEnt / Math.max(1, nSeen),
        valueLoss: vLoss / Math.max(1, nSeen),
        kl: meanKL,
        clipFrac: clipFrac / Math.max(1, nSeen),
        epochs: epochsRun,
        samples: N,
      };
      this.history.push(row);
      this.lastStats = row;
      this.clearBuffer();
      return row;
    }

    /** Train for at most `budgetMs` milliseconds. Returns any updates produced. */
    trainSlice(budgetMs) {
      const t0 = now();
      const rows = [];
      const perBatch = this.hp.batchEpisodes * (this.opponent === 'self' ? 2 : 1);
      let guard = 0;
      do {
        this.runEpisode();
        if (this.buf.trajs.length >= perBatch) {
          const row = this.update();
          if (row) rows.push(row);
        }
      } while (now() - t0 < budgetMs && ++guard < 500);
      return rows;
    }

    /* ----------------------------------------------------------------------
       Evaluation: greedy play against a fixed opponent. Training reward can be
       gamed by reward shaping; goals against a scripted bot cannot.
       ---------------------------------------------------------------------- */
    evaluate(nEpisodes = 12, opponent = 'scripted') {
      const arena = new Arena({ rewards: this.rewards, seed: 4242 });
      arena.solo = opponent === 'none';
      let scored = 0, conceded = 0, touches = 0, steps = 0;
      const obs = new Float32Array(OBS_SIZE);
      for (let e = 0; e < nEpisodes; e++) {
        arena.reset();
        while (!arena.done) {
          const actions = [4, 4];
          arena.observe(0, obs);
          actions[0] = argmax(softmaxInto(this.policy.forward(obs), this.pbuf));
          if (!arena.solo) {
            if (opponent === 'scripted') actions[1] = scriptedAction(arena, 1);
            else {
              arena.observe(1, obs);
              actions[1] = argmax(softmaxInto(this.policy.forward(obs), this.pbuf));
            }
          }
          for (let k = 0; k < this.hp.actionRepeat && !arena.done; k++) { arena.step(actions); steps++; }
        }
        if (arena.scorer === 0) scored++;
        else if (arena.scorer === 1) conceded++;
        touches += arena.touches[0];
      }
      return {
        scored, conceded, episodes: nEpisodes,
        scoreRate: scored / nEpisodes,
        touchesPerEp: touches / nEpisodes,
        avgLength: steps / nEpisodes,
      };
    }

    toJSON() {
      return {
        version: 2, policy: this.policy.toJSON(), value: this.value.toJSON(),
        hp: this.hp, rewards: this.rewards, episodes: this.episodes, updates: this.updates,
      };
    }

    loadJSON(obj) {
      this.hp = { ...this.hp, ...obj.hp };
      this.policy = MLP.fromJSON(obj.policy);
      this.value = MLP.fromJSON(obj.value);
      this.frozen = MLP.fromJSON(obj.policy);
      this.rewards = { ...this.rewards, ...obj.rewards };
      this.episodes = obj.episodes || 0;
      this.updates = obj.updates || 0;
      return this;
    }
  }

  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  function push(arr, v, max) { arr.push(v); if (arr.length > max) arr.shift(); }
  function avg(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
  function shuffle(arr, rand) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = (rand() * (i + 1)) | 0;
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
  }

  return { DEFAULT_HP, Trainer };
});
