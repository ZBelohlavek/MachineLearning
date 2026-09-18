/* ==========================================================================
   draft-env.js — a champion roster with a secret.

   Every other lesson learns something whose right answer nobody knows. Here I
   write the right answer first: each champion gets a hidden solo strength, a
   hidden synergy with certain team-mates and a hidden counter against certain
   opponents. Matches are then simulated from those hidden numbers.

   That makes this the only lesson on the site where you can check the model's
   homework. A network trained on match results has no access to any of it — it
   sees ten champion picks and a win or a loss — and afterwards you can ask
   whether it recovered the structure that actually generated the data.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const req = () => (typeof module === 'object' && module.exports
    ? require('../lib/nn.js') : root.ML);
  const { mulberry32 } = req();

  const ROLES = ['Fighter', 'Mage', 'Support', 'Marksman', 'Assassin'];
  const NAMES = [
    'Ashvane', 'Brambel', 'Corvex', 'Duskmaw', 'Emberly', 'Fenrik',
    'Gloamis', 'Hexley', 'Ironwood', 'Jadefall', 'Kestrel', 'Lumen',
    'Morrigan', 'Nyxara', 'Orvid', 'Pyrrha', 'Quill', 'Rooksbane',
    'Sablewing', 'Thornquist', 'Umbriel', 'Vandry', 'Wrenlow', 'Zephyra',
  ];
  const TEAM_SIZE = 5;

  class Roster {
    constructor(opts = {}) {
      this.seed = opts.seed ?? 99;
      this.rand = mulberry32(this.seed);
      this.n = opts.size ?? NAMES.length;
      this.build();
    }

    build() {
      const r = this.rand;
      this.champions = [];
      for (let i = 0; i < this.n; i++) {
        this.champions.push({
          id: i,
          name: NAMES[i % NAMES.length],
          role: ROLES[i % ROLES.length],
          // Solo strength is deliberately small compared with the interactions,
          // so a model that only learns "this champion is good" cannot do well.
          power: (r() - 0.5) * 0.8,
        });
      }
      // Synergy: pairs on the same team that are worth more together.
      this.synergy = new Float64Array(this.n * this.n);
      // Counter: how much champion i gains when j is on the other side.
      this.counter = new Float64Array(this.n * this.n);
      for (let i = 0; i < this.n; i++) {
        for (let j = i + 1; j < this.n; j++) {
          const s = r() < 0.16 ? (r() - 0.35) * 2.2 : 0;
          this.synergy[i * this.n + j] = s;
          this.synergy[j * this.n + i] = s;
        }
        for (let j = 0; j < this.n; j++) {
          if (i === j) continue;
          if (r() < 0.12) {
            const c = r() * 1.6;
            this.counter[i * this.n + j] = c;
            this.counter[j * this.n + i] = -c;     // a counter is someone else's problem
          }
        }
      }
    }

    /**
     * The truth: how much better team A is than team B, before luck.
     * Nothing outside this file is allowed to look at it.
     */
    advantage(teamA, teamB) {
      let a = 0;
      for (const i of teamA) a += this.champions[i].power;
      for (const j of teamB) a -= this.champions[j].power;
      for (let x = 0; x < teamA.length; x++) {
        for (let y = x + 1; y < teamA.length; y++) a += this.synergy[teamA[x] * this.n + teamA[y]];
      }
      for (let x = 0; x < teamB.length; x++) {
        for (let y = x + 1; y < teamB.length; y++) a -= this.synergy[teamB[x] * this.n + teamB[y]];
      }
      for (const i of teamA) for (const j of teamB) a += this.counter[i * this.n + j];
      return a;
    }

    /** True win probability for team A, via a logistic on the advantage. */
    winProb(teamA, teamB) {
      return 1 / (1 + Math.exp(-this.advantage(teamA, teamB)));
    }

    /**
     * Generate matches. Teams are drafted with a bias towards champions that
     * are individually strong, because real match data is never a uniform
     * sample of compositions — people pick what they think is good, and that
     * skew is part of what a model has to cope with.
     */
    matches(count, rand = this.rand) {
      const out = [];
      const weights = this.champions.map((c) => Math.exp(c.power * 1.2));
      const pick = (exclude) => {
        for (let guard = 0; guard < 200; guard++) {
          let total = 0;
          for (let i = 0; i < this.n; i++) if (!exclude.has(i)) total += weights[i];
          let r = rand() * total;
          for (let i = 0; i < this.n; i++) {
            if (exclude.has(i)) continue;
            r -= weights[i];
            if (r <= 0) return i;
          }
        }
        return 0;
      };
      for (let m = 0; m < count; m++) {
        const used = new Set();
        const a = [], b = [];
        for (let k = 0; k < TEAM_SIZE; k++) {
          const x = pick(used); used.add(x); a.push(x);
          const y = pick(used); used.add(y); b.push(y);
        }
        const p = this.winProb(a, b);
        out.push({ a, b, win: rand() < p ? 1 : 0, p });
      }
      return out;
    }

    /** One-hot of both teams: +1 for an ally, -1 for an enemy. */
    encode(teamA, teamB, out) {
      const v = out && out.length === this.n ? out : new Float32Array(this.n);
      v.fill(0);
      for (const i of teamA) v[i] = 1;
      for (const j of teamB) v[j] = -1;
      return v;
    }

    /** The strongest remaining pick, by the truth. Used to grade the model. */
    bestPick(teamA, teamB, taken) {
      let best = -1, bestP = -Infinity;
      for (let i = 0; i < this.n; i++) {
        if (taken.has(i)) continue;
        const p = this.winProb(teamA.concat([i]), teamB);
        if (p > bestP) { bestP = p; best = i; }
      }
      return { pick: best, prob: bestP };
    }
  }

  return { Roster, DRAFT: { ROLES, NAMES, TEAM_SIZE } };
});
