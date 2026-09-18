/* ==========================================================================
   factory-search.js — optimising something with no gradient in it.

   A layout is a pile of discrete choices. There is no derivative to follow, no
   loss to differentiate, and most random arrangements produce literally
   nothing — a flat landscape of zero with a few narrow ridges in it. Gradient
   descent has nothing to grip. What works instead is the oldest idea in
   optimisation: change something at random, keep it if it helped, and early on
   sometimes keep it even when it didn't.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const req = () => (typeof module === 'object' && module.exports
    ? { env: require('./factory-env.js'), nn: require('../lib/nn.js') }
    : { env: root.ML, nn: root.ML });
  const { env, nn } = req();
  const { Factory, FACTORY: F } = env;
  const { mulberry32 } = nn;

  const DEFAULT = {
    ticks: 300,           // how long each candidate factory is run for
    temp0: 3.0,           // starting temperature
    cooling: 0.9995,      // per-step multiplier
    restartAfter: 1200,   // steps without improvement before starting over
  };

  class FactorySearch {
    constructor(opts = {}) {
      this.opts = Object.assign({}, DEFAULT, opts);
      this.seed = opts.seed ?? 3;
      this.budget = opts.budget ?? 40;
      this.reset();
    }

    reset() {
      this.rand = mulberry32(this.seed);
      this.current = new Factory({ budget: this.budget });
      this.currentScore = this.evaluate(this.current);
      this.best = this.current.clone();
      this.bestScore = this.currentScore;
      this.steps = 0;
      this.sinceImproved = 0;
      this.temp = this.opts.temp0;
      this.restarts = 0;
      this.history = [];
      this.accepted = 0;
      this.acceptedWorse = 0;
      return this;
    }

    /** What the search climbs: shaped fitness, not raw throughput. */
    evaluate(f) {
      const s = f.score(this.opts.ticks);
      return s.invalid ? -1 : s.fitness;
    }

    /** The real objective, for reporting. */
    throughputOf(f) {
      const s = f.score(this.opts.ticks);
      return s.invalid ? 0 : s.throughput;
    }

    /** One random edit: build, rotate or clear a square. */
    mutate(f) {
      const r = this.rand();
      const x = (this.rand() * F.W) | 0;
      const y = (this.rand() * F.H) | 0;
      // Refuse a placement that would bust the budget rather than proposing a
      // candidate that is guaranteed invalid; nearly half of naive random
      // layouts were being thrown away for this alone.
      if (r < 0.45) {
        this.placeWithRoom(f, x, y, F.BELT, (this.rand() * 4) | 0);
      } else if (r < 0.6) {
        this.placeWithRoom(f, x, y, F.SMELTER, 1);
      } else if (r < 0.72) {
        this.placeWithRoom(f, x, y, F.ASSEMBLER, 1);
      } else if (r < 0.88) {
        f.remove(x, y);
      } else if (r < 0.94) {
        // rotate an existing belt rather than replacing it
        if (f.at(x, y) === F.BELT) f.dirs[F.idx(x, y)] = (this.rand() * 4) | 0;
      } else {
        this.extendLine(f);
      }
      return f;
    }

    /**
     * Build something, clearing belts elsewhere until it fits the budget.
     *
     * This is the move that unstuck the whole search. Refusing placements that
     * did not fit meant the layout filled up with cheap belts and could then
     * never afford the assembler it needed: the search reliably reached "makes
     * plates, cost exactly at budget, no assembler" and stopped there forever.
     * A structural change has to be allowed to pay for itself.
     */
    placeWithRoom(f, x, y, kind, dir) {
      const prev = f.at(x, y);
      const prevDir = f.dirAt(x, y);
      if (!f.place(x, y, kind, dir)) return;
      let guard = 0;
      while (f.overBudget() && guard++ < 40) {
        const cx = (this.rand() * F.W) | 0, cy = (this.rand() * F.H) | 0;
        if (cx === x && cy === y) continue;
        if (f.at(cx, cy) === F.BELT) f.remove(cx, cy);
      }
      if (f.overBudget()) {                      // could not make room; put it back
        if (prev === F.EMPTY) f.remove(x, y); else f.place(x, y, prev, prevDir);
      }
    }

    /**
     * A move that knows what a factory is: find an existing piece and put a
     * belt next to it pointing onwards towards the depot.
     *
     * Adding this mattered more than any tuning of the annealing schedule.
     * With only blind single-square edits the search reliably learned to make
     * plates and then stalled, because the remaining gap — an assembler plus a
     * belt run to the depot — is several coordinated changes away and every
     * one of them on its own scores nothing. The lesson is that the moves you
     * allow matter more than the algorithm that chooses between them.
     */
    extendLine(f) {
      const pieces = [];
      for (let y = 0; y < F.H; y++) {
        for (let x = 0; x < F.W; x++) {
          const t = f.at(x, y);
          if (t === F.BELT || t === F.SMELTER || t === F.ASSEMBLER || t === F.MINE) pieces.push([x, y]);
        }
      }
      if (!pieces.length) return;
      const [px, py] = pieces[(this.rand() * pieces.length) | 0];
      // step towards the depot, with a little noise so it can route round rocks
      const dx = f.depot.x - px, dy = f.depot.y - py;
      let d;
      if (this.rand() < 0.75) d = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0);
      else d = (this.rand() * 4) | 0;
      const nx = px + F.DIRS[d][0], ny = py + F.DIRS[d][1];
      if (!F.inside(nx, ny)) return;
      if (f.cost() + F.COSTS[F.BELT] > f.budget && f.at(nx, ny) !== F.BELT) return;
      f.place(nx, ny, F.BELT, d);
    }

    /**
     * One annealing step. Early on, a change that makes things worse is often
     * accepted anyway — which is the only way out of a layout that is locally
     * fine and globally mediocre. As the temperature falls it stops doing that
     * and settles.
     */
    step() {
      const candidate = this.current.clone();
      const edits = 1 + ((this.rand() * 2) | 0);
      for (let i = 0; i < edits; i++) this.mutate(candidate);

      const score = this.evaluate(candidate);
      const delta = score - this.currentScore;
      let accept = delta >= 0;
      if (!accept && this.temp > 1e-6) {
        accept = this.rand() < Math.exp(delta / this.temp);
        if (accept) this.acceptedWorse++;
      }

      if (accept) {
        this.current = candidate;
        this.currentScore = score;
        this.accepted++;
      }

      if (score > this.bestScore) {
        this.best = candidate.clone();
        this.bestScore = score;
        this.sinceImproved = 0;
      } else {
        this.sinceImproved++;
      }

      this.steps++;
      this.temp *= this.opts.cooling;

      // A run that has stopped getting anywhere is better spent starting again
      // from the best layout found so far, with the temperature turned back up.
      if (this.sinceImproved > this.opts.restartAfter) {
        // Alternate between resuming from the best layout found and starting
        // again from bare ground. Always resuming from the best keeps the
        // search in whatever basin it first fell into: on one seed that meant
        // eighty thousand steps of polishing a layout that could never make a
        // gear. A clean restart is the only way out of a bad basin.
        this.restarts++;
        if (this.restarts % 3 === 0) {
          this.current = new Factory({ budget: this.budget });
          this.currentScore = this.evaluate(this.current);
          this.temp = this.opts.temp0;
        } else {
          this.current = this.best.clone();
          this.currentScore = this.bestScore;
          this.temp = this.opts.temp0 * 0.6;
        }
        this.sinceImproved = 0;
      }
      return this.currentScore;
    }

    run(ms = 100) {
      const t0 = Date.now();
      let n = 0;
      while (Date.now() - t0 < ms) { this.step(); n++; }
      this.history.push({ steps: this.steps, best: this.bestScore });
      return n;
    }
  }

  /**
   * The control: try random layouts and keep the best. Worth running, because
   * it shows how much of the search's result is the search and how much is
   * simply having tried a lot of things.
   */
  function randomSearch(tries, budget, ticks, seed = 5) {
    const rand = mulberry32(seed);
    const search = new FactorySearch({ seed, budget, ticks });
    let best = -1, bestLayout = null;
    for (let t = 0; t < tries; t++) {
      const f = new Factory({ budget });
      const pieces = 6 + ((rand() * 14) | 0);
      for (let i = 0; i < pieces; i++) {
        const x = (rand() * F.W) | 0, y = (rand() * F.H) | 0;
        const r = rand();
        f.place(x, y, r < 0.7 ? F.BELT : r < 0.85 ? F.SMELTER : F.ASSEMBLER, (rand() * 4) | 0);
      }
      const s = search.evaluate(f);
      if (s > best) { best = s; bestLayout = f; }
    }
    // Report the real objective, not the shaped fitness the search climbed.
    // Printing fitness here and calling it throughput made random search look
    // like it was beating the annealer when it had produced nothing at all.
    const throughput = bestLayout ? search.throughputOf(bestLayout) : 0;
    return { best: throughput, fitness: best, layout: bestLayout };
  }

  /** A sensible human line: mine, belt, smelter, belt, assembler, belt, depot. */
  function handBuilt(budget = 40) {
    const f = new Factory({ budget });
    for (let x = 1; x < 3; x++) f.place(x, 3, F.BELT, 1);
    f.place(3, 3, F.SMELTER);
    for (let x = 4; x < 6; x++) f.place(x, 3, F.BELT, 1);
    f.place(6, 3, F.ASSEMBLER);
    for (let x = 7; x < 9; x++) f.place(x, 3, F.BELT, 1);
    return f;
  }

  return { FactorySearch, randomSearch, handBuilt, FACTORY_DEFAULT: DEFAULT };
});
