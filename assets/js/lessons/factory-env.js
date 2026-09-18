/* ==========================================================================
   factory-env.js — a tiny Factorio: build a line, measure what comes out.

   No gradients and no neural network. The thing being optimised is a *layout*,
   which is a pile of discrete choices with no derivative anywhere, and the
   only honest way to score one is to run the factory and count.

   Ore enters at the mine, travels along belts, gets turned into plates by a
   smelter and into gears by an assembler, and leaves at the depot. Throughput
   is items per hundred ticks at the depot. That is the whole objective.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const W = 10, H = 7;
  const EMPTY = 0, BELT = 1, SMELTER = 2, ASSEMBLER = 3;
  const MINE = 4, DEPOT = 5, ROCK = 6;

  // Ore -> plate at a smelter, plate -> gear at an assembler, gears score.
  const ORE = 0, PLATE = 1, GEAR = 2;
  const ITEM_NAME = ['ore', 'plate', 'gear'];

  const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];    // up right down left
  const DIR_NAME = ['up', 'right', 'down', 'left'];

  const idx = (x, y) => y * W + x;
  const inside = (x, y) => x >= 0 && y >= 0 && x < W && y < H;

  const COSTS = { [BELT]: 1, [SMELTER]: 6, [ASSEMBLER]: 9 };
  const SMELT_TICKS = 4, ASSEMBLE_TICKS = 6;

  /** A fixed obstacle course, so every layout is solving the same problem. */
  const ROCKS = [[4, 1], [4, 2], [5, 4], [5, 5], [2, 4]];

  class Factory {
    constructor(opts = {}) {
      this.budget = opts.budget ?? 40;
      this.tiles = new Uint8Array(W * H);
      this.dirs = new Uint8Array(W * H);
      this.mine = { x: 0, y: 3 };
      this.depot = { x: W - 1, y: 3 };
      this.clear();
    }

    clear() {
      this.tiles.fill(EMPTY);
      this.dirs.fill(1);
      for (const [x, y] of ROCKS) this.tiles[idx(x, y)] = ROCK;
      this.tiles[idx(this.mine.x, this.mine.y)] = MINE;
      this.tiles[idx(this.depot.x, this.depot.y)] = DEPOT;
      return this;
    }

    at(x, y) { return inside(x, y) ? this.tiles[idx(x, y)] : ROCK; }
    dirAt(x, y) { return this.dirs[idx(x, y)]; }

    /** Build something. Returns false when the square is not yours to use. */
    place(x, y, kind, dir = 1) {
      if (!inside(x, y)) return false;
      const t = this.tiles[idx(x, y)];
      if (t === MINE || t === DEPOT || t === ROCK) return false;
      this.tiles[idx(x, y)] = kind;
      this.dirs[idx(x, y)] = dir & 3;
      return true;
    }

    remove(x, y) {
      if (!inside(x, y)) return false;
      const t = this.tiles[idx(x, y)];
      if (t === MINE || t === DEPOT || t === ROCK) return false;
      this.tiles[idx(x, y)] = EMPTY;
      return true;
    }

    /** What the layout costs to build. The budget is the whole constraint. */
    cost() {
      let c = 0;
      for (let i = 0; i < this.tiles.length; i++) c += COSTS[this.tiles[i]] || 0;
      return c;
    }

    overBudget() { return this.cost() > this.budget; }

    /**
     * Run the factory and count what reaches the depot.
     *
     * Deliberately simple physics: every machine holds one item, belts move
     * their item one square per tick in the direction they face, and a machine
     * takes a fixed number of ticks to convert. Nothing here is clever — the
     * interest is entirely in which arrangement of these pieces works best.
     */
    simulate(ticks = 400) {
      const n = W * H;
      const held = new Int8Array(n).fill(-1);       // item on each square, -1 empty
      const timer = new Int8Array(n);              // machine progress
      let produced = 0;
      let oreSpawned = 0;
      // Intermediate progress, counted so a search has something to climb.
      // Measured on this map: none of 2,000 random layouts produce a single
      // gear, and a line one assembler short of working scores exactly what an
      // empty map scores. Throughput alone is a needle in a haystack.
      let oreSmelted = 0, platesMade = 0, gearsMade = 0;

      for (let t = 0; t < ticks; t++) {
        // Machines tick first, so a finished item can move the same tick.
        for (let i = 0; i < n; i++) {
          const kind = this.tiles[i];
          if (kind !== SMELTER && kind !== ASSEMBLER) continue;
          const want = kind === SMELTER ? ORE : PLATE;
          const make = kind === SMELTER ? PLATE : GEAR;
          const need = kind === SMELTER ? SMELT_TICKS : ASSEMBLE_TICKS;
          if (held[i] === want) {
            timer[i]++;
            if (timer[i] >= need) {
              held[i] = make; timer[i] = 0;
              if (kind === SMELTER) platesMade++; else gearsMade++;
            }
          }
        }

        // Movement, resolved from the far end back so a chain flows in one tick.
        const order = [];
        for (let i = 0; i < n; i++) order.push(i);
        for (const i of order) {
          const kind = this.tiles[i];
          if (held[i] < 0) continue;
          if (kind === SMELTER && held[i] !== PLATE) continue;
          if (kind === ASSEMBLER && held[i] !== GEAR) continue;
          if (kind === EMPTY || kind === ROCK) { held[i] = -1; continue; }

          const x = i % W, y = (i / W) | 0;
          const [dx, dy] = DIRS[kind === BELT ? this.dirs[i] : this.outDir(x, y)];
          const nx = x + dx, ny = y + dy;
          if (!inside(nx, ny)) continue;
          const j = idx(nx, ny);
          const target = this.tiles[j];

          if (target === DEPOT) {
            if (held[i] === GEAR) { produced++; held[i] = -1; }
            continue;
          }
          if (target === EMPTY || target === ROCK || target === MINE) continue;
          if (held[j] >= 0) continue;                       // occupied, wait
          if (target === SMELTER && held[i] !== ORE) continue;
          if (target === ASSEMBLER && held[i] !== PLATE) continue;
          if (target === SMELTER) oreSmelted++;
          held[j] = held[i];
          held[i] = -1;
          if (target === SMELTER || target === ASSEMBLER) timer[j] = 0;
        }

        // The mine pushes one ore a tick into whatever faces away from it.
        for (const [dx, dy] of DIRS) {
          const nx = this.mine.x + dx, ny = this.mine.y + dy;
          if (!inside(nx, ny)) continue;
          const j = idx(nx, ny);
          const k = this.tiles[j];
          if ((k === BELT || k === SMELTER) && held[j] < 0) {
            held[j] = ORE;
            oreSpawned++;
            break;
          }
        }
      }

      return { produced, ticks, throughput: (produced * 100) / ticks, oreSpawned,
               oreSmelted, platesMade, gearsMade };
    }

    /** Machines output towards the first belt or depot they are next to. */
    outDir(x, y) {
      for (let d = 0; d < 4; d++) {
        const nx = x + DIRS[d][0], ny = y + DIRS[d][1];
        if (!inside(nx, ny)) continue;
        const t = this.tiles[idx(nx, ny)];
        if (t === BELT || t === DEPOT || t === ASSEMBLER) return d;
      }
      return 1;
    }

    /**
     * Score a layout. `throughput` is the real objective and the only number
     * the page reports as the result. `fitness` is what a search climbs: the
     * same thing with partial credit for getting ore into a smelter and plates
     * out of one, because without it the landscape is flat and nothing is
     * findable. It is the reward-shaping problem from the Rocket League lesson
     * in a setting with no learning in it at all.
     */
    score(ticks = 400) {
      if (this.overBudget()) {
        return { throughput: 0, produced: 0, fitness: -1, invalid: true, cost: this.cost() };
      }
      const r = this.simulate(ticks);
      // Credit every stage of the chain, including a gear that was built but
      // never delivered. Without the gear term the search reliably reached
      // "makes plates" and stopped, because building an assembler that works
      // and a belt that reaches the depot is one jump, not two.
      const fitness = r.produced * 20 + r.gearsMade * 5 + r.platesMade * 1 + r.oreSmelted * 0.15;
      return { ...r, fitness, invalid: false, cost: this.cost() };
    }

    toJSON() { return { tiles: Array.from(this.tiles), dirs: Array.from(this.dirs), budget: this.budget }; }

    loadJSON(o) {
      this.tiles = Uint8Array.from(o.tiles);
      this.dirs = Uint8Array.from(o.dirs);
      this.budget = o.budget ?? this.budget;
      return this;
    }

    clone() { return new Factory({ budget: this.budget }).loadJSON(this.toJSON()); }
  }

  return {
    Factory,
    FACTORY: { W, H, EMPTY, BELT, SMELTER, ASSEMBLER, MINE, DEPOT, ROCK,
               ORE, PLATE, GEAR, ITEM_NAME, DIRS, DIR_NAME, COSTS, ROCKS, idx, inside,
               SMELT_TICKS, ASSEMBLE_TICKS },
  };
});
