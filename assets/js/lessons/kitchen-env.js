/* ==========================================================================
   kitchen-env.js — a very small Overcooked.

   Two cooks share one kitchen and one score. Nobody is competing: the only way
   to lose is to get in each other's way. That makes it a completely different
   problem from every other game on this site, where there was always someone
   to beat.

   The loop: fetch an onion, put it in the pot, wait for the soup, carry a
   plate to the pot, then carry the soup to the hatch. Five steps, two people,
   one narrow kitchen.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const W = 7, H = 5;
  const FLOOR = 0, COUNTER = 1, ONIONS = 2, PLATES = 3, POT = 4, HATCH = 5;

  // A deliberately awkward layout: one corridor, with the onions at one end and
  // the hatch at the other, so the two cooks must pass each other constantly.
  const LAYOUT = [
    1, 1, 1, 4, 1, 1, 1,
    2, 0, 0, 0, 0, 0, 5,
    1, 0, 0, 0, 0, 0, 1,
    3, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1,
  ];

  const HOLD = { NOTHING: 0, ONION: 1, PLATE: 2, SOUP: 3 };
  const HOLD_NAME = ['nothing', 'an onion', 'a plate', 'the soup'];

  // up, right, down, left, wait, interact
  const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0], null];
  const NUM_ACTIONS = 6;
  const INTERACT = 5;

  const COOK_TIME = 12;        // ticks the pot takes once it has enough onions
  const NEEDED = 3;            // onions per soup

  const idx = (x, y) => y * W + x;

  class Kitchen {
    constructor(opts = {}) {
      this.tiles = LAYOUT.slice();
      this.maxTicks = opts.maxTicks ?? 400;
      this.reset();
    }

    reset() {
      this.cooks = [
        { x: 2, y: 1, hold: HOLD.NOTHING },
        { x: 4, y: 3, hold: HOLD.NOTHING },
      ];
      this.potOnions = 0;
      this.potTimer = 0;          // counts down once full
      this.potReady = false;
      this.tick = 0;
      this.served = 0;
      this.shaped = [0, 0];       // per-cook shaped reward, for training
      this.collisions = 0;
      return this;
    }

    tileAt(x, y) {
      if (x < 0 || y < 0 || x >= W || y >= H) return COUNTER;
      return this.tiles[idx(x, y)];
    }

    walkable(x, y) { return this.tileAt(x, y) === FLOOR; }

    /** The station a cook is facing into, if they are standing next to one. */
    facing(cook) {
      // Interacting works on any adjacent station, which keeps the controls
      // simple: there is never more than one useful thing beside you here.
      const around = [[0, -1], [1, 0], [0, 1], [-1, 0]];
      for (const [dx, dy] of around) {
        const t = this.tileAt(cook.x + dx, cook.y + dy);
        if (t === ONIONS || t === PLATES || t === POT || t === HATCH) {
          return { tile: t, x: cook.x + dx, y: cook.y + dy };
        }
      }
      return null;
    }

    /**
     * One tick. Both cooks act simultaneously; if they want the same square,
     * neither gets it. That single rule is where all the coordination lives.
     */
    step(actions, rand = Math.random) {
      const before = this.served;
      this.shaped = [0, 0];

      const want = this.cooks.map((c, i) => {
        const a = actions[i];
        if (a === INTERACT || a === 4) return { x: c.x, y: c.y };
        const [dx, dy] = MOVES[a] || [0, 0];
        const nx = c.x + dx, ny = c.y + dy;
        return this.walkable(nx, ny) ? { x: nx, y: ny } : { x: c.x, y: c.y };
      });

      const sameSquare = want[0].x === want[1].x && want[0].y === want[1].y;
      const swapping = want[0].x === this.cooks[1].x && want[0].y === this.cooks[1].y &&
                       want[1].x === this.cooks[0].x && want[1].y === this.cooks[0].y;
      if (sameSquare || swapping) {
        // One of them gets through, chosen at random. Blocking both was the
        // first version and it deadlocked instantly: two cooks with the same
        // idea froze nose to nose for the whole shift. Letting one pass keeps
        // the cost of a clash real without making it terminal.
        this.collisions++;
        const winner = rand() < 0.5 ? 0 : 1;
        this.cooks[winner].x = want[winner].x;
        this.cooks[winner].y = want[winner].y;
      } else {
        for (let i = 0; i < 2; i++) { this.cooks[i].x = want[i].x; this.cooks[i].y = want[i].y; }
      }

      for (let i = 0; i < 2; i++) if (actions[i] === INTERACT) this.interact(i);

      if (this.potOnions >= NEEDED && !this.potReady) {
        this.potTimer--;
        if (this.potTimer <= 0) this.potReady = true;
      }

      this.tick++;
      return { served: this.served - before, done: this.tick >= this.maxTicks };
    }

    /** Pick something up, put something down, or serve. */
    interact(i) {
      const cook = this.cooks[i];
      const station = this.facing(cook);
      if (!station) return;

      if (station.tile === ONIONS && cook.hold === HOLD.NOTHING) {
        cook.hold = HOLD.ONION;
        this.shaped[i] += 0.1;
      } else if (station.tile === PLATES && cook.hold === HOLD.NOTHING) {
        cook.hold = HOLD.PLATE;
        this.shaped[i] += 0.1;
      } else if (station.tile === POT) {
        if (cook.hold === HOLD.ONION && this.potOnions < NEEDED && !this.potReady) {
          cook.hold = HOLD.NOTHING;
          this.potOnions++;
          this.shaped[i] += 1;
          if (this.potOnions === NEEDED) this.potTimer = COOK_TIME;
        } else if (cook.hold === HOLD.PLATE && this.potReady) {
          cook.hold = HOLD.SOUP;
          this.potOnions = 0;
          this.potReady = false;
          this.shaped[i] += 2;
        }
      } else if (station.tile === HATCH && cook.hold === HOLD.SOUP) {
        cook.hold = HOLD.NOTHING;
        this.served++;
        this.shaped[i] += 10;
      }
    }

    /**
     * What a cook sees: its own situation, its partner's, and the pot. Written
     * from that cook's point of view so one policy can play either role — the
     * same trick as the Rocket League agent and the Connect 4 encoder.
     */
    observe(i, out) {
      const o = out && out.length === OBS ? out : new Float32Array(OBS);
      const me = this.cooks[i], you = this.cooks[1 - i];
      let k = 0;
      o[k++] = me.x / (W - 1); o[k++] = me.y / (H - 1);
      o[k++] = you.x / (W - 1); o[k++] = you.y / (H - 1);
      o[k++] = (you.x - me.x) / (W - 1); o[k++] = (you.y - me.y) / (H - 1);
      for (let h = 0; h < 4; h++) o[k++] = me.hold === h ? 1 : 0;
      for (let h = 0; h < 4; h++) o[k++] = you.hold === h ? 1 : 0;
      o[k++] = this.potOnions / NEEDED;
      o[k++] = this.potReady ? 1 : 0;
      o[k++] = this.potOnions >= NEEDED && !this.potReady ? this.potTimer / COOK_TIME : 0;
      const st = this.facing(me);
      for (let t = 0; t < 6; t++) o[k++] = st && st.tile === t ? 1 : 0;
      o[k++] = this.tick / this.maxTicks;
      return o;
    }
  }

  const OBS = 2 + 2 + 2 + 4 + 4 + 3 + 6 + 1;   // 24

  /**
   * A cook that just follows the recipe. It is not clever and it does not look
   * at its partner, which is the point: it stands in for a person who knows
   * what they are doing and expects the same of everyone else.
   */
  function scriptedCook(kitchen, i, rand = Math.random) {
    const me = kitchen.cooks[i];
    const station = kitchen.facing(me);

    const mate = kitchen.cooks[1 - i];

    let target;
    if (me.hold === HOLD.SOUP) target = HATCH;
    else if (me.hold === HOLD.PLATE) target = kitchen.potReady ? POT : PLATES;
    else if (me.hold === HOLD.ONION) target = POT;
    else if (kitchen.potReady) {
      // Only one plate is needed, so the second cook should not go for one too.
      target = mate.hold === HOLD.PLATE || mate.hold === HOLD.SOUP ? ONIONS : PLATES;
    } else if (kitchen.potOnions < NEEDED) {
      // Don't both fetch the last onion the pot has room for.
      const incoming = mate.hold === HOLD.ONION ? 1 : 0;
      target = kitchen.potOnions + incoming < NEEDED ? ONIONS : PLATES;
    } else {
      target = PLATES;
    }

    // Standing next to the thing we want? Use it.
    if (station && station.tile === target) {
      if (target === PLATES && me.hold !== HOLD.NOTHING) { /* fall through to move */ }
      else return INTERACT;
    }

    const goal = findTile(kitchen, target);
    if (!goal) return 4;
    return stepToward(kitchen, me, goal, rand, mate);
  }

  function findTile(kitchen, tile) {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (kitchen.tileAt(x, y) === tile) return { x, y };
    }
    return null;
  }

  /** Breadth-first step towards a square adjacent to the goal station. */
  function stepToward(kitchen, from, goal, rand, avoid) {
    const dist = new Int16Array(W * H).fill(-1);
    const queue = [];
    // start from every floor square touching the goal
    for (const [dx, dy] of [[0, -1], [1, 0], [0, 1], [-1, 0]]) {
      const x = goal.x + dx, y = goal.y + dy;
      if (kitchen.walkable(x, y)) { dist[idx(x, y)] = 0; queue.push([x, y]); }
    }
    // Route around the other cook rather than through them, unless they are
    // standing on the only square that does the job.
    const blocked = avoid && !(dist[idx(avoid.x, avoid.y)] === 0) ? idx(avoid.x, avoid.y) : -1;
    for (let h = 0; h < queue.length; h++) {
      const [cx, cy] = queue[h];
      for (const [dx, dy] of [[0, -1], [1, 0], [0, 1], [-1, 0]]) {
        const nx = cx + dx, ny = cy + dy;
        if (!kitchen.walkable(nx, ny) || dist[idx(nx, ny)] !== -1) continue;
        if (idx(nx, ny) === blocked) continue;
        dist[idx(nx, ny)] = dist[idx(cx, cy)] + 1;
        queue.push([nx, ny]);
      }
    }
    const here = dist[idx(from.x, from.y)];
    if (here === 0) return INTERACT;
    let best = 4, bestD = here < 0 ? 1e9 : here;
    const order = [0, 1, 2, 3];
    for (let i = order.length - 1; i > 0; i--) {     // break ties randomly
      const j = (rand() * (i + 1)) | 0;
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const a of order) {
      const [dx, dy] = MOVES[a];
      const nx = from.x + dx, ny = from.y + dy;
      if (!kitchen.walkable(nx, ny)) continue;
      const d = dist[idx(nx, ny)];
      if (d >= 0 && d < bestD) { bestD = d; best = a; }
    }
    return best;
  }

  return {
    Kitchen,
    KITCHEN: { W, H, OBS, NUM_ACTIONS, INTERACT, MOVES, HOLD, HOLD_NAME, NEEDED, COOK_TIME,
               FLOOR, COUNTER, ONIONS, PLATES, POT, HATCH, LAYOUT, scriptedCook, idx },
  };
});
