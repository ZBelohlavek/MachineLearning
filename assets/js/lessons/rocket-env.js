/* ==========================================================================
   rocket-env.js — a top-down "Rocket League" arena.

   This file is the *environment* half of a reinforcement learning problem:
   physics, observations, rewards and episode bookkeeping. It knows nothing
   about neural networks. The learning half lives in rocket-train.js.

   Field coordinates: x in [-100, 100], y in [-65, 65].
   Blue attacks the +x goal, Orange attacks the -x goal.
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
    ? require('../lib/nn.js') : root.ML;
  const { mulberry32, clamp } = DEPS;

  const F = {
    halfW: 100,
    halfH: 65,
    goalHalf: 22,      // half-height of the goal mouth
    carR: 5.2,
    ballR: 4.2,
    carAccel: 300,     // forward acceleration, units/s^2
    carBrake: 240,
    carMax: 98,
    turnRate: 3.4,     // rad/s at full grip
    ballMax: 165,
    dt: 1 / 30,
    steps: 300,        // 10 seconds per episode
  };
  F.diag = Math.hypot(F.halfW, F.halfH);

  /* Nine discrete actions: every combination of throttle and steer.
     A tiny action space keeps the policy gradient signal strong. */
  const ACTIONS = [];
  for (const throttle of [1, 0, -1]) {
    for (const steer of [-1, 0, 1]) ACTIONS.push({ throttle, steer });
  }
  const ACTION_LABELS = ACTIONS.map(
    (a) => `${a.throttle > 0 ? 'fwd' : a.throttle < 0 ? 'rev' : 'coast'}·${a.steer < 0 ? 'L' : a.steer > 0 ? 'R' : '—'}`
  );

  const OBS_SIZE = 24;

  /** The reward weights the learner is trained on. Every one is user-editable. */
  const DEFAULT_REWARDS = {
    goal: 10,       // scoring (and conceding, symmetrically)
    push: 1.0,      // ball moving toward the opponent's goal
    touch: 0.35,    // making contact with the ball
    approach: 0.35, // closing distance to the ball
    face: 0.05,     // pointing at the ball
    speed: 0.02,    // driving fast rather than sitting still
    time: 0.0,      // per-step penalty (pressure to finish)
  };

  const REWARD_PRESETS = {
    recommended: { ...DEFAULT_REWARDS },
    sparse: { goal: 10, push: 0, touch: 0, approach: 0, face: 0, speed: 0, time: 0.005 },
    chaser: { goal: 2, push: 0, touch: 1.0, approach: 1.0, face: 0.15, speed: 0.05, time: 0 },
    striker: { goal: 12, push: 2.2, touch: 0.15, approach: 0.15, face: 0.02, speed: 0.02, time: 0.002 },
  };

  const TWO_PI = Math.PI * 2;
  function wrapAngle(a) {
    while (a > Math.PI) a -= TWO_PI;
    while (a < -Math.PI) a += TWO_PI;
    return a;
  }

  /* --------------------------------------------------------------------------
     The arena
     -------------------------------------------------------------------------- */

  class Arena {
    /**
     * @param {object} opts { seed, rewards, solo (no opponent), kickoffJitter }
     */
    constructor(opts = {}) {
      this.rand = mulberry32(opts.seed ?? (Math.random() * 1e9) | 0);
      this.rewards = { ...DEFAULT_REWARDS, ...(opts.rewards || {}) };
      this.solo = !!opts.solo;
      this.spread = opts.spread ?? 1;      // curriculum difficulty, see reset()
      // Two cars: index 0 = Blue (attacks +x), index 1 = Orange (attacks -x).
      this.cars = [makeCar(1), makeCar(-1)];
      this.ball = { x: 0, y: 0, vx: 0, vy: 0 };
      this.reset();
    }

    reset() {
      const r = this.rand;
      // `spread` is the curriculum dial: 0.15 spawns the car right next to the
      // ball facing it (easy), 1.0 is a full random kickoff anywhere (hard).
      const sp = clamp(this.spread ?? 1, 0.12, 1);

      this.ball.x = (r() - 0.5) * 70 * sp;
      this.ball.y = (r() - 0.5) * 80 * sp;
      this.ball.vx = (r() - 0.5) * 60 * sp;
      this.ball.vy = (r() - 0.5) * 60 * sp;

      for (let i = 0; i < 2; i++) {
        const c = this.cars[i];
        // Each car starts on the side of the ball nearest its own goal, so
        // "drive forwards" is at least sometimes the right idea.
        const baseAng = i === 0 ? Math.PI : 0;
        const ang = baseAng + (r() - 0.5) * 2.4 * sp;
        const d = 13 + 52 * sp;
        c.x = clamp(this.ball.x + Math.cos(ang) * d, -F.halfW + F.carR, F.halfW - F.carR);
        c.y = clamp(this.ball.y + Math.sin(ang) * d, -F.halfH + F.carR, F.halfH - F.carR);
        c.vx = 0; c.vy = 0;
        c.angle = Math.atan2(this.ball.y - c.y, this.ball.x - c.x) + (r() - 0.5) * 2.6 * sp;
        c.lastTouch = 0;
      }

      this.step_ = 0;
      this.done = false;
      this.scorer = -1;            // 0 blue, 1 orange, -1 none
      this.touches = [0, 0];
      this.prevCarBall = [this.dist(this.cars[0], this.ball), this.dist(this.cars[1], this.ball)];
      this.prevBallGoal = [this.ballGoalDist(0), this.ballGoalDist(1)];
      this.termSums = [blankTerms(), blankTerms()];
      return this;
    }

    dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

    /** Distance from the ball to the goal that agent `i` is attacking. */
    ballGoalDist(i) {
      const gx = i === 0 ? F.halfW : -F.halfW;
      const dy = Math.max(0, Math.abs(this.ball.y) - F.goalHalf * 0.5);
      return Math.hypot(gx - this.ball.x, dy);
    }

    /* ----------------------------------------------------------------------
       Observation: what the agent actually gets to "see".

       Everything is expressed from the agent's own point of view. Orange's
       world is rotated 180 degrees so that both agents always attack the +x
       goal — that is what lets a single network play both sides in self-play.
       Rotating (rather than mirroring) keeps left and right consistent, so
       "steer left" means the same thing for both cars.
       ---------------------------------------------------------------------- */
    observe(i, out) {
      const s = i === 0 ? 1 : -1;                       // team sign
      const me = this.cars[i], op = this.cars[1 - i], b = this.ball;

      const mx = s * me.x, my = s * me.y;
      const mvx = s * me.vx, mvy = s * me.vy;
      const ang = me.angle + (s < 0 ? Math.PI : 0);
      const ca = Math.cos(ang), sa = Math.sin(ang);

      const bx = s * b.x, by = s * b.y;
      const bvx = s * b.vx, bvy = s * b.vy;
      const ox = s * op.x, oy = s * op.y;

      // Rotate a world vector into the car's frame: +x is "in front of me".
      const fwd = (vx, vy) => vx * ca + vy * sa;
      const lat = (vx, vy) => -vx * sa + vy * ca;

      const dbx = bx - mx, dby = by - my;
      const dball = Math.hypot(dbx, dby) || 1e-6;
      const gx = F.halfW - mx, gy = 0 - my;             // toward the goal I attack
      const bgx = F.halfW - bx, bgy = 0 - by;           // ball -> goal
      const bgLen = Math.hypot(bgx, bgy) || 1e-6;
      const dox = ox - mx, doy = oy - my;

      const o = out || new Float32Array(OBS_SIZE);
      let k = 0;
      o[k++] = mx / F.halfW;                            // where am I
      o[k++] = my / F.halfH;
      o[k++] = fwd(mvx, mvy) / F.carMax;                // how fast, forward/sideways
      o[k++] = lat(mvx, mvy) / F.carMax;
      o[k++] = ca;                                      // which way am I pointing
      o[k++] = sa;
      o[k++] = fwd(dbx, dby) / F.halfW;                 // ball, in my frame
      o[k++] = lat(dbx, dby) / F.halfW;
      o[k++] = dball / F.diag;
      o[k++] = Math.exp(-dball / 20);                   // "am I basically on it"
      o[k++] = fwd(bvx, bvy) / F.ballMax;               // ball velocity, my frame
      o[k++] = lat(bvx, bvy) / F.ballMax;
      o[k++] = bx / F.halfW;                            // ball on the field
      o[k++] = by / F.halfH;
      o[k++] = fwd(gx, gy) / F.halfW;                   // target goal, my frame
      o[k++] = lat(gx, gy) / F.halfW;
      o[k++] = fwd(bgx / bgLen, bgy / bgLen);           // ball->goal direction
      o[k++] = lat(bgx / bgLen, bgy / bgLen);
      o[k++] = dbx / dball * ca + dby / dball * sa;     // cos angle-to-ball
      o[k++] = -dbx / dball * sa + dby / dball * ca;    // sin angle-to-ball
      o[k++] = this.solo ? 0 : fwd(dox, doy) / F.halfW; // opponent, my frame
      o[k++] = this.solo ? 0 : lat(dox, doy) / F.halfW;
      o[k++] = this.solo ? 0 : Math.hypot(dox, doy) / F.diag;
      o[k++] = 1 - (this.step_ / F.steps) * 2;          // clock, so it knows time is short
      return o;
    }

    /* ----------------------------------------------------------------------
       One physics tick + reward calculation.
       `actions` is [blueActionIndex, orangeActionIndex].
       ---------------------------------------------------------------------- */
    step(actions) {
      const dt = F.dt;
      const rewards = [0, 0];
      const terms = [blankTerms(), blankTerms()];

      // --- cars ---
      for (let i = 0; i < 2; i++) {
        if (this.solo && i === 1) continue;
        const c = this.cars[i];
        const a = ACTIONS[actions[i] | 0] || ACTIONS[4];
        const ca = Math.cos(c.angle), sa = Math.sin(c.angle);

        // Split velocity into "along the car" and "sideways".
        let vf = c.vx * ca + c.vy * sa;
        let vl = -c.vx * sa + c.vy * ca;

        if (a.throttle > 0) vf += F.carAccel * dt;
        else if (a.throttle < 0) vf -= F.carBrake * dt;
        else vf *= Math.exp(-1.4 * dt);                 // coast

        vf = clamp(vf, -F.carMax * 0.55, F.carMax);
        vl *= Math.exp(-9 * dt);                        // tyres grip: kill sideways slide

        // You can only turn while moving — same as a real car (and a real Rocket League car).
        const grip = Math.min(1, Math.abs(vf) / 28);
        c.angle = wrapAngle(c.angle + a.steer * F.turnRate * grip * Math.sign(vf || 1) * dt);

        const na = c.angle, nca = Math.cos(na), nsa = Math.sin(na);
        c.vx = vf * nca - vl * nsa;
        c.vy = vf * nsa + vl * nca;
        c.x += c.vx * dt;
        c.y += c.vy * dt;
        c.throttle = a.throttle; c.steer = a.steer;

        // Walls
        if (c.x < -F.halfW + F.carR) { c.x = -F.halfW + F.carR; c.vx *= -0.25; }
        if (c.x > F.halfW - F.carR) { c.x = F.halfW - F.carR; c.vx *= -0.25; }
        if (c.y < -F.halfH + F.carR) { c.y = -F.halfH + F.carR; c.vy *= -0.25; }
        if (c.y > F.halfH - F.carR) { c.y = F.halfH - F.carR; c.vy *= -0.25; }
        if (c.lastTouch > 0) c.lastTouch--;
      }

      // --- car / ball contact ---
      const b = this.ball;
      const touched = [false, false];
      for (let i = 0; i < 2; i++) {
        if (this.solo && i === 1) continue;
        const c = this.cars[i];
        const dx = b.x - c.x, dy = b.y - c.y;
        const d = Math.hypot(dx, dy);
        const minD = F.carR + F.ballR;
        if (d < minD && d > 1e-6) {
          const nx = dx / d, ny = dy / d;
          b.x = c.x + nx * minD;
          b.y = c.y + ny * minD;
          // Closing speed of car into ball, plus a base bump so gentle touches still count.
          const rel = (c.vx - b.vx) * nx + (c.vy - b.vy) * ny;
          const impulse = 42 + Math.max(0, rel) * 1.35;
          b.vx += nx * impulse; b.vy += ny * impulse;
          c.vx -= nx * impulse * 0.10; c.vy -= ny * impulse * 0.10;
          touched[i] = true;
          this.touches[i]++;
          c.lastTouch = 6;
        }
      }

      // --- ball ---
      b.vx *= Math.exp(-0.30 * dt);
      b.vy *= Math.exp(-0.30 * dt);
      const bs = Math.hypot(b.vx, b.vy);
      if (bs > F.ballMax) { b.vx *= F.ballMax / bs; b.vy *= F.ballMax / bs; }
      b.x += b.vx * dt;
      b.y += b.vy * dt;

      if (b.y < -F.halfH + F.ballR) { b.y = -F.halfH + F.ballR; b.vy = Math.abs(b.vy) * 0.8; }
      if (b.y > F.halfH - F.ballR) { b.y = F.halfH - F.ballR; b.vy = -Math.abs(b.vy) * 0.8; }

      let scored = -1;
      if (b.x > F.halfW - F.ballR) {
        if (Math.abs(b.y) < F.goalHalf) scored = 0;     // blue scores in the +x goal
        else { b.x = F.halfW - F.ballR; b.vx = -Math.abs(b.vx) * 0.8; }
      } else if (b.x < -F.halfW + F.ballR) {
        if (Math.abs(b.y) < F.goalHalf) scored = 1;
        else { b.x = -F.halfW + F.ballR; b.vx = Math.abs(b.vx) * 0.8; }
      }

      /* ------------------------------------------------------------------
         Rewards. Every term is scaled to be roughly O(1) per step so that the
         user's weight sliders mean the same thing across terms.
         ------------------------------------------------------------------ */
      const W = this.rewards;
      for (let i = 0; i < 2; i++) {
        if (this.solo && i === 1) continue;
        const c = this.cars[i];
        const t = terms[i];

        const dCB = this.dist(c, b);
        const dBG = this.ballGoalDist(i);

        // Shaping term 1: did I get closer to the ball this tick?
        t.approach = clamp((this.prevCarBall[i] - dCB) / 4, -1, 1) * W.approach;
        // Shaping term 2: did the ball get closer to the goal I'm attacking?
        t.push = clamp((this.prevBallGoal[i] - dBG) / 5, -1.5, 1.5) * W.push;
        // Contact bonus
        t.touch = touched[i] ? W.touch : 0;
        // Am I pointing at the ball?
        const ang = Math.atan2(b.y - c.y, b.x - c.x);
        t.face = Math.cos(wrapAngle(ang - c.angle)) * W.face;
        // Am I moving at all?
        t.speed = (Math.hypot(c.vx, c.vy) / F.carMax) * W.speed;
        t.time = -W.time;

        if (scored >= 0) t.goal = (scored === i ? 1 : -1) * W.goal;

        this.prevCarBall[i] = dCB;
        this.prevBallGoal[i] = dBG;

        let sum = 0;
        for (const key in t) { sum += t[key]; this.termSums[i][key] += t[key]; }
        rewards[i] = sum;
      }

      this.step_++;
      if (scored >= 0) { this.done = true; this.scorer = scored; }
      else if (this.step_ >= F.steps) this.done = true;

      this.lastTerms = terms;
      return { rewards, done: this.done, scored, touched };
    }

    /** Snapshot for rendering / replay. */
    snapshot() {
      return {
        cars: this.cars.map((c) => ({ ...c })),
        ball: { ...this.ball },
        step: this.step_, scorer: this.scorer, done: this.done,
      };
    }
  }

  function makeCar(team) {
    return { x: 0, y: 0, vx: 0, vy: 0, angle: 0, team, throttle: 0, steer: 0, lastTouch: 0 };
  }

  function blankTerms() {
    return { goal: 0, push: 0, touch: 0, approach: 0, face: 0, speed: 0, time: 0 };
  }

  /* --------------------------------------------------------------------------
     A hand-written opponent, so there is something to compare the learner to.
     Pure geometry: drive to the point behind the ball that lines it up with
     the goal, then accelerate through it.
     -------------------------------------------------------------------------- */
  function scriptedAction(arena, i) {
    const s = i === 0 ? 1 : -1;
    const c = arena.cars[i], b = arena.ball;
    const gx = s * F.halfW;
    // Aim point: a spot on the far side of the ball from the goal.
    const dx = gx - b.x, dy = 0 - b.y;
    const len = Math.hypot(dx, dy) || 1;
    const targetX = b.x - (dx / len) * (F.carR + F.ballR + 2);
    const targetY = b.y - (dy / len) * (F.carR + F.ballR + 2);

    const want = Math.atan2(targetY - c.y, targetX - c.x);
    const err = wrapAngle(want - c.angle);
    const steer = err > 0.12 ? 1 : err < -0.12 ? -1 : 0;
    // Back up if we are badly misaligned and close, otherwise drive.
    const d = Math.hypot(targetX - c.x, targetY - c.y);
    const throttle = Math.abs(err) > 2.2 && d < 22 ? -1 : 1;
    const idx = ACTIONS.findIndex((a) => a.throttle === throttle && a.steer === steer);
    return idx < 0 ? 4 : idx;
  }

  return { F, ACTIONS, ACTION_LABELS, OBS_SIZE, DEFAULT_REWARDS, REWARD_PRESETS, Arena, scriptedAction };
});
