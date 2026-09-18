/* ==========================================================================
   tests/gradcheck.cjs — numerical gradient checks for every hand-written
   backward pass on this site.

       node tests/gradcheck.cjs

   Two checks per model:

   1. DIRECTIONAL (the pass/fail criterion). Pick a random unit direction v
      across every parameter and compare the analytic directional derivative
      g·v against a central finite difference of the loss. Summing over
      thousands of parameters gives a large signal, so this catches any real
      error in a backward pass with a very tight tolerance.

   2. PER-PARAMETER (reported for information). Finite differences on
      individual weights. Two effects make a handful of these unreliable, and
      both are detected and excluded rather than papered over:
        * Float32 parameters mean the loss carries rounding noise, which
          swamps gradients near zero.
        * ReLU has a kink. If a perturbation flips a unit on or off, the
          finite difference measures the corner, not the derivative — visible
          as strongly unequal one-sided slopes.
   ========================================================================== */

const path = require('path');
const nn = require(path.join(__dirname, '../assets/js/lib/nn.js'));
const cnnLib = require(path.join(__dirname, '../assets/js/lib/cnn.js'));
const vitLib = require(path.join(__dirname, '../assets/js/lib/vit.js'));
const charLib = require(path.join(__dirname, '../assets/js/lib/charlm.js'));

let failures = 0;

/** g·v vs a central difference along a random direction over all parameters. */
function directional(loss, blocks, rand, eps = 0.005) {
  const dirs = blocks.map((b) => Float32Array.from({ length: b.values.length }, () => nn.randn(rand)));
  let norm = 0;
  for (const d of dirs) for (const x of d) norm += x * x;
  norm = Math.sqrt(norm) || 1;

  let analytic = 0;
  blocks.forEach((b, i) => {
    for (let k = 0; k < b.values.length; k++) analytic += b.grads[k] * dirs[i][k] / norm;
  });

  const orig = blocks.map((b) => Float32Array.from(b.values));
  const move = (sign) => blocks.forEach((b, i) => {
    for (let k = 0; k < b.values.length; k++) b.values[k] = orig[i][k] + sign * eps * dirs[i][k] / norm;
  });
  move(1); const lp = loss();
  move(-1); const lm = loss();
  move(0);
  const numeric = (lp - lm) / (2 * eps);
  return {
    analytic, numeric,
    rel: Math.abs(numeric - analytic) / Math.max(1e-12, Math.abs(numeric) + Math.abs(analytic)),
  };
}

/** Per-parameter finite differences, skipping the two unreliable cases. */
function perParam(loss, blocks, eps = 0.01, floor = 1e-4) {
  let worst = 0, checked = 0, skipped = 0;
  for (const b of blocks) {
    const n = Math.min(b.values.length, 24);
    for (let k = 0; k < n; k++) {
      const i = (k * 7919) % b.values.length;
      const o = b.values[i];
      b.values[i] = o + eps; const lp = loss();
      b.values[i] = o;       const l0 = loss();
      b.values[i] = o - eps; const lm = loss();
      b.values[i] = o;
      const num = (lp - lm) / (2 * eps), ana = b.grads[i];
      if (Math.abs(num) < floor && Math.abs(ana) < floor) { skipped++; continue; }
      const oneSided = Math.abs((lp - l0) / eps - (l0 - lm) / eps);
      if (oneSided > 0.12 * (Math.abs(num) + Math.abs(ana)) + 1e-7) { skipped++; continue; }
      checked++;
      worst = Math.max(worst, Math.abs(num - ana) / Math.max(1e-12, Math.abs(num) + Math.abs(ana)));
    }
  }
  return { worst, checked, skipped };
}

function run(name, loss, blocks, rand, opts = {}) {
  const dir = directional(loss, blocks, rand, opts.eps ?? 0.005);
  const per = perParam(loss, blocks, opts.perEps ?? 0.01);
  const ok = dir.rel <= (opts.limit ?? 2e-3);
  if (!ok) failures++;
  console.log(
    `${ok ? '  ok  ' : ' FAIL '} ${name.padEnd(36)} ` +
    `directional ${dir.rel.toExponential(2)}  ·  per-parameter worst ${per.worst.toExponential(2)} ` +
    `over ${per.checked}${per.skipped ? ` (${per.skipped} skipped: noise / ReLU kinks)` : ''}`
  );
}

/* ------------------------------------------------------- dense network */
{
  const rand = nn.mulberry32(1);
  const net = new nn.MLP([4, 8, 6, 3], { hidden: 'tanh', out: 'linear', rand });
  const x = Float32Array.from([0.4, -0.7, 1.1, 0.2]);
  const target = [0.3, -0.5, 0.8];
  const loss = () => {
    const o = net.forward(x);
    let l = 0;
    for (let i = 0; i < 3; i++) l += 0.5 * (o[i] - target[i]) ** 2;
    return l;
  };
  net.zeroGrad();
  const out = net.forward(x);
  const d = new Float32Array(3);
  for (let i = 0; i < 3; i++) d[i] = out[i] - target[i];
  net.backward(d);
  const blocks = [];
  net.layers.forEach((l) => {
    blocks.push({ values: l.W, grads: l.dW });
    blocks.push({ values: l.b, grads: l.db });
  });
  run('nn.js  MLP (tanh + linear)', loss, blocks, rand);
}

/* ------------------------------------------------ convolutional network */
{
  const rand = nn.mulberry32(5);
  const c1 = new cnnLib.Conv2D([1, 8, 8], 3, 3, rand);
  const r1 = new cnnLib.ReLU(c1.outShape);
  const p1 = new cnnLib.MaxPool2(r1.outShape);
  const c2 = new cnnLib.Conv2D(p1.outShape, 4, 3, rand);
  const r2 = new cnnLib.ReLU(c2.outShape);
  const p2 = new cnnLib.MaxPool2(r2.outShape);
  const fl = new cnnLib.Flatten(p2.outShape);
  const fc = new cnnLib.FC(fl.outShape[0], 5, 'linear', rand);
  const net = new cnnLib.ConvNet([c1, r1, p1, c2, r2, p2, fl, fc]);
  const x = new Float32Array(64).map(() => rand() * 2 - 1);
  const label = 2;
  const loss = () => net.lossAndGrad(net.forward(x), label).loss;
  net.zeroGrad();
  net.backward(net.lossAndGrad(net.forward(x), label).dLogits);
  const blocks = [];
  for (const layer of [c1, c2, fc]) {
    for (const p of layer.params()) blocks.push({ values: p.v, grads: p.g });
  }
  run('cnn.js  Conv2D + MaxPool + FC', loss, blocks, rand);
}

/* -------------------------------------------------------- layer norm */
{
  const rand = nn.mulberry32(3);
  const T = 4, d = 6;
  const ln = new vitLib.LayerNorm(T, d);
  for (let i = 0; i < d; i++) { ln.g.v[i] = 0.7 + rand(); ln.b.v[i] = rand() - 0.5; }
  const x = new Float32Array(T * d).map(() => (rand() * 2 - 1) * 1.5);
  const w = new Float32Array(T * d).map(() => rand() * 2 - 1);
  const loss = () => {
    const o = ln.forward(x);
    let s = 0;
    for (let i = 0; i < o.length; i++) s += w[i] * o[i] * o[i] * 0.5;
    return s;
  };
  ln.g.zero(); ln.b.zero();
  const o = ln.forward(x);
  const dO = new Float32Array(T * d);
  for (let i = 0; i < dO.length; i++) dO[i] = w[i] * o[i];
  ln.backward(dO);
  run('vit.js  LayerNorm', loss, [
    { values: ln.g.v, grads: ln.g.g },
    { values: ln.b.v, grads: ln.b.g },
  ], rand, { perEps: 1e-3 });
}

/* ---------------------------- the whole vision transformer, several variants */
for (const cfg of [
  { useCLS: true, useNorm: true },
  { useCLS: false, useNorm: false },
  { useCLS: false, useNorm: false, heads: 2 },
  { useCLS: true, useNorm: true, heads: 4 },
]) {
  const rand = nn.mulberry32(17);
  const net = new vitLib.ViT({ imgSize: 8, patch: 2, dim: 16, mlpHidden: 32, classes: 4, rand, ...cfg });
  const img = new Float32Array(64).map(() => rand());
  const label = 2;
  const loss = () => net.lossAndGrad(net.forward(img), label).loss;
  net.zeroGrad();
  net.backward(net.lossAndGrad(net.forward(img), label).dLogits);
  const blocks = net.params().map((p) => ({ values: p.v, grads: p.g }));
  run(`vit.js  ViT (${cfg.useCLS ? 'cls' : 'mean'}, ${cfg.useNorm ? 'norm' : 'no norm'}, ` +
      `${cfg.heads || 1} head${(cfg.heads || 1) > 1 ? 's' : ''})`, loss, blocks, rand);
}

/* ------------------- the character-level language model ------------------ */
{
  const rand = nn.mulberry32(11);
  const text = 'the sea is calm tonight, the tide is full and the moon lies fair. ';
  const vocab = charLib.buildVocab(text);
  const net = new charLib.CharLM({ vocab: vocab.size, context: 8, dim: 12, mlpHidden: 16, rand, heads: 2 });
  const ids = Int32Array.from(vocab.encode(text.slice(0, 8)));
  const targets = Int32Array.from(vocab.encode(text.slice(1, 9)));
  const loss = () => net.lossAndGrad(net.forward(ids), targets).loss;
  net.zeroGrad();
  net.backward(net.lossAndGrad(net.forward(ids), targets).dLogits);
  const blocks = net.params().map((p) => ({ values: p.v, grads: p.g }));
  run('charlm.js  causal transformer (2 heads)', loss, blocks, rand);

  // causality: a later character must not change an earlier position's logits
  const before = Array.from(net.forward(ids).slice(0, vocab.size));
  const poked = Int32Array.from(ids);
  poked[7] = (poked[7] + 3) % vocab.size;
  const after = Array.from(net.forward(poked).slice(0, vocab.size));
  const leaked = before.some((v, i) => Math.abs(v - after[i]) > 1e-6);
  if (leaked) failures++;
  console.log(`${leaked ? ' FAIL ' : '  ok  '} ${'charlm.js  no peeking at the future'.padEnd(36)} ` +
    'changing the last character leaves position 0 untouched');
}

/* --------------- the RL environment: sanity rather than gradients -------- */
{
  const { Arena, scriptedAction } = require(path.join(__dirname, '../assets/js/lessons/rocket-env.js'));
  let goals = 0, finite = true, touches = 0;
  for (let e = 0; e < 60; e++) {
    const a = new Arena({ seed: e + 1 });
    while (!a.done) {
      a.step([scriptedAction(a, 0), scriptedAction(a, 1)]);
      if (!isFinite(a.ball.x) || !isFinite(a.cars[0].x) || !isFinite(a.cars[0].vx)) finite = false;
    }
    if (a.scorer >= 0) goals++;
    touches += a.touches[0] + a.touches[1];
  }
  const ok = finite && goals > 20 && touches > 500;
  if (!ok) failures++;
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${'rocket-env.js  physics + scripted play'.padEnd(36)} ` +
    `${goals}/60 episodes ended in a goal, ${touches} ball touches, no NaNs`);
}

/* ------------------------- PPO actually improves a policy ---------------- */
{
  const { Trainer } = require(path.join(__dirname, '../assets/js/lessons/rocket-train.js'));
  const t = new Trainer({ opponent: 'none', seed: 3 });
  const before = t.evaluate(12, 'none');
  const start = Date.now();
  while (Date.now() - start < 20000) t.trainSlice(500);
  const after = t.evaluate(12, 'none');
  const ok = after.touchesPerEp > before.touchesPerEp && t.updates > 5;
  if (!ok) failures++;
  console.log(`${ok ? '  ok  ' : ' FAIL '} ${'rocket-train.js  PPO improves play'.padEnd(36)} ` +
    `touches/episode ${before.touchesPerEp.toFixed(2)} → ${after.touchesPerEp.toFixed(2)} ` +
    `after ${t.episodes} episodes (${t.updates} updates)`);
}

/* ---------------------------------------------------------------- arcade.js
   The game-feel layer has real logic in it — medal thresholds, combo decay,
   personal bests — and it is the sort of thing that breaks quietly. */
{
  const arcade = require('../assets/js/lib/arcade.js');
  const cases = [];

  cases.push(['medal: a score at the gold threshold is gold',
    arcade.medalFor(250, { gold: 250, silver: 140, bronze: 60 }) === 'gold']);
  cases.push(['medal: below every tier earns nothing',
    arcade.medalFor(12, { gold: 250, silver: 140, bronze: 60 }) === null]);
  cases.push(['medal: for times, lower wins',
    arcade.medalFor(12.5, { gold: 13, silver: 17, bronze: 24 }, true) === 'gold']);
  cases.push(['medal: a slow time still takes bronze',
    arcade.medalFor(23, { gold: 13, silver: 17, bronze: 24 }, true) === 'bronze']);

  const c = arcade.combo({ windowMs: 5000, cap: 5 });
  const first = c.hit(1000);
  c.hit(1500);
  const third = c.hit(2000);
  cases.push(['combo: the first hit is worth ×1', first === 1]);
  cases.push(['combo: the streak raises the multiplier', third === 2 && c.streak === 3]);
  c.hit(60000);
  cases.push(['combo: a long gap resets the streak', c.streak === 1]);
  c.miss();
  cases.push(['combo: a miss resets the streak', c.streak === 0]);

  const f = arcade.fx();
  f.burst(0, 0, { count: 6 });
  cases.push(['fx: a burst makes the field busy', f.busy === true]);
  f.clear();
  cases.push(['fx: clear empties it', f.busy === false]);

  for (const [name, ok] of cases) {
    if (!ok) failures++;
    console.log(`${ok ? '  ok  ' : ' FAIL '} ${('arcade.js  ' + name).padEnd(60)}`);
  }
}

/* -------------------------------------------------- connect 4 + tree search
   The rules and the search are pure logic, so they can be checked exactly
   rather than statistically. A search that cannot see a win one move away is
   broken no matter what the win rate says. */
{
  const { C4 } = require('../assets/js/lessons/connect4-env.js');
  const { MCTS, pickMove } = require('../assets/js/lib/mcts.js');
  const cases = [];

  // --- rules ---
  {
    const b = C4.newBoard();
    let y = 0;
    for (let i = 0; i < 4; i++) y = C4.play(b, 3, C4.P1);
    cases.push(['rules: four in a column wins', C4.winsAt(b, 3, y)]);
  }
  {
    const b = C4.newBoard();
    let y = 0;
    for (let x = 0; x < 4; x++) y = C4.play(b, x, C4.P2);
    cases.push(['rules: four in a row wins', C4.winsAt(b, 3, y)]);
    const line = C4.winningLine(b, 3, y);
    cases.push(['rules: the winning line is four cells', !!line && line.length === 4]);
  }
  {
    const b = C4.newBoard();
    for (let i = 0; i < 6; i++) C4.play(b, 0, C4.P1);
    cases.push(['rules: a full column is not a legal move', !C4.legalMoves(b).includes(0)]);
    cases.push(['rules: dropping into a full column is refused', C4.play(b, 0, C4.P2) === -1]);
  }
  {
    // A board is encoded from the mover's side, so the same position seen by
    // each player must produce mirrored planes. That is what lets one network
    // play both colours.
    const b = C4.newBoard();
    C4.play(b, 2, C4.P1); C4.play(b, 4, C4.P2);
    const a = C4.encode(b, C4.P1), c = C4.encode(b, C4.P2);
    let mirrored = true;
    for (let i = 0; i < C4.N; i++) {
      if (a[i] !== c[C4.N + i] || a[C4.N + i] !== c[i]) mirrored = false;
    }
    cases.push(['encoding: the two players see mirrored planes', mirrored]);
  }
  {
    // Making and unmaking a move must leave the board byte-for-byte identical,
    // or the search corrupts the position it is searching.
    const b = C4.newBoard();
    C4.play(b, 1, C4.P1); C4.play(b, 5, C4.P2);
    const before = Array.from(b).join('');
    const row = C4.play(b, 3, C4.P1);
    C4.undo(b, 3, row);
    cases.push(['search: make and unmake restores the board', Array.from(b).join('') === before]);
  }

  // --- the search itself, with a network that knows nothing ---
  const game = {
    nActions: C4.W,
    legal: (b) => C4.legalMoves(b),
    apply: (b, a, p) => C4.play(b, a, p),
    undo: (b, a, row) => C4.undo(b, a, row),
    justWon: (b, a, row) => C4.winsAt(b, a, row),
    isDraw: (b) => C4.isFull(b),
    other: (p) => (p === C4.P1 ? C4.P2 : C4.P1),
  };
  const blind = () => ({ P: Float32Array.from({ length: C4.W }, () => 1 / C4.W), v: 0 });
  const mcts = new MCTS(game, blind, { rand: Math.random });

  {
    const b = C4.newBoard();
    C4.play(b, 0, C4.P1); C4.play(b, 1, C4.P1); C4.play(b, 2, C4.P1);
    C4.play(b, 0, C4.P2); C4.play(b, 1, C4.P2);
    const r = mcts.search(b, C4.P1, 400);
    cases.push(['search: takes a win one move away', pickMove(r.visits, 0) === 3]);
  }
  {
    const b = C4.newBoard();
    C4.play(b, 0, C4.P2); C4.play(b, 1, C4.P2); C4.play(b, 2, C4.P2);
    C4.play(b, 5, C4.P1); C4.play(b, 6, C4.P1);
    const r = mcts.search(b, C4.P1, 800);
    cases.push(['search: blocks a loss one move away', pickMove(r.visits, 0) === 3]);
  }
  {
    const b = C4.newBoard();
    const r = mcts.search(b, C4.P1, 200);
    let total = 0;
    for (const v of r.visits) total += v;
    cases.push(['search: every simulation is accounted for', Math.abs(total - 200) < 1e-6]);
    let sum = 0;
    for (const p of r.policy) sum += p;
    cases.push(['search: the visit policy sums to one', Math.abs(sum - 1) < 1e-5]);
  }
  {
    // A full column must never be searched, whatever the priors say.
    const b = C4.newBoard();
    for (let i = 0; i < 6; i++) C4.play(b, 3, i % 2 ? C4.P1 : C4.P2);
    const r = mcts.search(b, C4.P1, 120);
    cases.push(['search: never spends a visit on an illegal move', r.visits[3] === 0]);
  }
  {
    // Temperature 0 is deterministic; temperature 1 must be able to pick others.
    const v = Float32Array.from([1, 60, 30, 5, 0, 0, 0]);
    const greedy = new Set();
    for (let i = 0; i < 30; i++) greedy.add(pickMove(v, 0));
    const sampled = new Set();
    for (let i = 0; i < 200; i++) sampled.add(pickMove(v, 1));
    cases.push(['search: temperature 0 always picks the most visited',
      greedy.size === 1 && greedy.has(1)]);
    cases.push(['search: temperature 1 explores other moves', sampled.size > 1]);
  }

  // --- the combined policy/value gradient, against finite differences ---
  {
    const { C4Trainer } = require('../assets/js/lessons/connect4-train.js');
    const t = new C4Trainer({ seed: 2, hp: { batch: 8 } });
    for (let i = 0; i < 3; i++) t.selfPlayGame();

    const sample = t.buffer.slice(0, 8);
    const loss = () => {
      let L = 0;
      for (const s of sample) {
        const out = t.net.forward(s.x);
        const logits = new Float32Array(C4.W);
        for (let a = 0; a < C4.W; a++) logits[a] = out[a];
        let max = -Infinity;
        for (const l of logits) max = Math.max(max, l);
        let z = 0;
        for (const l of logits) z += Math.exp(l - max);
        for (let a = 0; a < C4.W; a++) {
          if (s.pi[a] > 0) L -= s.pi[a] * (logits[a] - max - Math.log(z));
        }
        const v = Math.tanh(out[C4.W]);
        L += (v - s.z) * (v - s.z);
      }
      return L / sample.length;
    };

    // analytic gradient for exactly this batch
    t.net.zeroGrad();
    const d = new Float32Array(C4.W + 1);
    for (const s of sample) {
      const out = t.net.forward(s.x);
      const logits = new Float32Array(C4.W);
      for (let a = 0; a < C4.W; a++) logits[a] = out[a];
      let max = -Infinity;
      for (const l of logits) max = Math.max(max, l);
      let z = 0;
      for (const l of logits) z += Math.exp(l - max);
      for (let a = 0; a < C4.W; a++) {
        d[a] = (Math.exp(logits[a] - max) / z - s.pi[a]) / sample.length;
      }
      const v = Math.tanh(out[C4.W]);
      d[C4.W] = (2 * (v - s.z) * (1 - v * v)) / sample.length;
      t.net.backward(d);
    }

    // Directional derivative, the same criterion the other backward passes use.
    // eps matters here: the hidden layers are ReLU, so a larger step crosses
    // kinks and the "error" it reports is the kink, not the gradient. Checked
    // by sweeping eps — the discrepancy falls 1.2e-2 → 5.7e-5 from 1e-2 to
    // 1e-4, which is convergence rather than a bug.
    const check = (layer, eps, seedFn) => {
      const dir = new Float64Array(layer.W.length);
      let dot = 0;
      for (let i = 0; i < dir.length; i++) { dir[i] = seedFn(i); dot += dir[i] * layer.dW[i]; }
      const orig = Float32Array.from(layer.W);
      for (let i = 0; i < dir.length; i++) layer.W[i] = orig[i] + eps * dir[i];
      const lPlus = loss();
      for (let i = 0; i < dir.length; i++) layer.W[i] = orig[i] - eps * dir[i];
      const lMinus = loss();
      layer.W.set(orig);
      const numeric = (lPlus - lMinus) / (2 * eps);
      return Math.abs(numeric - dot) / Math.max(1e-9, Math.abs(numeric) + Math.abs(dot));
    };

    // The output layer carries both heads and has no ReLU above it, so this is
    // the clean test of the policy and value gradient maths itself.
    const outErr = check(t.net.layers[t.net.layers.length - 1], 1e-3, (i) => Math.cos(i * 7.13) % 1);
    cases.push([`training: policy+value head gradient exact (rel err ${outErr.toExponential(1)})`,
      outErr < 1e-4]);

    const deepErr = check(t.net.layers[0], 1e-4, (i) => Math.sin(i * 12.9898) * 43758.5453 % 1);
    cases.push([`training: gradient reaches the first layer (rel err ${deepErr.toExponential(1)})`,
      deepErr < 1e-3]);
  }

  for (const [name, ok] of cases) {
    if (!ok) failures++;
    console.log(`${ok ? '  ok  ' : ' FAIL '} ${('connect4  ' + name).padEnd(72)}`);
  }
}

/* ----------------------------------------------------- wordle + information
   No gradients here at all, which is the point of the lesson. What there is:
   a scoring rule everyone gets wrong on duplicate letters, and an entropy
   calculation that has to agree with what actually happens when you play the
   guess. Both are checked exactly. */
{
  const { wordle: W } = require('../assets/js/lib/wordle.js');
  const { WORDLE_WORDS: WORDS } = require('../assets/js/lib/wordle-words.js');
  const cases = [];
  const pat = (g, a) => W.patternDigits(W.score(g, a)).map((d) => '.yG'[d]).join('');

  cases.push(['word list: every entry is five lowercase letters',
    WORDS.every((w) => /^[a-z]{5}$/.test(w)) && new Set(WORDS).size === WORDS.length]);

  // The duplicate-letter rule: greens claim their letters first, and only the
  // answer's leftovers can make a later position yellow.
  cases.push(['scoring: an exact match is all green', pat('crane', 'crane') === 'GGGGG']);
  cases.push(['scoring: geese/these leaves the first e grey', pat('geese', 'these') === '..GGG']);
  cases.push(['scoring: sassy/class spends both s letters correctly', pat('sassy', 'class') === 'yy.G.']);
  cases.push(['scoring: array/radar', pat('array', 'radar') === 'yyyG.']);
  cases.push(['scoring: abbey/babes', pat('abbey', 'babes') === 'yyGG.']);
  cases.push(['scoring: a letter absent from the answer is grey', pat('vivid', 'crane') === '.....']);

  // Scoring must be consistent with filtering: the answer always survives its
  // own colours, and everything that survives really does produce them.
  {
    let ok = true;
    for (let i = 0; i < 40; i++) {
      const answer = WORDS[(i * 37) % WORDS.length];
      const guess = WORDS[(i * 101 + 5) % WORDS.length];
      const code = W.score(guess, answer);
      const kept = W.filter(WORDS, guess, code);
      if (!kept.includes(answer)) ok = false;
      for (const k of kept) if (W.score(guess, k) !== code) ok = false;
    }
    cases.push(['filtering: the answer always survives, and survivors all match', ok]);
  }

  // Entropy has to mean what the lesson says it means.
  {
    const single = ['crane'];
    cases.push(['entropy: with one candidate left, every guess is worth 0 bits',
      Math.abs(W.entropy('slate', single)) < 1e-12]);

    // A guess that splits n candidates into n equally sized buckets is worth
    // exactly log2(n) bits; the opener cannot beat the total uncertainty.
    const bits = W.entropy('trace', WORDS);
    cases.push([`entropy: an opener cannot exceed log2(list) (${bits.toFixed(2)} < ${Math.log2(WORDS.length).toFixed(2)})`,
      bits > 0 && bits < Math.log2(WORDS.length)]);

    // N / 2^H is not the expected number of survivors — it is their weighted
    // GEOMETRIC mean, which is always the smaller of the two. Worth pinning
    // down, because quoting one and calling it the other overstates how much a
    // guess achieves.
    const counts = new Int32Array(W.PATTERNS);
    for (const w of WORDS) counts[W.score('trace', w)]++;
    let arithmetic = 0, logGeo = 0;
    for (let c = 0; c < W.PATTERNS; c++) {
      if (!counts[c]) continue;
      const p = counts[c] / WORDS.length;
      arithmetic += p * counts[c];
      logGeo += p * Math.log(counts[c]);
    }
    const geometric = Math.exp(logGeo);
    const fromBits = WORDS.length / Math.pow(2, bits);
    cases.push([`entropy: N/2^bits is exactly the geometric mean (${geometric.toFixed(2)} vs ${fromBits.toFixed(2)})`,
      Math.abs(geometric - fromBits) < 1e-6]);
    cases.push([`entropy: the true expected remainder is larger (${arithmetic.toFixed(1)} > ${geometric.toFixed(1)})`,
      arithmetic > geometric]);

    // A word of five identical letters can barely split anything.
    const flat = W.entropy('vivid', WORDS);
    cases.push([`entropy: repeated letters score worse (${flat.toFixed(2)} < ${bits.toFixed(2)})`, flat < bits]);
  }

  // The solver must actually solve, from any answer, without ever exceeding six.
  {
    let worst = 0, total = 0, failed = 0;
    for (let i = 0; i < WORDS.length; i += 7) {
      const g = W.solve(WORDS[i], WORDS, 'trace');
      const last = g[g.length - 1];
      if (!last || last.code !== W.ALL_GREEN) failed++;
      worst = Math.max(worst, g.length);
      total += g.length;
    }
    const n = Math.ceil(WORDS.length / 7);
    cases.push([`solver: always finds the word (mean ${(total / n).toFixed(2)}, worst ${worst})`,
      failed === 0 && worst <= 6]);
  }

  // Ranking must prefer information, and must be able to name a word that
  // cannot win when that word splits the survivors better.
  {
    const ranked = W.rankGuesses(WORDS, WORDS, 5);
    const descending = ranked.every((r, i) => i === 0 || r.bits <= ranked[i - 1].bits);
    cases.push(['ranking: guesses come back in descending order of information', descending]);
  }

  for (const [name, ok] of cases) {
    if (!ok) failures++;
    console.log(`${ok ? '  ok  ' : ' FAIL '} ${('wordle  ' + name).padEnd(72)}`);
  }
}

/* ------------------------------------------------- game theory + regret
   The claims this lesson makes are strong and checkable: the game is zero-sum,
   the equilibrium is unexploitable, and being unexploitable earns nothing. */
{
  const { GT } = require('../assets/js/lib/gametheory.js');
  const cases = [];
  const N = GT.N;

  {
    let anti = true;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (GT.PAYOFF[i][j] !== -GT.PAYOFF[j][i]) anti = false;
    }
    cases.push(['the game is zero-sum (payoffs antisymmetric)', anti]);
  }
  {
    // No action may dominate another, or the game has no mixture to find.
    let dominated = false;
    for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) {
      if (i === k) continue;
      let alwaysBetter = true;
      for (let j = 0; j < N; j++) if (GT.PAYOFF[i][j] <= GT.PAYOFF[k][j]) alwaysBetter = false;
      if (alwaysBetter) dominated = true;
    }
    cases.push(['no move beats another whatever the opponent does', !dominated]);
  }

  const eq = GT.solve(120000);
  cases.push([`equilibrium is unexploitable (${eq.exploitability.toFixed(4)} per round)`,
    Math.abs(eq.exploitability) < 0.05]);
  cases.push(['equilibrium uses every option', eq.mix.every((p) => p > 0.02)]);
  {
    // The rarest option should be the throw: it loses to both strikes and only
    // beats blocks. If that ever stops being true the payoffs have drifted.
    let rarest = 0;
    for (let i = 1; i < N; i++) if (eq.mix[i] < eq.mix[rarest]) rarest = i;
    cases.push([`the throw is the rarest option at equilibrium (${(eq.mix[2] * 100).toFixed(1)}%)`,
      GT.ACTIONS[rarest].id === 'throw']);
  }

  {
    // Unexploitable means it earns nothing from anyone, including a sucker.
    const habit = new Float64Array(N);
    habit[3] = 0.55; habit[0] = 0.15; habit[1] = 0.15; habit[2] = 0.1; habit[4] = 0.05;
    const v = GT.value(eq.mix, habit);
    cases.push([`equilibrium earns nothing even against a habit (${v.toFixed(3)} per round)`,
      Math.abs(v) < 0.06]);
    // …while a perfect reader takes real money off that same habit.
    const br = GT.bestResponse(habit);
    cases.push([`a best response does punish that habit (+${br.value.toFixed(2)} per round)`,
      br.value > 0.5]);
  }

  {
    // A pure strategy is always exploitable; that is what makes mixing necessary.
    let allPunishable = true;
    for (let i = 0; i < N; i++) {
      const pure = new Float64Array(N);
      pure[i] = 1;
      if (GT.exploitability(pure) <= 0) allPunishable = false;
    }
    cases.push(['every single fixed move can be punished', allPunishable]);
  }

  {
    // Regret matching must beat a fixed opponent, which is the other half of
    // the lesson: against something that does not adapt, adapting pays.
    const rm = new GT.RegretMatcher();
    let total = 0;
    const fixed = 3;
    for (let t = 0; t < 4000; t++) {
      const a = rm.act();
      total += GT.PAYOFF[a.action][fixed];
      rm.observe(a.action, fixed);
    }
    cases.push([`regret matching learns to punish a fixed opponent (+${(total / 4000).toFixed(2)} per round)`,
      total / 4000 > 1]);
  }

  {
    // The running average must be less exploitable than the current mixture:
    // this is the point the lesson makes, so it should be measured.
    const a = new GT.RegretMatcher(), b = new GT.RegretMatcher();
    for (let t = 0; t < 40000; t++) {
      const x = a.act(), y = b.act();
      a.observe(x.action, y.action);
      b.observe(y.action, x.action);
    }
    const curE = GT.exploitability(a.strategy());
    const avgE = GT.exploitability(a.average());
    cases.push([`the average is less exploitable than the current mixture (${avgE.toFixed(3)} vs ${curE.toFixed(2)})`,
      avgE < curE && avgE < 0.08]);
  }

  for (const [name, ok] of cases) {
    if (!ok) failures++;
    console.log(`${ok ? '  ok  ' : ' FAIL '} ${('fighter  ' + name).padEnd(72)}`);
  }
}

console.log(failures === 0
  ? '\nAll checks passed.'
  : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
