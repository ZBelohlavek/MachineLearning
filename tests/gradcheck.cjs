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

/* ---------------------------- the whole vision transformer, both variants */
for (const cfg of [{ useCLS: true, useNorm: true }, { useCLS: false, useNorm: false }]) {
  const rand = nn.mulberry32(17);
  const net = new vitLib.ViT({ imgSize: 8, patch: 2, dim: 16, mlpHidden: 32, classes: 4, rand, ...cfg });
  const img = new Float32Array(64).map(() => rand());
  const label = 2;
  const loss = () => net.lossAndGrad(net.forward(img), label).loss;
  net.zeroGrad();
  net.backward(net.lossAndGrad(net.forward(img), label).dLogits);
  const blocks = net.params().map((p) => ({ values: p.v, grads: p.g }));
  run(`vit.js  ViT (${cfg.useCLS ? 'class token' : 'mean pool'}, ${cfg.useNorm ? 'norm' : 'no norm'})`,
      loss, blocks, rand);
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

console.log(failures === 0
  ? '\nAll checks passed.'
  : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
