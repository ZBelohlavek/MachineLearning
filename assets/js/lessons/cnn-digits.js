/* ==========================================================================
   cnn-digits.js — build and train a convolutional network on digits that are
   generated in your browser, then draw one yourself and watch every feature
   map light up.

   There is no dataset download. Digits are rendered with the canvas text API
   in random fonts, sizes, rotations and offsets — which makes the training
   set genuinely varied, and lets you turn the augmentation up and down to see
   what it does to generalisation.
   ========================================================================== */

;(function () {
  'use strict';
  const { Conv2D, ReLU, MaxPool2, Flatten, FC, ConvNet,
          mulberry32, softmax, argmax, clamp, hidpi, fit, LineChart, heat, diverging, achieve,
          chrome, nextLinks, slider, pills, checkbox, statGrid, rafLoop,
          renderDigit, makeDigitSet, imageFromCanvas } = window.ML;
  const arcade = window.ML.arcade;

  const S = 20;                       // digit images are S x S

  /* ------------------------------------------------------------- page */
  document.addEventListener('DOMContentLoaded', () => {
    chrome('cnn', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.cnn);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'cnn');
    nextLinks(document.getElementById('next-links'), 'convolutions', 'attention', '../');

    let f1 = 8, f2 = 16, hiddenUnits = 32;
    let lr = 0.004, batchSize = 16, aug = 1;
    let net, epoch = 0, seen = 0, running = false, pretrained = false, fromScratch = true;
    let seed = 1234;

    function updateBanner() {
      const el = document.getElementById('model-banner');
      if (!el) return;
      el.innerHTML = pretrained
        ? '<b class="label">This network arrives pre-trained</b><p class="mb0">It has already seen ' +
          `${(seen || 0).toLocaleString()} generated digits, so you can draw one right now and it will ` +
          'work. Everything below still trains live on top of that — press <b>Start over</b> for ' +
          'random weights and the full from-scratch experience.</p>'
        : '<b class="label">Starting from random weights</b><p class="mb0">Nothing has been learned ' +
          'yet: the ten confidence bars are all about 10% and the drawing pad will be nonsense until ' +
          'you press <b>Train the network</b>.</p>';
      el.className = pretrained ? 'note good' : 'note warn';
    }
    let trainSet, testSet;
    let rng = mulberry32(seed);

    /* ------------------------- model ------------------------- */
    function build() {
      // Re-seeded here rather than at the call site, so every path that rebuilds
      // the network (a slider, a reset, a new seed) starts from the same weights.
      rng = mulberry32(seed + 7919);
      const l = [];
      let shape = [1, S, S];
      const c1 = new Conv2D(shape, f1, 3, rng); l.push(c1);
      l.push(new ReLU(c1.outShape));
      const p1 = new MaxPool2(c1.outShape); l.push(p1);
      const c2 = new Conv2D(p1.outShape, f2, 3, rng); l.push(c2);
      l.push(new ReLU(c2.outShape));
      const p2 = new MaxPool2(c2.outShape); l.push(p2);
      const fl = new Flatten(p2.outShape); l.push(fl);
      l.push(new FC(fl.outShape[0], hiddenUnits, 'relu', rng));
      l.push(new FC(hiddenUnits, 10, 'linear', rng));
      net = new ConvNet(l);
      net.conv1 = c1; net.conv2 = c2; net.pool1 = p1; net.pool2 = p2;
      net.arch = { f1, f2, hiddenUnits, S };
      window.__cnn = net;                      // used by tools/train-vision.cjs

      // A pre-trained network ships with the site so the drawing pad works the
      // moment the page opens. It is only loaded when the architecture on screen
      // still matches the one it was trained with.
      pretrained = false;
      fromScratch = true;
      const saved = window.ML_CNN_MODEL;
      if (saved && saved.arch && saved.arch.f1 === f1 && saved.arch.f2 === f2 &&
          saved.arch.hiddenUnits === hiddenUnits) {
        try { net.loadJSON(saved.model); pretrained = true; fromScratch = false; epoch = saved.epoch || 0; seen = saved.seen || 0; }
        catch (err) { console.warn('could not load the pre-trained CNN:', err); }
      }
      updateBanner();
      epoch = 0; seen = 0;
      chart.clear();
      setStat('parameters', net.numParams().toLocaleString());
      drawArchitecture();
    }

    function regenerateData() {
      const dataRng = mulberry32(seed);            // data has its own stream
      trainSet = makeDigitSet(1000, dataRng, aug, S);
      testSet = makeDigitSet(300, mulberry32(seed + 999), aug, S);
      drawSamples();
    }

    /* ------------------------- training ------------------------- */
    let order = [], cursor = 0;
    function shuffleOrder() {
      order = trainSet.xs.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = (rng() * (i + 1)) | 0;
        [order[i], order[j]] = [order[j], order[i]];
      }
      cursor = 0;
    }

    let runningLoss = 0, runningCorrect = 0, runningCount = 0;

    function trainBatch() {
      if (cursor >= order.length) { shuffleOrder(); epoch++; }
      const end = Math.min(order.length, cursor + batchSize);
      net.zeroGrad();
      let loss = 0, correct = 0;
      for (let i = cursor; i < end; i++) {
        const idx = order[i];
        const logits = net.forward(trainSet.xs[idx]);
        const { loss: L, probs, dLogits } = net.lossAndGrad(logits, trainSet.ys[idx]);
        loss += L;
        if (argmax(probs) === trainSet.ys[idx]) correct++;
        net.backward(dLogits);
      }
      const size = end - cursor;
      net.step(lr, 1 / size);
      cursor = end;
      seen += size;
      if (pretrained && seen > (window.ML_CNN_MODEL?.seen || 0) + 2000) { pretrained = false; updateBanner(); }
      runningLoss += loss; runningCorrect += correct; runningCount += size;
    }

    function evaluate() {
      let correct = 0;
      const conf = new Int32Array(100);
      const wrong = [];
      for (let i = 0; i < testSet.n; i++) {
        const probs = softmax(net.forward(testSet.xs[i]));
        const pred = argmax(probs);
        const truth = testSet.ys[i];
        conf[truth * 10 + pred]++;
        if (pred === truth) correct++;
        else if (wrong.length < 10) wrong.push({ i, pred, truth, p: probs[pred] });
      }
      return { acc: correct / testSet.n, conf, wrong };
    }

    /* ------------------------- rendering helpers ------------------------- */
    function paintTile(canvas, data, w, h, opts = {}) {
      const ctx = canvas._ctx || (canvas._ctx = hidpi(canvas, canvas.clientWidth || 60, canvas.clientHeight || 60));
      const buf = canvas._buf || (canvas._buf = document.createElement('canvas'));
      buf.width = w; buf.height = h;
      const g = buf.getContext('2d');
      const im = g.createImageData(w, h);
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < w * h; i++) { if (data[i] < lo) lo = data[i]; if (data[i] > hi) hi = data[i]; }
      const span = Math.max(1e-6, hi - lo);
      for (let i = 0; i < w * h; i++) {
        const t = opts.raw ? clamp(data[i], 0, 1) : (data[i] - lo) / span;
        let r, gg, b;
        if (opts.raw) {
          // raw pixel data is shown as plain grey levels — that is what it is
          r = gg = b = Math.round(t * 255);
        } else if (opts.signed) {
          const c = diverging(data[i], Math.max(Math.abs(lo), Math.abs(hi)) || 1);
          const m = c.match(/\d+/g);
          r = +m[0]; gg = +m[1]; b = +m[2];
        } else {
          const c = heat(t).match(/\d+/g);
          r = +c[0]; gg = +c[1]; b = +c[2];
        }
        im.data[i * 4] = r; im.data[i * 4 + 1] = gg; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = 255;
      }
      g.putImageData(im, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, ctx._cssW, ctx._cssH);
      ctx.drawImage(buf, 0, 0, ctx._cssW, ctx._cssH);
    }

    function tileCanvas(parent, px, cls = '') {
      const c = document.createElement('canvas');
      c.className = 'tile ' + cls;
      c.style.width = px + 'px'; c.style.height = px + 'px';
      parent.appendChild(c);
      c._ctx = hidpi(c, px, px);
      return c;
    }

    /* ------------------------- sample gallery ------------------------- */
    function drawSamples() {
      const host = document.getElementById('samples');
      host.innerHTML = '';
      for (let i = 0; i < 30; i++) {
        const idx = (i * 37) % trainSet.n;
        const wrap = document.createElement('div');
        wrap.className = 'sample';
        host.appendChild(wrap);
        const c = tileCanvas(wrap, 42);
        paintTile(c, trainSet.xs[idx], S, S, { raw: true });
        const lab = document.createElement('div');
        lab.className = 'sample-label';
        lab.textContent = trainSet.ys[idx];
        wrap.appendChild(lab);
      }
    }

    /* ------------------------- architecture diagram ------------------------- */
    function drawArchitecture() {
      const host = document.getElementById('arch-diagram');
      const shapes = [
        ['input', [1, S, S]],
        ['conv 3×3', net.conv1.outShape],
        ['pool', net.pool1.outShape],
        ['conv 3×3', net.conv2.outShape],
        ['pool', net.pool2.outShape],
        ['dense', [hiddenUnits]],
        ['output', [10]],
      ];
      host.innerHTML = shapes.map(([name, s], i) => `
        <div class="arch-step">
          <div class="arch-box"><b>${name}</b><span>${s.join('×')}</span></div>
          ${i < shapes.length - 1 ? '<div class="arch-arrow">→</div>' : ''}
        </div>`).join('');
    }

    /* ------------------------- feature map panels ------------------------- */
    let fmTiles = null;
    function buildFeatureTiles() {
      const host = document.getElementById('feature-maps');
      host.innerHTML = '';
      fmTiles = { conv1: [], conv2: [], kernels: [] };
      const section = (title, sub) => {
        const h = document.createElement('div');
        h.className = 'fm-section';
        h.innerHTML = `<div class="fm-title">${title}</div><div class="fm-sub">${sub}</div>`;
        host.appendChild(h);
        const row = document.createElement('div');
        row.className = 'fm-row';
        host.appendChild(row);
        return row;
      };
      let row = section(`Layer 1 kernels · ${f1} × 3×3`, 'the weights the network invented for itself');
      for (let i = 0; i < f1; i++) fmTiles.kernels.push(tileCanvas(row, 34, 'kern'));
      row = section(`Layer 1 feature maps · ${f1} × ${S}×${S}`, 'each kernel’s response to the digit you drew');
      for (let i = 0; i < f1; i++) fmTiles.conv1.push(tileCanvas(row, 46));
      row = section(`Layer 2 feature maps · ${f2} × 10×10`, 'combinations of layer 1, at half the resolution');
      for (let i = 0; i < f2; i++) fmTiles.conv2.push(tileCanvas(row, 36));
    }

    function drawFeatureMaps() {
      if (!fmTiles) return;
      const c1 = net.conv1, c2 = net.conv2;
      const k = 3;
      for (let i = 0; i < f1; i++) {
        paintTile(fmTiles.kernels[i], c1.w.v.subarray(i * k * k, (i + 1) * k * k), k, k, { signed: true });
        paintTile(fmTiles.conv1[i], net.activations[2].subarray(i * S * S, (i + 1) * S * S), S, S);
      }
      const h2 = c2.outShape[1], w2 = c2.outShape[2];
      for (let i = 0; i < f2; i++) {
        paintTile(fmTiles.conv2[i], net.activations[5].subarray(i * h2 * w2, (i + 1) * h2 * w2), h2, w2);
      }
    }

    /* ------------------------- draw-your-own ------------------------- */
    const padCanvas = document.getElementById('pad');
    const padSize = 224;
    const pctx = hidpi(padCanvas, padSize, padSize);
    const padBuf = document.createElement('canvas');
    padBuf.width = padSize; padBuf.height = padSize;
    const pbctx = padBuf.getContext('2d', { willReadFrequently: true });
    function clearPad() {
      pbctx.fillStyle = '#000';
      pbctx.fillRect(0, 0, padSize, padSize);
      renderPad();
      predictPad();
    }
    function renderPad() {
      pctx.clearRect(0, 0, padSize, padSize);
      pctx.drawImage(padBuf, 0, 0, padSize, padSize);
      pctx.strokeStyle = 'rgba(255,255,255,.06)';
      pctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        pctx.beginPath();
        pctx.moveTo((i * padSize) / 4, 0); pctx.lineTo((i * padSize) / 4, padSize);
        pctx.moveTo(0, (i * padSize) / 4); pctx.lineTo(padSize, (i * padSize) / 4);
        pctx.stroke();
      }
    }
    let drawingPad = false, lastPt = null;
    const padPos = (ev) => {
      const r = padCanvas.getBoundingClientRect();
      const p = ev.touches ? ev.touches[0] : ev;
      return { x: ((p.clientX - r.left) / r.width) * padSize, y: ((p.clientY - r.top) / r.height) * padSize };
    };
    padCanvas.addEventListener('pointerdown', (e) => { drawingPad = true; lastPt = padPos(e); strokePad(lastPt, lastPt); });
    padCanvas.addEventListener('pointermove', (e) => {
      if (!drawingPad) return;
      const p = padPos(e);
      strokePad(lastPt, p);
      lastPt = p;
    });
    window.addEventListener('pointerup', () => {
      if (!drawingPad) return;
      drawingPad = false;
      predictPad();
      if (duelPendingClear) { duelPendingClear = false; setTimeout(clearPad, 150); }
    });
    function strokePad(a, b) {
      pbctx.strokeStyle = '#fff';
      pbctx.lineWidth = 20;
      pbctx.lineCap = 'round';
      pbctx.lineJoin = 'round';
      pbctx.beginPath();
      pbctx.moveTo(a.x, a.y);
      pbctx.lineTo(b.x, b.y);
      pbctx.stroke();
      renderPad();
      predictPad();
    }

    const smallCanvas = document.getElementById('pad-small');
    smallCanvas._ctx = hidpi(smallCanvas, 92, 92);
    const barsHost = document.getElementById('prob-bars');
    const bars = [];
    for (let d = 0; d < 10; d++) {
      const row = document.createElement('div');
      row.className = 'pbar';
      row.innerHTML = `<span class="d">${d}</span><span class="track"><i></i></span><span class="v">0%</span>`;
      barsHost.appendChild(row);
      bars.push({ row, fill: row.querySelector('i'), val: row.querySelector('.v') });
    }

    function predictPad() {
      // downscale the 224px pad to the 20x20 the network expects
      const img = imageFromCanvas(padBuf, S);
      paintTile(smallCanvas, img, S, S, { raw: true });
      let ink = 0;
      for (let i = 0; i < img.length; i++) ink += img[i];

      const probs = softmax(net.forward(img));
      const best = argmax(probs);
      bars.forEach((b, i) => {
        b.fill.style.width = (probs[i] * 100).toFixed(1) + '%';
        b.val.textContent = (probs[i] * 100).toFixed(0) + '%';
        b.row.classList.toggle('best', i === best);
      });
      document.getElementById('pad-verdict').innerHTML =
        `I think that's a <b>${best}</b> <span class="muted">(${(probs[best] * 100).toFixed(1)}% confident)</span>`;
      duelCheck(probs, best, ink);
      drawFeatureMaps();
    }

    document.getElementById('btn-clear-pad').addEventListener('click', clearPad);
    document.getElementById('btn-random-digit').addEventListener('click', () => {
      // paint a generated digit onto the pad so you can test without drawing
      const d = (Math.random() * 10) | 0;
      const img = renderDigit(d, mulberry32((Math.random() * 1e9) | 0), aug, S);
      pbctx.fillStyle = '#000';
      pbctx.fillRect(0, 0, padSize, padSize);
      const tmp = document.createElement('canvas');
      tmp.width = S; tmp.height = S;
      const tg = tmp.getContext('2d');
      const im = tg.createImageData(S, S);
      for (let i = 0; i < S * S; i++) {
        const v = Math.round(img[i] * 255);
        im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v;
        im.data[i * 4 + 3] = 255;
      }
      tg.putImageData(im, 0, 0);
      pbctx.imageSmoothingEnabled = true;
      pbctx.drawImage(tmp, 0, 0, padSize, padSize);
      renderPad();
      predictPad();
    });

    /* ------------------------- the digit duel -------------------------
       A 60-second game: the network names a digit, you draw it, and it has to
       recognise your handwriting. It is a much better test of the model than
       any accuracy number, because your handwriting is not in its training set.

       Scoring rewards two things the plain point-per-digit version did not:
       drawing quickly, and stringing hits together. Both push you to draw the
       clear, unambiguous shape the network is most confident about, which is
       the behaviour that actually teaches you what it learned.
       ------------------------------------------------------------------- */
    const DUEL_SECONDS = 60;
    const SKIP_PENALTY_MS = 3000;
    const DUEL_TIERS = { gold: 250, silver: 140, bronze: 60 };
    let duel = null, duelPendingClear = false;
    const duelTarget = document.getElementById('duel-target');
    const duelTimer = document.getElementById('duel-timer');
    const duelMsg = document.getElementById('duel-msg');
    const padFx = arcade.fx();

    const duelHud = arcade.hud(document.getElementById('duel-hud'), [
      { name: 'score', label: 'score', value: '0' },
      { name: 'combo', label: 'multiplier', value: '×1' },
      { name: 'hits', label: 'digits', value: '0' },
      { name: 'best', label: 'best', value: '—' },
    ]);
    arcade.soundToggle(document.getElementById('duel-sound'));

    function showBest() {
      const b = arcade.readBest('cnn-duel');
      duelHud.set('best', b === null ? '—' : b);
      const track = document.getElementById('duel-medals');
      if (!track) return;
      track.innerHTML = ['gold', 'silver', 'bronze'].map((t) =>
        `<span class="${b !== null && b >= DUEL_TIERS[t] ? 'won' : ''}">` +
        `${arcade.MEDALS[t].icon} <b>${DUEL_TIERS[t]}</b></span>`).join('') +
        '<span>drawing faster and chaining hits is where the points are</span>';
    }

    function nextTarget() {
      let d;
      do { d = (Math.random() * 10) | 0; } while (duel && d === duel.target && Math.random() < 0.8);
      duel.target = d;
      duel.shownAt = performance.now();
      duelTarget.textContent = d;
    }

    function startDuel() {
      duelPendingClear = false;
      duel = {
        target: 0, shownAt: 0, score: 0, hits: 0, bestStreak: 0,
        ends: performance.now() + DUEL_SECONDS * 1000, done: false,
        combo: arcade.combo({ windowMs: 7000, cap: 5 }),
      };
      nextTarget();
      clearPad();
      duelMsg.innerHTML = '';
      duelHud.set('score', '0');
      duelHud.set('hits', '0');
      duelHud.set('combo', '×1');
      document.getElementById('btn-duel').textContent = 'Restart duel';
      document.getElementById('duel-panel').classList.add('live');
      arcade.sfx('start');
    }

    function endDuel() {
      duel.done = true;
      document.getElementById('duel-panel').classList.remove('live');
      const { score, hits, bestStreak } = duel;
      const res = arcade.resultCard(score, {
        tiers: DUEL_TIERS, bestKey: 'cnn-duel',
        note: `${hits} digit${hits === 1 ? '' : 's'} recognised` +
              (bestStreak > 1 ? `, best streak ${bestStreak}` : '') + '. ' +
              (hits >= 8 ? 'The network is holding up well against real handwriting.'
               : hits >= 4 ? 'Train it a little longer, or draw bigger and bolder.'
               : 'If it misses easy digits it needs more training, or more augmentation.'),
      });
      duelMsg.innerHTML = res.html;
      showBest();
      arcade.sfx(res.tier === 'gold' ? 'win' : hits >= 4 ? 'score' : 'fail');
      if (res.tier && !arcade.isMuted()) padFx.burst(padSize / 2, padSize / 2,
        { count: 60, speed: 190, colors: ['#ffd166', '#38d39f', '#4da3ff'], gravity: 120 });
      if (hits >= 8) achieve('cnn-duel', `${hits} digits, ${score} points`);
      if (res.tier === 'gold') achieve('cnn-duel-gold', `${score} points in the digit duel`);
      duelTimer.style.width = '0%';
      duel = null;
    }

    document.getElementById('btn-duel').addEventListener('click', startDuel);
    document.getElementById('btn-duel-skip').addEventListener('click', () => {
      if (!duel || duel.done) return;
      // Skipping is allowed but never free, so "which digit is this?" stays a
      // decision rather than a button you hold down.
      duel.ends -= SKIP_PENALTY_MS;
      duel.combo.miss();
      duelHud.set('combo', '×1');
      duelHud.flash('combo', 'bad');
      arcade.sfx('tick');
      nextTarget();
      clearPad();
    });
    showBest();

    /**
     * Called after every prediction; scores a hit when the network is convinced.
     * `ink` guards against the blank pad: a network will happily name a digit
     * when shown an empty square, and that must not count as a point.
     */
    function duelCheck(probs, best, ink) {
      if (!duel || duel.done) return;
      if (ink < 5) return;
      if (best !== duel.target || probs[best] <= 0.6) return;

      const mult = duel.combo.hit();
      const seconds = (performance.now() - duel.shownAt) / 1000;
      const speed = Math.max(0, Math.round(10 * (1 - seconds / 6)));   // 10 → 0 over six seconds
      const points = (10 + speed) * mult;

      duel.score += points;
      duel.hits++;
      duel.bestStreak = Math.max(duel.bestStreak, duel.combo.streak);
      duelHud.set('score', duel.score);
      duelHud.set('hits', duel.hits);
      duelHud.set('combo', '×' + mult);
      duelHud.flash('score');
      if (mult > 1) duelHud.flash('combo');

      duelMsg.innerHTML = `<span style="color:var(--good)">Yes, that's a ${best}.</span> ` +
        `<span class="muted">+${points}${mult > 1 ? ` (×${mult} streak)` : ''}` +
        `${speed >= 7 ? ' · fast' : ''}</span>`;

      padFx.burst(padSize / 2, padSize / 2, {
        count: 14 + mult * 6, speed: 90 + mult * 30,
        colors: mult >= 3 ? ['#ffd166', '#ff9f45'] : ['#38d39f', '#4da3ff'],
      });
      padFx.flash('#38d39f', 0.14);
      arcade.sfx(mult > 1 ? 'combo' : 'hit', duel.combo.streak);

      nextTarget();
      // Wait for the pen to come up: clearing mid-stroke leaves the tail of
      // the stroke behind on the fresh pad.
      if (drawingPad) duelPendingClear = true;
      else setTimeout(clearPad, 180);
    }

    /* ------------------------- predict, then check -------------------------
       Scored against this run's own confusion matrix rather than a stored
       answer, so it is a real prediction about a real model.
       ---------------------------------------------------------------------- */
    const PAIR_OPTS = [
      { label: '4 and 9', value: '4-9' },
      { label: '3 and 8', value: '3-8' },
      { label: '1 and 7', value: '1-7' },
      { label: '5 and 6', value: '5-6' },
      { label: 'something else', value: 'other' },
    ];
    const confusionPredict = window.ML.predictBox(document.getElementById('confusion-predict'), {
      id: 'cnn-confusion-pair',
      question: 'Which two digits will this network mix up most often? Commit before you train it.',
      hint: 'Scored against your own confusion matrix.',
      options: PAIR_OPTS,
    });

    /** The pair with the most errors in both directions. */
    function worstPair(conf) {
      let best = null, bestN = 0, total = 0;
      for (let t = 0; t < 10; t++) {
        for (let p = t + 1; p < 10; p++) {
          const n = conf[t * 10 + p] + conf[p * 10 + t];
          total += n;
          if (n > bestN) { bestN = n; best = [t, p]; }
        }
      }
      return { pair: best, count: bestN, total };
    }

    function scoreConfusionPrediction(conf) {
      if (!confusionPredict || confusionPredict.choice === null) return;
      const { pair, count, total } = worstPair(conf);
      if (!pair || total < 6) return;                 // too few mistakes to call it yet
      const key = `${pair[0]}-${pair[1]}`;
      const known = PAIR_OPTS.some((o) => o.value === key);
      confusionPredict.reveal(known ? key : 'other',
        `Your network's most confused pair is <b>${pair[0]} and ${pair[1]}</b>: ${count} of its ` +
        `${total} mistakes. Which pairs collide depends on the fonts this run happened to draw, so ` +
        `train it again with a different seed and the answer can genuinely change.`);
    }

    /* ------------------------- confusion matrix ------------------------- */
    function drawConfusion(conf) {
      const host = document.getElementById('confusion');
      let html = '<table class="conf"><tr><th></th>';
      for (let p = 0; p < 10; p++) html += `<th>${p}</th>`;
      html += '</tr>';
      for (let t = 0; t < 10; t++) {
        html += `<tr><th>${t}</th>`;
        let rowTotal = 0;
        for (let p = 0; p < 10; p++) rowTotal += conf[t * 10 + p];
        for (let p = 0; p < 10; p++) {
          const v = conf[t * 10 + p];
          const frac = rowTotal ? v / rowTotal : 0;
          const bg = t === p
            ? `rgba(56,211,159,${0.08 + frac * 0.6})`
            : `rgba(255,107,107,${v ? 0.15 + frac * 0.7 : 0})`;
          html += `<td style="background:${bg}">${v || ''}</td>`;
        }
        html += '</tr>';
      }
      html += '</table>';
      host.innerHTML = html;
    }

    function drawMistakes(wrong) {
      const host = document.getElementById('mistakes');
      host.innerHTML = '';
      if (!wrong.length) {
        host.innerHTML = '<span class="muted">No mistakes on the test set. Turn the augmentation up.</span>';
        return;
      }
      for (const w of wrong) {
        const wrap = document.createElement('div');
        wrap.className = 'sample';
        host.appendChild(wrap);
        const c = tileCanvas(wrap, 42);
        paintTile(c, testSet.xs[w.i], S, S, { raw: true });
        const lab = document.createElement('div');
        lab.className = 'sample-label bad';
        lab.innerHTML = `${w.pred} <span class="muted">(${w.truth})</span>`;
        wrap.appendChild(lab);
      }
    }

    /* ------------------------- controls ------------------------- */
    const chart = new LineChart(document.getElementById('cnn-chart'), {
      height: 160, xLabel: 'images seen', yMin: 0, yMax: 1,
      series: [
        { name: 'train accuracy', color: '#4da3ff' },
        { name: 'test accuracy', color: '#38d39f' },
      ],
    });
    const setStat = statGrid(document.getElementById('cnn-stats'),
      ['epoch', 'images seen', 'train acc', 'test acc', 'loss', 'parameters']);

    const runControls = window.ML.runBar(document.getElementById('run-bar'), {
      seed,
      charts: () => [chart],
      onSeed: (v) => {
        seed = v;
        const keep = window.ML_CNN_MODEL;
        window.ML_CNN_MODEL = null;          // a chosen seed means training it yourself
        build();
        window.ML_CNN_MODEL = keep;
        epoch = 0; seen = 0;
        regenerateData(); shuffleOrder(); buildFeatureTiles(); predictPad();
      },
      note: 'Same seed, same digits and same starting weights.',
    });

    const say = window.ML.announcer();
    const btnTrain = document.getElementById('btn-train');
    btnTrain.addEventListener('click', () => {
      running = !running;
      btnTrain.textContent = running ? '⏸ Pause training' : '▶ Train the network';
      btnTrain.classList.toggle('primary', !running);
    });
    document.getElementById('btn-reset').addEventListener('click', () => {
      const keep = window.ML_CNN_MODEL;
      window.ML_CNN_MODEL = null;              // force genuinely random weights
      build();
      window.ML_CNN_MODEL = keep;
      epoch = 0; seen = 0;
      shuffleOrder(); buildFeatureTiles(); predictPad();
      setStat('epoch', 0); setStat('images seen', 0);
      setStat('train acc', '–'); setStat('test acc', '–'); setStat('loss', '–');
      chart.clear(); chart.draw();
    });

    const cfg = document.getElementById('cnn-config');
    slider(cfg, {
      label: 'layer 1 filters', min: 2, max: 16, step: 1, value: f1,
      format: (v) => v.toFixed(0), onInput: (v) => { f1 = v | 0; build(); shuffleOrder(); buildFeatureTiles(); },
      desc: 'How many different patterns the first layer may look for. Two is not enough to describe a digit; sixteen is plenty.',
    });
    slider(cfg, {
      label: 'layer 2 filters', min: 4, max: 32, step: 1, value: f2,
      format: (v) => v.toFixed(0), onInput: (v) => { f2 = v | 0; build(); shuffleOrder(); buildFeatureTiles(); },
    });
    slider(cfg, {
      label: 'learning rate', min: 0.0005, max: 0.02, step: 0.0005, value: lr,
      format: (v) => v.toFixed(4), onInput: (v) => { lr = v; },
    });
    slider(cfg, {
      label: 'batch size', min: 1, max: 64, step: 1, value: batchSize,
      format: (v) => v.toFixed(0), onInput: (v) => { batchSize = v | 0; },
    });
    slider(cfg, {
      label: 'augmentation', min: 0, max: 2, step: 0.05, value: aug,
      format: (v) => v.toFixed(2), onInput: (v) => { aug = v; },
      desc: 'How much the generated digits are rotated, resized, shifted and noised. Zero gives a clean, easy, unrealistic dataset.',
    });
    document.getElementById('btn-regen').addEventListener('click', () => { regenerateData(); shuffleOrder(); });

    /* ------------------------- main loop ------------------------- */
    let sinceEval = 0;
    rafLoop(() => {
      if (duel && !duel.done) {
        const left = duel.ends - performance.now();
        duelTimer.style.width = clamp(left / (DUEL_SECONDS * 1000), 0, 1) * 100 + '%';
        // The multiplier decays if you stall, so the HUD must show it going.
        if (duel.combo.stale() && duel.combo.streak) {
          duel.combo.reset();
          duelHud.set('combo', '×1');
        }
        if (left <= 0) endDuel();
      }
      if (padFx.busy) {
        renderPad();
        padFx.begin(pctx);
        padFx.end(pctx, 1 / 60, padSize, padSize);
      }
      if (!running) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 22) trainBatch();

      setStat('epoch', epoch);
      setStat('images seen', seen.toLocaleString());
      if (runningCount > 200) {
        const trainAcc = runningCorrect / runningCount;
        setStat('train acc', (trainAcc * 100).toFixed(1) + '%');
        setStat('loss', (runningLoss / runningCount).toFixed(3));
        runningLoss = runningCorrect = runningCount = 0;

        sinceEval += 1;
        if (sinceEval >= 3) {
          sinceEval = 0;
          const ev = evaluate();
          setStat('test acc', (ev.acc * 100).toFixed(1) + '%', ev.acc > 0.9 ? 'good' : '');
          say(`${seen.toLocaleString()} images seen. Test accuracy ${(ev.acc * 100).toFixed(0)} percent.`);
          if (ev.acc >= 0.95 && fromScratch) {
            achieve('cnn-trained', `${(ev.acc * 100).toFixed(1)}% test accuracy from random weights`);
          }
          chart.push(seen, [trainAcc, ev.acc]);
          drawConfusion(ev.conf);
          drawMistakes(ev.wrong);
          scoreConfusionPrediction(ev.conf);
        } else {
          chart.push(seen, [trainAcc, null]);
        }
        chart.draw();
      }
      drawFeatureMaps();
    }).start();

    /* ------------------------- go ------------------------- */
    build();
    regenerateData();
    shuffleOrder();
    buildFeatureTiles();
    clearPad();
    drawConfusion(new Int32Array(100));
  });
})();
