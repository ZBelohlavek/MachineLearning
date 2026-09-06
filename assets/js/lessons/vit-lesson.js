/* ==========================================================================
   vit-lesson.js — patches, attention and a Vision Transformer you can train.
   ========================================================================== */

;(function () {
  'use strict';
  const { ViT, mulberry32, softmax, argmax, clamp, hidpi, LineChart, heat, diverging, achieve,
          chrome, nextLinks, slider, pills, statGrid, rafLoop,
          renderDigit, makeDigitSet, imageFromCanvas } = window.ML;

  const S = 20;

  document.addEventListener('DOMContentLoaded', () => {
    chrome('attention', '../');
    nextLinks(document.getElementById('next-links'), 'cnn', 'gridworld', '../');

    let patch = 4, dim = 24, lr = 0.004, batchSize = 16, aug = 1;
    let useCLS = false, useNorm = false, trainSize = 3000;
    let net, trainSet, testSet, running = false, seen = 0, epoch = 0, pretrained = false, fromScratch = true;
    const trainedArch = { cls: false, mean: false };

    function updateBanner() {
      const el = document.getElementById('model-banner');
      if (!el) return;
      el.innerHTML = pretrained
        ? '<b class="label">This transformer arrives pre-trained</b><p class="mb0">Its attention ' +
          'maps are already meaningful, so the overlay below shows something real rather than noise. ' +
          'Press <b>Start over</b> — or change any architecture setting — for random weights and the ' +
          'from-scratch version.</p>'
        : '<b class="label">Random weights</b><p class="mb0">Attention is currently near-uniform: ' +
          'every patch is mildly interested in everything. That flat map is exactly what an untrained ' +
          'transformer looks like, and it is worth a look before you train it.</p>';
      el.className = pretrained ? 'note good' : 'note warn';
    }
    let current = null;               // the image currently being inspected
    let selected = 0;                 // which patch is the query
    let attnMode = 'cls';
    const rng = mulberry32(4242);

    /* ---------------------------------------------------------- model */
    function build() {
      net = new ViT({ imgSize: S, patch, dim, mlpHidden: dim * 2, classes: 10, rand: rng, useCLS, useNorm });
      window.__vit = net;                      // used by tools/train-vision.cjs
      net.arch = { patch, dim, useCLS, useNorm };

      pretrained = false;
      fromScratch = true;
      const saved = window.ML_VIT_MODEL;
      if (saved && saved.arch && saved.arch.patch === patch && saved.arch.dim === dim &&
          saved.arch.useCLS === useCLS && saved.arch.useNorm === useNorm) {
        try { net.loadJSON(saved.model); pretrained = true; fromScratch = false; seen = saved.seen || 0; epoch = saved.epoch || 0; }
        catch (err) { console.warn('could not load the pre-trained ViT:', err); }
      }
      updateBanner();
      seen = 0; epoch = 0;
      selected = Math.floor(net.Tp / 2);            // a patch index, 0..Tp-1
      chart.clear();
      setStat('parameters', net.numParams().toLocaleString());
      setStat('patches', `${net.grid}×${net.grid}${net.useCLS ? ' + cls' : ''}`);
      buildPatchStrip();
      buildPosPanel();
      refreshAll();
    }

    function regenerateData() {
      trainSet = makeDigitSet(trainSize, rng, aug, S);
      testSet = makeDigitSet(300, mulberry32(31337), aug, S);
      shuffleOrder();
    }

    let order = [], cursor = 0;
    function shuffleOrder() {
      order = trainSet.xs.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = (rng() * (i + 1)) | 0;
        [order[i], order[j]] = [order[j], order[i]];
      }
      cursor = 0;
    }

    let runLoss = 0, runCorrect = 0, runCount = 0;
    function trainBatch() {
      if (cursor >= order.length) { shuffleOrder(); epoch++; }
      const end = Math.min(order.length, cursor + batchSize);
      net.zeroGrad();
      for (let i = cursor; i < end; i++) {
        const idx = order[i];
        const logits = net.forward(trainSet.xs[idx]);
        const { loss, probs, dLogits } = net.lossAndGrad(logits, trainSet.ys[idx]);
        runLoss += loss;
        if (argmax(probs) === trainSet.ys[idx]) runCorrect++;
        net.backward(dLogits);
      }
      const size = end - cursor;
      net.step(lr, 1 / size);
      runCount += size;
      seen += size;
      if (pretrained && seen > (window.ML_VIT_MODEL?.seen || 0) + 3000) { pretrained = false; updateBanner(); }
      cursor = end;
    }

    function evaluate() {
      let correct = 0;
      for (let i = 0; i < testSet.n; i++) {
        if (argmax(softmax(net.forward(testSet.xs[i]))) === testSet.ys[i]) correct++;
      }
      return correct / testSet.n;
    }

    /* How many tokens sit in front of the patch tokens: 1 with a class token,
       0 when the model averages the patches instead. */
    const clsOff = () => net.T - net.Tp;

    /* ------------------------------------------------------- drawing */
    function grayTile(canvas, data, w, h, opts = {}) {
      const ctx = canvas._ctx;
      const buf = canvas._buf || (canvas._buf = document.createElement('canvas'));
      buf.width = w; buf.height = h;
      const g = buf.getContext('2d');
      const im = g.createImageData(w, h);
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < w * h; i++) { if (data[i] < lo) lo = data[i]; if (data[i] > hi) hi = data[i]; }
      const span = Math.max(1e-6, hi - lo);
      for (let i = 0; i < w * h; i++) {
        let r, gg, b;
        if (opts.mode === 'heat') {
          const c = heat((data[i] - lo) / span).match(/\d+/g);
          r = +c[0]; gg = +c[1]; b = +c[2];
        } else if (opts.mode === 'signed') {
          const c = diverging(data[i], Math.max(Math.abs(lo), Math.abs(hi)) || 1).match(/\d+/g);
          r = +c[0]; gg = +c[1]; b = +c[2];
        } else {
          r = gg = b = Math.round(clamp(data[i], 0, 1) * 255);
        }
        im.data[i * 4] = r; im.data[i * 4 + 1] = gg; im.data[i * 4 + 2] = b; im.data[i * 4 + 3] = 255;
      }
      g.putImageData(im, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, ctx._cssW, ctx._cssH);
      ctx.drawImage(buf, 0, 0, ctx._cssW, ctx._cssH);
    }

    function tile(parent, px, cls = '') {
      const c = document.createElement('canvas');
      c.className = 'tile ' + cls;
      c.style.width = px + 'px'; c.style.height = px + 'px';
      parent.appendChild(c);
      c._ctx = hidpi(c, px, px);
      return c;
    }

    const mainCanvas = document.getElementById('vit-image');
    const MAIN = 260;
    mainCanvas._ctx = hidpi(mainCanvas, MAIN, MAIN);
    const attnCanvas = document.getElementById('attn-matrix');
    attnCanvas._ctx = hidpi(attnCanvas, 220, 220);

    function drawMain() {
      if (!current) return;
      grayTile(mainCanvas, current, S, S);
      const ctx = mainCanvas._ctx;
      const g = net.grid, cell = MAIN / g;

      // Attention over the patches. Token 0 is the class token, so patch p
      // lives at token p+1 and the overlays drop that first column.
      let weights = null;
      const off = clsOff();
      if (attnMode === 'from') weights = net.attentionFrom(selected + off).subarray(off);
      else if (attnMode === 'to') weights = net.attentionTo(selected + off).subarray(off);
      else weights = net.clsAttention();
      let max = 0;
      for (const w of weights) max = Math.max(max, w);
      for (let t = 0; t < net.Tp; t++) {
        const gx = t % g, gy = (t / g) | 0;
        const a = weights[t] / (max || 1);
        ctx.fillStyle = `rgba(124, 92, 255, ${(a * 0.72).toFixed(3)})`;
        ctx.fillRect(gx * cell, gy * cell, cell, cell);
      }

      // grid + the selected patch
      ctx.strokeStyle = 'rgba(255,255,255,.18)';
      ctx.lineWidth = 1;
      for (let i = 1; i < g; i++) {
        ctx.beginPath();
        ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, MAIN);
        ctx.moveTo(0, i * cell); ctx.lineTo(MAIN, i * cell);
        ctx.stroke();
      }
      if (attnMode !== 'cls') {
        const sx = selected % g, sy = (selected / g) | 0;
        ctx.strokeStyle = '#ffd166';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(sx * cell + 1, sy * cell + 1, cell - 2, cell - 2);
      }
    }

    function drawAttnMatrix() {
      const T = net.T;
      grayTile(attnCanvas, net.attn.A, T, T, { mode: 'heat' });
      const ctx = attnCanvas._ctx;
      const row = attnMode === 'cls' ? 0 : selected + clsOff();
      ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 2;
      if (attnMode === 'to') ctx.strokeRect((row / T) * 220, 0, 220 / T, 220);
      else ctx.strokeRect(0, (row / T) * 220, 220, 220 / T);
    }

    /* -------- the patch sequence strip -------- */
    let stripTiles = [];
    function buildPatchStrip() {
      const host = document.getElementById('patch-strip');
      host.innerHTML = '';
      stripTiles = [];
      const px = Math.max(20, Math.min(38, Math.floor(560 / net.grid)));
      for (let t = 0; t < net.Tp; t++) {
        const wrap = document.createElement('div');
        wrap.className = 'patch-item';
        host.appendChild(wrap);
        const c = tile(wrap, px, 'patchtile');
        c.addEventListener('click', () => { selected = t; refreshAll(); });
        const lab = document.createElement('div');
        lab.className = 'patch-idx';
        lab.textContent = t;
        wrap.appendChild(lab);
        stripTiles.push({ canvas: c, wrap });
      }
    }
    function drawPatchStrip() {
      const pd = net.patchDim, p = net.patch;
      for (let t = 0; t < net.Tp; t++) {
        grayTile(stripTiles[t].canvas, net.patches.subarray(t * pd, (t + 1) * pd), p, p);
        stripTiles[t].wrap.classList.toggle('sel', t === selected);
      }
    }

    /* -------- position embedding similarity -------- */
    let posTiles = [];
    function buildPosPanel() {
      const host = document.getElementById('pos-grid');
      host.innerHTML = '';
      host.style.gridTemplateColumns = `repeat(${net.grid}, auto)`;
      posTiles = [];
      const px = Math.max(16, Math.min(38, Math.floor(420 / net.grid)));
      for (let t = 0; t < net.Tp; t++) posTiles.push(tile(host, px, 'postile'));
    }
    function drawPosPanel() {
      const Tp = net.Tp, d = net.d, g = net.grid;
      const P = net.pos.v.subarray(clsOff() * d);   // skip the class token's position, if any
      const norms = new Float32Array(Tp);
      for (let t = 0; t < Tp; t++) {
        let n = 0;
        for (let i = 0; i < d; i++) n += P[t * d + i] ** 2;
        norms[t] = Math.sqrt(n) + 1e-9;
      }
      const sim = new Float32Array(Tp);
      for (let t = 0; t < Tp; t++) {
        for (let u = 0; u < Tp; u++) {
          let dot = 0;
          for (let i = 0; i < d; i++) dot += P[t * d + i] * P[u * d + i];
          sim[u] = dot / (norms[t] * norms[u]);
        }
        grayTile(posTiles[t], sim, g, g, { mode: 'signed' });
      }
    }

    /* -------- prediction read-out -------- */
    const barsHost = document.getElementById('vit-probs');
    const bars = [];
    for (let d = 0; d < 10; d++) {
      const row = document.createElement('div');
      row.className = 'pbar';
      row.innerHTML = `<span class="d">${d}</span><span class="track"><i></i></span><span class="v">0%</span>`;
      barsHost.appendChild(row);
      bars.push({ row, fill: row.querySelector('i'), val: row.querySelector('.v') });
    }

    function refreshAll() {
      if (!current) return;
      const probs = softmax(net.forward(current));
      const best = argmax(probs);
      bars.forEach((b, i) => {
        b.fill.style.width = (probs[i] * 100).toFixed(1) + '%';
        b.val.textContent = (probs[i] * 100).toFixed(0) + '%';
        b.row.classList.toggle('best', i === best);
      });
      document.getElementById('vit-verdict').innerHTML =
        `Prediction: <b>${best}</b> <span class="muted">(${(probs[best] * 100).toFixed(1)}%)</span>`;
      drawMain();
      drawAttnMatrix();
      drawPatchStrip();
      drawPosPanel();
      const off = clsOff();
      const w = attnMode === 'from' ? net.attentionFrom(selected + off).subarray(off)
              : attnMode === 'to' ? net.attentionTo(selected + off).subarray(off)
              : net.clsAttention();
      const idx = Array.from(w).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 4);
      document.getElementById('attn-top').innerHTML = idx
        .map(([v, i]) => `<span class="chip">patch ${i} <b>${(v * 100).toFixed(1)}%</b></span>`)
        .join('');
    }

    mainCanvas.addEventListener('pointerdown', (e) => {
      const r = mainCanvas.getBoundingClientRect();
      const g = net.grid;
      const gx = clamp(Math.floor(((e.clientX - r.left) / r.width) * g), 0, g - 1);
      const gy = clamp(Math.floor(((e.clientY - r.top) / r.height) * g), 0, g - 1);
      selected = gy * g + gx;
      refreshAll();
    });

    /* -------- drawing pad -------- */
    const pad = document.getElementById('vit-pad');
    const PAD = 168;
    const padCtx = hidpi(pad, PAD, PAD);
    const padBuf = document.createElement('canvas');
    padBuf.width = PAD; padBuf.height = PAD;
    const pb = padBuf.getContext('2d', { willReadFrequently: true });
    function clearPad() {
      pb.fillStyle = '#000'; pb.fillRect(0, 0, PAD, PAD);
      padCtx.clearRect(0, 0, PAD, PAD);
      padCtx.drawImage(padBuf, 0, 0, PAD, PAD);
    }
    let padDown = false, last = null;
    const padPos = (e) => {
      const r = pad.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * PAD, y: ((e.clientY - r.top) / r.height) * PAD };
    };
    pad.addEventListener('pointerdown', (e) => { padDown = true; last = padPos(e); stroke(last, last); });
    pad.addEventListener('pointermove', (e) => { if (padDown) { const p = padPos(e); stroke(last, p); last = p; } });
    window.addEventListener('pointerup', () => { padDown = false; });
    function stroke(a, b) {
      pb.strokeStyle = '#fff'; pb.lineWidth = 15; pb.lineCap = 'round'; pb.lineJoin = 'round';
      pb.beginPath(); pb.moveTo(a.x, a.y); pb.lineTo(b.x, b.y); pb.stroke();
      padCtx.clearRect(0, 0, PAD, PAD);
      padCtx.drawImage(padBuf, 0, 0, PAD, PAD);
      current = imageFromCanvas(padBuf, S);
      refreshAll();
    }
    document.getElementById('btn-clear-pad').addEventListener('click', () => { clearPad(); });
    document.getElementById('btn-next-digit').addEventListener('click', () => {
      current = renderDigit((Math.random() * 10) | 0, mulberry32((Math.random() * 1e9) | 0), aug, S);
      refreshAll();
    });

    /* ---------------------------------------------------- controls */
    const chart = new LineChart(document.getElementById('vit-chart'), {
      height: 150, xLabel: 'images seen', yMin: 0, yMax: 1,
      series: [
        { name: 'train accuracy', color: '#4da3ff' },
        { name: 'test accuracy', color: '#38d39f' },
      ],
    });
    const setStat = statGrid(document.getElementById('vit-stats'),
      ['epoch', 'images seen', 'train acc', 'test acc', 'patches', 'parameters']);

    pills(document.getElementById('patch-pills'), [
      { value: 10, label: '10×10 patches (2×2 grid)' },
      { value: 5, label: '5×5 patches (4×4 grid)' },
      { value: 4, label: '4×4 patches (5×5 grid)' },
      { value: 2, label: '2×2 patches (10×10 grid)' },
    ], 4, (v) => { patch = v; build(); });

    pills(document.getElementById('arch-toggles'), [
      { value: 'mean', label: 'Average the patch tokens' },
      { value: 'cls', label: 'Class token' },
    ], 'mean', (v) => { useCLS = v === 'cls'; build(); });
    pills(document.getElementById('arch-toggles'), [
      { value: 'nonorm', label: 'LayerNorm off' },
      { value: 'norm', label: 'LayerNorm on' },
    ], 'nonorm', (v) => { useNorm = v === 'norm'; build(); });

    pills(document.getElementById('attn-mode'), [
      { value: 'cls', label: 'What the classifier reads' },
      { value: 'from', label: 'Where this patch looks' },
      { value: 'to', label: 'Who looks at this patch' },
    ], 'cls', (v) => { attnMode = v; refreshAll(); });

    const cfg = document.getElementById('vit-config');
    slider(cfg, {
      label: 'embedding size d', min: 8, max: 48, step: 4, value: dim,
      format: (v) => v.toFixed(0), onInput: (v) => { dim = v | 0; build(); },
      desc: 'The width of every token vector. Attention compares tokens in this space, so it sets how much a patch can say about itself.',
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
    });
    slider(cfg, {
      label: 'training images', min: 250, max: 6000, step: 250, value: trainSize,
      format: (v) => v.toLocaleString(), onInput: (v) => { trainSize = v | 0; },
      desc: 'Press "New data" after changing this. A transformer has fewer built-in assumptions than a CNN, so it leans harder on examples — this slider is the cheapest way to see that.',
    });

    const btnTrain = document.getElementById('btn-train');
    btnTrain.addEventListener('click', () => {
      running = !running;
      btnTrain.textContent = running ? '⏸ Pause training' : '▶ Train the transformer';
      btnTrain.classList.toggle('primary', !running);
    });
    document.getElementById('btn-reset').addEventListener('click', () => {
      const keep = window.ML_VIT_MODEL;
      window.ML_VIT_MODEL = null;
      build();
      window.ML_VIT_MODEL = keep;
      seen = 0; epoch = 0;
      setStat('train acc', '–'); setStat('test acc', '–');
      refreshAll();
    });
    document.getElementById('btn-regen').addEventListener('click', () => { regenerateData(); });

    /* ---------------------------------------------------- main loop */
    let evalCountdown = 0;
    rafLoop(() => {
      if (!running) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 20) trainBatch();
      setStat('epoch', epoch);
      setStat('images seen', seen.toLocaleString());
      if (runCount > 200) {
        const acc = runCorrect / runCount;
        setStat('train acc', (acc * 100).toFixed(1) + '%');
        runLoss = runCorrect = runCount = 0;
        if (--evalCountdown <= 0) {
          evalCountdown = 3;
          const te = evaluate();
          setStat('test acc', (te * 100).toFixed(1) + '%', te > 0.9 ? 'good' : '');
          if (te >= 0.8 && fromScratch) {
            achieve('vit-trained', `${(te * 100).toFixed(1)}% test accuracy on ten classes`);
          }
          // trying both pooling arrangements on the same task is an ablation study
          trainedArch[useCLS ? 'cls' : 'mean'] = true;
          if (trainedArch.cls && trainedArch.mean) {
            achieve('vit-ablation', 'You compared a class token against mean pooling');
          }
          chart.push(seen, [acc, te]);
        } else chart.push(seen, [acc, null]);
        chart.draw();
      }
      refreshAll();
    }).start();

    /* ---------------------------------------------------- go */
    build();
    regenerateData();
    clearPad();
    current = renderDigit(3, mulberry32(7), aug, S);
    refreshAll();
  });
})();
