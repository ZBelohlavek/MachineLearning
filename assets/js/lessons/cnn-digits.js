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
          mulberry32, softmax, argmax, clamp, hidpi, fit, LineChart, heat, diverging,
          chrome, nextLinks, slider, pills, checkbox, statGrid, rafLoop,
          renderDigit, makeDigitSet, imageFromCanvas } = window.ML;

  const S = 20;                       // digit images are S x S

  /* ------------------------------------------------------------- page */
  document.addEventListener('DOMContentLoaded', () => {
    chrome('cnn', '../');
    nextLinks(document.getElementById('next-links'), 'convolutions', 'attention', '../');

    let f1 = 8, f2 = 16, hiddenUnits = 32;
    let lr = 0.004, batchSize = 16, aug = 1;
    let net, epoch = 0, seen = 0, running = false;
    let trainSet, testSet;
    const rng = mulberry32(1234);

    /* ------------------------- model ------------------------- */
    function build() {
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
      epoch = 0; seen = 0;
      chart.clear();
      setStat('parameters', net.numParams().toLocaleString());
      drawArchitecture();
    }

    function regenerateData() {
      trainSet = makeDigitSet(1000, rng, aug, S);
      testSet = makeDigitSet(300, mulberry32(999), aug, S);
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
    window.addEventListener('pointerup', () => { if (drawingPad) { drawingPad = false; predictPad(); } });
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

      const probs = softmax(net.forward(img));
      const best = argmax(probs);
      bars.forEach((b, i) => {
        b.fill.style.width = (probs[i] * 100).toFixed(1) + '%';
        b.val.textContent = (probs[i] * 100).toFixed(0) + '%';
        b.row.classList.toggle('best', i === best);
      });
      document.getElementById('pad-verdict').innerHTML =
        `I think that's a <b>${best}</b> <span class="muted">(${(probs[best] * 100).toFixed(1)}% confident)</span>`;
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

    const btnTrain = document.getElementById('btn-train');
    btnTrain.addEventListener('click', () => {
      running = !running;
      btnTrain.textContent = running ? '⏸ Pause training' : '▶ Train the network';
      btnTrain.classList.toggle('primary', !running);
    });
    document.getElementById('btn-reset').addEventListener('click', () => {
      build(); shuffleOrder(); buildFeatureTiles(); predictPad();
      setStat('epoch', 0); setStat('images seen', 0);
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
          chart.push(seen, [trainAcc, ev.acc]);
          drawConfusion(ev.conf);
          drawMistakes(ev.wrong);
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
