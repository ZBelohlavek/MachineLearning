/* ==========================================================================
   foundations.js — a 2D classification playground.

   Draw a dataset, build a network, and watch gradient descent bend a decision
   boundary around your points while every hidden neuron shows its own slice
   of the problem.
   ========================================================================== */

;(function () {
  'use strict';
  const { MLP, mulberry32, randn, clamp, hidpi, fit, LineChart, achieve, heat,
          chrome, nextLinks, slider, pills, checkbox, statGrid, rafLoop } = window.ML;

  /* ------------------------------------------------------------ datasets */
  function makeData(kind, n, noise, rand) {
    const pts = [];
    const add = (x, y, l) => pts.push({ x: clamp(x, -1.15, 1.15), y: clamp(y, -1.15, 1.15), label: l });
    const jit = () => randn(rand) * noise;
    for (let i = 0; i < n; i++) {
      const t = i / n;
      switch (kind) {
        case 'moons': {
          const l = i % 2;
          const a = Math.PI * (i / n);            // half a circle per moon
          const s = 0.72;
          if (l === 0) add((Math.cos(a) - 0.5) * s + jit(), (Math.sin(a) - 0.25) * s + jit(), 0);
          else add((0.5 - Math.cos(a)) * s + jit(), (0.25 - Math.sin(a)) * s + jit(), 1);
          break;
        }
        case 'circles': {
          const l = i % 2;
          const a = rand() * Math.PI * 2;
          const r = l === 0 ? 0.32 : 0.85;
          add(Math.cos(a) * r + jit(), Math.sin(a) * r + jit(), l);
          break;
        }
        case 'spiral': {
          const l = i % 2;
          const r = 0.12 + 0.9 * t;
          const a = t * 5.2 + (l === 0 ? 0 : Math.PI);
          add(Math.cos(a) * r + jit(), Math.sin(a) * r + jit(), l);
          break;
        }
        case 'xor': {
          const x = rand() * 2 - 1, y = rand() * 2 - 1;
          add(x + jit() * 0.4, y + jit() * 0.4, (x > 0) === (y > 0) ? 1 : 0);
          break;
        }
        case 'blobs':
        default: {
          const l = i % 2;
          add((l ? 0.45 : -0.45) + randn(rand) * 0.28 + jit(),
              (l ? 0.4 : -0.4) + randn(rand) * 0.28 + jit(), l);
        }
      }
    }
    return pts;
  }

  /* ------------------------------------------------------------- helpers */
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  document.addEventListener('DOMContentLoaded', () => {
    chrome('foundations', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.foundations);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'foundations');
    nextLinks(document.getElementById('next-links'), null, 'convolutions', '../');

    /* ---------------- state ---------------- */
    let dataset = 'moons', noise = 0.12, nPoints = 180;
    // One seed drives the dataset, the train/test split and the initial weights,
    // so two runs with the same seed differ only by what you changed.
    let seed = 1234;
    let rand = mulberry32(seed);
    let all = [], train = [], test = [];
    let hidden = [6, 6], activation = 'tanh';
    let lr = 0.03, weightDecay = 0, batchSize = 16;
    let net = null, epoch = 0, running = false;
    let paintLabel = 0;

    const canvas = document.getElementById('data-canvas');
    let ctx, size;
    function resizeMain() {
      canvas.style.width = '';
      const w = canvas.clientWidth || canvas.parentElement.clientWidth;
      size = w;
      ctx = hidpi(canvas, w, w);
    }
    resizeMain();
    window.addEventListener('resize', () => { resizeMain(); draw(); });

    const toPx = (v) => ((v + 1.2) / 2.4) * size;
    const toData = (p) => (p / size) * 2.4 - 1.2;

    const chart = new LineChart(document.getElementById('loss-chart'), {
      height: 150, xLabel: 'epoch', yMin: 0,
      series: [
        { name: 'training loss', color: '#4da3ff' },
        { name: 'test loss', color: '#ff9f45', dashed: true },
      ],
    });
    const setStat = statGrid(document.getElementById('fd-stats'),
      ['epoch', 'train loss', 'test loss', 'train acc', 'test acc', 'parameters']);

    /* ---------------- data + model plumbing ---------------- */
    function regenerate() {
      rand = mulberry32(seed);
      all = makeData(dataset, nPoints, noise, rand);
      splitData();
      rebuild();
    }
    /* Split into train and test. The generators alternate classes as they go,
       so this has to shuffle first — splitting on the raw order would hand one
       class to training and the other to test. */
    function splitData() {
      const order = all.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = (rand() * (i + 1)) | 0;
        [order[i], order[j]] = [order[j], order[i]];
      }
      train = []; test = [];
      order.forEach((idx, k) => {
        const p = all[idx];
        p.test = k % 2 === 1;
        (p.test ? test : train).push(p);
      });
      if (!train.length) { train = all.slice(); all.forEach((p) => (p.test = false)); }
    }
    function rebuild() {
      minTestLoss = Infinity; trainLossAtMin = Infinity;
      landTrail = []; landDirs = null;
      const sizes = [2, ...hidden.filter((h) => h > 0), 1];
      net = new MLP(sizes, { hidden: activation, out: 'linear', rand: mulberry32(seed + 7919) });
      epoch = 0;
      chart.clear();
      buildNeuronPanels();
      draw();
      report();
      if (landCtx) computeLandscape();
    }

    const xbuf = new Float32Array(2);
    function predict(x, y) {
      xbuf[0] = x; xbuf[1] = y;
      return sigmoid(net.forward(xbuf)[0]);
    }

    /** One pass over the training set, in shuffled minibatches. */
    function trainEpoch() {
      const order = train.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = (rand() * (i + 1)) | 0;
        [order[i], order[j]] = [order[j], order[i]];
      }
      const d = new Float32Array(1);
      for (let b = 0; b < order.length; b += batchSize) {
        const end = Math.min(order.length, b + batchSize);
        net.zeroGrad();
        for (let k = b; k < end; k++) {
          const p = train[order[k]];
          xbuf[0] = p.x; xbuf[1] = p.y;
          const z = net.forward(xbuf)[0];
          // Binary cross-entropy on a sigmoid output has the tidiest gradient
          // in all of machine learning: dL/dz is just (prediction - target).
          d[0] = sigmoid(z) - p.label;
          net.backward(d);
        }
        net.step(lr, 1 / (end - b), weightDecay);
      }
      epoch++;
    }

    function metrics(set) {
      if (!set.length) return { loss: 0, acc: 0 };
      let loss = 0, correct = 0;
      for (const p of set) {
        const q = predict(p.x, p.y);
        loss += -(p.label * Math.log(Math.max(q, 1e-9)) + (1 - p.label) * Math.log(Math.max(1 - q, 1e-9)));
        if ((q > 0.5 ? 1 : 0) === p.label) correct++;
      }
      return { loss: loss / set.length, acc: correct / set.length };
    }

    /* ---------------- drawing ---------------- */
    const GRID = 64;
    const boundary = document.createElement('canvas');
    boundary.width = GRID; boundary.height = GRID;
    const bctx = boundary.getContext('2d');
    const bimg = bctx.createImageData(GRID, GRID);

    function drawBoundary() {
      for (let j = 0; j < GRID; j++) {
        for (let i = 0; i < GRID; i++) {
          const x = (i / (GRID - 1)) * 2.4 - 1.2;
          const y = (j / (GRID - 1)) * 2.4 - 1.2;
          const q = predict(x, y);
          const k = (j * GRID + i) * 4;
          // orange for class 1, blue for class 0, pale where the model is unsure
          const t = q * 2 - 1;
          bimg.data[k]     = 90 + 150 * Math.max(0, t);
          bimg.data[k + 1] = 120 + 40 * Math.abs(t);
          bimg.data[k + 2] = 90 + 150 * Math.max(0, -t);
          bimg.data[k + 3] = 90 + 90 * Math.abs(t);
        }
      }
      bctx.putImageData(bimg, 0, 0);
    }

    function draw() {
      if (!net) return;
      drawBoundary();
      ctx.clearRect(0, 0, size, size);
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, size, size);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(boundary, 0, 0, size, size);

      // axes
      ctx.strokeStyle = 'rgba(255,255,255,.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(toPx(0), 0); ctx.lineTo(toPx(0), size);
      ctx.moveTo(0, toPx(0)); ctx.lineTo(size, toPx(0));
      ctx.stroke();

      for (const p of all) {
        const isTest = p.test;
        const wrong = (predict(p.x, p.y) > 0.5 ? 1 : 0) !== p.label;
        if (wrong) {
          ctx.beginPath();
          ctx.arc(toPx(p.x), toPx(p.y), 8, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(255,107,107,.85)';
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(toPx(p.x), toPx(p.y), 4.2, 0, Math.PI * 2);
        ctx.fillStyle = p.label ? '#ff9f45' : '#4da3ff';
        ctx.fill();
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = isTest ? 'rgba(255,255,255,.85)' : 'rgba(0,0,0,.55)';
        ctx.stroke();
      }
    }

    /* ---------------- the loss landscape ----------------
       Gradient descent is always drawn as a ball rolling into a valley. This is
       that picture, computed live from your model and your data.

       Slicing along two individual weights turns out to be dull — once the rest
       of the network has fit the data, one neuron's weights barely matter and
       the surface is flat. So this follows the standard recipe instead: take two
       random directions in the *whole* weight space, walk along them, and colour
       each point by the training loss there. The white line is the optimiser's
       real trajectory projected onto that plane.
       ---------------------------------------------------------------- */
    const landCanvas = document.getElementById('landscape');
    const LAND = 36;
    let landCtx = null, landImg = null;
    let landCentre = null, landDirs = null, landScale = 1, landTrail = [];

    function weightVector() {
      const out = [];
      for (const l of net.layers) { out.push(...l.W); out.push(...l.b); }
      return Float64Array.from(out);
    }
    function setWeights(vec) {
      let k = 0;
      for (const l of net.layers) {
        for (let i = 0; i < l.W.length; i++) l.W[i] = vec[k++];
        for (let i = 0; i < l.b.length; i++) l.b[i] = vec[k++];
      }
    }
    function randomDirections(n) {
      const mk = () => {
        const d = Float64Array.from({ length: n }, () => randn(rand));
        let norm = 0;
        for (const v of d) norm += v * v;
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < n; i++) d[i] /= norm;
        return d;
      };
      const d1 = mk();
      let d2 = mk();
      // orthogonalise, so the two axes are not secretly the same direction
      let dot = 0;
      for (let i = 0; i < n; i++) dot += d1[i] * d2[i];
      let norm = 0;
      for (let i = 0; i < n; i++) { d2[i] -= dot * d1[i]; norm += d2[i] * d2[i]; }
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < n; i++) d2[i] /= norm;
      return [d1, d2];
    }

    function trainLoss() {
      let loss = 0;
      for (const p of train) {
        xbuf[0] = p.x; xbuf[1] = p.y;
        const q = sigmoid(net.forward(xbuf)[0]);
        loss += -(p.label * Math.log(Math.max(q, 1e-9)) + (1 - p.label) * Math.log(Math.max(1 - q, 1e-9)));
      }
      return loss / Math.max(1, train.length);
    }

    function computeLandscape() {
      if (!landCtx || !net) return;
      const theta = weightVector();
      const n = theta.length;
      if (!landDirs || landDirs[0].length !== n) landDirs = randomDirections(n);
      landCentre = theta;
      let normTheta = 0;
      for (const v of theta) normTheta += v * v;
      landScale = Math.max(0.6, Math.sqrt(normTheta) * 0.55);   // sweep a real fraction of the weights

      const grid = new Float64Array(LAND * LAND);
      const probe = new Float64Array(n);
      let lo = Infinity, hi = -Infinity;
      for (let j = 0; j < LAND; j++) {
        const b = ((j / (LAND - 1)) * 2 - 1) * landScale;
        for (let i = 0; i < LAND; i++) {
          const a = ((i / (LAND - 1)) * 2 - 1) * landScale;
          for (let k = 0; k < n; k++) probe[k] = theta[k] + a * landDirs[0][k] + b * landDirs[1][k];
          setWeights(probe);
          const L = trainLoss();
          grid[j * LAND + i] = L;
          if (L < lo) lo = L;
          if (L > hi) hi = L;
        }
      }
      setWeights(theta);                                        // put the model back

      // log scaling: loss spans orders of magnitude and a linear ramp hides the valley
      const buf = landCtx.createImageData(LAND, LAND);
      const ll = Math.log(lo + 1e-3), lh = Math.log(hi + 1e-3);
      for (let k = 0; k < LAND * LAND; k++) {
        const t = (Math.log(grid[k] + 1e-3) - ll) / Math.max(1e-6, lh - ll);
        const c = heat(1 - t).match(/\d+/g);                   // bright = low loss
        buf.data[k * 4] = +c[0]; buf.data[k * 4 + 1] = +c[1];
        buf.data[k * 4 + 2] = +c[2]; buf.data[k * 4 + 3] = 255;
      }
      landImg = buf;
      const el = document.getElementById('landscape-range');
      if (el) el.textContent = `loss ${lo.toFixed(3)} – ${hi.toFixed(2)}`;
      drawLandscape();
    }

    /** Project a stored weight vector onto the current slice. */
    function project(vec) {
      let a = 0, b = 0;
      for (let k = 0; k < vec.length; k++) {
        const d = vec[k] - landCentre[k];
        a += d * landDirs[0][k];
        b += d * landDirs[1][k];
      }
      return [a, b];
    }

    function drawLandscape() {
      if (!landCtx || !landImg) return;
      const off = landCanvas._buf || (landCanvas._buf = document.createElement('canvas'));
      off.width = LAND; off.height = LAND;
      off.getContext('2d').putImageData(landImg, 0, 0);
      const W = landCtx._cssW, H = landCtx._cssH;
      landCtx.imageSmoothingEnabled = true;
      landCtx.clearRect(0, 0, W, H);
      landCtx.drawImage(off, 0, 0, W, H);

      const px = (a) => (a / landScale * 0.5 + 0.5) * W;
      const py = (b) => (b / landScale * 0.5 + 0.5) * H;
      const pts = landTrail.map(project);
      if (pts.length > 1) {
        landCtx.strokeStyle = 'rgba(255,255,255,.9)';
        landCtx.lineWidth = 1.6;
        landCtx.beginPath();
        pts.forEach(([a, b], i) => (i ? landCtx.lineTo(px(a), py(b)) : landCtx.moveTo(px(a), py(b))));
        landCtx.stroke();
      }
      landCtx.fillStyle = '#fff';
      landCtx.beginPath();
      landCtx.arc(px(0), py(0), 4.5, 0, Math.PI * 2);
      landCtx.fill();
      landCtx.strokeStyle = '#0a0f19';
      landCtx.lineWidth = 1.5;
      landCtx.stroke();
    }

    function initLandscape() {
      if (!landCanvas) return;
      landCanvas.style.width = '';
      const w = landCanvas.clientWidth || 260;
      landCtx = hidpi(landCanvas, w, w);
      landDirs = null;
      computeLandscape();
    }

    /* ---------------- what each hidden neuron learned ---------------- */
    let neuronCanvases = [];
    function buildNeuronPanels() {
      const host = document.getElementById('neurons');
      host.innerHTML = '';
      neuronCanvases = [];
      net.layers.slice(0, -1).forEach((layer, li) => {
        const row = document.createElement('div');
        row.className = 'neuron-row';
        const label = document.createElement('div');
        label.className = 'neuron-label';
        label.textContent = `hidden layer ${li + 1} · ${layer.nOut} neurons`;
        host.appendChild(label);
        for (let u = 0; u < layer.nOut; u++) {
          const c = document.createElement('canvas');
          c.width = 44; c.height = 44;
          c.className = 'neuron';
          c.title = `layer ${li + 1}, neuron ${u + 1}`;
          row.appendChild(c);
          neuronCanvases.push({ canvas: c, layer: li, unit: u, img: c.getContext('2d').createImageData(44, 44) });
        }
        host.appendChild(row);
      });
    }

    function drawNeurons() {
      if (!neuronCanvases.length) return;
      const N = 44;
      // one forward pass per grid point gives every neuron's value at once
      const cache = [];
      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          xbuf[0] = (i / (N - 1)) * 2.4 - 1.2;
          xbuf[1] = (j / (N - 1)) * 2.4 - 1.2;
          net.forward(xbuf);
          cache.push(net.activations.map((a) => Float32Array.from(a)));
        }
      }
      for (const nc of neuronCanvases) {
        const img = nc.img;
        for (let k = 0; k < N * N; k++) {
          const v = cache[k][nc.layer + 1][nc.unit];
          const t = clamp(v, -1, 1);
          const o = k * 4;
          img.data[o]     = 90 + 150 * Math.max(0, t);
          img.data[o + 1] = 120 + 40 * Math.abs(t);
          img.data[o + 2] = 90 + 150 * Math.max(0, -t);
          img.data[o + 3] = 235;
        }
        nc.canvas.getContext('2d').putImageData(img, 0, 0);
      }
    }

    let minTestLoss = Infinity, trainLossAtMin = Infinity;
    function report() {
      const tr = metrics(train), te = metrics(test);

      if (dataset === 'spiral' && te.acc >= 0.9 && epoch > 20) {
        achieve('foundations-spiral', `${(te.acc * 100).toFixed(1)}% on held-out spiral points`);
        const units = hidden.filter((h) => h > 0).reduce((a, b) => a + b, 0);
        if (units > 0 && units <= 8) {
          achieve('foundations-lean',
            `${(te.acc * 100).toFixed(1)}% on the spiral with only ${units} hidden neurons`);
        }
      }
      // Overfitting: test loss has climbed well past its best while training loss kept falling.
      if (te.loss < minTestLoss) { minTestLoss = te.loss; trainLossAtMin = tr.loss; }
      else if (epoch > 60 && te.loss > minTestLoss * 1.4 && tr.loss < trainLossAtMin * 0.6) {
        achieve('foundations-overfit',
                `test loss ${minTestLoss.toFixed(2)} → ${te.loss.toFixed(2)} while training loss fell`);
      }
      setStat('epoch', epoch);
      setStat('train loss', tr.loss.toFixed(3));
      setStat('test loss', te.loss.toFixed(3), te.loss > tr.loss * 1.6 && epoch > 30 ? 'bad' : '');
      setStat('train acc', (tr.acc * 100).toFixed(1) + '%');
      setStat('test acc', (te.acc * 100).toFixed(1) + '%', te.acc > 0.9 ? 'good' : '');
      setStat('parameters', net.numParams());
      return { tr, te };
    }

    /* ---------------- controls ---------------- */
    pills(document.getElementById('dataset-pills'), [
      { value: 'moons', label: 'Two moons' },
      { value: 'circles', label: 'Circles' },
      { value: 'spiral', label: 'Spiral' },
      { value: 'xor', label: 'XOR' },
      { value: 'blobs', label: 'Blobs' },
    ], 'moons', (v) => { dataset = v; regenerate(); });

    const dataCtl = document.getElementById('data-controls');
    slider(dataCtl, {
      label: 'noise', min: 0, max: 0.4, step: 0.01, value: noise,
      format: (v) => v.toFixed(2), onInput: (v) => { noise = v; regenerate(); },
      desc: 'Messier data makes the gap between training and test loss — overfitting — much easier to see.',
    });
    slider(dataCtl, {
      label: 'points', min: 40, max: 400, step: 10, value: nPoints,
      format: (v) => v.toFixed(0), onInput: (v) => { nPoints = v; regenerate(); },
    });

    let painting = false;
    pills(document.getElementById('paint-pills'), [
      { value: 0, label: 'Add blue points' },
      { value: 1, label: 'Add orange points' },
      { value: -1, label: 'Erase' },
    ], 0, (v) => { paintLabel = v; });

    const paintAt = (ev) => {
      const r = canvas.getBoundingClientRect();
      const p = ev.touches ? ev.touches[0] : ev;
      const x = toData(((p.clientX - r.left) / r.width) * size);
      const y = toData(((p.clientY - r.top) / r.height) * size);
      if (paintLabel === -1) {
        all = all.filter((q) => Math.hypot(q.x - x, q.y - y) > 0.09);
      } else {
        all.push({ x, y, label: paintLabel });
      }
      splitData();
      draw();
    };
    canvas.addEventListener('pointerdown', (e) => { painting = true; paintAt(e); });
    canvas.addEventListener('pointermove', (e) => { if (painting) paintAt(e); });
    window.addEventListener('pointerup', () => { painting = false; });

    const archCtl = document.getElementById('arch-controls');
    slider(archCtl, {
      label: 'hidden layers', min: 0, max: 3, step: 1, value: 2,
      format: (v) => v.toFixed(0), onInput: (v) => {
        const n = v | 0;
        const width = hidden[0] || 6;
        hidden = Array.from({ length: n }, () => width);
        rebuild();
      },
      desc: 'Zero hidden layers is plain logistic regression: it can only ever draw a straight line. Try it on the spiral.',
    });
    slider(archCtl, {
      label: 'neurons per layer', min: 1, max: 16, step: 1, value: 6,
      format: (v) => v.toFixed(0), onInput: (v) => { hidden = hidden.map(() => v | 0); rebuild(); },
    });
    pills(archCtl, [
      { value: 'tanh', label: 'tanh' },
      { value: 'relu', label: 'ReLU' },
      { value: 'sigmoid', label: 'sigmoid' },
    ], 'tanh', (v) => { activation = v; rebuild(); });

    const optCtl = document.getElementById('opt-controls');
    slider(optCtl, {
      label: 'learning rate', min: 0.001, max: 0.3, step: 0.001, value: lr,
      format: (v) => v.toFixed(3), onInput: (v) => { lr = v; },
      desc: 'The size of each downhill step. Too large and the loss bounces; too small and you wait all day.',
    });
    slider(optCtl, {
      label: 'batch size', min: 1, max: 64, step: 1, value: batchSize,
      format: (v) => v.toFixed(0), onInput: (v) => { batchSize = v; },
      desc: 'How many points contribute to each step. Small batches are noisy — which sometimes helps.',
    });
    slider(optCtl, {
      label: 'weight decay (L2)', min: 0, max: 0.02, step: 0.0005, value: 0,
      format: (v) => v.toFixed(4), onInput: (v) => { weightDecay = v; },
      desc: 'Pulls weights toward zero, which smooths the boundary. The simplest cure for overfitting.',
    });

    const runControls = window.ML.runBar(document.getElementById('run-bar'), {
      seed,
      charts: () => [chart],
      onSeed: (v) => { seed = v; regenerate(); },
      note: 'Same seed, same points and same starting weights.',
    });

    const btnRun = document.getElementById('btn-run');
    btnRun.addEventListener('click', () => {
      running = !running;
      btnRun.textContent = running ? '⏸ Pause' : '▶ Train';
      btnRun.classList.toggle('primary', !running);
    });
    document.getElementById('btn-step').addEventListener('click', () => {
      trainEpoch(); draw(); drawNeurons();
      const m = report();
      chart.push(epoch, [m.tr.loss, m.te.loss]); chart.draw();
    });
    document.getElementById('btn-reinit').addEventListener('click', () => {
      runControls.set(Math.floor(Math.random() * 100000));    // a new seed is a new init
    });
    document.getElementById('btn-regen').addEventListener('click', () => regenerate());

    /* ---------------- loop ---------------- */
    let frame = 0;
    rafLoop(() => {
      if (running) {
        for (let i = 0; i < 3; i++) trainEpoch();
        const m = report();
        chart.push(epoch, [m.tr.loss, m.te.loss]);
        chart.draw();
        draw();
        landTrail.push(weightVector());
        if (landTrail.length > 240) landTrail.shift();
        if (frame % 3 === 0) drawNeurons();
        // The surface itself shifts as training moves the other weights, so it
        // is recomputed about once a second rather than every frame.
        if (frame % 45 === 0) computeLandscape(); else drawLandscape();
        frame++;
      }
    }).start();

    regenerate();
    drawNeurons();
    initLandscape();
    window.addEventListener('resize', initLandscape);
  });
})();
