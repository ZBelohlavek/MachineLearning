/* ==========================================================================
   convolutions.js — the sliding-window arithmetic behind every vision model.

   One 48x48 grayscale image, one 3x3 kernel, and an animation that walks the
   window across the image showing all nine multiplications as they happen.
   ========================================================================== */

;(function () {
  'use strict';
  const { hidpi, fit, chrome, nextLinks, slider, pills, checkbox, statGrid, rafLoop, clamp,
          mulberry32, achieve } = window.ML;

  const N = 48;                       // image is N x N, single channel, values 0..1
  const rand = mulberry32(11);

  /* ------------------------------------------------------- image sources */
  function blankImage() { return new Float32Array(N * N); }

  function renderToImage(drawFn) {
    // Draw with the normal 2D canvas API, then read the pixels back as grey levels.
    const c = document.createElement('canvas');
    c.width = N; c.height = N;
    const g = c.getContext('2d');
    g.fillStyle = '#000';
    g.fillRect(0, 0, N, N);
    drawFn(g);
    const d = g.getImageData(0, 0, N, N).data;
    const img = blankImage();
    for (let i = 0; i < N * N; i++) {
      img[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
    }
    return img;
  }

  const SOURCES = {
    shapes: () => renderToImage((g) => {
      const grd = g.createLinearGradient(0, 0, N, N);
      grd.addColorStop(0, '#333'); grd.addColorStop(1, '#111');
      g.fillStyle = grd; g.fillRect(0, 0, N, N);
      g.fillStyle = '#fff';
      g.fillRect(6, 6, 16, 16);
      g.beginPath(); g.arc(33, 16, 10, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.moveTo(24, 44); g.lineTo(12, 27); g.lineTo(36, 27); g.closePath(); g.fill();
    }),
    letter: () => renderToImage((g) => {
      g.fillStyle = '#fff';
      g.font = 'bold 42px Georgia, serif';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText('A', N / 2, N / 2 + 2);
    }),
    checker: () => renderToImage((g) => {
      for (let y = 0; y < N; y += 8) for (let x = 0; x < N; x += 8) {
        g.fillStyle = ((x / 8 + y / 8) % 2) ? '#fff' : '#1a1a1a';
        g.fillRect(x, y, 8, 8);
      }
    }),
    stripes: () => renderToImage((g) => {
      g.save(); g.translate(N / 2, N / 2); g.rotate(0.6); g.translate(-N / 2, -N / 2);
      for (let x = -N; x < N * 2; x += 10) { g.fillStyle = '#fff'; g.fillRect(x, -N, 5, N * 3); }
      g.restore();
    }),
    gradient: () => renderToImage((g) => {
      const grd = g.createRadialGradient(N * 0.35, N * 0.35, 2, N * 0.5, N * 0.5, N * 0.7);
      grd.addColorStop(0, '#fff'); grd.addColorStop(1, '#000');
      g.fillStyle = grd; g.fillRect(0, 0, N, N);
      g.fillStyle = '#888'; g.fillRect(0, 34, N, 3);
    }),
    noise: () => {
      const img = blankImage();
      for (let i = 0; i < img.length; i++) img[i] = rand();
      return img;
    },
  };

  /* ----------------------------------------------------------- kernels */
  const KERNELS = {
    identity:   [0, 0, 0, 0, 1, 0, 0, 0, 0],
    blur:       [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
    gaussian:   [1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16],
    sharpen:    [0, -1, 0, -1, 5, -1, 0, -1, 0],
    sobelX:     [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    sobelY:     [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    laplacian:  [0, 1, 0, 1, -4, 1, 0, 1, 0],
    emboss:     [-2, -1, 0, -1, 1, 1, 0, 1, 2],
  };

  /** 'same' convolution with edge clamping. Returns raw (possibly signed) values. */
  function convolve(img, k) {
    const out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const sx = clamp(x + kx, 0, N - 1), sy = clamp(y + ky, 0, N - 1);
            sum += img[sy * N + sx] * k[(ky + 1) * 3 + (kx + 1)];
          }
        }
        out[y * N + x] = sum;
      }
    }
    return out;
  }

  function patchAt(img, x, y) {
    const p = new Float32Array(9);
    for (let ky = -1; ky <= 1; ky++)
      for (let kx = -1; kx <= 1; kx++)
        p[(ky + 1) * 3 + (kx + 1)] = img[clamp(y + ky, 0, N - 1) * N + clamp(x + kx, 0, N - 1)];
    return p;
  }

  /** Paint a single-channel image into a canvas, optionally centring signed values. */
  function paint(canvas, data, centred) {
    const c = canvas._buf || (canvas._buf = document.createElement('canvas'));
    c.width = N; c.height = N;
    const g = c.getContext('2d');
    const im = g.createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const v = centred ? data[i] * 0.5 + 0.5 : data[i];
      const b = clamp(Math.round(v * 255), 0, 255);
      im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = b;
      im.data[i * 4 + 3] = 255;
    }
    g.putImageData(im, 0, 0);
    const ctx = canvas._ctx;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, ctx._cssW, ctx._cssH);
    ctx.drawImage(c, 0, 0, ctx._cssW, ctx._cssH);
  }

  function setupCanvas(canvas, size) {
    canvas._ctx = hidpi(canvas, size, size);
    return canvas;
  }

  /* ------------------------------------------------------------- page */
  document.addEventListener('DOMContentLoaded', () => {
    chrome('convolutions', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.convolutions);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'convolutions');
    nextLinks(document.getElementById('next-links'), 'foundations', 'cnn', '../');

    let img = SOURCES.shapes();
    let kernel = KERNELS.sobelX.slice();
    let out = convolve(img, kernel);
    let cursor = { x: 1, y: 1 };
    let playing = false, speed = 8, acc = 0;

    const inCanvas = document.getElementById('conv-in');
    const outCanvas = document.getElementById('conv-out');
    const inCtx = () => inCanvas._ctx;

    function sizeCanvases() {
      const w = Math.min(360, inCanvas.parentElement.clientWidth);
      setupCanvas(inCanvas, w);
      setupCanvas(outCanvas, w);
      redraw();
    }

    /* ---------------- kernel editor ---------------- */
    const kEditor = document.getElementById('kernel-editor');
    const kInputs = [];
    for (let i = 0; i < 9; i++) {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = '0.25';
      inp.value = kernel[i];
      inp.addEventListener('input', () => {
        const v = parseFloat(inp.value);
        kernel[i] = isFinite(v) ? v : 0;
        kernelTyped = true;
        achieve('conv-custom', 'You edited the kernel by hand');
        recompute();
      });
      kEditor.appendChild(inp);
      kInputs.push(inp);
    }
    let kernelTyped = false;

    /**
     * The challenge: type a kernel that finds horizontal edges. Graded by
     * correlating its output with a Sobel-Y reference on the current image, so
     * any scaling or sign flip of a genuine horizontal edge detector counts —
     * and clicking the preset does not.
     */
    function checkEdgeChallenge() {
      if (!kernelTyped) return;
      const ref = convolve(img, KERNELS.sobelY);
      let ma = 0, mb = 0;
      for (let i = 0; i < ref.length; i++) { ma += out[i]; mb += ref[i]; }
      ma /= ref.length; mb /= ref.length;
      let num = 0, da = 0, db = 0;
      for (let i = 0; i < ref.length; i++) {
        const x = out[i] - ma, y = ref[i] - mb;
        num += x * y; da += x * x; db += y * y;
      }
      if (da < 1e-6 || db < 1e-6) return;                  // a flat image proves nothing
      const r = num / Math.sqrt(da * db);
      if (Math.abs(r) > 0.9) {
        achieve('conv-challenge',
          `Your kernel matches a Sobel-Y reference at r = ${Math.abs(r).toFixed(2)}`);
      }
    }

    function setKernel(k) {
      kernelTyped = false;
      kernel = k.slice();
      kInputs.forEach((inp, i) => (inp.value = +kernel[i].toFixed(4)));
      recompute();
    }

    pills(document.getElementById('kernel-presets'), [
      { value: 'sobelX', label: 'Sobel X (vertical edges)' },
      { value: 'sobelY', label: 'Sobel Y (horizontal edges)' },
      { value: 'laplacian', label: 'Laplacian (all edges)' },
      { value: 'sharpen', label: 'Sharpen' },
      { value: 'blur', label: 'Box blur' },
      { value: 'gaussian', label: 'Gaussian blur' },
      { value: 'emboss', label: 'Emboss' },
      { value: 'identity', label: 'Identity' },
    ], 'sobelX', (v) => setKernel(KERNELS[v]));

    pills(document.getElementById('image-presets'), [
      { value: 'shapes', label: 'Shapes' },
      { value: 'letter', label: 'Letter A' },
      { value: 'checker', label: 'Checkerboard' },
      { value: 'stripes', label: 'Diagonal stripes' },
      { value: 'gradient', label: 'Soft gradient' },
      { value: 'noise', label: 'Noise' },
    ], 'shapes', (v) => { img = SOURCES[v](); recompute(); });

    /* ---------------- upload / draw your own ---------------- */
    document.getElementById('file-input').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => {
        img = renderToImage((g) => {
          const s = Math.min(image.width, image.height);
          g.drawImage(image, (image.width - s) / 2, (image.height - s) / 2, s, s, 0, 0, N, N);
        });
        URL.revokeObjectURL(url);
        recompute();
      };
      image.src = url;
    });

    let drawing = false, drawMode = 1;
    const drawAt = (ev) => {
      const r = inCanvas.getBoundingClientRect();
      const p = ev.touches ? ev.touches[0] : ev;
      const x = Math.floor(((p.clientX - r.left) / r.width) * N);
      const y = Math.floor(((p.clientY - r.top) / r.height) * N);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= N || yy >= N) continue;
        const falloff = dx === 0 && dy === 0 ? 1 : 0.45;
        img[yy * N + xx] = drawMode
          ? Math.min(1, img[yy * N + xx] + falloff)
          : Math.max(0, img[yy * N + xx] - falloff);
      }
      recompute();
    };
    inCanvas.addEventListener('pointerdown', (e) => {
      if (!paintOn.get()) {                        // in inspect mode, clicking moves the window
        const r = inCanvas.getBoundingClientRect();
        cursor.x = clamp(Math.floor(((e.clientX - r.left) / r.width) * N), 0, N - 1);
        cursor.y = clamp(Math.floor(((e.clientY - r.top) / r.height) * N), 0, N - 1);
        playing = false;
        redraw();
        return;
      }
      drawing = true; drawAt(e);
    });
    inCanvas.addEventListener('pointermove', (e) => { if (drawing) drawAt(e); });
    window.addEventListener('pointerup', () => { drawing = false; });

    const paintOn = checkbox(document.getElementById('draw-controls'), 'Draw on the image with the mouse', false, () => {});
    pills(document.getElementById('draw-controls'), [
      { value: 1, label: 'Brush: white' },
      { value: 0, label: 'Brush: black' },
    ], 1, (v) => { drawMode = v; });
    document.getElementById('btn-clear').addEventListener('click', () => { img = blankImage(); recompute(); });

    /* ---------------- the arithmetic panel ---------------- */
    const mathEl = document.getElementById('math-panel');
    function drawMath() {
      const p = patchAt(img, cursor.x, cursor.y);
      let sum = 0;
      let cells = '';
      for (let i = 0; i < 9; i++) {
        const prod = p[i] * kernel[i];
        sum += prod;
        cells += `<div class="mcell">
            <span class="pix" style="background:rgba(255,255,255,${p[i].toFixed(3)})"></span>
            <span class="mnum">${p[i].toFixed(2)}</span>
            <span class="mop">×</span>
            <span class="mk ${kernel[i] > 0 ? 'pos' : kernel[i] < 0 ? 'neg' : ''}">${fmtK(kernel[i])}</span>
            <span class="mop">=</span>
            <span class="mres">${prod >= 0 ? '' : '−'}${Math.abs(prod).toFixed(2)}</span>
          </div>`;
      }
      const shown = clamp(sum * 0.5 + 0.5, 0, 1);
      mathEl.innerHTML = `
        <div class="mgrid">${cells}</div>
        <div class="msum">
          sum = <b>${sum >= 0 ? '' : '−'}${Math.abs(sum).toFixed(3)}</b>
          → output pixel at (${cursor.x}, ${cursor.y})
          <span class="swatch" style="background:rgba(255,255,255,${shown.toFixed(3)})"></span>
        </div>`;
    }
    const fmtK = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, ''));

    /* ---------------- filter bank ---------------- */
    const bankHost = document.getElementById('filter-bank');
    const BANK = ['sobelX', 'sobelY', 'laplacian', 'blur', 'sharpen', 'emboss'];
    const bankCanvases = BANK.map((name) => {
      const wrap = document.createElement('div');
      wrap.className = 'bank-item';
      const c = document.createElement('canvas');
      c.className = 'bank-canvas';
      wrap.appendChild(c);
      const label = document.createElement('div');
      label.className = 'bank-label';
      label.textContent = name;
      wrap.appendChild(label);
      bankHost.appendChild(wrap);
      setupCanvas(c, 104);
      return { name, canvas: c };
    });
    function drawBank() {
      for (const b of bankCanvases) {
        const k = KERNELS[b.name];
        const o = convolve(img, k);
        paint(b.canvas, o, Math.abs(k.reduce((a, v) => a + v, 0)) < 0.01);
      }
    }

    /* ---------------- max pooling demo ---------------- */
    const poolIn = setupCanvas(document.getElementById('pool-in'), 168);
    const poolOut = setupCanvas(document.getElementById('pool-out'), 168);
    function drawPool() {
      const half = N / 2;
      const pooled = new Float32Array(half * half);
      const source = out;
      const centred = Math.abs(kernel.reduce((a, v) => a + v, 0)) < 0.01;
      for (let y = 0; y < half; y++) {
        for (let x = 0; x < half; x++) {
          let m = -Infinity;
          for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
            const v = source[(y * 2 + dy) * N + (x * 2 + dx)];
            if (v > m) m = v;
          }
          pooled[y * half + x] = m;
        }
      }
      // paint a half-resolution image
      const c = document.createElement('canvas');
      c.width = half; c.height = half;
      const g = c.getContext('2d');
      const im = g.createImageData(half, half);
      for (let i = 0; i < pooled.length; i++) {
        const v = centred ? pooled[i] * 0.5 + 0.5 : pooled[i];
        const b = clamp(Math.round(v * 255), 0, 255);
        im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = b;
        im.data[i * 4 + 3] = 255;
      }
      g.putImageData(im, 0, 0);
      const ctx = poolOut._ctx;
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, ctx._cssW, ctx._cssH);
      ctx.drawImage(c, 0, 0, ctx._cssW, ctx._cssH);
      paint(poolIn, out, centred);
    }

    /* ---------------- redraw everything ---------------- */
    function redraw() {
      const centred = Math.abs(kernel.reduce((a, v) => a + v, 0)) < 0.01;
      paint(inCanvas, img, false);
      paint(outCanvas, out, centred);

      // the sliding window, drawn over the input, and the pixel it writes
      const ctx = inCtx(), s = ctx._cssW / N;
      ctx.strokeStyle = '#4da3ff';
      ctx.lineWidth = 2;
      ctx.strokeRect((cursor.x - 1) * s, (cursor.y - 1) * s, s * 3, s * 3);
      ctx.strokeStyle = 'rgba(255,209,102,.95)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cursor.x * s, cursor.y * s, s, s);

      const octx = outCanvas._ctx, os = octx._cssW / N;
      octx.strokeStyle = 'rgba(255,209,102,.95)';
      octx.lineWidth = 1.5;
      octx.strokeRect(cursor.x * os, cursor.y * os, os, os);

      drawMath();
    }

    function recompute() {
      out = convolve(img, kernel);
      drawBank();
      drawPool();
      redraw();
      checkEdgeChallenge();
    }

    /* ---------------- animation controls ---------------- */
    const btnPlay = document.getElementById('btn-play');
    btnPlay.addEventListener('click', () => {
      playing = !playing;
      btnPlay.textContent = playing ? '⏸ Pause scan' : '▶ Scan the image';
      btnPlay.classList.toggle('primary', !playing);
    });
    document.getElementById('btn-step-px').addEventListener('click', () => { advance(); redraw(); });
    slider(document.getElementById('scan-controls'), {
      label: 'scan speed', min: 1, max: 120, step: 1, value: 8,
      format: (v) => v.toFixed(0) + ' px/s', onInput: (v) => { speed = v; },
    });

    let scannedRow = 0;
    function advance() {
      cursor.x++;
      scannedRow++;
      if (cursor.x >= N) {
        cursor.x = 0;
        cursor.y = (cursor.y + 1) % N;
        if (scannedRow >= N) achieve('conv-scan', 'You scanned a full row of the image');
        scannedRow = 0;
      }
    }

    sizeCanvases();
    window.addEventListener('resize', sizeCanvases);
    recompute();

    rafLoop((dt) => {
      if (!playing) return;
      acc += dt * speed;
      let moved = false;
      while (acc >= 1) { acc -= 1; advance(); moved = true; }
      if (moved) redraw();
    }).start();
  });
})();
