/* ==========================================================================
   plot.js — minimal canvas charting (line charts + helpers).
   No chart library; just enough to watch learning curves move in real time.
   ========================================================================== */

/* Loaded either as a plain <script> (exports land on window.ML) or via
   require() in Node for the test suite — so the pages work from file:// too. */
;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  /**
   * Size a canvas to the width its CSS gives it (not its parent's padding box)
   * and a fixed pixel height, then return a high-DPI context.
   */
  function fit(canvas, height) {
    canvas.style.width = '';                    // drop any inline width we set before
    const w = canvas.clientWidth || (canvas.parentElement ? canvas.parentElement.clientWidth : 400) || 400;
    return hidpi(canvas, w, height);
  }

  /** Make a canvas crisp on high-DPI screens. Returns the 2d context. */
  function hidpi(canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    const w = cssW ?? canvas.clientWidth ?? canvas.width;
    const h = cssH ?? canvas.clientHeight ?? canvas.height;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx._cssW = w; ctx._cssH = h;
    return ctx;
  }

  const fmt = (v) => {
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a >= 0.01) return v.toFixed(3);
    if (a === 0) return '0';
    return v.toExponential(1);
  };

  class LineChart {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {object} opts  { series:[{name,color,width}], yLabel, xLabel, maxPoints,
     *                         yMin, yMax, zeroLine, height }
     */
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.opts = Object.assign({
        maxPoints: 600, yLabel: '', xLabel: '', zeroLine: false,
        pad: { l: 46, r: 10, t: 10, b: 24 },
      }, opts);
      this.series = (opts.series || [{ name: '', color: '#4da3ff' }]).map((s) => ({
        name: s.name || '', color: s.color || '#4da3ff', width: s.width || 2,
        dashed: !!s.dashed, data: [],
      }));
      this.xs = [];
      this.resize();
      window.addEventListener('resize', () => { this.resize(); this.draw(); });
    }

    resize() {
      this.ctx = fit(this.canvas, this.opts.height || 180);
    }

    /** push(x, [y0, y1, ...]) — one y per series (null to skip a point). */
    push(x, ys) {
      this.xs.push(x);
      this.series.forEach((s, i) => s.data.push(ys[i]));
      const max = this.opts.maxPoints;
      if (this.xs.length > max * 2) {
        // Down-sample by 2 instead of dropping history, so the whole run stays visible.
        const keep = (arr) => arr.filter((_, i) => i % 2 === 0);
        this.xs = keep(this.xs);
        this.series.forEach((s) => (s.data = keep(s.data)));
      }
    }

    clear() { this.xs = []; this.series.forEach((s) => (s.data = [])); this.draw(); }

    draw() {
      const ctx = this.ctx, W = ctx._cssW, H = ctx._cssH, P = this.opts.pad;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, W, H);

      const plotW = W - P.l - P.r, plotH = H - P.t - P.b;
      if (plotW <= 4 || plotH <= 4) return;

      // --- y range ---
      let lo = this.opts.yMin ?? Infinity, hi = this.opts.yMax ?? -Infinity;
      if (this.opts.yMin == null || this.opts.yMax == null) {
        for (const s of this.series) {
          for (const v of s.data) {
            if (v == null || !isFinite(v)) continue;
            if (this.opts.yMin == null && v < lo) lo = v;
            if (this.opts.yMax == null && v > hi) hi = v;
          }
        }
      }
      if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
      if (hi - lo < 1e-9) { hi = lo + 1; lo -= 0.5; }
      const padY = (hi - lo) * 0.08; lo -= padY; hi += padY;

      const n = this.xs.length;
      const x0 = this.xs[0] ?? 0, x1 = this.xs[n - 1] ?? 1;
      const spanX = Math.max(1e-9, x1 - x0);
      const px = (x) => P.l + ((x - x0) / spanX) * plotW;
      const py = (y) => P.t + (1 - (y - lo) / (hi - lo)) * plotH;

      // --- grid + y ticks ---
      ctx.strokeStyle = '#1e2739'; ctx.lineWidth = 1;
      ctx.fillStyle = '#6d7f9c';
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let i = 0; i <= 4; i++) {
        const v = lo + ((hi - lo) * i) / 4;
        const y = Math.round(py(v)) + 0.5;
        ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(W - P.r, y); ctx.stroke();
        ctx.fillText(fmt(v), P.l - 6, y);
      }
      if (this.opts.zeroLine && lo < 0 && hi > 0) {
        ctx.strokeStyle = '#3a4666'; ctx.setLineDash([3, 3]);
        const y = Math.round(py(0)) + 0.5;
        ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(W - P.r, y); ctx.stroke();
        ctx.setLineDash([]);
      }

      // --- x ticks ---
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      if (n > 1) {
        const lastTick = this.opts.xLabel ? 2 : 3;
        for (let i = 0; i <= lastTick; i++) {
          const xv = x0 + (spanX * i) / 3;
          ctx.fillText(xv >= 1000 ? (xv / 1000).toFixed(1) + 'k' : xv.toFixed(0), px(xv), H - P.b + 5);
        }
      }
      if (this.opts.xLabel) {
        ctx.textAlign = 'right';
        ctx.fillText(this.opts.xLabel, W - P.r, H - P.b + 5);
      }

      // --- series ---
      ctx.save();
      ctx.beginPath(); ctx.rect(P.l, P.t, plotW, plotH); ctx.clip();
      for (const s of this.series) {
        ctx.strokeStyle = s.color; ctx.lineWidth = s.width;
        ctx.setLineDash(s.dashed ? [4, 4] : []);
        ctx.lineJoin = 'round';
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < n; i++) {
          const v = s.data[i];
          if (v == null || !isFinite(v)) { started = false; continue; }
          const X = px(this.xs[i]), Y = py(v);
          if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.restore();

      // --- legend ---
      if (this.series.some((s) => s.name)) {
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        let lx = P.l + 6;
        for (const s of this.series) {
          if (!s.name) continue;
          const last = [...s.data].reverse().find((v) => v != null && isFinite(v));
          const label = s.name + (last != null ? '  ' + fmt(last) : '');
          const wNeeded = ctx.measureText(label).width + 13;
          if (lx + wNeeded > W - P.r) break;      // never spill out of the plot
          ctx.fillStyle = s.color;
          ctx.fillRect(lx, P.t + 6, 8, 3);
          ctx.fillStyle = '#9fb0cc';
          ctx.fillText(label, lx + 13, P.t + 8);
          lx += wNeeded + 16;
        }
      }
    }
  }

  /** Draw a horizontal bar showing a signed value in [-max, max]. */
  function signedBar(el, value, max, colorPos = '#38d39f', colorNeg = '#ff6b6b') {
    const pct = Math.max(-1, Math.min(1, value / max)) * 50;
    el.style.background = value >= 0 ? colorPos : colorNeg;
    if (value >= 0) { el.style.left = '50%'; el.style.width = pct + '%'; }
    else { el.style.left = (50 + pct) + '%'; el.style.width = (-pct) + '%'; }
  }

  /** Blue→white→red diverging colour, for weights and feature maps. */
  function diverging(v, scale = 1) {
    const t = Math.max(-1, Math.min(1, v / scale));
    if (t >= 0) {
      const c = Math.round(255 * (1 - t));
      return `rgb(${255 - Math.round(60 * t)},${c + 20 > 255 ? 255 : c + 20},${c})`;
    }
    const c = Math.round(255 * (1 + t));
    return `rgb(${c},${c + 15 > 255 ? 255 : c + 15},${255 - Math.round(30 * t)})`;
  }

  /** Perceptually simple heat colour for [0,1] activations. */
  function heat(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255 * Math.min(1, Math.max(0, 1.6 * t - 0.2)));
    const g = Math.round(255 * Math.min(1, Math.max(0, 1.5 * t - 0.35)));
    const b = Math.round(255 * Math.min(1, Math.max(0, 0.9 - 1.4 * t)) + 40 * t);
    return `rgb(${r},${g},${b})`;
  }

  return { hidpi, fit, LineChart, signedBar, diverging, heat };
});
