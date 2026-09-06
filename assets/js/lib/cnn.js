/* ==========================================================================
   cnn.js — convolutional layers with backpropagation, in plain JavaScript.

   Tensors are flat Float32Arrays in channel-major order:
       value at channel c, row y, column x  ==  data[(c * H + y) * W + x]

   Layers implement forward(x) and backward(dOut), exactly like the Dense
   layer in nn.js, so they stack the same way.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';
  const DEPS = (typeof module === 'object' && module.exports) ? require('./nn.js') : root.ML;
  const { randn, softmax } = DEPS;

  /** A block of learnable numbers plus its gradient and Adam state. */
  class Param {
    constructor(size) {
      this.v = new Float32Array(size);
      this.g = new Float32Array(size);
      this.m = new Float32Array(size);
      this.s = new Float32Array(size);
    }
    zero() { this.g.fill(0); }
    adam(lr, t, scale, wd, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
      const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
      const { v, g, m, s } = this;
      for (let i = 0; i < v.length; i++) {
        let grad = g[i] * scale;
        if (wd) grad += wd * v[i];
        m[i] = b1 * m[i] + (1 - b1) * grad;
        s[i] = b2 * s[i] + (1 - b2) * grad * grad;
        v[i] -= lr * (m[i] / c1) / (Math.sqrt(s[i] / c2) + eps);
      }
    }
  }

  /* ----------------------------------------------------------- Conv2D */
  class Conv2D {
    /** @param inShape [C,H,W]; stride is always 1 and padding keeps the size. */
    constructor(inShape, outC, k = 3, rand = Math.random) {
      const [C, H, W] = inShape;
      this.inShape = inShape;
      this.outShape = [outC, H, W];
      this.C = C; this.H = H; this.W = W; this.outC = outC; this.k = k;
      this.pad = (k - 1) >> 1;
      this.w = new Param(outC * C * k * k);
      this.b = new Param(outC);
      // He initialisation: variance 2/fan_in keeps ReLU activations alive.
      const gain = Math.sqrt(2 / (C * k * k));
      for (let i = 0; i < this.w.v.length; i++) this.w.v[i] = randn(rand) * gain;
      this.out = new Float32Array(outC * H * W);
      this.dIn = new Float32Array(C * H * W);
    }

    forward(x) {
      const { C, H, W, outC, k, pad, out } = this;
      const wv = this.w.v, bv = this.b.v;
      this.x = x;
      out.fill(0);
      for (let oc = 0; oc < outC; oc++) {
        const obase = oc * H * W;
        const bias = bv[oc];
        for (let y = 0; y < H; y++) {
          for (let x2 = 0; x2 < W; x2++) out[obase + y * W + x2] = bias;
        }
        for (let ic = 0; ic < C; ic++) {
          const wbase = ((oc * C + ic) * k) * k;
          const ibase = ic * H * W;
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              const weight = wv[wbase + ky * k + kx];
              if (weight === 0) continue;
              const dy = ky - pad, dx = kx - pad;
              const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
              const x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
              for (let y = y0; y < y1; y++) {
                const orow = obase + y * W;
                const irow = ibase + (y + dy) * W + dx;
                for (let x2 = x0; x2 < x1; x2++) out[orow + x2] += weight * x[irow + x2];
              }
            }
          }
        }
      }
      return out;
    }

    backward(dOut, needInputGrad = true) {
      const { C, H, W, outC, k, pad, x, dIn } = this;
      const wv = this.w.v, wg = this.w.g, bg = this.b.g;
      dIn.fill(0);
      for (let oc = 0; oc < outC; oc++) {
        const obase = oc * H * W;
        let bsum = 0;
        for (let i = 0; i < H * W; i++) bsum += dOut[obase + i];
        bg[oc] += bsum;
        for (let ic = 0; ic < C; ic++) {
          const wbase = ((oc * C + ic) * k) * k;
          const ibase = ic * H * W;
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              const dy = ky - pad, dx = kx - pad;
              const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
              const x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
              let gsum = 0;
              const weight = wv[wbase + ky * k + kx];
              for (let y = y0; y < y1; y++) {
                const orow = obase + y * W;
                const irow = ibase + (y + dy) * W + dx;
                for (let x2 = x0; x2 < x1; x2++) {
                  const g = dOut[orow + x2];
                  gsum += g * x[irow + x2];
                  if (needInputGrad) dIn[irow + x2] += g * weight;
                }
              }
              wg[wbase + ky * k + kx] += gsum;
            }
          }
        }
      }
      return dIn;
    }

    params() { return [this.w, this.b]; }
  }

  /* ------------------------------------------------------------- ReLU */
  class ReLU {
    constructor(shape) {
      this.inShape = this.outShape = shape;
      const n = shape.reduce((a, b) => a * b, 1);
      this.out = new Float32Array(n);
      this.dIn = new Float32Array(n);
    }
    forward(x) {
      this.x = x;
      for (let i = 0; i < x.length; i++) this.out[i] = x[i] > 0 ? x[i] : 0;
      return this.out;
    }
    backward(dOut) {
      for (let i = 0; i < dOut.length; i++) this.dIn[i] = this.x[i] > 0 ? dOut[i] : 0;
      return this.dIn;
    }
    params() { return []; }
  }

  /* --------------------------------------------------------- MaxPool2 */
  class MaxPool2 {
    constructor(inShape) {
      const [C, H, W] = inShape;
      this.inShape = inShape;
      this.H = H; this.W = W; this.C = C;
      this.oH = H >> 1; this.oW = W >> 1;
      this.outShape = [C, this.oH, this.oW];
      this.out = new Float32Array(C * this.oH * this.oW);
      this.arg = new Int32Array(C * this.oH * this.oW);
      this.dIn = new Float32Array(C * H * W);
    }
    forward(x) {
      const { C, H, W, oH, oW, out, arg } = this;
      for (let c = 0; c < C; c++) {
        for (let y = 0; y < oH; y++) {
          for (let px = 0; px < oW; px++) {
            let best = -Infinity, bi = 0;
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                const i = (c * H + y * 2 + dy) * W + px * 2 + dx;
                if (x[i] > best) { best = x[i]; bi = i; }
              }
            }
            const o = (c * oH + y) * oW + px;
            out[o] = best;
            arg[o] = bi;          // remember which pixel won: only it gets the gradient
          }
        }
      }
      return out;
    }
    backward(dOut) {
      this.dIn.fill(0);
      for (let o = 0; o < dOut.length; o++) this.dIn[this.arg[o]] += dOut[o];
      return this.dIn;
    }
    params() { return []; }
  }

  /* ------------------------------------------------ Flatten and Dense */
  class Flatten {
    constructor(inShape) {
      this.inShape = inShape;
      this.outShape = [inShape.reduce((a, b) => a * b, 1)];
    }
    forward(x) { return x; }
    backward(d) { return d; }
    params() { return []; }
  }

  class FC {
    constructor(nIn, nOut, act = 'linear', rand = Math.random) {
      this.nIn = nIn; this.nOut = nOut; this.act = act;
      this.inShape = [nIn]; this.outShape = [nOut];
      this.w = new Param(nIn * nOut);
      this.b = new Param(nOut);
      const gain = act === 'relu' ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
      for (let i = 0; i < this.w.v.length; i++) this.w.v[i] = randn(rand) * gain;
      this.out = new Float32Array(nOut);
      this.dIn = new Float32Array(nIn);
    }
    forward(x) {
      const { nIn, nOut, out } = this;
      const wv = this.w.v, bv = this.b.v;
      this.x = x;
      for (let o = 0; o < nOut; o++) {
        let s = bv[o];
        const row = o * nIn;
        for (let i = 0; i < nIn; i++) s += wv[row + i] * x[i];
        out[o] = this.act === 'relu' ? (s > 0 ? s : 0) : s;
      }
      return out;
    }
    backward(dOut, needInputGrad = true) {
      const { nIn, nOut, x, dIn, out } = this;
      const wv = this.w.v, wg = this.w.g, bg = this.b.g;
      dIn.fill(0);
      for (let o = 0; o < nOut; o++) {
        let d = dOut[o];
        if (this.act === 'relu' && out[o] <= 0) d = 0;
        if (d === 0) continue;
        bg[o] += d;
        const row = o * nIn;
        for (let i = 0; i < nIn; i++) {
          wg[row + i] += d * x[i];
          if (needInputGrad) dIn[i] += wv[row + i] * d;
        }
      }
      return dIn;
    }
    params() { return [this.w, this.b]; }
  }

  /* -------------------------------------------------------------- Net */
  class ConvNet {
    constructor(layers) {
      this.layers = layers;
      this.t = 0;
    }
    forward(x) {
      this.activations = [x];
      let a = x;
      for (const l of this.layers) { a = l.forward(a); this.activations.push(a); }
      return a;
    }
    backward(dOut) {
      let d = dOut;
      for (let i = this.layers.length - 1; i >= 0; i--) {
        d = this.layers[i].backward(d, i > 0);
      }
      return d;
    }
    zeroGrad() { for (const l of this.layers) for (const p of l.params()) p.zero(); }
    step(lr, scale = 1, wd = 0) {
      this.t++;
      for (const l of this.layers) for (const p of l.params()) p.adam(lr, this.t, scale, wd);
    }
    numParams() {
      let n = 0;
      for (const l of this.layers) for (const p of l.params()) n += p.v.length;
      return n;
    }
    /** Cross-entropy loss on softmax outputs; returns { loss, probs, dLogits }. */
    lossAndGrad(logits, label) {
      const probs = softmax(logits);
      const loss = -Math.log(Math.max(probs[label], 1e-9));
      const d = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) d[i] = probs[i] - (i === label ? 1 : 0);
      return { loss, probs, dLogits: d };
    }
    toJSON() {
      return {
        params: this.layers.map((l) => l.params().map((p) => Array.from(p.v))),
      };
    }
    loadJSON(o) {
      this.layers.forEach((l, i) => l.params().forEach((p, j) => p.v.set(o.params[i][j])));
      return this;
    }
  }

  return { Param, Conv2D, ReLU, MaxPool2, Flatten, FC, ConvNet };
});
