/* ==========================================================================
   nn.js — a tiny, dependency-free neural network library.

   Everything here is plain JavaScript and plain arrays: no WebGL, no
   frameworks. The whole point is that you can open this file and read every
   line of the maths that trains the agents and classifiers on this site.

   Contents:
     - RNG helpers (seedable, so runs are reproducible)
     - Dense layer with forward + backward (backpropagation)
     - Adam optimiser
     - MLP: a stack of Dense layers
     - softmax / cross-entropy helpers
   ========================================================================== */

/* Loaded either as a plain <script> (exports land on window.ML) or via
   require() in Node for the test suite — so the pages work from file:// too. */
;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  /** Deterministic 32-bit PRNG. Same seed => same sequence => reproducible runs. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Standard normal sample (Box–Muller) drawn from a uniform generator. */
  function randn(rand = Math.random) {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
  const lerp = (a, b, t) => a + (b - a) * t;

  /* -------------------------------------------------------------------------
     Activations. Each is a pair: f(z) and f'(z) expressed via the output a,
     which is the cheap way to do it for tanh / relu / sigmoid.
     ------------------------------------------------------------------------- */
  const ACT = {
    linear:  { f: (z) => z,                          df: (a) => 1 },
    relu:    { f: (z) => (z > 0 ? z : 0),            df: (a) => (a > 0 ? 1 : 0) },
    lrelu:   { f: (z) => (z > 0 ? z : 0.01 * z),     df: (a) => (a > 0 ? 1 : 0.01) },
    tanh:    { f: (z) => Math.tanh(z),               df: (a) => 1 - a * a },
    sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)),     df: (a) => a * (1 - a) },
  };

  /**
   * A fully connected layer: out = act(W · x + b)
   *
   * Weights live in one flat Float32Array of length nOut*nIn, row-major:
   * W[o * nIn + i] is the weight from input i to output o.
   */
  class Dense {
    constructor(nIn, nOut, act = 'tanh', rand = Math.random, gainScale = 1) {
      this.nIn = nIn;
      this.nOut = nOut;
      this.act = act;
      this.W = new Float32Array(nIn * nOut);
      this.b = new Float32Array(nOut);
      this.dW = new Float32Array(nIn * nOut);
      this.db = new Float32Array(nOut);
      // Adam moment estimates
      this.mW = new Float32Array(nIn * nOut); this.vW = new Float32Array(nIn * nOut);
      this.mb = new Float32Array(nOut);       this.vb = new Float32Array(nOut);

      // He init for relu-family, Xavier otherwise. gainScale lets a policy's
      // output layer start small, so the first policy is near-uniform.
      const gain = (act === 'relu' || act === 'lrelu') ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
      for (let i = 0; i < this.W.length; i++) this.W[i] = randn(rand) * gain * gainScale;

      // Output and input-gradient buffers are allocated once and reused on every
      // forward/backward pass. Allocation inside the loop is the single biggest
      // cost in a JS training loop, and RL runs millions of forward passes.
      this.a = new Float32Array(nOut);   // last output (also needed by backward)
      this.dX = new Float32Array(nIn);   // gradient w.r.t. this layer's input
      this.x = null;                     // last input (kept by reference)
    }

    forward(x) {
      const { nIn, nOut, W, b, a } = this;
      const fn = ACT[this.act].f;
      for (let o = 0; o < nOut; o++) {
        let s = b[o];
        const row = o * nIn;
        for (let i = 0; i < nIn; i++) s += W[row + i] * x[i];
        a[o] = fn(s);
      }
      this.x = x;
      return a;
    }

    /**
     * Given dL/d(output), accumulate weight gradients and return dL/d(input).
     * `needInputGrad=false` skips the dL/dx accumulation, which is pure waste
     * on the first layer of a network — nothing downstream consumes it.
     */
    backward(dA, needInputGrad = true) {
      const { nIn, nOut, W, dW, db, x, a, dX } = this;
      const df = ACT[this.act].df;
      dX.fill(0);
      for (let o = 0; o < nOut; o++) {
        const dz = dA[o] * df(a[o]);      // through the activation
        if (dz === 0) continue;
        db[o] += dz;
        const row = o * nIn;
        if (needInputGrad) {
          for (let i = 0; i < nIn; i++) {
            dW[row + i] += dz * x[i];     // dL/dW = dz * input
            dX[i] += W[row + i] * dz;     // dL/dx = W^T · dz
          }
        } else {
          for (let i = 0; i < nIn; i++) dW[row + i] += dz * x[i];
        }
      }
      return dX;
    }

    zeroGrad() { this.dW.fill(0); this.db.fill(0); }

    /** Adam update. `scale` divides gradients (e.g. by batch size). */
    adam(lr, t, scale = 1, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0) {
      const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
      const upd = (P, G, M, V) => {
        for (let i = 0; i < P.length; i++) {
          let g = G[i] * scale;
          if (wd) g += wd * P[i];
          M[i] = b1 * M[i] + (1 - b1) * g;
          V[i] = b2 * V[i] + (1 - b2) * g * g;
          P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + eps);
        }
      };
      upd(this.W, this.dW, this.mW, this.vW);
      upd(this.b, this.db, this.mb, this.vb);
    }
  }

  /**
   * A stack of Dense layers.
   *   new MLP([8, 32, 32, 4], { hidden: 'tanh', out: 'linear' })
   * gives 8 inputs -> two 32-unit hidden layers -> 4 outputs.
   */
  class MLP {
    constructor(sizes, opts = {}) {
      const hidden = opts.hidden || 'tanh';
      const out = opts.out || 'linear';
      const rand = opts.rand || Math.random;
      const outScale = opts.outScale ?? 1;
      this.sizes = sizes.slice();
      this.layers = [];
      for (let i = 0; i < sizes.length - 1; i++) {
        const isLast = i === sizes.length - 2;
        this.layers.push(new Dense(sizes[i], sizes[i + 1], isLast ? out : hidden, rand,
                                   isLast ? outScale : 1));
      }
      this.t = 0; // Adam timestep
    }

    /** Runs the network. Also records every layer activation for visualisation. */
    forward(x) {
      let a = x;
      this.activations = [x];
      for (const l of this.layers) { a = l.forward(a); this.activations.push(a); }
      return a;
    }

    backward(dOut) {
      let d = dOut;
      for (let i = this.layers.length - 1; i >= 0; i--) d = this.layers[i].backward(d, i > 0);
      return d;
    }

    zeroGrad() { for (const l of this.layers) l.zeroGrad(); }

    step(lr, scale = 1, wd = 0) {
      this.t++;
      for (const l of this.layers) l.adam(lr, this.t, scale, 0.9, 0.999, 1e-8, wd);
    }

    /** Global gradient-norm clipping — keeps RL updates from exploding. */
    clipGradients(maxNorm) {
      let sum = 0;
      for (const l of this.layers) {
        for (let i = 0; i < l.dW.length; i++) sum += l.dW[i] * l.dW[i];
        for (let i = 0; i < l.db.length; i++) sum += l.db[i] * l.db[i];
      }
      const norm = Math.sqrt(sum);
      if (norm > maxNorm && norm > 0) {
        const s = maxNorm / norm;
        for (const l of this.layers) {
          for (let i = 0; i < l.dW.length; i++) l.dW[i] *= s;
          for (let i = 0; i < l.db.length; i++) l.db[i] *= s;
        }
      }
      return norm;
    }

    numParams() {
      return this.layers.reduce((n, l) => n + l.W.length + l.b.length, 0);
    }

    toJSON() {
      return {
        sizes: this.sizes,
        acts: this.layers.map((l) => l.act),
        layers: this.layers.map((l) => ({ W: Array.from(l.W), b: Array.from(l.b) })),
      };
    }

    loadJSON(obj) {
      obj.layers.forEach((src, i) => {
        this.layers[i].W.set(src.W);
        this.layers[i].b.set(src.b);
      });
      return this;
    }

    static fromJSON(obj) {
      const net = new MLP(obj.sizes, { hidden: obj.acts[0], out: obj.acts[obj.acts.length - 1] });
      return net.loadJSON(obj);
    }
  }

  /* ---------------- probability helpers ---------------- */

  /** Numerically stable softmax: turns scores ("logits") into probabilities. */
  function softmax(logits, temperature = 1) {
    return softmaxInto(logits, new Float32Array(logits.length), temperature);
  }

  /** Same, but writes into a buffer you own — no allocation in hot loops. */
  function softmaxInto(logits, out, temperature = 1) {
    const n = logits.length;
    let max = -Infinity;
    for (let i = 0; i < n; i++) { const v = logits[i] / temperature; if (v > max) max = v; }
    let sum = 0;
    for (let i = 0; i < n; i++) { out[i] = Math.exp(logits[i] / temperature - max); sum += out[i]; }
    for (let i = 0; i < n; i++) out[i] /= sum;
    return out;
  }

  /** Draw an index from a probability vector. */
  function sampleFrom(probs, rand = Math.random) {
    let r = rand();
    for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) return i; }
    return probs.length - 1;
  }

  function argmax(arr) {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > bv) { bv = arr[i]; bi = i; }
    return bi;
  }

  function entropy(probs) {
    let h = 0;
    for (let i = 0; i < probs.length; i++) if (probs[i] > 1e-9) h -= probs[i] * Math.log(probs[i]);
    return h;
  }

  /** Exponential moving average, for smoothing noisy training curves. */
  class EMA {
    constructor(alpha = 0.02) { this.alpha = alpha; this.value = null; }
    push(x) {
      this.value = this.value === null ? x : this.value + this.alpha * (x - this.value);
      return this.value;
    }
  }

  return { mulberry32, randn, clamp, lerp, Dense, MLP, softmax, softmaxInto, sampleFrom, argmax, entropy, EMA };
});
