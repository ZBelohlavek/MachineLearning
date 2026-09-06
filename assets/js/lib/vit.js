/* ==========================================================================
   vit.js — a very small Vision Transformer, written out longhand.

   Pipeline (one image):
       image  ->  patches  ->  linear embedding + position embedding
              ->  self-attention block (residual)
              ->  per-token MLP (residual)
              ->  average over tokens
              ->  linear classifier
   Tokens live in one flat Float32Array of length T*d: token t, feature i is
   at [t*d + i]. Every backward pass here is written by hand so the attention
   gradient is visible rather than hidden in a framework.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';
  const DEPS = (typeof module === 'object' && module.exports)
    ? Object.assign({}, require('./nn.js'), require('./cnn.js')) : root.ML;
  const { randn, softmax, Param } = DEPS;

  /** A linear layer applied independently to every token: [T,dIn] -> [T,dOut]. */
  class TokenLinear {
    constructor(T, dIn, dOut, rand = Math.random, gain = null) {
      this.T = T; this.dIn = dIn; this.dOut = dOut;
      this.w = new Param(dIn * dOut);
      this.b = new Param(dOut);
      const g = gain ?? Math.sqrt(1 / dIn);
      for (let i = 0; i < this.w.v.length; i++) this.w.v[i] = randn(rand) * g;
      this.out = new Float32Array(T * dOut);
      this.dIn_ = new Float32Array(T * dIn);
    }
    forward(x) {
      const { T, dIn, dOut, out } = this;
      const wv = this.w.v, bv = this.b.v;
      this.x = x;
      for (let t = 0; t < T; t++) {
        const xo = t * dIn, oo = t * dOut;
        for (let o = 0; o < dOut; o++) {
          let s = bv[o];
          const row = o * dIn;
          for (let i = 0; i < dIn; i++) s += wv[row + i] * x[xo + i];
          out[oo + o] = s;
        }
      }
      return out;
    }
    /** Returns dL/dx. Set accumulate=true to add into the existing buffer. */
    backward(dOut, needInputGrad = true) {
      const { T, dIn, dOut: D, x } = this;
      const wv = this.w.v, wg = this.w.g, bg = this.b.g, dx = this.dIn_;
      dx.fill(0);
      for (let t = 0; t < T; t++) {
        const xo = t * dIn, oo = t * D;
        for (let o = 0; o < D; o++) {
          const g = dOut[oo + o];
          if (g === 0) continue;
          bg[o] += g;
          const row = o * dIn;
          for (let i = 0; i < dIn; i++) {
            wg[row + i] += g * x[xo + i];
            if (needInputGrad) dx[xo + i] += wv[row + i] * g;
          }
        }
      }
      return dx;
    }
    params() { return [this.w, this.b]; }
  }

  /**
   * Single-head self-attention.
   *   scores = Q·Kᵀ / sqrt(d)   ->   A = softmax(scores)   ->   out = Wo(A·V)
   * Every token compares itself against every other token and takes a weighted
   * average of their values. The weights are the attention map.
   */
  class SelfAttention {
    constructor(T, d, rand = Math.random) {
      this.T = T; this.d = d;
      this.scale = 1 / Math.sqrt(d);
      this.q = new TokenLinear(T, d, d, rand, 0.5 / Math.sqrt(d));
      this.k = new TokenLinear(T, d, d, rand, 0.5 / Math.sqrt(d));
      this.v = new TokenLinear(T, d, d, rand, 0.5 / Math.sqrt(d));
      this.o = new TokenLinear(T, d, d, rand, 0.5 / Math.sqrt(d));
      this.S = new Float32Array(T * T);
      this.A = new Float32Array(T * T);      // the attention map, kept for drawing
      this.C = new Float32Array(T * d);
      this.dC = new Float32Array(T * d);
      this.dA = new Float32Array(T * T);
      this.dS = new Float32Array(T * T);
      this.dQ = new Float32Array(T * d);
      this.dK = new Float32Array(T * d);
      this.dV = new Float32Array(T * d);
      this.dX = new Float32Array(T * d);
    }

    forward(x) {
      const { T, d, scale, S, A, C } = this;
      // Three different linear views of the same tokens: what each token is
      // looking for (Q), what it offers as a label (K), and what it passes on (V).
      const Qv = this.q.forward(x);
      const K = this.k.forward(x);
      const V = this.v.forward(x);
      this.Qv = Qv; this.Kv = K; this.Vv = V;

      for (let t = 0; t < T; t++) {
        let max = -Infinity;
        for (let u = 0; u < T; u++) {
          let s = 0;
          for (let i = 0; i < d; i++) s += Qv[t * d + i] * K[u * d + i];
          s *= scale;
          S[t * T + u] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let u = 0; u < T; u++) { const e = Math.exp(S[t * T + u] - max); A[t * T + u] = e; sum += e; }
        for (let u = 0; u < T; u++) A[t * T + u] /= sum;
        for (let i = 0; i < d; i++) {
          let acc = 0;
          for (let u = 0; u < T; u++) acc += A[t * T + u] * V[u * d + i];
          C[t * d + i] = acc;
        }
      }
      return this.o.forward(C);
    }

    backward(dOut) {
      const { T, d, scale, A, dA, dS, dQ, dK, dV, dX } = this;
      const Qv = this.Qv, Kv = this.Kv, Vv = this.Vv;
      const dC = this.o.backward(dOut);

      dA.fill(0); dV.fill(0);
      for (let t = 0; t < T; t++) {
        for (let u = 0; u < T; u++) {
          let s = 0;
          for (let i = 0; i < d; i++) s += dC[t * d + i] * Vv[u * d + i];
          dA[t * T + u] = s;                                  // dL/dA
        }
        for (let u = 0; u < T; u++) {
          const a = A[t * T + u];
          for (let i = 0; i < d; i++) dV[u * d + i] += a * dC[t * d + i];
        }
      }

      // softmax jacobian, row by row: dS = A * (dA - sum(A*dA))
      for (let t = 0; t < T; t++) {
        let dot = 0;
        for (let u = 0; u < T; u++) dot += A[t * T + u] * dA[t * T + u];
        for (let u = 0; u < T; u++) dS[t * T + u] = A[t * T + u] * (dA[t * T + u] - dot);
      }

      dQ.fill(0); dK.fill(0);
      for (let t = 0; t < T; t++) {
        for (let u = 0; u < T; u++) {
          const g = dS[t * T + u] * scale;
          if (g === 0) continue;
          for (let i = 0; i < d; i++) {
            dQ[t * d + i] += g * Kv[u * d + i];
            dK[u * d + i] += g * Qv[t * d + i];
          }
        }
      }

      const dxq = this.q.backward(dQ);
      dX.set(dxq);
      const dxk = this.k.backward(dK);
      for (let i = 0; i < dX.length; i++) dX[i] += dxk[i];
      const dxv = this.v.backward(dV);
      for (let i = 0; i < dX.length; i++) dX[i] += dxv[i];
      return dX;
    }

    params() { return [...this.q.params(), ...this.k.params(), ...this.v.params(), ...this.o.params()]; }
  }

  /**
   * LayerNorm, applied independently to each token: subtract that token's mean,
   * divide by its standard deviation, then scale and shift with learned
   * parameters. Without it the residual stream drifts to wildly different
   * scales at different depths and training becomes fragile — which is why
   * every real transformer has one in front of every block.
   */
  class LayerNorm {
    constructor(T, d) {
      this.T = T; this.d = d;
      this.g = new Param(d); this.b = new Param(d);
      this.g.v.fill(1);
      this.out = new Float32Array(T * d);
      this.xhat = new Float32Array(T * d);
      this.inv = new Float32Array(T);     // 1/sigma per token
      this.dX = new Float32Array(T * d);
    }
    forward(x) {
      const { T, d, out, xhat, inv } = this;
      const gv = this.g.v, bv = this.b.v;
      this.x = x;
      for (let t = 0; t < T; t++) {
        const o = t * d;
        let mean = 0;
        for (let i = 0; i < d; i++) mean += x[o + i];
        mean /= d;
        let varsum = 0;
        for (let i = 0; i < d; i++) { const c = x[o + i] - mean; varsum += c * c; }
        const iv = 1 / Math.sqrt(varsum / d + 1e-5);
        inv[t] = iv;
        for (let i = 0; i < d; i++) {
          const h = (x[o + i] - mean) * iv;
          xhat[o + i] = h;
          out[o + i] = gv[i] * h + bv[i];
        }
      }
      return out;
    }
    backward(dOut) {
      const { T, d, xhat, inv, dX } = this;
      const gv = this.g.v, gg = this.g.g, bg = this.b.g;
      for (let t = 0; t < T; t++) {
        const o = t * d;
        let mDh = 0, mDhH = 0;
        for (let i = 0; i < d; i++) {
          const dy = dOut[o + i];
          gg[i] += dy * xhat[o + i];
          bg[i] += dy;
          const dh = dy * gv[i];
          mDh += dh;
          mDhH += dh * xhat[o + i];
        }
        mDh /= d; mDhH /= d;
        for (let i = 0; i < d; i++) {
          const dh = dOut[o + i] * gv[i];
          dX[o + i] = inv[t] * (dh - mDh - xhat[o + i] * mDhH);
        }
      }
      return dX;
    }
    params() { return [this.g, this.b]; }
  }

  /** Per-token two-layer MLP with a ReLU in the middle. */
  class TokenMLP {
    constructor(T, d, hidden, rand = Math.random) {
      this.T = T; this.d = d; this.hidden = hidden;
      this.l1 = new TokenLinear(T, d, hidden, rand);
      this.l2 = new TokenLinear(T, hidden, d, rand, 0.5 / Math.sqrt(hidden));
      this.h = new Float32Array(T * hidden);
      this.dh = new Float32Array(T * hidden);
    }
    forward(x) {
      const a = this.l1.forward(x);
      for (let i = 0; i < a.length; i++) this.h[i] = a[i] > 0 ? a[i] : 0;
      return this.l2.forward(this.h);
    }
    backward(dOut) {
      const dh = this.l2.backward(dOut);
      for (let i = 0; i < dh.length; i++) this.dh[i] = this.h[i] > 0 ? dh[i] : 0;
      return this.l1.backward(this.dh);
    }
    params() { return [...this.l1.params(), ...this.l2.params()]; }
  }

  /* ---------------------------------------------------------------- ViT */
  class ViT {
    /**
     * @param opts { imgSize, patch, dim, mlpHidden, classes, rand }
     */
    constructor(opts = {}) {
      const imgSize = opts.imgSize ?? 20;
      const patch = opts.patch ?? 4;
      const grid = Math.floor(imgSize / patch);
      const useCLS = opts.useCLS ?? true;
      const useNorm = opts.useNorm ?? true;
      const Tp = grid * grid;          // one token per patch
      const T = Tp + (useCLS ? 1 : 0); // plus the class token, which sits at index 0
      const d = opts.dim ?? 24;
      const rand = opts.rand || Math.random;

      Object.assign(this, { imgSize, patch, grid, Tp, T, d, useCLS, useNorm, classes: opts.classes ?? 10 });
      this.patchDim = patch * patch;

      this.embed = new TokenLinear(Tp, this.patchDim, d, rand);
      // The class token is a learned vector that belongs to no patch. It attends
      // to the whole image and its output is what the classifier reads — so the
      // model has one slot whose only job is to summarise.
      this.cls = new Param(d);
      for (let i = 0; i < d; i++) this.cls.v[i] = randn(rand) * 0.02;
      this.pos = new Param(T * d);                       // learned position embeddings
      for (let i = 0; i < this.pos.v.length; i++) this.pos.v[i] = randn(rand) * 0.02;
      this.ln1 = new LayerNorm(T, d);
      this.attn = new SelfAttention(T, d, rand);
      this.ln2 = new LayerNorm(T, d);
      this.mlp = new TokenMLP(T, d, opts.mlpHidden ?? 48, rand);
      this.ln3 = new LayerNorm(T, d);
      this.head = new TokenLinear(1, d, this.classes, rand);

      this.patches = new Float32Array(Tp * this.patchDim);
      this.x0 = new Float32Array(T * d);
      this.x1 = new Float32Array(T * d);
      this.x2 = new Float32Array(T * d);
      this.pooled = new Float32Array(d);
      this.dx2 = new Float32Array(T * d);
      this.dx1 = new Float32Array(T * d);
      this.dx0 = new Float32Array(T * d);
      this.t = 0;
    }

    /** Cut the image into non-overlapping square patches, flattened row-major. */
    patchify(img) {
      const { imgSize, patch, grid, patchDim, patches } = this;
      for (let gy = 0; gy < grid; gy++) {
        for (let gx = 0; gx < grid; gx++) {
          const base = (gy * grid + gx) * patchDim;
          for (let py = 0; py < patch; py++) {
            for (let px = 0; px < patch; px++) {
              patches[base + py * patch + px] =
                img[(gy * patch + py) * imgSize + (gx * patch + px)];
            }
          }
        }
      }
      return patches;
    }

    forward(img) {
      const { T, Tp, d } = this;
      this.patchify(img);
      const e = this.embed.forward(this.patches);
      const off = this.useCLS ? 1 : 0;
      if (this.useCLS) for (let i = 0; i < d; i++) this.x0[i] = this.cls.v[i] + this.pos.v[i];
      for (let t = 0; t < Tp; t++) {
        for (let i = 0; i < d; i++) {
          this.x0[(t + off) * d + i] = e[t * d + i] + this.pos.v[(t + off) * d + i];
        }
      }

      // Pre-norm blocks: normalise, transform, add back to the residual stream.
      const a = this.attn.forward(this.useNorm ? this.ln1.forward(this.x0) : this.x0);
      for (let i = 0; i < T * d; i++) this.x1[i] = this.x0[i] + a[i];

      const m = this.mlp.forward(this.useNorm ? this.ln2.forward(this.x1) : this.x1);
      for (let i = 0; i < T * d; i++) this.x2[i] = this.x1[i] + m[i];

      const n3 = this.useNorm ? this.ln3.forward(this.x2) : this.x2;
      if (this.useCLS) {
        for (let i = 0; i < d; i++) this.pooled[i] = n3[i];  // read the class token
      } else {
        this.pooled.fill(0);
        for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) this.pooled[i] += n3[t * d + i] / T;
      }
      return this.head.forward(this.pooled);
    }

    backward(dLogits) {
      const { T, Tp, d } = this;
      const dPooled = this.head.backward(dLogits);
      const dn3 = this.dn3 || (this.dn3 = new Float32Array(T * d));
      dn3.fill(0);
      if (this.useCLS) {
        for (let i = 0; i < d; i++) dn3[i] = dPooled[i];     // gradient enters at the class token
      } else {
        for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) dn3[t * d + i] = dPooled[i] / T;
      }
      this.dx2.set(this.useNorm ? this.ln3.backward(dn3) : dn3);

      const mg = this.mlp.backward(this.dx2);
      const dm = this.useNorm ? this.ln2.backward(mg) : mg;
      for (let i = 0; i < T * d; i++) this.dx1[i] = this.dx2[i] + dm[i];   // through the residual

      const ag = this.attn.backward(this.dx1);
      const da = this.useNorm ? this.ln1.backward(ag) : ag;
      for (let i = 0; i < T * d; i++) this.dx0[i] = this.dx1[i] + da[i];

      for (let i = 0; i < T * d; i++) this.pos.g[i] += this.dx0[i];
      if (this.useCLS) {
        for (let i = 0; i < d; i++) this.cls.g[i] += this.dx0[i];
        // patch tokens start at index 1, so the embedding only sees the tail
        this.embed.backward(this.dx0.subarray(d), false);
      } else {
        this.embed.backward(this.dx0, false);
      }
    }

    params() {
      const ps = [...this.embed.params(), this.pos, ...this.attn.params(),
                  ...this.mlp.params(), ...this.head.params()];
      if (this.useCLS) ps.push(this.cls);
      if (this.useNorm) ps.push(...this.ln1.params(), ...this.ln2.params(), ...this.ln3.params());
      return ps;
    }
    zeroGrad() { for (const p of this.params()) p.zero(); }
    step(lr, scale = 1, wd = 0) {
      this.t++;
      for (const p of this.params()) p.adam(lr, this.t, scale, wd);
    }
    numParams() { return this.params().reduce((n, p) => n + p.v.length, 0); }

    lossAndGrad(logits, label) {
      const probs = softmax(logits);
      const loss = -Math.log(Math.max(probs[label], 1e-9));
      const d = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) d[i] = probs[i] - (i === label ? 1 : 0);
      return { loss, probs, dLogits: d };
    }

    /** What the class token attends to, per patch (drops its self-attention). */
    clsAttention() {
      const out = new Float32Array(this.Tp);
      if (!this.useCLS) return this.attentionMass().subarray(0, this.Tp);
      for (let u = 0; u < this.Tp; u++) out[u] = this.attn.A[u + 1];
      return out;
    }

    /** Attention weights from one token to all tokens (after a forward pass). */
    attentionFrom(token) {
      return this.attn.A.subarray(token * this.T, (token + 1) * this.T);
    }
    /** How much attention every patch pays to one particular patch. */
    attentionTo(token) {
      const out = new Float32Array(this.T);
      for (let t = 0; t < this.T; t++) out[t] = this.attn.A[t * this.T + token];
      return out;
    }
    /** Average incoming attention per patch — "what did the image look at?" */
    attentionMass() {
      const out = new Float32Array(this.T);
      for (let t = 0; t < this.T; t++)
        for (let u = 0; u < this.T; u++) out[u] += this.attn.A[t * this.T + u] / this.T;
      return out;
    }
    toJSON() { return { params: this.params().map((p) => Array.from(p.v)) }; }
    loadJSON(o) { this.params().forEach((p, i) => p.v.set(o.params[i])); return this; }
  }

  return { TokenLinear, SelfAttention, TokenMLP, LayerNorm, ViT };
});
