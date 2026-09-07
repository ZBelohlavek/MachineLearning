/* ==========================================================================
   charlm.js — a character-level language model: a small causal transformer.

   The same attention block as the Vision Transformer, pointed at text instead
   of patches. Three differences, and they are the whole idea:

     1. Tokens are characters, looked up in an embedding table rather than
        projected from pixels.
     2. Attention is *causal*: position t may look at positions 0..t and no
        further, so predicting the next character can never peek at it.
     3. Every position predicts the character that follows it, so one pass over
        a 24-character window produces 24 training signals rather than one.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';
  const DEPS = (typeof module === 'object' && module.exports)
    ? Object.assign({}, require('./nn.js'), require('./cnn.js'), require('./vit.js')) : root.ML;
  const { randn, softmax, sampleFrom, Param, TokenLinear, SelfAttention, TokenMLP, LayerNorm } = DEPS;

  /** Build a character vocabulary from a corpus. */
  function buildVocab(text) {
    const chars = [...new Set(text)].sort();
    const toId = new Map(chars.map((c, i) => [c, i]));
    return { chars, size: chars.length, toId, encode: (s) => Array.from(s, (c) => toId.get(c) ?? 0),
             decode: (ids) => Array.from(ids, (i) => chars[i] ?? '?').join('') };
  }

  class CharLM {
    constructor(opts = {}) {
      const T = opts.context ?? 24;
      const d = opts.dim ?? 32;
      const V = opts.vocab;
      const rand = opts.rand || Math.random;
      Object.assign(this, { T, d, V, heads: opts.heads ?? 1 });

      // A character embedding is a lookup table: one learned vector per symbol.
      this.emb = new Param(V * d);
      for (let i = 0; i < this.emb.v.length; i++) this.emb.v[i] = randn(rand) * 0.08;
      this.pos = new Param(T * d);
      for (let i = 0; i < this.pos.v.length; i++) this.pos.v[i] = randn(rand) * 0.02;

      this.ln1 = new LayerNorm(T, d);
      this.attn = new SelfAttention(T, d, rand, this.heads, true);   // causal
      this.ln2 = new LayerNorm(T, d);
      this.mlp = new TokenMLP(T, d, opts.mlpHidden ?? d * 2, rand);
      this.ln3 = new LayerNorm(T, d);
      this.head = new TokenLinear(T, d, V, rand, 0.5 / Math.sqrt(d));

      this.x0 = new Float32Array(T * d);
      this.x1 = new Float32Array(T * d);
      this.x2 = new Float32Array(T * d);
      this.dx0 = new Float32Array(T * d);
      this.dx1 = new Float32Array(T * d);
      this.dx2 = new Float32Array(T * d);
      this.dn3 = new Float32Array(T * d);
      this.ids = new Int32Array(T);
      this.t = 0;
    }

    /** ids: Int32Array(T) of character ids. Returns logits for all T positions. */
    forward(ids) {
      const { T, d } = this;
      this.ids.set(ids);
      for (let t = 0; t < T; t++) {
        const e = ids[t] * d, p = t * d;
        for (let i = 0; i < d; i++) this.x0[p + i] = this.emb.v[e + i] + this.pos.v[p + i];
      }
      const a = this.attn.forward(this.ln1.forward(this.x0));
      for (let i = 0; i < T * d; i++) this.x1[i] = this.x0[i] + a[i];
      const m = this.mlp.forward(this.ln2.forward(this.x1));
      for (let i = 0; i < T * d; i++) this.x2[i] = this.x1[i] + m[i];
      return this.head.forward(this.ln3.forward(this.x2));
    }

    /**
     * Cross-entropy against the next character at every position at once.
     * targets[t] is the character that follows position t.
     */
    lossAndGrad(logits, targets) {
      const { T, V } = this;
      const d = new Float32Array(T * V);
      let loss = 0;
      const row = new Float32Array(V);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < V; i++) row[i] = logits[t * V + i];
        const p = softmax(row);
        const y = targets[t];
        loss += -Math.log(Math.max(p[y], 1e-9));
        for (let i = 0; i < V; i++) d[t * V + i] = (p[i] - (i === y ? 1 : 0)) / T;
      }
      return { loss: loss / T, dLogits: d };
    }

    backward(dLogits) {
      const { T, d } = this;
      const dn3 = this.head.backward(dLogits);
      this.dn3.set(dn3);
      this.dx2.set(this.ln3.backward(this.dn3));
      const dm = this.ln2.backward(this.mlp.backward(this.dx2));
      for (let i = 0; i < T * d; i++) this.dx1[i] = this.dx2[i] + dm[i];
      const da = this.ln1.backward(this.attn.backward(this.dx1));
      for (let i = 0; i < T * d; i++) this.dx0[i] = this.dx1[i] + da[i];

      for (let i = 0; i < T * d; i++) this.pos.g[i] += this.dx0[i];
      // an embedding row collects gradient from every position that used it
      for (let t = 0; t < T; t++) {
        const e = this.ids[t] * d, p = t * d;
        for (let i = 0; i < d; i++) this.emb.g[e + i] += this.dx0[p + i];
      }
    }

    params() {
      return [this.emb, this.pos, ...this.ln1.params(), ...this.attn.params(),
              ...this.ln2.params(), ...this.mlp.params(), ...this.ln3.params(),
              ...this.head.params()];
    }
    zeroGrad() { for (const p of this.params()) p.zero(); }
    step(lr, scale = 1, wd = 0) {
      this.t++;
      for (const p of this.params()) p.adam(lr, this.t, scale, wd);
    }
    numParams() { return this.params().reduce((n, p) => n + p.v.length, 0); }

    /** Probabilities for the character following a full context window. */
    nextProbs(ids, temperature = 1) {
      const logits = this.forward(ids);
      const { T, V } = this;
      const row = new Float32Array(V);
      for (let i = 0; i < V; i++) row[i] = logits[(T - 1) * V + i];
      return softmax(row, Math.max(0.05, temperature));
    }

    /** Continue a prompt, sliding the window one character at a time. */
    sample(vocab, prompt, length, temperature = 1, rand = Math.random) {
      const { T } = this;
      const ids = new Int32Array(T);
      const seed = vocab.encode(prompt.slice(-T));
      for (let i = 0; i < T; i++) {
        const k = seed.length - T + i;
        ids[i] = k >= 0 ? seed[k] : (vocab.toId.get(' ') ?? 0);
      }
      let out = '';
      for (let n = 0; n < length; n++) {
        const p = this.nextProbs(ids, temperature);
        const next = sampleFrom(p, rand);
        out += vocab.chars[next];
        ids.copyWithin(0, 1);
        ids[T - 1] = next;
      }
      return out;
    }

    toJSON() { return { params: this.params().map((p) => Array.from(p.v)) }; }
    loadJSON(o) { this.params().forEach((p, i) => p.v.set(o.params[i])); return this; }
  }

  return { CharLM, buildVocab };
});
