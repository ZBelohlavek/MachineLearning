/* ==========================================================================
   language.js — "Teach a model to write": a character-level transformer.

   The same attention block as the Vision Transformer, pointed at text. You
   supply the corpus, watch the loss fall, and read the samples turn from
   noise into words into sentences.
   ========================================================================== */

;(function () {
  'use strict';
  const { CharLM, buildVocab, mulberry32, softmax, argmax, clamp, hidpi, LineChart, heat,
          chrome, nextLinks, achieve, goalPanel, slider, pills, statGrid, rafLoop } = window.ML;

  /* Original corpora, written for this lesson. Short, repetitive and heavily
     patterned, because a 10,000-parameter model learns structure long before
     it learns anything resembling meaning. */
  const CORPORA = {
    harbour: `the harbour wakes early. the boats go out at four, and the water is black and flat.
the gulls follow the boats out past the wall, and the light comes up behind the hill.
by six the harbour is empty and the town is quiet. the tide goes out and the mud shines.
the boats come back at noon with the catch, and the gulls follow them in again.
the men unload the boxes onto the stone, and the ice runs off into the water.
the harbour is loud for an hour, and then the boats are tied and the town is quiet again.
the tide comes in at three and lifts the boats off the mud. the water is green in the sun.
the light goes down behind the hill at seven, and the harbour is dark and still.
the boats knock against the wall all night, and the water moves under them.
the harbour wakes early. the boats go out at four, and the water is black and flat.`,

    recipe: `to make bread you need flour, water, salt and yeast.
mix the flour and the salt in a bowl. warm the water and stir in the yeast.
pour the water into the flour and mix until the dough comes together.
knead the dough for ten minutes until it is smooth and springs back.
put the dough in a bowl, cover it, and leave it for an hour until it doubles.
knock the dough back and shape it into a round. leave it to rise for another hour.
heat the oven as hot as it will go. slash the top of the loaf with a knife.
bake the loaf for thirty minutes until it is dark and sounds hollow underneath.
cool the loaf on a rack for an hour before you cut it. do not cut it while it is hot.
to make bread you need flour, water, salt and yeast, and you need to wait.`,

    notes: `a model is a function with numbers in it that you can change.
training is the process of changing those numbers so the answers get better.
the loss says how wrong the model is. the gradient says which way to change each number.
gradient descent takes a small step downhill and then measures the slope again.
a small step is slow. a large step overshoots the bottom and climbs the other side.
the training set is what the model learns from. the test set is what you judge it by.
if the training loss falls and the test loss rises, the model is memorising the training set.
more data helps. a smaller model helps. weight decay helps. training longer does not help.
a model can only learn the variation it has seen, and it cannot tell you when it has not seen something.
a model is a function with numbers in it, and every one of those numbers came from data.`,
  };

  document.addEventListener('DOMContentLoaded', () => {
    chrome('language', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.language);
    goalPanel(document.getElementById('lesson-goals'), 'language');
    nextLinks(document.getElementById('next-links'), 'attention', 'gridworld', '../');

    let seed = 1234;
    let rng = mulberry32(seed);
    let context = 24, dim = 32, heads = 1, lr = 0.01, batch = 16, temperature = 0.6;
    let net = null, vocab = null, data = null, running = false;
    let step = 0, charsSeen = 0, customText = false;

    const corpusEl = document.getElementById('corpus');
    corpusEl.value = CORPORA.harbour;

    /* ------------------------------------------------ model + data ------ */
    function rebuild() {
      const text = corpusEl.value.length > 40 ? corpusEl.value : CORPORA.harbour;
      vocab = buildVocab(text);
      data = Int32Array.from(vocab.encode(text));
      rng = mulberry32(seed + 7919);
      net = new CharLM({ vocab: vocab.size, context, dim, mlpHidden: dim * 2, heads, rand: rng });
      window.__charlm = net;
      step = 0; charsSeen = 0;
      chart.clear();
      setStat('parameters', net.numParams().toLocaleString());
      setStat('vocabulary', vocab.size + ' characters');
      setStat('corpus', text.length.toLocaleString() + ' chars');
      setStat('loss', '–'); setStat('chars seen', '0'); setStat('step', '0');
      drawVocab();
      refreshSample();
      drawAttention();
      drawNextProbs();
    }

    function trainBatch() {
      if (!net || data.length < context + 2) return;
      net.zeroGrad();
      let loss = 0;
      for (let b = 0; b < batch; b++) {
        const i = Math.floor(rng() * (data.length - context - 1));
        const out = net.forward(data.subarray(i, i + context));
        const g = net.lossAndGrad(out, data.subarray(i + 1, i + context + 1));
        loss += g.loss;
        net.backward(g.dLogits);
        charsSeen += context;
      }
      net.step(lr, 1 / batch);
      step++;
      return loss / batch;
    }

    function evalLoss(n = 24) {
      let loss = 0;
      for (let k = 0; k < n; k++) {
        const i = ((k * 9973) % (data.length - context - 1));
        loss += net.lossAndGrad(net.forward(data.subarray(i, i + context)),
                                data.subarray(i + 1, i + context + 1)).loss;
      }
      return loss / n;
    }

    /* ------------------------------------------------ views -------------- */
    const chart = new LineChart(document.getElementById('lm-chart'), {
      height: 150, xLabel: 'training steps', yMin: 0,
      series: [{ name: 'loss (bits per character, natural log)', color: '#4da3ff' }],
    });
    const setStat = statGrid(document.getElementById('lm-stats'),
      ['step', 'loss', 'chars seen', 'vocabulary', 'corpus', 'parameters']);

    function drawVocab() {
      const host = document.getElementById('vocab-view');
      host.innerHTML = vocab.chars.map((c, i) =>
        `<span class="vchip"><b>${c === ' ' ? '␣' : c === '\\n' ? '⏎' : c}</b><i>${i}</i></span>`).join('');
    }

    const sampleEl = document.getElementById('sample-out');
    function refreshSample() {
      const prompt = document.getElementById('prompt').value || 'the ';
      const text = net.sample(vocab, prompt, 220, temperature, rng);
      sampleEl.innerHTML = `<span class="prompt">${escapeHtml(prompt)}</span>${escapeHtml(text)}`;
    }
    const escapeHtml = (t) => t.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

    /* the distribution over the next character, for the prompt as typed */
    const probHost = document.getElementById('next-probs');
    function drawNextProbs() {
      const prompt = document.getElementById('prompt').value || 'the ';
      const ids = new Int32Array(context);
      const enc = vocab.encode(prompt.slice(-context));
      for (let i = 0; i < context; i++) {
        const k = enc.length - context + i;
        ids[i] = k >= 0 ? enc[k] : (vocab.toId.get(' ') ?? 0);
      }
      const p = net.nextProbs(ids, 1);
      const order = Array.from(p, (v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
      probHost.innerHTML = order.map(([v, i]) => {
        const ch = vocab.chars[i];
        const label = ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch;
        return `<div class="pbar"><span class="d">${label}</span>
          <span class="track"><i style="width:${(v * 100).toFixed(1)}%"></i></span>
          <span class="v">${(v * 100).toFixed(0)}%</span></div>`;
      }).join('');
      return ids;
    }

    /* the causal attention map: a triangle, because nothing may look forward */
    const attnCanvas = document.getElementById('lm-attn');
    attnCanvas._ctx = hidpi(attnCanvas, attnCanvas.clientWidth || 300, 300);
    function drawAttention() {
      const ids = drawNextProbs();
      net.forward(ids);
      const T = net.T;
      const A = net.attn.H > 1 ? net.attn.mean() : net.attn.head(0);
      const ctx = attnCanvas._ctx;
      const W = ctx._cssW, H = ctx._cssH;
      const pad = 22;
      const cell = (Math.min(W, H) - pad) / T;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, W, H);
      let max = 0;
      for (let i = 0; i < T * T; i++) max = Math.max(max, A[i]);
      for (let t = 0; t < T; t++) {
        for (let u = 0; u < T; u++) {
          const v = A[t * T + u] / (max || 1);
          ctx.fillStyle = u > t ? '#0d1320' : heat(v);
          ctx.fillRect(pad + u * cell, pad + t * cell, cell - 1, cell - 1);
        }
      }
      ctx.fillStyle = '#6d7f9c';
      ctx.font = `${Math.max(7, Math.min(11, cell * 0.8))}px ui-monospace, monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let t = 0; t < T; t++) {
        const c = vocab.chars[ids[t]];
        const label = c === ' ' ? '␣' : c === '\n' ? '⏎' : c;
        ctx.fillText(label, pad + t * cell + cell / 2, pad / 2);          // keys, along the top
        ctx.fillText(label, pad / 2, pad + t * cell + cell / 2);          // queries, down the side
      }
    }

    /* ------------------------------------------------ controls ----------- */
    pills(document.getElementById('corpus-presets'), [
      { value: 'harbour', label: 'Harbour' },
      { value: 'recipe', label: 'Bread recipe' },
      { value: 'notes', label: 'ML notes' },
    ], 'harbour', (v) => { corpusEl.value = CORPORA[v]; customText = false; rebuild(); });

    corpusEl.addEventListener('change', () => {
      customText = true;
      achieve('lm-own', 'You trained it on text you supplied');
      rebuild();
    });

    document.getElementById('prompt').addEventListener('input', () => {
      drawNextProbs();
      drawAttention();
    });

    const cfg = document.getElementById('lm-config');
    slider(cfg, {
      label: 'context window', min: 8, max: 48, step: 4, value: context,
      format: (v) => v.toFixed(0) + ' characters', onInput: (v) => { context = v | 0; rebuild(); },
      desc: 'How far back the model can see. Everything before this simply does not exist to it — which is why a short context produces text that forgets what sentence it is in.',
    });
    slider(cfg, {
      label: 'embedding size', min: 8, max: 64, step: 8, value: dim,
      format: (v) => v.toFixed(0), onInput: (v) => { dim = v | 0; rebuild(); },
    });
    slider(cfg, {
      label: 'learning rate', min: 0.001, max: 0.03, step: 0.001, value: lr,
      format: (v) => v.toFixed(3), onInput: (v) => { lr = v; },
    });
    slider(cfg, {
      label: 'temperature', min: 0.1, max: 1.6, step: 0.05, value: temperature,
      format: (v) => v.toFixed(2), onInput: (v) => { temperature = v; refreshSample(); },
      desc: 'How adventurously it samples. Near zero it repeats its single most likely continuation forever; above 1 it starts inventing spellings.',
    });
    pills(cfg, [
      { value: 1, label: '1 head' },
      { value: 2, label: '2 heads' },
      { value: 4, label: '4 heads' },
    ], 1, (v) => { heads = v; rebuild(); });

    window.ML.runBar(document.getElementById('run-bar'), {
      seed,
      charts: () => [chart],
      onSeed: (v) => { seed = v; rebuild(); },
      note: 'Same seed, same starting weights and the same batches.',
    });

    const say = window.ML.announcer();
    const btnTrain = document.getElementById('btn-train');
    btnTrain.addEventListener('click', () => {
      running = !running;
      btnTrain.textContent = running ? '⏸ Pause training' : '▶ Train the model';
      btnTrain.classList.toggle('primary', !running);
    });
    document.getElementById('btn-reset').addEventListener('click', () => rebuild());
    document.getElementById('btn-sample').addEventListener('click', refreshSample);

    /* ------------------------------------------------ loop --------------- */
    let sinceSample = 0, frame = 0;
    rafLoop(() => {
      if (!running) return;
      const t0 = performance.now();
      let last = 0;
      while (performance.now() - t0 < 22) last = trainBatch();
      setStat('step', step.toLocaleString());
      setStat('chars seen', charsSeen.toLocaleString());
      if (frame % 4 === 0) {
        const l = evalLoss(12);
        setStat('loss', l.toFixed(3), l < 1 ? 'good' : '');
        say(`Step ${step.toLocaleString()}, loss ${l.toFixed(2)}.`);
        chart.push(step, [l]);
        chart.draw();
        if (l < 1.0) achieve('lm-words', `Loss ${l.toFixed(2)} — it is writing real words now`);
      }
      if (++sinceSample > 25) { sinceSample = 0; refreshSample(); drawAttention(); }
      frame++;
    }).start();

    rebuild();
  });
})();
