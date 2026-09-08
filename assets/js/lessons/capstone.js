/* ==========================================================================
   capstone.js — the final challenge.

   Two games, neither of which trains anything:

     1. Pick the approach. Nine situations, each with four plausible methods.
        The wrong answers are the ones people actually reach for.
     2. Diagnose the run. Four training curves, drawn rather than described,
        each showing a failure you have already produced somewhere on this site.
   ========================================================================== */

;(function () {
  'use strict';
  const { hidpi, chrome, nextLinks, achieve, goalPanel } = window.ML;

  /* ------------------------------------------------ part one: approaches */
  const SITUATIONS = [
    {
      q: 'You have 4,000 labelled photographs of machine parts and need to sort them into eight categories.',
      options: ['A convolutional network', 'Q-learning', 'A Vision Transformer trained from scratch', 'Neuroevolution'],
      answer: 0,
      why: 'Labelled images, a fixed answer per image, and not much data: this is exactly what convolution\'s built-in assumptions are for. A ViT would need far more examples to learn locality from scratch, and both RL methods need an environment to act in, which you do not have.',
    },
    {
      q: 'You have the same task but 400 million images and a large compute budget.',
      options: ['A convolutional network', 'A Vision Transformer', 'Tabular Q-learning', 'Logistic regression'],
      answer: 1,
      why: 'At that scale the transformer\'s lack of built-in assumptions turns from a handicap into freedom: it can learn relationships a small kernel structurally cannot express. This reversal with data scale is the actual finding of the ViT paper, and you can see the small end of it in the attention lesson.',
    },
    {
      q: 'A warehouse robot must find its way between 200 known locations. You can simulate it cheaply and there is no learned perception involved.',
      options: ['A Vision Transformer', 'Tabular Q-learning', 'A convolutional network', 'PPO with a deep network'],
      answer: 1,
      why: 'A few hundred discrete states is exactly where a table beats a network: it is exact, needs no tuning, and you can inspect every value it holds. Reaching for deep RL here adds hyperparameters and failure modes to a problem that does not have them.',
    },
    {
      q: 'You want an agent to play a continuous control game well, you can run millions of simulated steps, and your reward arrives every frame.',
      options: ['Neuroevolution', 'Policy gradients such as PPO', 'A Q-table', 'Supervised learning on human replays'],
      answer: 1,
      why: 'A dense per-step reward is information you should not waste. Policy gradients learn from every one of those steps, while evolution learns one number per lifetime. A table cannot represent continuous states at all.',
    },
    {
      q: 'Your objective is "a human rates the result 1–5", the model has about 200 parameters, and nothing about the score is differentiable.',
      options: ['Policy gradients', 'Neuroevolution', 'Backpropagation through the rating', 'A convolutional network'],
      answer: 1,
      why: 'No derivative exists to follow, so any gradient method is out. Evolution only needs to rank candidates, which a human rating does perfectly well, and 200 parameters is small enough for selection to search effectively.',
    },
    {
      q: 'Your image classifier scores 99% on its test set and fails badly in the factory it was built for.',
      options: [
        'Train for longer',
        'Add more layers',
        'Look at how the test set was collected; it probably shares the training set\'s blind spots',
        'Lower the learning rate',
      ],
      answer: 2,
      why: 'A test set drawn from the same narrow distribution as the training set certifies nothing about the factory. This is the single most common way a real project fails, and no amount of extra training or capacity touches it. You need data that looks like deployment.',
    },
    {
      q: 'Your reinforcement learning agent has learned to collect a shaping reward endlessly and never completes the actual task.',
      options: [
        'Increase the learning rate',
        'Rebalance the reward: the shaping term is worth more to it than finishing',
        'Train for longer',
        'Add more hidden units',
      ],
      answer: 1,
      why: 'The agent is optimising exactly what you wrote. Reward hacking is a specification problem rather than a training one, which is why the Rocket League lesson hands you the weights and invites you to break it on purpose.',
    },
    {
      q: 'Two agents train against each other and both look competent, but you cannot tell whether either is improving.',
      options: [
        'Watch the reward curve for longer',
        'Increase the batch size',
        'Evaluate both against a fixed opponent that does not change',
        'Lower the discount factor',
      ],
      answer: 2,
      why: 'In symmetric self-play, wins and losses cancel however good the agents get, and both can settle into a mutually terrible equilibrium. Only an opponent that stays still, a scripted bot or a frozen past self, measures absolute progress.',
    },
    {
      q: 'You need to explain to a colleague why your network predicts "9" for a particular image.',
      options: [
        'Read the weights of the final layer',
        'Look at what the layers respond to (feature maps, attention, saliency) and accept a partial answer',
        'Retrain with a smaller network so it is simpler',
        'It cannot be done at all',
      ],
      answer: 1,
      why: 'First-layer filters are legible, and attention maps show where a transformer looked, but deeper layers resist plain-language explanation and an honest answer says so. Interpretability is a live research area precisely because "read the weights" does not work.',
    },
  ];

  /* ------------------------------------------------ part two: diagnosis */
  const CURVES = [
    {
      title: 'Run A',
      draw: (ctx, W, H, plot) => plot(ctx, W, H, [
        { colour: '#4da3ff', label: 'train loss', fn: (t) => 1.4 * Math.exp(-3.2 * t) + 0.02 },
        { colour: '#ff9f45', label: 'test loss', fn: (t) => 1.4 * Math.exp(-3.6 * t) + 0.05 + 2.4 * Math.max(0, t - 0.28) ** 1.6 },
      ]),
      q: 'What is happening in run A?',
      options: ['It is learning well', 'It is overfitting', 'The learning rate is too high', 'It has too little capacity'],
      answer: 1,
      why: 'Training loss keeps falling while test loss turns and climbs: the model is fitting detail that exists only in the training set. Weight decay, more or noisier data, or a smaller model all pull the curves back together.',
    },
    {
      title: 'Run B',
      draw: (ctx, W, H, plot) => plot(ctx, W, H, [
        { colour: '#4da3ff', label: 'train loss',
          fn: (t) => 0.9 + 0.55 * Math.sin(t * 46) * Math.exp(-0.4 * t) + 0.25 * Math.sin(t * 17) },
      ]),
      q: 'What is happening in run B?',
      options: ['Overfitting', 'The batches are too large', 'The learning rate is too high, so steps overshoot the minimum', 'It has converged'],
      answer: 2,
      why: 'A sawtooth that never settles is the signature of steps so large they jump past the bottom of the valley and back up the far side. Halve the learning rate and the same run usually descends smoothly.',
    },
    {
      title: 'Run C',
      draw: (ctx, W, H, plot) => plot(ctx, W, H, [
        { colour: '#4da3ff', label: 'train loss', fn: (t) => 0.95 - 0.06 * t + 0.01 * Math.sin(t * 30) },
        { colour: '#ff9f45', label: 'test loss', fn: (t) => 0.97 - 0.05 * t + 0.01 * Math.cos(t * 26) },
      ]),
      q: 'Both curves are flat and high, and they agree with each other. What is wrong?',
      options: [
        'Overfitting',
        'The model cannot represent the answer, or is barely learning at all',
        'The test set is too small',
        'Nothing, it has converged well',
      ],
      answer: 1,
      why: 'When training and test loss agree and neither falls, the model is not memorising anything. It is failing to fit even the data sitting in front of it. That is underfitting: too little capacity, a learning rate near zero, or a problem the architecture simply cannot express, like a spiral through a model with no hidden layer.',
    },
    {
      title: 'Run D',
      draw: (ctx, W, H, plot) => plot(ctx, W, H, [
        { colour: '#38d39f', label: 'reward', fn: (t) => 0.06 + 0.02 * Math.sin(t * 20), lo: 0, hi: 1 },
        { colour: '#7c5cff', label: 'policy entropy', fn: (t) => Math.max(0.02, 1 - 3.2 * t), lo: 0, hi: 1 },
      ]),
      q: 'A reinforcement learning run: entropy collapses almost immediately and reward never moves. What happened?',
      options: [
        'The discount factor is too high',
        'The critic is broken',
        'The policy stopped exploring before it found anything worth doing',
        'The environment is too simple',
      ],
      answer: 2,
      why: 'Entropy falling to nearly zero means the policy has committed to one habit. If reward has not risen by then, it committed to a bad one and can no longer sample anything else long enough to discover better. More entropy bonus, or a curriculum that makes early reward easier to find, gets the run moving again.',
    },
  ];

  /* ------------------------------------------------------------- page */
  document.addEventListener('DOMContentLoaded', () => {
    chrome('capstone', '../');
    goalPanel(document.getElementById('lesson-goals'), 'capstone');
    nextLinks(document.getElementById('next-links'), 'rocket', null, '../');

    const state = load();
    function load() {
      try { return JSON.parse(localStorage.getItem('mlbb-capstone')) || { approach: {}, curves: {} }; }
      catch (err) { return { approach: {}, curves: {} }; }
    }
    function save() {
      try { localStorage.setItem('mlbb-capstone', JSON.stringify(state)); } catch (err) { /* private mode */ }
    }

    /** A small line chart, drawn from formulas so the shapes are exact. */
    function plot(ctx, W, H, series) {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, W, H);
      const pad = { l: 10, r: 10, t: 22, b: 16 };
      ctx.strokeStyle = '#1e2739';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 3; i++) {
        const y = pad.t + ((H - pad.t - pad.b) * i) / 3;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      }
      let lx = pad.l + 4;
      series.forEach((s) => {
        const lo = s.lo ?? 0, hi = s.hi ?? 1.6;
        ctx.strokeStyle = s.colour;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const t = i / 120;
          const v = Math.max(lo, Math.min(hi, s.fn(t)));
          const x = pad.l + t * (W - pad.l - pad.r);
          const y = pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
        ctx.fillStyle = s.colour;
        ctx.fillRect(lx, 8, 8, 3);
        ctx.fillStyle = '#9fb0cc';
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(s.label, lx + 13, 10);
        lx += ctx.measureText(s.label).width + 30;
      });
      ctx.fillStyle = '#6d7f9c';
      ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('training time →', W - pad.r, H - 2);
    }

    /** Shared renderer for both games. */
    function buildGame(host, items, bucket, badge, opts = {}) {
      const scoreEl = document.getElementById(opts.scoreId);
      const cards = items.map((item, i) => {
        const card = document.createElement('div');
        card.className = 'quiz-q';
        card.innerHTML = `
          <div class="quiz-num">${item.title || `Situation ${i + 1}`}</div>
          ${item.draw ? '<canvas class="curve-canvas"></canvas>' : ''}
          <div class="quiz-text">${item.q}</div>
          <div class="quiz-options"></div>
          <div class="quiz-why" hidden></div>`;
        const opts_ = card.querySelector('.quiz-options');
        item.options.forEach((text, oi) => {
          const b = document.createElement('button');
          b.className = 'quiz-opt';
          b.innerHTML = `<i>${'ABCD'[oi]}</i><span>${text}</span>`;
          b.addEventListener('click', () => {
            if (bucket[i] !== undefined) return;
            bucket[i] = oi;
            save();
            render();
          });
          opts_.appendChild(b);
        });
        host.appendChild(card);
        if (item.draw) {
          const c = card.querySelector('.curve-canvas');
          const ctx = hidpi(c, c.clientWidth || 460, 150);
          item.draw(ctx, ctx._cssW, ctx._cssH, plot);
          window.addEventListener('resize', () => {
            const ctx2 = hidpi(c, c.clientWidth || 460, 150);
            item.draw(ctx2, ctx2._cssW, ctx2._cssH, plot);
          });
        }
        return { card, opts: opts_, why: card.querySelector('.quiz-why'), item };
      });

      function render() {
        let correct = 0;
        cards.forEach(({ opts: o, why, item }, i) => {
          const pick = bucket[i];
          const answered = pick !== undefined;
          [...o.children].forEach((b, oi) => {
            b.classList.toggle('correct', answered && oi === item.answer);
            b.classList.toggle('wrong', answered && oi === pick && pick !== item.answer);
            b.disabled = answered;
          });
          why.hidden = !answered;
          if (answered) {
            const right = pick === item.answer;
            if (right) correct++;
            why.className = 'quiz-why ' + (right ? 'right' : 'nope');
            why.innerHTML = `<b>${right ? 'Correct.' : 'The answer is ' + 'ABCD'[item.answer] + '.'}</b> ${item.why}`;
          }
        });
        if (scoreEl) scoreEl.textContent = `${correct} / ${items.length}`;
        if (correct === items.length) achieve(badge, `${correct} out of ${items.length}, first time or not`);
      }

      const reset = document.getElementById(opts.resetId);
      if (reset) reset.addEventListener('click', () => {
        Object.keys(bucket).forEach((k) => delete bucket[k]);
        save();
        render();
      });
      render();
    }

    buildGame(document.getElementById('approach-game'), SITUATIONS, state.approach,
              'capstone-approach', { scoreId: 'approach-score', resetId: 'approach-reset' });
    buildGame(document.getElementById('curve-game'), CURVES, state.curves,
              'capstone-diagnose', { scoreId: 'curve-score', resetId: 'curve-reset' });
  });
})();
