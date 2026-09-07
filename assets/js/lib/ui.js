/* ==========================================================================
   ui.js — shared page chrome and small control builders.
   ========================================================================== */

/* Loaded either as a plain <script> (exports land on window.ML) or via
   require() in Node for the test suite — so the pages work from file:// too. */
;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  /** The course, defined once and reused by the nav and the home page. */
  const LESSONS = [
    {
      id: 'foundations',
      href: 'lessons/foundations.html',
      nav: 'Foundations',
      title: 'How a network learns',
      blurb: 'Draw your own dataset, then watch a neural network carve it up. Gradient descent, layers, activations and overfitting — all live.',
      tags: [['core', 'core idea'], ['', 'backprop'], ['', 'gradient descent']],
      time: '15 min',
    },
    {
      id: 'convolutions',
      href: 'lessons/convolutions.html',
      nav: 'Convolutions',
      title: 'How a computer sees an edge',
      blurb: 'Slide a 3×3 kernel over an image by hand, watch the arithmetic, and discover why edge detectors fall out of simple multiplication.',
      tags: [['cv', 'vision'], ['', 'kernels'], ['', 'feature maps']],
      time: '15 min',
    },
    {
      id: 'cnn',
      href: 'lessons/cnn-digits.html',
      nav: 'CNNs',
      title: 'Build and train a digit CNN',
      blurb: 'Assemble a convolutional network layer by layer, train it in your browser on generated digits, then draw your own and watch every feature map light up.',
      tags: [['cv', 'vision'], ['', 'CNN'], ['', 'training']],
      time: '25 min',
    },
    {
      id: 'attention',
      href: 'lessons/vision-transformer.html',
      nav: 'Transformers',
      title: 'Patches, attention and ViTs',
      blurb: 'Cut an image into patches, compute queries and keys yourself, and see the attention map that a Vision Transformer builds on top of them.',
      tags: [['cv', 'vision'], ['', 'attention'], ['', 'ViT']],
      time: '20 min',
    },
    {
      id: 'language',
      href: 'lessons/language.html',
      nav: 'Language',
      title: 'Teach a model to write',
      blurb: 'The same attention block, pointed at text. Feed it a page of your own writing and watch noise turn into words, then into sentences, in about a minute.',
      tags: [['cv', 'attention'], ['', 'transformer'], ['', 'generation']],
      time: '25 min',
    },
    {
      id: 'gridworld',
      href: 'lessons/gridworld.html',
      nav: 'Q-Learning',
      title: 'Reinforcement learning from zero',
      blurb: 'Build a maze, set the rewards, and watch Q-learning fill in the value of every square until a policy emerges out of nothing.',
      tags: [['rl', 'RL'], ['', 'Q-learning'], ['', 'exploration']],
      time: '20 min',
    },
    {
      id: 'racer',
      href: 'lessons/evolve-a-driver.html',
      nav: 'Evolution',
      title: 'Evolve a driver from scratch',
      blurb: 'Sixty cars, sixty tiny brains, no teacher. Watch a population learn to drive a track you drew — then race the champion yourself.',
      tags: [['rl', 'game'], ['', 'neuroevolution'], ['', 'no gradients']],
      time: '20 min',
    },
    {
      id: 'rocket',
      href: 'lessons/rocket-league.html',
      nav: 'Rocket League',
      title: 'Train a self-play Rocket League bot',
      blurb: 'The flagship project: design the rewards, train a policy-gradient agent by self-play in a top-down arena, and then take it on yourself.',
      tags: [['rl', 'RL'], ['', 'policy gradient'], ['', 'self-play']],
      time: '40 min',
    },
    {
      id: 'capstone',
      capstone: true,
      href: 'lessons/final-challenge.html',
      nav: 'Final Challenge',
      title: 'The final challenge',
      blurb: 'No sliders, no training — just judgement. Pick the right approach for nine real situations, then diagnose four training runs from their curves alone.',
      tags: [['core', 'capstone'], ['', 'judgement'], ['', 'diagnosis']],
      time: '15 min',
    },
  ];

  /**
   * Inject the site header and footer.
   * @param {string} active  lesson id (or 'home')
   * @param {string} base    path prefix back to the site root ('' or '../')
   */
  function chrome(active = 'home', base = '') {
    const nav = LESSONS.filter((l) => !l.capstone).map(
      (l) => `<a href="${base}${l.href}" class="${l.id === active ? 'active' : ''}">${l.nav}</a>`
    ).join('');

    const header = document.createElement('header');
    header.className = 'site';
    header.innerHTML = `
      <div class="wrap">
        <a class="brand" href="${base}index.html"><span class="dot"></span> Learn ML by Building</a>
        <nav class="site-nav">${nav}</nav>
      </div>`;
    document.body.prepend(header);

    /* A reading-progress bar. These pages are long and heavily sectioned, and a
       thin line is enough to say how much is left without taking any space. */
    const bar = document.createElement('div');
    bar.className = 'read-bar';
    bar.innerHTML = '<i></i>';
    document.body.appendChild(bar);
    const fill = bar.firstChild;
    const onScroll = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      fill.style.width = (h > 0 ? Math.min(1, window.scrollY / h) * 100 : 0) + '%';
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    onScroll();

    /* Sections fade in as they arrive. Applied from JS so that with scripting
       off, or without IntersectionObserver, nothing is ever hidden. */
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduce && 'IntersectionObserver' in window) {
      requestAnimationFrame(() => {
        const io = new IntersectionObserver((entries) => {
          for (const e of entries) {
            if (!e.isIntersecting) continue;
            e.target.classList.add('in');
            io.unobserve(e.target);
          }
        }, { rootMargin: '0px 0px -60px 0px' });
        document.querySelectorAll('.section').forEach((el, i) => {
          if (i === 0) return;                 // the first screen should never wait
          el.classList.add('reveal');
          io.observe(el);
        });
      });
    }

    const footer = document.createElement('footer');
    footer.className = 'site';
    footer.innerHTML = `
      <div class="wrap">
        Learn ML by Building — every model on this site is trained live in your browser
        in plain JavaScript. No servers, no pre-trained weights, no dependencies.
        <a href="${base}index.html">Back to all lessons</a>
      </div>`;
    document.body.append(footer);
  }

  /** Two "what to read next" links at the bottom of a lesson. */
  function nextLinks(container, prevId, nextId, base = '../') {
    const find = (id) => LESSONS.find((l) => l.id === id);
    const out = [];
    const p = find(prevId), n = find(nextId);
    if (p) out.push(`<a href="${base}${p.href}"><span>Previous</span>${p.title}</a>`);
    if (n) out.push(`<a href="${base}${n.href}"><span>Next lesson</span>${n.title}</a>`);
    container.innerHTML = out.join('');
    container.className = 'next-links';
  }

  /* ==========================================================================
     Badges — small goals that only unlock when a model actually works.
     Stored in localStorage, shown on the home page and toasted when earned.
     ========================================================================== */

  const BADGES = [
    { id: 'foundations-spiral', lesson: 'foundations', label: 'Untangled the spiral',
      hint: 'Get 90% test accuracy on the spiral dataset' },
    { id: 'foundations-overfit', lesson: 'foundations', label: 'Caught it overfitting',
      hint: 'Make test loss climb while training loss keeps falling' },
    { id: 'foundations-lean', lesson: 'foundations', label: 'Solved the spiral with 8 neurons',
      hint: 'Reach 90% on the spiral using 8 hidden neurons or fewer' },
    { id: 'conv-custom', lesson: 'convolutions', label: 'Wrote your own kernel',
      hint: 'Type your own numbers into the 3×3 kernel' },
    { id: 'conv-scan', lesson: 'convolutions', label: 'Followed the window',
      hint: 'Scan a whole row of the image one pixel at a time' },
    { id: 'conv-challenge', lesson: 'convolutions', label: 'Built a horizontal edge detector',
      hint: 'Type a kernel that finds horizontal edges — no presets' },
    { id: 'cnn-trained', lesson: 'cnn', label: 'Trained a CNN to 95%',
      hint: 'Reach 95% test accuracy from random weights' },
    { id: 'cnn-duel', lesson: 'cnn', label: 'Won a digit duel',
      hint: 'Score 8 or more in the 60-second drawing challenge' },
    { id: 'vit-trained', lesson: 'attention', label: 'Trained a transformer',
      hint: 'Get the Vision Transformer past 80% test accuracy' },
    { id: 'vit-ablation', lesson: 'attention', label: 'Ran an ablation',
      hint: 'Try both the class token and mean pooling on the same task' },
    { id: 'grid-solved', lesson: 'gridworld', label: 'Q-learning solved the maze',
      hint: 'Let the agent find a route of 20 steps or fewer' },
    { id: 'grid-beaten', lesson: 'gridworld', label: 'Beat the agent',
      hint: 'Reach the goal yourself in fewer steps than the agent’s best' },
    { id: 'racer-finish', lesson: 'racer', label: 'Evolved a finisher',
      hint: 'Evolve a car that completes the course' },
    { id: 'racer-beaten', lesson: 'racer', label: 'Outdrove the champion',
      hint: 'Beat the evolved driver in a head-to-head race' },
    { id: 'rocket-touch', lesson: 'rocket', label: 'It found the ball',
      hint: 'Train until the agent averages 3 touches an episode' },
    { id: 'rocket-scores', lesson: 'rocket', label: 'It learned to score',
      hint: 'Train an agent that beats the scripted bot' },
    { id: 'rocket-human', lesson: 'rocket', label: 'Scored on your own bot',
      hint: 'Beat your trained agent in a head-to-head match' },
    { id: 'quiz-foundations', lesson: 'foundations', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'quiz-convolutions', lesson: 'convolutions', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'quiz-cnn', lesson: 'cnn', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'quiz-attention', lesson: 'attention', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'lm-words', lesson: 'language', label: 'Taught it to write words',
      hint: 'Train until the loss drops below 1.0' },
    { id: 'lm-own', lesson: 'language', label: 'Trained it on your own text',
      hint: 'Paste your own writing into the corpus box' },
    { id: 'quiz-language', lesson: 'language', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'quiz-gridworld', lesson: 'gridworld', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'quiz-racer', lesson: 'racer', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'quiz-rocket', lesson: 'rocket', label: 'Passed the quiz',
      hint: 'Answer all three questions correctly' },
    { id: 'capstone-approach', lesson: 'capstone', label: 'Picked every approach correctly',
      hint: 'Choose the right method for all nine situations' },
    { id: 'capstone-diagnose', lesson: 'capstone', label: 'Diagnosed every training run',
      hint: 'Read all four training curves correctly' },
  ];

  const BADGE_KEY = 'mlbb-badges';
  const badgeListeners = [];

  function loadBadges() {
    try { return JSON.parse(localStorage.getItem(BADGE_KEY)) || {}; }
    catch (err) { return {}; }
  }

  /** Award a badge. Safe to call repeatedly — only the first call counts. */
  function achieve(id, detail) {
    const badge = BADGES.find((b) => b.id === id);
    if (!badge) return false;
    const store = loadBadges();
    if (store[id]) return false;
    store[id] = { at: Date.now(), detail: detail || '' };
    try { localStorage.setItem(BADGE_KEY, JSON.stringify(store)); } catch (err) { /* private mode */ }
    toast(badge, detail);

    // Finishing every goal in a lesson deserves more than another small toast.
    const siblings = BADGES.filter((b) => b.lesson === badge.lesson);
    if (siblings.length > 1 && siblings.every((b) => store[b.id])) {
      const lesson = LESSONS.find((l) => l.id === badge.lesson);
      setTimeout(() => toast(
        { label: `${lesson ? lesson.nav : 'Lesson'} complete`, hint: '' },
        `Every goal in this lesson. ${remainingBadges()} left across the site.`, true), 700);
    }

    badgeListeners.forEach((fn) => fn(id));
    return true;
  }

  function remainingBadges() {
    const store = loadBadges();
    return BADGES.filter((b) => !store[b.id]).length;
  }

  function onBadge(fn) { badgeListeners.push(fn); }

  function badgesEarned() { return loadBadges(); }

  function toast(badge, detail, gold) {
    let host = document.querySelector('.toast-host');
    if (!host) {
      host = document.createElement('div');
      host.className = 'toast-host';
      document.body.appendChild(host);
    }
    while (host.children.length >= 3) host.firstChild.remove();
    const el = document.createElement('div');
    el.className = 'toast' + (gold ? ' gold' : '');
    el.innerHTML = `<span class="medal">${gold ? '🏆' : '★'}</span><div><b>${badge.label}</b>` +
      `<span>${detail || badge.hint}</span></div>`;
    host.appendChild(el);
    requestAnimationFrame(() => el.classList.add('in'));
    setTimeout(() => {
      el.classList.remove('in');
      setTimeout(() => el.remove(), 400);
    }, 4200);
  }

  /**
   * The goals for one lesson, rendered inside that lesson and kept up to date
   * as they are earned. Seeing what is worth aiming for turns a page you read
   * into a page you do something with.
   */
  function goalPanel(container, lessonId) {
    if (!container) return;
    const list = BADGES.filter((b) => b.lesson === lessonId);
    if (!list.length) return;
    const render = () => {
      const earned = loadBadges();
      const done = list.filter((b) => earned[b.id]).length;
      container.className = 'goals' + (done === list.length ? ' complete' : '');
      container.innerHTML =
        `<div class="goals-head">
           <b>${done === list.length ? 'Every goal in this lesson is done' : 'Goals in this lesson'}</b>
           <span class="mono">${done} / ${list.length}</span>
         </div>
         <div class="badge-row">` +
        list.map((b) => `<span class="badge ${earned[b.id] ? 'earned' : ''}" title="${b.label}">
            <i>${earned[b.id] ? '★' : '☆'}</i>${earned[b.id] ? b.label : b.hint}</span>`).join('') +
        '</div>';
    };
    render();
    onBadge(render);
  }

  /* ==========================================================================
     Quiz checkpoints — a few questions per lesson about what you just did,
     with the reasoning shown after you answer. Answers persist, so returning
     to a lesson shows what you already worked out.
     ========================================================================== */

  const QUIZ_KEY = 'mlbb-quiz';

  function quizState() {
    try { return JSON.parse(localStorage.getItem(QUIZ_KEY)) || {}; }
    catch (err) { return {}; }
  }
  function saveQuizState(state) {
    try { localStorage.setItem(QUIZ_KEY, JSON.stringify(state)); } catch (err) { /* private mode */ }
  }

  /**
   * quiz(container, { id, title, intro, badge, questions:[{ q, options, answer, why }] })
   * `answer` is the index of the correct option; `why` is shown either way.
   */
  function quiz(container, spec) {
    if (!container) return;
    const store = quizState();
    const saved = store[spec.id] || {};
    const chosen = {};
    Object.keys(saved).forEach((k) => (chosen[k] = saved[k]));

    const head = document.createElement('div');
    head.className = 'quiz-head';
    head.innerHTML = `
      <div>
        <h3>${spec.title || 'Check yourself'}</h3>
        <p class="muted">${spec.intro || 'Three questions about what you just watched happen. ' +
          'Getting one wrong is more useful than getting it right — the explanation is the point.'}</p>
      </div>
      <div class="quiz-score"><span id="${spec.id}-score">0</span> / ${spec.questions.length}</div>`;
    container.appendChild(head);
    const scoreEl = head.querySelector('.quiz-score span');

    const cards = spec.questions.map((question, qi) => {
      const card = document.createElement('div');
      card.className = 'quiz-q';
      card.innerHTML = `
        <div class="quiz-num">Question ${qi + 1}</div>
        <div class="quiz-text">${question.q}</div>
        <div class="quiz-options"></div>
        <div class="quiz-why" hidden></div>`;
      const opts = card.querySelector('.quiz-options');
      const why = card.querySelector('.quiz-why');

      question.options.forEach((text, oi) => {
        const b = document.createElement('button');
        b.className = 'quiz-opt';
        b.innerHTML = `<i>${'ABCD'[oi]}</i><span>${text}</span>`;
        b.addEventListener('click', () => answer(qi, oi));
        opts.appendChild(b);
      });
      container.appendChild(card);
      return { card, opts, why, question };
    });

    function render() {
      let correct = 0;
      cards.forEach(({ opts, why, question }, qi) => {
        const pick = chosen[qi];
        const answered = pick !== undefined;
        [...opts.children].forEach((b, oi) => {
          b.classList.toggle('correct', answered && oi === question.answer);
          b.classList.toggle('wrong', answered && oi === pick && pick !== question.answer);
          b.disabled = answered;
        });
        why.hidden = !answered;
        if (answered) {
          const right = pick === question.answer;
          if (right) correct++;
          why.className = 'quiz-why ' + (right ? 'right' : 'nope');
          why.innerHTML = `<b>${right ? 'Correct.' : 'Not quite — ' +
            'the answer is ' + 'ABCD'[question.answer] + '.'}</b> ${question.why}`;
        }
      });
      scoreEl.textContent = String(correct);
      const done = Object.keys(chosen).length === spec.questions.length;
      footer.hidden = !done;
      if (done) {
        footerText.innerHTML = correct === spec.questions.length
          ? '<b>All three.</b> You can explain this lesson to someone else now, which is the real test.'
          : `<b>${correct} of ${spec.questions.length}.</b> Re-read the explanations above, then reset and try again — ` +
            'the ones you got wrong are the ones worth going back to the sandbox for.';
        if (correct === spec.questions.length && spec.badge) {
          achieve(spec.badge, `You answered every question in ${spec.title || 'the quiz'}`);
        }
      }
    }

    function answer(qi, oi) {
      if (chosen[qi] !== undefined) return;
      chosen[qi] = oi;
      const state = quizState();
      state[spec.id] = chosen;
      saveQuizState(state);
      render();
    }

    const footer = document.createElement('div');
    footer.className = 'quiz-footer';
    footer.hidden = true;
    footer.innerHTML = '<span></span><button class="small">Reset these questions</button>';
    const footerText = footer.querySelector('span');
    footer.querySelector('button').addEventListener('click', () => {
      Object.keys(chosen).forEach((k) => delete chosen[k]);
      const state = quizState();
      delete state[spec.id];
      saveQuizState(state);
      render();
    });
    container.appendChild(footer);

    render();
  }

  /**
   * Run controls: a seed and a pin.
   *
   * Every lesson invites you to change one setting and see what happens, which
   * is only an experiment if everything else is held constant and you can see
   * the before and after together. The seed fixes the random draw; the pin
   * freezes the current curves as a ghost for the next run to be measured
   * against.
   *
   * runBar(parent, { seed, onSeed, charts, note })
   */
  function runBar(parent, opts = {}) {
    if (!parent) return null;
    let seed = opts.seed ?? 1234;
    const wrap = document.createElement('div');
    wrap.className = 'run-bar';
    wrap.innerHTML = `
      <label class="run-seed">
        <span>seed</span>
        <input type="number" min="0" max="999999" step="1" value="${seed}">
      </label>
      <button class="small" data-role="dice" title="Random seed">🎲</button>
      <button class="small" data-role="pin">Pin this run</button>
      <span class="run-note muted"></span>`;
    parent.appendChild(wrap);

    const input = wrap.querySelector('input');
    const pinBtn = wrap.querySelector('[data-role=pin]');
    const note = wrap.querySelector('.run-note');
    const charts = () => (typeof opts.charts === 'function' ? opts.charts() : (opts.charts || []));

    const apply = (v) => {
      seed = Math.max(0, Math.round(v) || 0);
      input.value = String(seed);
      if (opts.onSeed) opts.onSeed(seed);
    };
    input.addEventListener('change', () => apply(parseInt(input.value, 10)));
    wrap.querySelector('[data-role=dice]').addEventListener('click',
      () => apply(Math.floor(Math.random() * 100000)));

    const refresh = () => {
      const anyPinned = charts().some((c) => c && c.pinned);
      pinBtn.textContent = anyPinned ? 'Clear pinned run' : 'Pin this run';
      pinBtn.classList.toggle('toggle', anyPinned);
      pinBtn.classList.toggle('on', anyPinned);
      note.textContent = anyPinned
        ? 'The dashed curves are the pinned run — change one setting and compare.'
        : (opts.note || 'Same seed, same data and same starting weights.');
    };
    pinBtn.addEventListener('click', () => {
      const cs = charts().filter(Boolean);
      const anyPinned = cs.some((c) => c.pinned);
      cs.forEach((c) => (anyPinned ? c.unpin() : c.pin()));
      refresh();
    });
    refresh();
    return { get seed() { return seed; }, set: apply, refresh };
  }

  /**
   * Predict, then check.
   *
   * Commit to an answer before the page reveals it. Being wrong on a
   * prediction you actually made is far stickier than reading the right answer
   * first — and unlike the quizzes, these are scored against what your own run
   * really does, not against a stored answer.
   *
   * predictBox(parent, { id, question, options:[{label,value}], hint })
   *   .reveal(actualValue, message)   scores the prediction once the run knows
   */
  function predictBox(parent, opts) {
    if (!parent) return null;
    const KEY = 'mlbb-predict';
    const store = () => {
      try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (err) { return {}; }
    };
    const saved = store()[opts.id];
    let choice = saved === undefined ? null : saved;
    let revealed = null;

    const box = document.createElement('div');
    box.className = 'predict';
    box.innerHTML = `
      <div class="predict-head"><b>Predict first</b><span>${opts.hint || 'Then run it and find out.'}</span></div>
      <div class="predict-q">${opts.question}</div>
      <div class="predict-opts"></div>
      <div class="predict-result" hidden></div>`;
    const optHost = box.querySelector('.predict-opts');
    const result = box.querySelector('.predict-result');
    parent.appendChild(box);

    const buttons = opts.options.map((o) => {
      const b = document.createElement('button');
      b.className = 'pill';
      b.textContent = o.label;
      b.addEventListener('click', () => {
        if (choice !== null) return;
        choice = o.value;
        const s = store();
        s[opts.id] = choice;
        try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (err) { /* private mode */ }
        render();
      });
      optHost.appendChild(b);
      return b;
    });

    function render() {
      buttons.forEach((b, i) => {
        const v = opts.options[i].value;
        b.classList.toggle('active', choice === v);
        b.classList.toggle('correct', revealed !== null && v === revealed);
        b.classList.toggle('wrong', revealed !== null && v === choice && choice !== revealed);
        b.disabled = choice !== null;
      });
      result.hidden = choice === null;
      if (choice !== null && revealed === null) {
        result.className = 'predict-result waiting';
        result.innerHTML = '<b>Locked in.</b> Now train it and see whether you were right.';
      } else if (revealed !== null) {
        const right = choice === revealed;
        result.className = 'predict-result ' + (right ? 'right' : 'nope');
        result.innerHTML = `<b>${right ? 'You called it.' : 'Not what happened.'}</b> ${revealMsg}`;
      }
    }

    let revealMsg = '';
    render();
    return {
      get choice() { return choice; },
      reveal(actualValue, message) {
        if (choice === null || revealed !== null) return;   // no prediction, or already scored
        revealed = actualValue;
        revealMsg = message || '';
        render();
      },
    };
  }

  /**
   * A polite live region.
   *
   * Everything interesting on this site happens inside a canvas, which is
   * invisible to a screen reader however well it is labelled. This announces
   * the numbers that matter — accuracy, goals, loss — at a pace that informs
   * rather than interrupts.
   */
  function announcer(throttleMs = 9000) {
    let node = document.querySelector('.sr-live');
    if (!node) {
      node = document.createElement('div');
      node.className = 'sr-live';
      node.setAttribute('role', 'status');
      node.setAttribute('aria-live', 'polite');
      document.body.appendChild(node);
    }
    let last = 0, lastText = '';
    return (text) => {
      const now = Date.now();
      if (text === lastText || now - last < throttleMs) return;
      last = now;
      lastText = text;
      node.textContent = text;
    };
  }

  /* ------------------------- controls ------------------------- */

  /**
   * A labelled slider.
   * slider(parent, { label, min, max, step, value, format, desc, onInput })
   * Returns { el, input, set(v), get() }.
   */
  function slider(parent, o) {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    const format = o.format || ((v) => String(v));
    wrap.innerHTML = `
      <label class="row"><span>${o.label}</span><span class="val"></span></label>
      <input type="range" min="${o.min}" max="${o.max}" step="${o.step ?? 0.01}" value="${o.value}">
      ${o.desc ? `<div class="desc">${o.desc}</div>` : ''}`;
    const input = wrap.querySelector('input');
    const val = wrap.querySelector('.val');
    const sync = (fire) => {
      const v = parseFloat(input.value);
      val.textContent = format(v);
      if (fire && o.onInput) o.onInput(v);
    };
    input.addEventListener('input', () => sync(true));
    sync(false);
    parent.appendChild(wrap);
    return {
      el: wrap, input,
      get: () => parseFloat(input.value),
      set: (v) => { input.value = v; sync(true); },
    };
  }

  /** A row of mutually exclusive pills. Returns { get, set }. */
  function pills(parent, options, initial, onChange) {
    const row = document.createElement('div');
    row.className = 'pill-row';
    let current = initial;
    const els = options.map((opt) => {
      const b = document.createElement('button');
      b.className = 'pill' + (opt.value === initial ? ' active' : '');
      b.textContent = opt.label;
      b.title = opt.title || '';
      b.addEventListener('click', () => {
        current = opt.value;
        els.forEach((e) => e.classList.toggle('active', e === b));
        onChange && onChange(current);
      });
      row.appendChild(b);
      return b;
    });
    parent.appendChild(row);
    return {
      get: () => current,
      set: (v) => {
        current = v;
        els.forEach((e, i) => e.classList.toggle('active', options[i].value === v));
      },
    };
  }

  /** A checkbox with a label. */
  function checkbox(parent, label, checked, onChange) {
    const l = document.createElement('label');
    l.className = 'checkline';
    l.innerHTML = `<input type="checkbox" ${checked ? 'checked' : ''}><span>${label}</span>`;
    const input = l.querySelector('input');
    input.addEventListener('change', () => onChange && onChange(input.checked));
    parent.appendChild(l);
    return { get: () => input.checked, set: (v) => { input.checked = v; onChange && onChange(v); } };
  }

  /* plot.js owns the high-DPI helper; fall back if it has not loaded. */
  function fitCanvas(canvas, w, h) {
    if (root.ML && root.ML.hidpi) return root.ML.hidpi(canvas, w, h);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  /**
   * A grid of read-only stat tiles; returns a setter keyed by name.
   *
   * Numeric stats also grow a sparkline of their own recent history, drawn
   * behind the number — a stat that is climbing looks different from one that
   * has stalled, without anyone having to watch it continuously.
   */
  function statGrid(parent, names) {
    const grid = document.createElement('div');
    grid.className = 'stat-grid';
    const map = {};
    for (const n of names) {
      const d = document.createElement('div');
      d.className = 'stat';
      d.innerHTML = `<canvas class="spark"></canvas><div class="k">${n}</div><div class="v">–</div>`;
      grid.appendChild(d);
      map[n] = { value: d.querySelector('.v'), canvas: d.querySelector('.spark'), history: [], ctx: null };
    }
    parent.appendChild(grid);

    const numeric = (v) => {
      if (typeof v === 'number') return isFinite(v) ? v : null;
      const m = String(v).replace(/,/g, '').match(/^-?\d+(\.\d+)?/);
      return m ? parseFloat(m[0]) : null;
    };

    function drawSpark(entry) {
      const { canvas, history } = entry;
      if (history.length < 3) return;
      const w = canvas.clientWidth || 0, h = canvas.clientHeight || 0;
      if (!w || !h) return;
      if (!entry.ctx || entry.w !== w) { entry.ctx = fitCanvas(canvas, w, h); entry.w = w; }
      const ctx = entry.ctx;
      ctx.clearRect(0, 0, w, h);
      let lo = Infinity, hi = -Infinity;
      for (const v of history) { if (v < lo) lo = v; if (v > hi) hi = v; }
      if (hi - lo < 1e-9) { hi = lo + 1; lo -= 0.5; }
      ctx.beginPath();
      history.forEach((v, i) => {
        const x = (i / (history.length - 1)) * w;
        const y = h - 2 - ((v - lo) / (hi - lo)) * (h - 6);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.strokeStyle = 'rgba(124, 148, 190, .38)';
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
      ctx.fillStyle = 'rgba(77, 163, 255, .07)';
      ctx.fill();
    }

    return (name, value, cls = '') => {
      const entry = map[name];
      if (!entry) return;
      entry.value.textContent = value;
      entry.value.className = 'v ' + cls;
      const n = numeric(value);
      if (n !== null) {
        entry.history.push(n);
        if (entry.history.length > 48) entry.history.shift();
        drawSpark(entry);
      }
    };
  }

  /** requestAnimationFrame loop with start/stop and a dt in seconds. */
  function rafLoop(fn) {
    let running = false, last = 0, id = 0;
    const tick = (t) => {
      if (!running) return;
      const dt = last ? Math.min(0.05, (t - last) / 1000) : 1 / 60;
      last = t;
      try {
        fn(dt, t);
      } catch (err) {
        // Keep the loop alive: one bad frame should not freeze the whole page.
        console.error('animation frame failed:', err);
      }
      id = requestAnimationFrame(tick);
    };
    return {
      start() { if (!running) { running = true; last = 0; id = requestAnimationFrame(tick); } },
      stop() { running = false; cancelAnimationFrame(id); },
      get running() { return running; },
      toggle() { running ? this.stop() : this.start(); return running; },
    };
  }

  /**
   * On-screen direction pad, for the games. Without it none of the driving is
   * playable on a phone, since every one of them expects arrow keys.
   *
   * dpad(container, { onPress(dir, down), keys, discrete, always })
   *   keys      an object to mirror the arrow state into ({up,down,left,right})
   *   discrete  fire onPress once per tap rather than tracking hold state
   *   always    show on desktop too (default: only on touch devices)
   */
  function dpad(container, opts = {}) {
    if (!container) return null;
    const touch = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
    if (!touch && !opts.always) return null;

    const wrap = document.createElement('div');
    wrap.className = 'dpad';
    const mk = (dir, label, cls) => {
      const b = document.createElement('button');
      b.className = 'dpad-btn ' + cls;
      b.type = 'button';
      b.textContent = label;
      b.setAttribute('aria-label', dir);
      const set = (down) => (ev) => {
        ev.preventDefault();
        if (opts.keys) opts.keys[dir] = down;
        if (opts.onPress && (down || !opts.discrete)) opts.onPress(dir, down);
        b.classList.toggle('is-pressed', down);
      };
      if (opts.discrete) {
        b.addEventListener('pointerdown', set(true));
        b.addEventListener('pointerup', (e) => { e.preventDefault(); b.classList.remove('is-pressed'); });
      } else {
        b.addEventListener('pointerdown', set(true));
        b.addEventListener('pointerup', set(false));
        b.addEventListener('pointerleave', set(false));
        b.addEventListener('pointercancel', set(false));
      }
      return b;
    };
    wrap.appendChild(mk('up', '▲', 'up'));
    wrap.appendChild(mk('left', '◀', 'left'));
    wrap.appendChild(mk('down', '▼', 'down'));
    wrap.appendChild(mk('right', '▶', 'right'));
    container.appendChild(wrap);
    return wrap;
  }

  /** Canvas pointer position in canvas CSS pixels. */
  function pointerPos(canvas, ev) {
    const r = canvas.getBoundingClientRect();
    const p = ev.touches ? ev.touches[0] : ev;
    return {
      x: ((p.clientX - r.left) / r.width) * (canvas._logicalW ?? canvas.clientWidth),
      y: ((p.clientY - r.top) / r.height) * (canvas._logicalH ?? canvas.clientHeight),
    };
  }

  return { LESSONS, BADGES, chrome, nextLinks, achieve, badgesEarned, onBadge,
           quiz, goalPanel, dpad, runBar, predictBox, announcer, slider, pills, checkbox, statGrid,
           rafLoop, pointerPos };
});
