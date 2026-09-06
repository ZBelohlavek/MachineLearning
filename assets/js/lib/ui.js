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
      nav: 'Train a CNN',
      title: 'Build and train a digit CNN',
      blurb: 'Assemble a convolutional network layer by layer, train it in your browser on generated digits, then draw your own and watch every feature map light up.',
      tags: [['cv', 'vision'], ['', 'CNN'], ['', 'training']],
      time: '25 min',
    },
    {
      id: 'attention',
      href: 'lessons/vision-transformer.html',
      nav: 'Vision Transformers',
      title: 'Patches, attention and ViTs',
      blurb: 'Cut an image into patches, compute queries and keys yourself, and see the attention map that a Vision Transformer builds on top of them.',
      tags: [['cv', 'vision'], ['', 'attention'], ['', 'ViT']],
      time: '20 min',
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
      id: 'rocket',
      href: 'lessons/rocket-league.html',
      nav: 'Rocket League Bot',
      title: 'Train a self-play Rocket League bot',
      blurb: 'The flagship project: design the rewards, train a policy-gradient agent by self-play in a top-down arena, and then take it on yourself.',
      tags: [['rl', 'RL'], ['', 'policy gradient'], ['', 'self-play']],
      time: '40 min',
    },
  ];

  /**
   * Inject the site header and footer.
   * @param {string} active  lesson id (or 'home')
   * @param {string} base    path prefix back to the site root ('' or '../')
   */
  function chrome(active = 'home', base = '') {
    const nav = LESSONS.map(
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

  /** A grid of read-only stat tiles; returns a setter keyed by name. */
  function statGrid(parent, names) {
    const grid = document.createElement('div');
    grid.className = 'stat-grid';
    const map = {};
    for (const n of names) {
      const d = document.createElement('div');
      d.className = 'stat';
      d.innerHTML = `<div class="k">${n}</div><div class="v">–</div>`;
      grid.appendChild(d);
      map[n] = d.querySelector('.v');
    }
    parent.appendChild(grid);
    return (name, value, cls = '') => {
      if (!map[name]) return;
      map[name].textContent = value;
      map[name].className = 'v ' + cls;
    };
  }

  /** requestAnimationFrame loop with start/stop and a dt in seconds. */
  function rafLoop(fn) {
    let running = false, last = 0, id = 0;
    const tick = (t) => {
      if (!running) return;
      const dt = last ? Math.min(0.05, (t - last) / 1000) : 1 / 60;
      last = t;
      fn(dt, t);
      id = requestAnimationFrame(tick);
    };
    return {
      start() { if (!running) { running = true; last = 0; id = requestAnimationFrame(tick); } },
      stop() { running = false; cancelAnimationFrame(id); },
      get running() { return running; },
      toggle() { running ? this.stop() : this.start(); return running; },
    };
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

  return { LESSONS, chrome, nextLinks, slider, pills, checkbox, statGrid, rafLoop, pointerPos };
});
