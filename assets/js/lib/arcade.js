/* ==========================================================================
   arcade.js — the bits that make a game feel like a game.

   Personal bests, medal tiers, combo streaks, particles, screen shake, a
   countdown and a small synthesised sound kit. Nothing here knows anything
   about machine learning; the lessons supply the game and borrow the feel.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, { arcade: api });
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const hasDOM = typeof document !== 'undefined';
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (err) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (err) { /* private mode */ } },
  };

  /* ----------------------------- personal bests -----------------------------
     Every game keeps one number. `lower` flips the comparison for times, where
     a smaller number is the better one. */

  function readBest(key) {
    const raw = store.get('mlbb-best-' + key);
    if (raw === null) return null;
    const n = parseFloat(raw);
    return Number.isFinite(n) ? n : null;
  }

  /** Record a result. Returns { best, prev, isNew } without needing a reload. */
  function submitBest(key, value, lower = false) {
    const prev = readBest(key);
    const better = prev === null || (lower ? value < prev : value > prev);
    if (better) store.set('mlbb-best-' + key, String(value));
    return { best: better ? value : prev, prev, isNew: better && prev !== null, first: prev === null };
  }

  function clearBest(key) { store.set('mlbb-best-' + key, ''); }

  /* -------------------------------- medals ---------------------------------
     Tiers are given best-first. For times pass lower = true. A result that
     beats nothing returns null rather than a participation medal. */

  const MEDALS = {
    gold:   { icon: '🥇', label: 'Gold',   color: '#ffd166' },
    silver: { icon: '🥈', label: 'Silver', color: '#c9d6e8' },
    bronze: { icon: '🥉', label: 'Bronze', color: '#e08a53' },
  };

  function medalFor(value, tiers, lower = false) {
    const ok = (t) => (lower ? value <= t : value >= t);
    if (tiers.gold !== undefined && ok(tiers.gold)) return 'gold';
    if (tiers.silver !== undefined && ok(tiers.silver)) return 'silver';
    if (tiers.bronze !== undefined && ok(tiers.bronze)) return 'bronze';
    return null;
  }

  function medalHTML(tier) {
    const m = MEDALS[tier];
    return m ? `<span class="medal-chip ${tier}">${m.icon} ${m.label}</span>` : '';
  }

  /* -------------------------------- combos ---------------------------------
     A streak that decays if you take too long, which is what turns a list of
     points into a run you don't want to break. */

  function combo(opts = {}) {
    const window_ = opts.windowMs ?? 6000;
    const cap = opts.cap ?? 5;
    let streak = 0, last = 0;
    return {
      get streak() { return streak; },
      get multiplier() { return Math.min(cap, 1 + Math.floor(streak / 2)); },
      /** Register a success; returns the multiplier the hit was worth. */
      hit(now = Date.now()) {
        if (last && now - last > window_) streak = 0;
        last = now;
        streak++;
        return Math.min(cap, 1 + Math.floor((streak - 1) / 2));
      },
      miss() { streak = 0; last = 0; },
      reset() { streak = 0; last = 0; },
      /** True when the streak has gone stale and should stop being advertised. */
      stale(now = Date.now()) { return !!last && now - last > window_; },
    };
  }

  /* --------------------------------- sound ---------------------------------
     Synthesised on the fly, so there are no audio files to ship. Sound only
     ever plays inside a game the reader deliberately started, but the toggle
     is there because someone always wants it off. */

  const SOUND_KEY = 'mlbb-sound';
  let audio = null;
  let muted = store.get(SOUND_KEY) === 'off';
  const soundListeners = [];

  function ctxFor() {
    if (muted) return null;
    const AC = root.AudioContext || root.webkitAudioContext;
    if (!AC) return null;
    if (!audio) { try { audio = new AC(); } catch (err) { return null; } }
    if (audio.state === 'suspended') audio.resume().catch(() => {});
    return audio;
  }

  /** One note. Everything in the kit is a few of these stacked up. */
  function tone(ac, { freq, dur = 0.12, type = 'square', gain = 0.05, delay = 0, slideTo = null }) {
    const t0 = ac.currentTime + delay;
    const osc = ac.createOscillator();
    const amp = ac.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t0);
    if (slideTo) osc.frequency.exponentialRampToValueAtTime(slideTo, t0 + dur);
    amp.gain.setValueAtTime(0.0001, t0);
    amp.gain.exponentialRampToValueAtTime(gain, t0 + 0.012);
    amp.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.connect(amp).connect(ac.destination);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  const KIT = {
    tick:  (ac) => tone(ac, { freq: 440, dur: 0.04, gain: 0.025, type: 'triangle' }),
    hit:   (ac) => tone(ac, { freq: 660, dur: 0.09, gain: 0.045, slideTo: 880 }),
    score: (ac) => [0, 0.07, 0.14].forEach((d, i) =>
             tone(ac, { freq: 523 * Math.pow(1.26, i), dur: 0.1, gain: 0.04, delay: d })),
    goal:  (ac) => [523, 659, 784, 1047].forEach((f, i) =>
             tone(ac, { freq: f, dur: 0.16, gain: 0.05, delay: i * 0.075, type: 'square' })),
    win:   (ac) => [523, 659, 784, 1047, 1318].forEach((f, i) =>
             tone(ac, { freq: f, dur: 0.3, gain: 0.055, delay: i * 0.09, type: 'triangle' })),
    fail:  (ac) => tone(ac, { freq: 300, dur: 0.28, gain: 0.05, type: 'sawtooth', slideTo: 90 }),
    start: (ac) => tone(ac, { freq: 880, dur: 0.18, gain: 0.05, type: 'triangle' }),
    combo: (ac, n) => tone(ac, { freq: 520 * Math.pow(1.12, Math.min(n, 8)), dur: 0.1, gain: 0.045 }),
  };

  function sfx(name, arg) {
    const ac = ctxFor();
    if (!ac || !KIT[name]) return;
    try { KIT[name](ac, arg); } catch (err) { /* audio can fail; the game continues */ }
  }

  function setMuted(v) {
    muted = !!v;
    store.set(SOUND_KEY, muted ? 'off' : 'on');
    soundListeners.forEach((fn) => fn(muted));
  }
  function isMuted() { return muted; }
  function onSound(fn) { soundListeners.push(fn); }

  /** A small speaker button. Drop it beside any game's controls. */
  function soundToggle(parent) {
    if (!hasDOM || !parent) return null;
    const b = document.createElement('button');
    b.className = 'sound-toggle';
    b.type = 'button';
    const sync = () => {
      b.textContent = muted ? '🔇' : '🔊';
      b.title = muted ? 'Sound off — click for sound' : 'Sound on — click to mute';
      b.setAttribute('aria-label', b.title);
      b.classList.toggle('off', muted);
    };
    b.addEventListener('click', () => { setMuted(!muted); if (!muted) sfx('tick'); });
    onSound(sync);
    sync();
    parent.appendChild(b);
    return b;
  }

  /* ------------------------------ canvas juice -----------------------------
     Particles and screen shake, drawn in whatever coordinate space the game
     already uses. Wrap the game's drawing in begin()/end() and call burst()
     whenever something good happens. */

  /* Canvas effects are invisible to the CSS reduced-motion rule, so the fx
     system checks the preference itself and simply stops moving. */
  function reducedMotion() {
    return hasDOM && typeof root.matchMedia === 'function'
      && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function fx() {
    const parts = [];
    let shakeMag = 0, shakeX = 0, shakeY = 0;
    let flashAlpha = 0, flashColor = '#fff';

    return {
      /** A spray of particles from a point, in game coordinates. */
      burst(x, y, o = {}) {
        if (reducedMotion()) return;
        const n = o.count ?? 18;
        const speed = o.speed ?? 120;
        const colors = o.colors || [o.color || '#ffd166'];
        for (let i = 0; i < n; i++) {
          const a = o.angle !== undefined
            ? o.angle + (Math.random() - 0.5) * (o.spread ?? Math.PI * 2)
            : Math.random() * Math.PI * 2;
          const s = speed * (0.35 + Math.random() * 0.9);
          parts.push({
            x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s,
            life: o.life ?? (0.5 + Math.random() * 0.5), age: 0,
            r: o.size ?? (1.6 + Math.random() * 2.4),
            color: colors[(Math.random() * colors.length) | 0],
            gravity: o.gravity ?? 0,
          });
        }
        if (parts.length > 400) parts.splice(0, parts.length - 400);
      },
      shake(mag) { if (!reducedMotion()) shakeMag = Math.max(shakeMag, mag); },
      flash(color = '#fff', alpha = 0.5) { flashColor = color; flashAlpha = Math.max(flashAlpha, alpha); },
      get busy() { return parts.length > 0 || shakeMag > 0.05 || flashAlpha > 0.01; },
      clear() { parts.length = 0; shakeMag = 0; flashAlpha = 0; },

      /** Save the context and offset it by the current shake. */
      begin(ctx) {
        ctx.save();
        ctx.translate(shakeX, shakeY);
      },
      /** Advance the simulation, draw the particles, restore the context. */
      end(ctx, dt, w, h) {
        for (let i = parts.length - 1; i >= 0; i--) {
          const p = parts[i];
          p.age += dt;
          if (p.age >= p.life) { parts.splice(i, 1); continue; }
          p.x += p.vx * dt; p.y += p.vy * dt;
          p.vy += p.gravity * dt;
          p.vx *= 0.98; p.vy *= 0.98;
          const k = 1 - p.age / p.life;
          ctx.globalAlpha = k;
          ctx.fillStyle = p.color;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.r * (0.4 + k * 0.6), 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
        ctx.restore();

        if (flashAlpha > 0.01 && w && h) {
          ctx.globalAlpha = flashAlpha;
          ctx.fillStyle = flashColor;
          ctx.fillRect(0, 0, w, h);
          ctx.globalAlpha = 1;
          flashAlpha *= Math.pow(0.02, dt);
          if (flashAlpha < 0.01) flashAlpha = 0;
        }

        if (shakeMag > 0.05) {
          shakeX = (Math.random() - 0.5) * shakeMag * 2;
          shakeY = (Math.random() - 0.5) * shakeMag * 2;
          shakeMag *= Math.pow(0.02, dt);
        } else { shakeMag = 0; shakeX = 0; shakeY = 0; }
      },
    };
  }

  /* ------------------------------- countdown -------------------------------
     Three seconds of anticipation, which is most of what separates "press a
     button and it starts" from "a race". */

  function countdown(host, onGo, opts = {}) {
    if (!hasDOM || !host) { onGo(); return () => {}; }
    const words = opts.words || ['3', '2', '1', 'GO'];
    const step = opts.stepMs ?? 650;
    const el = document.createElement('div');
    el.className = 'countdown';
    host.appendChild(el);
    let i = 0, timer = null, cancelled = false;

    const show = () => {
      if (cancelled) return;
      if (i >= words.length) {
        el.remove();
        onGo();
        return;
      }
      el.textContent = words[i];
      el.classList.remove('pop');
      void el.offsetWidth;                       // restart the animation
      el.classList.add('pop');
      el.classList.toggle('go', i === words.length - 1);
      sfx(i === words.length - 1 ? 'start' : 'tick');
      i++;
      timer = setTimeout(show, step);
    };
    show();
    return () => { cancelled = true; clearTimeout(timer); el.remove(); };
  }

  /* ---------------------------------- HUD ----------------------------------
     One row of big readouts, because a score you have to hunt for is not a
     score. Values are set by name and flash when they change. */

  function hud(parent, fields) {
    if (!hasDOM || !parent) return { set() {}, flash() {} };
    parent.classList.add('arcade-hud');
    parent.innerHTML = '';
    const map = {};
    fields.forEach((f) => {
      const cell = document.createElement('div');
      cell.className = 'hud-cell' + (f.wide ? ' wide' : '');
      cell.innerHTML = `<span class="hud-label">${f.label}</span>` +
                       `<span class="hud-value">${f.value ?? '—'}</span>`;
      parent.appendChild(cell);
      map[f.name] = cell.querySelector('.hud-value');
    });
    return {
      set(name, value) {
        const el = map[name];
        if (!el || el.innerHTML === String(value)) return;
        el.innerHTML = String(value);
      },
      flash(name, cls = 'good') {
        const el = map[name];
        if (!el) return;
        el.classList.remove('flash-good', 'flash-bad');
        void el.offsetWidth;
        el.classList.add(cls === 'bad' ? 'flash-bad' : 'flash-good');
      },
      el(name) { return map[name]; },
    };
  }

  /* ------------------------------ result card ------------------------------
     What you see when a run ends: the number, the medal, and whether it beat
     the number you were carrying. */

  function resultCard(value, o = {}) {
    const unit = o.unit || '';
    const tier = o.tiers ? medalFor(value, o.tiers, o.lower) : null;
    const rec = o.bestKey ? submitBest(o.bestKey, value, o.lower) : null;
    const parts = [`<b class="result-score">${o.format ? o.format(value) : value}${unit}</b>`];
    if (tier) parts.push(medalHTML(tier));
    if (rec && rec.isNew) parts.push('<span class="result-new">new best</span>');
    else if (rec && rec.best !== null && !rec.first) {
      parts.push(`<span class="muted">best ${o.format ? o.format(rec.best) : rec.best}${unit}</span>`);
    }
    if (o.note) parts.push(`<span class="result-note">${o.note}</span>`);
    return { html: parts.join(' '), tier, record: rec };
  }

  return {
    readBest, submitBest, clearBest,
    medalFor, medalHTML, MEDALS,
    combo, fx, countdown, hud, resultCard,
    sfx, setMuted, isMuted, onSound, soundToggle,
  };
});
