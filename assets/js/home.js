/* ==========================================================================
   home.js — builds the lesson grid and paints a small preview for each card.
   ========================================================================== */

;(function () {
  'use strict';
  const { LESSONS, BADGES, chrome, hidpi, rafLoop, mulberry32, badgesEarned } = window.ML;

  const THUMB = {
    /* a decision boundary with two clouds of points */
    foundations(ctx, W, H) {
      const r = mulberry32(4);
      const img = ctx.createImageData(W, H);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const nx = (x / W) * 2 - 1, ny = (y / H) * 2 - 1;
          const v = Math.tanh(2.6 * (ny - 0.55 * Math.sin(nx * 2.4) - 0.1));
          const i = (y * W + x) * 4;
          img.data[i] = 90 + 150 * Math.max(0, v);
          img.data[i + 1] = 120 + 40 * Math.abs(v);
          img.data[i + 2] = 90 + 150 * Math.max(0, -v);
          img.data[i + 3] = 120 + 80 * Math.abs(v);
        }
      }
      ctx.putImageData(img, 0, 0);
      for (let k = 0; k < 46; k++) {
        const x = r() * W, y = r() * H;
        const nx = (x / W) * 2 - 1, ny = (y / H) * 2 - 1;
        const cls = ny - 0.55 * Math.sin(nx * 2.4) - 0.1 > 0;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = cls ? '#ff9f45' : '#4da3ff';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    },

    /* an image with a 3x3 kernel window on it */
    convolutions(ctx, W, H) {
      const N = 16, cell = Math.min(W, H) / N;
      const ox = (W - cell * N) / 2, oy = (H - cell * N) / 2;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const d = Math.hypot(x - 7, y - 7);
          const v = d < 4.5 ? 0.88 : 0.10 + 0.05 * Math.sin(x * 0.9 + y * 0.5);
          const g = Math.round(v * 255);
          ctx.fillStyle = `rgb(${g},${g},${g})`;
          ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
        }
      }
      ctx.strokeStyle = '#4da3ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(ox + 4 * cell, oy + 5 * cell, cell * 3, cell * 3);
      ctx.strokeStyle = 'rgba(255,209,102,.95)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(ox + 5 * cell, oy + 6 * cell, cell, cell);
    },

    /* a digit plus a strip of feature maps */
    cnn(ctx, W, H) {
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${H * 0.72}px Georgia, serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('5', W * 0.22, H * 0.5);
      const cols = ['#7c5cff', '#4da3ff', '#38d39f', '#ffb547', '#ff6b6b', '#4da3ff'];
      const s = H / 3.4;
      for (let i = 0; i < 6; i++) {
        const x = W * 0.44 + (i % 3) * (s + 6);
        const y = H * 0.22 + ((i / 3) | 0) * (s + 8);
        ctx.globalAlpha = 0.28 + 0.12 * i;
        ctx.fillStyle = cols[i];
        ctx.fillRect(x, y, s, s);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = 'rgba(255,255,255,.16)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, s, s);
      }
    },

    /* patch grid with an attention overlay */
    attention(ctx, W, H) {
      const N = 5, cell = Math.min(W, H) / N;
      const ox = (W - cell * N) / 2, oy = (H - cell * N) / 2;
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, W, H);
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const d = Math.hypot(x - 2, y - 2);
          const a = Math.exp(-d * d / 2.6);
          ctx.fillStyle = `rgba(124,92,255,${(0.12 + a * 0.75).toFixed(3)})`;
          ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
          ctx.strokeStyle = 'rgba(255,255,255,.16)';
          ctx.lineWidth = 1;
          ctx.strokeRect(ox + x * cell, oy + y * cell, cell, cell);
        }
      }
      ctx.strokeStyle = '#ffd166';
      ctx.lineWidth = 2;
      ctx.strokeRect(ox + 2 * cell + 1, oy + 2 * cell + 1, cell - 2, cell - 2);
    },

    /* text turning into a probability distribution over the next character */
    language(ctx, W, H) {
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, W, H);
      ctx.font = '11px ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      const line = 'the harbour wakes early. the boats go';
      ctx.fillStyle = '#6d7f9c';
      ctx.fillText(line.slice(0, 28), 12, H * 0.28);
      ctx.fillStyle = '#e6ecf7';
      ctx.fillText(line.slice(28, 36), 12 + ctx.measureText(line.slice(0, 28)).width, H * 0.28);
      // a small distribution over the next character
      const bars = [['o', .52], ['e', .21], ['a', .12], ['i', .08], ['u', .04]];
      bars.forEach(([ch, p], i) => {
        const y = H * 0.45 + i * 11;
        ctx.fillStyle = '#9fb0cc';
        ctx.fillText(ch, 14, y);
        ctx.fillStyle = 'rgba(124,92,255,.30)';
        ctx.fillRect(26, y - 4, (W - 46), 8);
        ctx.fillStyle = '#7c5cff';
        ctx.fillRect(26, y - 4, (W - 46) * p, 8);
      });
    },

    /* q-values and a policy arrow per cell */
    gridworld(ctx, W, H) {
      const NX = 7, NY = 4, cell = Math.min(W / NX, H / NY);
      const ox = (W - cell * NX) / 2, oy = (H - cell * NY) / 2;
      for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
          const px = ox + x * cell, py = oy + y * cell;
          const goalish = 1 - (Math.abs(x - (NX - 1)) + y) / (NX + NY);
          ctx.fillStyle = `rgba(56,211,159,${(0.08 + goalish * 0.6).toFixed(3)})`;
          ctx.fillRect(px, py, cell, cell);
          ctx.strokeStyle = 'rgba(255,255,255,.09)';
          ctx.lineWidth = 1;
          ctx.strokeRect(px, py, cell, cell);
          ctx.fillStyle = 'rgba(230,236,247,.75)';
          ctx.font = `${cell * 0.42}px system-ui`;
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(y === 0 ? '→' : '↗'.replace('↗', y === NY - 1 ? '↑' : '→'), px + cell / 2, py + cell / 2);
        }
      }
      ctx.fillStyle = 'rgba(255,107,107,.55)';
      ctx.fillRect(ox + 2 * cell, oy + cell, cell, cell);
      ctx.fillStyle = '#4da3ff';
      ctx.beginPath();
      ctx.arc(ox + cell * 0.5, oy + cell * (NY - 0.5), cell * 0.24, 0, Math.PI * 2);
      ctx.fill();
    },

    /* a winding track with a pack of cars on it */
    racer(ctx, W, H) {
      ctx.fillStyle = '#0c1220';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = '#2c3a55';
      ctx.lineWidth = H * 0.20;
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(W * 0.1, H * 0.78);
      ctx.bezierCurveTo(W * 0.32, H * 0.78, W * 0.3, H * 0.24, W * 0.52, H * 0.26);
      ctx.bezierCurveTo(W * 0.74, H * 0.28, W * 0.68, H * 0.8, W * 0.92, H * 0.7);
      ctx.stroke();
      const dots = [[0.18, 0.76, '#7c5cff'], [0.27, 0.66, '#7c5cff'], [0.36, 0.42, '#7c5cff'],
                    [0.47, 0.28, '#38d39f'], [0.6, 0.34, '#7c5cff'], [0.7, 0.6, '#7c5cff']];
      for (const [x, y, col] of dots) {
        ctx.fillStyle = col;
        ctx.globalAlpha = col === '#38d39f' ? 1 : 0.55;
        ctx.fillRect(x * W - 5, y * H - 3, 10, 6);
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = 'rgba(255,209,102,.55)';
      ctx.lineWidth = 1;
      for (const a of [-0.6, -0.2, 0.2, 0.6]) {
        ctx.beginPath();
        ctx.moveTo(0.47 * W, 0.28 * H);
        ctx.lineTo(0.47 * W + Math.cos(a - 0.3) * 34, 0.28 * H + Math.sin(a - 0.3) * 34);
        ctx.stroke();
      }
    },

    /* the arena, with a ball trail */
    rocket(ctx, W, H) {
      const g = ctx.createLinearGradient(0, 0, W, H);
      g.addColorStop(0, '#0d1526'); g.addColorStop(1, '#0a1120');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = 'rgba(255,255,255,.12)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
      ctx.beginPath(); ctx.arc(W / 2, H / 2, H * 0.17, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = 'rgba(77,163,255,.5)'; ctx.fillRect(0, H * 0.32, 4, H * 0.36);
      ctx.fillStyle = 'rgba(255,159,69,.5)'; ctx.fillRect(W - 4, H * 0.32, 4, H * 0.36);
      const car = (x, y, ang, col) => {
        ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
        ctx.fillStyle = col;
        ctx.fillRect(-9, -5.5, 18, 11);
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.moveTo(9, 0); ctx.lineTo(4, -3); ctx.lineTo(4, 3); ctx.closePath(); ctx.fill();
        ctx.restore();
      };
      ctx.strokeStyle = 'rgba(255,209,102,.35)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(W * 0.3, H * 0.7);
      ctx.quadraticCurveTo(W * 0.55, H * 0.3, W * 0.78, H * 0.46);
      ctx.stroke();
      car(W * 0.26, H * 0.72, -0.7, '#4da3ff');
      car(W * 0.66, H * 0.68, 2.5, '#ff9f45');
      ctx.save();
      ctx.shadowColor = '#ffd166'; ctx.shadowBlur = 12;
      ctx.fillStyle = '#ffd166';
      ctx.beginPath(); ctx.arc(W * 0.78, H * 0.46, 6, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    },
  };

  document.addEventListener('DOMContentLoaded', () => {
    chrome('home', '');

    /* ---- progress across the whole course ---- */
    const earned = badgesEarned();
    const nEarned = BADGES.filter((b) => earned[b.id]).length;
    const strip = document.getElementById('progress-strip');
    if (strip) {
      strip.className = 'progress-strip';
      strip.innerHTML = `
        <span class="count">${nEarned} / ${BADGES.length} badges</span>
        <span class="track"><i style="width:${(nEarned / BADGES.length) * 100}%"></i></span>
        <span class="muted" style="font-size:.82rem">
          ${nEarned === 0 ? 'Badges unlock when a model of yours actually works.'
            : nEarned === BADGES.length ? 'Every badge earned. Genuinely well done.'
            : 'Kept in this browser only.'}
        </span>
        <button class="small" id="export-progress">Export</button>
        <label class="btn small" style="display:inline-block;margin:0">Import
          <input type="file" id="import-progress" accept="application/json" style="display:none">
        </label>
        ${nEarned ? '<button class="small reset" id="reset-badges">Reset</button>' : ''}`;
      const reset = document.getElementById('reset-badges');
      if (reset) reset.addEventListener('click', () => {
        localStorage.removeItem('mlbb-badges');
        location.reload();
      });

      /* Progress lives in localStorage, which is per-browser and easy to lose.
         These move it as a small file. */
      // Badges, quiz answers and predictions are fixed keys. Game records are
      // not: every personal best and saved ghost lap gets its own key, so they
      // are collected by prefix rather than listed here and forgotten later.
      const FIXED = ['mlbb-badges', 'mlbb-quiz', 'mlbb-capstone', 'mlbb-predict', 'mlbb-sound'];
      const PREFIXES = ['mlbb-best-', 'mlbb-ghost-'];
      const savedKeys = () => {
        const out = new Set(FIXED);
        try {
          for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            if (PREFIXES.some((p) => k && k.startsWith(p))) out.add(k);
          }
        } catch (err) { /* private mode */ }
        return [...out];
      };
      const exp = document.getElementById('export-progress');
      if (exp) exp.addEventListener('click', () => {
        const out = {};
        for (const k of savedKeys()) { const v = localStorage.getItem(k); if (v !== null) out[k] = v; }
        const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'learn-ml-progress.json';
        a.click();
        URL.revokeObjectURL(a.href);
      });
      const imp = document.getElementById('import-progress');
      if (imp) imp.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const obj = JSON.parse(reader.result);
            // Accept anything the site owns, so a file written by a later
            // version with more games still restores everything in it.
            for (const k of Object.keys(obj)) {
              if (FIXED.includes(k) || PREFIXES.some((p) => k.startsWith(p))) {
                localStorage.setItem(k, obj[k]);
              }
            }
            location.reload();
          } catch (err) {
            alert('That file could not be read as saved progress.');
          }
        };
        reader.readAsText(file);
      });
    }

    const host = document.getElementById('lesson-cards');
    for (const l of LESSONS) {
      if (l.capstone) continue;
      const a = document.createElement('a');
      a.className = 'card';
      a.href = l.href;
      a.innerHTML = `
        <canvas class="thumb" data-id="${l.id}"></canvas>
        <h3>${l.title}</h3>
        <p>${l.blurb}</p>
        <div class="tag-row">
          ${l.tags.map(([cls, t]) => `<span class="tag ${cls}">${t}</span>`).join('')}
          <span class="tag">${l.time}</span>
        </div>
        <div class="badge-row">
          ${BADGES.filter((b) => b.lesson === l.id).map((b) => `
            <span class="badge ${earned[b.id] ? 'earned' : ''}" title="${earned[b.id] ? b.label : b.hint}">
              <i>${earned[b.id] ? '★' : '☆'}</i>${earned[b.id] ? b.label : b.hint}
            </span>`).join('')}
        </div>`;
      host.appendChild(a);
    }

    // the capstone sits after the lessons, framed as the thing you do last
    const capstone = LESSONS.find((l) => l.capstone);
    const capHost = document.getElementById('capstone-card');
    if (capstone && capHost) {
      const done = BADGES.filter((b) => b.lesson === 'capstone');
      const got = done.filter((b) => earned[b.id]).length;
      capHost.innerHTML = `
        <a class="card capstone" href="${capstone.href}">
          <div class="cap-flag">Capstone</div>
          <h3>${capstone.title}</h3>
          <p>${capstone.blurb}</p>
          <div class="badge-row">
            ${done.map((b) => `<span class="badge ${earned[b.id] ? 'earned' : ''}">
              <i>${earned[b.id] ? '★' : '☆'}</i>${earned[b.id] ? b.label : b.hint}</span>`).join('')}
          </div>
          <div class="muted" style="font-size:.82rem;margin-top:10px">
            ${got === done.length ? 'Both parts complete.' : 'Thirteen decisions, no training, ' + capstone.time + '.'}
          </div>
        </a>`;
    }

    const paint = () => {
      document.querySelectorAll('canvas.thumb').forEach((c) => {
        const ctx = hidpi(c, c.clientWidth, 108);
        ctx.fillStyle = '#0d1320';
        ctx.fillRect(0, 0, ctx._cssW, ctx._cssH);
        const fn = THUMB[c.dataset.id];
        if (fn) fn(ctx, ctx._cssW, ctx._cssH);
      });
    };
    paint();
    window.addEventListener('resize', paint);
  });
})();
