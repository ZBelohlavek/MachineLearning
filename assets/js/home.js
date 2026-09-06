/* ==========================================================================
   home.js — builds the lesson grid and paints a small preview for each card.
   ========================================================================== */

;(function () {
  'use strict';
  const { LESSONS, chrome, hidpi, rafLoop, mulberry32 } = window.ML;

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
    const host = document.getElementById('lesson-cards');
    for (const l of LESSONS) {
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
        </div>`;
      host.appendChild(a);
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
