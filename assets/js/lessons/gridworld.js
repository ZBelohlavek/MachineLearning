/* ==========================================================================
   gridworld.js — tabular Q-learning you can watch fill in, square by square.

   No neural network here on purpose. Q-learning on a small grid is the one
   place where you can see the *entire* contents of an agent's mind at once:
   four numbers per square, and a policy that falls out of them.
   ========================================================================== */

;(function () {
  'use strict';
  const { hidpi, fit, LineChart, chrome, nextLinks, slider, pills, checkbox, achieve,
          statGrid, rafLoop, mulberry32, clamp } = window.ML;

  const EMPTY = 0, WALL = 1, GOAL = 2, PIT = 3, START = 4;
  const ACTIONS = [[0, -1], [1, 0], [0, 1], [-1, 0]];      // up, right, down, left
  const ANAMES = ['↑', '→', '↓', '←'];

  /* ---------------------------------------------------------------- world */
  class Grid {
    constructor(w, h) {
      this.w = w; this.h = h;
      this.cells = new Uint8Array(w * h);
      this.stepReward = -0.02;
      this.slip = 0;
      this.maxSteps = 220;
      this.rand = mulberry32(7);
      this.loadPreset('maze');
    }
    idx(x, y) { return y * this.w + x; }
    get(x, y) { return this.cells[this.idx(x, y)]; }
    set(x, y, v) {
      if (v === START) {                       // only one start square
        for (let i = 0; i < this.cells.length; i++) if (this.cells[i] === START) this.cells[i] = EMPTY;
      }
      this.cells[this.idx(x, y)] = v;
      this.startPos = this.findStart();
    }
    inside(x, y) { return x >= 0 && y >= 0 && x < this.w && y < this.h; }
    findStart() {
      for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++)
        if (this.get(x, y) === START) return { x, y };
      return { x: 0, y: this.h - 1 };
    }

    loadPreset(name) {
      this.cells.fill(EMPTY);
      const W = this.w, H = this.h;
      if (name === 'maze') {
        const walls = [[3,0],[3,1],[3,2],[3,4],[3,5],[3,6],[6,7],[6,6],[6,5],[6,3],[6,2],[6,1],[9,0],[9,1],[9,2],[9,3],[9,5],[9,6],[9,7]];
        for (const [x, y] of walls) if (this.inside(x, y)) this.cells[this.idx(x, y)] = WALL;
        this.cells[this.idx(W - 1, 0)] = GOAL;
        this.cells[this.idx(7, 4)] = PIT;
        this.cells[this.idx(4, 3)] = PIT;
        this.cells[this.idx(0, H - 1)] = START;
      } else if (name === 'cliff') {
        for (let x = 1; x < W - 1; x++) this.cells[this.idx(x, H - 1)] = PIT;
        this.cells[this.idx(W - 1, H - 1)] = GOAL;
        this.cells[this.idx(0, H - 1)] = START;
      } else if (name === 'rooms') {
        for (let y = 0; y < H; y++) if (y !== 2 && y !== 5) this.cells[this.idx(5, y)] = WALL;
        for (let x = 0; x < W; x++) if (x !== 2 && x !== 8 && x !== 5) this.cells[this.idx(x, 4)] = WALL;
        this.cells[this.idx(W - 1, 0)] = GOAL;
        this.cells[this.idx(W - 2, H - 1)] = PIT;
        this.cells[this.idx(0, H - 1)] = START;
      } else if (name === 'open') {
        this.cells[this.idx(W - 1, 0)] = GOAL;
        this.cells[this.idx(0, H - 1)] = START;
      } else if (name === 'random') {
        for (let i = 0; i < this.cells.length; i++) if (this.rand() < 0.22) this.cells[i] = WALL;
        for (let i = 0; i < 3; i++) {
          const x = 1 + ((this.rand() * (W - 2)) | 0), y = 1 + ((this.rand() * (H - 2)) | 0);
          this.cells[this.idx(x, y)] = PIT;
        }
        this.cells[this.idx(W - 1, 0)] = GOAL;
        this.cells[this.idx(0, H - 1)] = START;
      }
      this.startPos = this.findStart();
    }

    /** One transition of the MDP: where do I land, and what do I get paid? */
    step(s, a) {
      let dir = a;
      // "Slip": with some probability the floor is icy and you go sideways.
      if (this.slip > 0 && this.rand() < this.slip) dir = (a + (this.rand() < 0.5 ? 1 : 3)) % 4;
      const [dx, dy] = ACTIONS[dir];
      let nx = s.x + dx, ny = s.y + dy;
      if (!this.inside(nx, ny) || this.get(nx, ny) === WALL) { nx = s.x; ny = s.y; }
      const t = this.get(nx, ny);
      let reward = this.stepReward, done = false;
      if (t === GOAL) { reward += 1; done = true; }
      else if (t === PIT) { reward += -1; done = true; }
      return { next: { x: nx, y: ny }, reward, done };
    }
  }

  /* ---------------------------------------------------------------- agent */
  class QAgent {
    constructor(grid) {
      this.grid = grid;
      this.Q = new Float32Array(grid.w * grid.h * 4);
      this.visits = new Uint32Array(grid.w * grid.h);
      this.alpha = 0.25; this.gamma = 0.95; this.eps = 0.3;
      this.epsDecay = 0.999; this.epsMin = 0.02;
      this.rand = mulberry32(99);
    }
    qi(s, a) { return (s.y * this.grid.w + s.x) * 4 + a; }
    qs(s) { const b = (s.y * this.grid.w + s.x) * 4; return this.Q.subarray(b, b + 4); }
    maxQ(s) { const q = this.qs(s); return Math.max(q[0], q[1], q[2], q[3]); }
    bestA(s) {
      const q = this.qs(s);
      let bi = 0;
      for (let i = 1; i < 4; i++) if (q[i] > q[bi]) bi = i;
      return bi;
    }
    chooseAction(s) {
      // epsilon-greedy: mostly exploit what you know, occasionally try anything.
      if (this.rand() < this.eps) return (this.rand() * 4) | 0;
      return this.bestA(s);
    }
    /** The Q-learning update, exactly as written in the textbook. */
    learn(s, a, r, s2, done) {
      const i = this.qi(s, a);
      const old = this.Q[i];
      const future = done ? 0 : this.maxQ(s2);
      const target = r + this.gamma * future;
      const tdError = target - old;
      this.Q[i] = old + this.alpha * tdError;
      this.visits[s.y * this.grid.w + s.x]++;
      return { old, r, future, target, tdError, next: this.Q[i], a, s: { ...s }, s2: { ...s2 } };
    }
    reset() { this.Q.fill(0); this.visits.fill(0); }
  }

  /* ---------------------------------------------------------------- page */
  document.addEventListener('DOMContentLoaded', () => {
    chrome('gridworld', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.gridworld);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'gridworld');
    nextLinks(document.getElementById('next-links'), 'attention', 'racer', '../');

    const GW = 11, GH = 8;
    const grid = new Grid(GW, GH);
    const agent = new QAgent(grid);

    const canvas = document.getElementById('grid-canvas');
    let ctx, cell;
    function resize() {
      canvas.style.width = '';
      const w = canvas.clientWidth || canvas.parentElement.clientWidth;
      cell = Math.floor(w / GW);
      ctx = hidpi(canvas, cell * GW, cell * GH);
    }
    resize();
    window.addEventListener('resize', () => { resize(); draw(); });

    /* ------------------------------- state of the running episode ------- */
    let pos = { ...grid.startPos };
    let prevPos = { ...pos };
    let animT = 1;
    let episode = 0, epSteps = 0, epReward = 0, done = false;
    let lastUpdate = null;
    let totalSteps = 0;
    let running = false;
    let stepsPerFrame = 4;
    let showValues = true, showPolicy = true;

    const chart = new LineChart(document.getElementById('gw-chart'), {
      height: 150, xLabel: 'episode', yMin: 0,
      series: [{ name: 'steps to finish', color: '#ffb547' }],
    });
    const chartR = new LineChart(document.getElementById('gw-chart-r'), {
      height: 150, xLabel: 'episode', zeroLine: true,
      series: [{ name: 'reward collected', color: '#38d39f' }],
    });
    const setStat = statGrid(document.getElementById('gw-stats'),
      ['episode', 'steps this ep', 'reward this ep', 'epsilon', 'total steps', 'best route']);

    let bestRoute = Infinity;

    function newEpisode() {
      pos = { ...grid.startPos };
      prevPos = { ...pos };
      animT = 1;
      epSteps = 0; epReward = 0; done = false;
      episode++;
    }

    /** One environment step + one Q-learning update. */
    function stepOnce() {
      if (done) {
        if (epSteps > 0) {
          chart.push(episode, [epSteps]);
          chartR.push(episode, [epReward]);
          if (grid.get(pos.x, pos.y) === GOAL) {
            bestRoute = Math.min(bestRoute, epSteps);
            if (bestRoute <= 20) achieve('grid-solved', `A ${bestRoute}-step route, learned from nothing`);
          }
        }
        agent.eps = Math.max(agent.epsMin, agent.eps * agent.epsDecay);
        newEpisode();
        return;
      }
      const a = agent.chooseAction(pos);
      const res = grid.step(pos, a);
      lastUpdate = agent.learn(pos, a, res.reward, res.next, res.done);
      prevPos = { ...pos };
      pos = res.next;
      animT = 0;
      epReward += res.reward;
      epSteps++;
      totalSteps++;
      if (res.done || epSteps >= grid.maxSteps) done = true;
    }

    /* ------------------------------- race the agent ---------------------
       You play the same maze with the arrow keys. It is a surprisingly good
       way to feel what the Q-table is worth: the agent needed thousands of
       steps to learn a route you can see at a glance — and you cannot beat it
       on a maze you have not looked at either.
       -------------------------------------------------------------------- */
    let racing = false, player = null, playerSteps = 0, raceMsg = '';

    function startRace() {
      racing = true;
      running = false;
      btnRun.textContent = '▶ Run';
      btnRun.classList.add('primary');
      player = { ...grid.startPos };
      playerSteps = 0;
      raceMsg = 'Use the arrow keys.';
      updateRaceUI();
      draw();
    }

    function movePlayer(dir) {
      if (!racing || !player) return;
      const res = grid.step(player, dir);
      player = res.next;
      playerSteps++;
      if (res.done) {
        const won = grid.get(player.x, player.y) === GOAL;
        if (won) {
          raceMsg = `Goal in ${playerSteps} steps. ` +
            (bestRoute === Infinity ? 'The agent has not finished a run yet.'
             : playerSteps < bestRoute ? `You beat the agent's best of ${bestRoute}.`
             : `The agent's best is ${bestRoute}.`);
          if (bestRoute !== Infinity && playerSteps < bestRoute) {
            achieve('grid-beaten', `${playerSteps} steps against the agent's ${bestRoute}`);
          }
        } else raceMsg = `You fell in a pit after ${playerSteps} steps. The agent has done that too.`;
        racing = false;
      }
      updateRaceUI();
      draw();
    }

    function updateRaceUI() {
      const el = document.getElementById('race-readout');
      if (!el) return;
      el.innerHTML = player
        ? `<span class="chip">your steps <b>${playerSteps}</b></span>` +
          `<span class="chip">agent's best <b>${bestRoute === Infinity ? '–' : bestRoute}</b></span>` +
          (raceMsg ? ` <span style="color:var(--text-dim)">${raceMsg}</span>` : '')
        : '';
    }

    window.addEventListener('keydown', (e) => {
      const dir = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3,
                    w: 0, d: 1, s: 2, a: 3 }[e.key];
      if (dir === undefined || !racing) return;
      e.preventDefault();
      movePlayer(dir);
    });
    document.getElementById('btn-race').addEventListener('click', startRace);

    /* ------------------------------- drawing ---------------------------- */
    function qColor(v, scale) {
      const t = clamp(v / scale, -1, 1);
      if (t >= 0) return `rgba(56, 211, 159, ${0.10 + 0.75 * t})`;
      return `rgba(255, 107, 107, ${0.10 + 0.75 * -t})`;
    }

    function draw() {
      const W = ctx._cssW, H = ctx._cssH;
      ctx.clearRect(0, 0, W, H);
      let scale = 0.15;
      for (let i = 0; i < agent.Q.length; i++) scale = Math.max(scale, Math.abs(agent.Q[i]));

      for (let y = 0; y < GH; y++) {
        for (let x = 0; x < GW; x++) {
          const px = x * cell, py = y * cell, t = grid.get(x, y);
          ctx.fillStyle = '#0d1320';
          ctx.fillRect(px, py, cell, cell);

          if (t === WALL) {
            ctx.fillStyle = '#3a4a6d';
            ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
            ctx.save();
            ctx.beginPath();
            ctx.rect(px + 1, py + 1, cell - 2, cell - 2);
            ctx.clip();                                      // keep hatching inside the cell
            ctx.strokeStyle = 'rgba(255,255,255,.10)';
            ctx.lineWidth = 1;
            for (let k = -cell; k < cell; k += 6) {
              ctx.beginPath();
              ctx.moveTo(px + k, py + cell);
              ctx.lineTo(px + k + cell, py);
              ctx.stroke();
            }
            ctx.restore();
          } else if (t === GOAL || t === PIT) {
            const good = t === GOAL;
            ctx.fillStyle = good ? 'rgba(56,211,159,.22)' : 'rgba(255,107,107,.20)';
            ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
            ctx.fillStyle = good ? '#38d39f' : '#ff6b6b';
            ctx.font = `600 ${cell * 0.3}px ui-monospace, monospace`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(good ? '+1' : '−1', px + cell / 2, py + cell / 2);
          } else {
            // four triangles, one per action, shaded by that action's Q value
            const q = agent.qs({ x, y });
            const cx = px + cell / 2, cy = py + cell / 2;
            const tri = [
              [[px, py], [px + cell, py]],                       // up
              [[px + cell, py], [px + cell, py + cell]],         // right
              [[px + cell, py + cell], [px, py + cell]],         // down
              [[px, py + cell], [px, py]],                       // left
            ];
            for (let a = 0; a < 4; a++) {
              ctx.fillStyle = qColor(q[a], scale);
              ctx.beginPath();
              ctx.moveTo(cx, cy);
              ctx.lineTo(tri[a][0][0], tri[a][0][1]);
              ctx.lineTo(tri[a][1][0], tri[a][1][1]);
              ctx.closePath();
              ctx.fill();
            }
            if (showValues) {
              const v = agent.maxQ({ x, y });
              if (Math.abs(v) > 0.001) {
                ctx.fillStyle = 'rgba(230,236,247,.75)';
                ctx.font = `${cell * 0.2}px ui-monospace, monospace`;
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(v.toFixed(2), cx, cy);
              }
            }
            if (showPolicy && Math.abs(agent.maxQ({ x, y })) > 0.001) {
              ctx.fillStyle = 'rgba(230,236,247,.9)';
              ctx.font = `${cell * 0.3}px system-ui, sans-serif`;
              ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText(ANAMES[agent.bestA({ x, y })], cx, cy - cell * 0.28);
            }
          }
          if (t === START) {
            ctx.strokeStyle = 'rgba(77,163,255,.8)';
            ctx.lineWidth = 2;
            ctx.strokeRect(px + 2, py + 2, cell - 4, cell - 4);
          }
          ctx.strokeStyle = 'rgba(255,255,255,.05)';
          ctx.lineWidth = 1;
          ctx.strokeRect(px + .5, py + .5, cell - 1, cell - 1);
        }
      }

      if (player) {
        ctx.save();
        ctx.strokeStyle = '#ffd166';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#ffd166'; ctx.shadowBlur = 10;
        ctx.strokeRect(player.x * cell + 4, player.y * cell + 4, cell - 8, cell - 8);
        ctx.restore();
        ctx.fillStyle = '#ffd166';
        ctx.font = `${cell * 0.22}px ui-monospace, monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('YOU', player.x * cell + cell / 2, player.y * cell + cell * 0.78);
      }

      // the agent, sliding between squares
      const ax = (prevPos.x + (pos.x - prevPos.x) * animT + 0.5) * cell;
      const ay = (prevPos.y + (pos.y - prevPos.y) * animT + 0.5) * cell;
      ctx.save();
      ctx.shadowColor = '#4da3ff'; ctx.shadowBlur = 14;
      ctx.fillStyle = '#4da3ff';
      ctx.beginPath(); ctx.arc(ax, ay, cell * 0.22, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    /* ------------------------------- the update read-out ---------------- */
    const upd = document.getElementById('update-readout');
    function drawUpdate() {
      if (!lastUpdate) { upd.innerHTML = '<span class="muted">Run the agent to see an update.</span>'; return; }
      const u = lastUpdate;
      const f = (v) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);
      upd.innerHTML = `
        <div class="mono" style="font-size:.82rem;line-height:1.9">
          <div>state <b>(${u.s.x},${u.s.y})</b>, action <b>${ANAMES[u.a]}</b>
               → <b>(${u.s2.x},${u.s2.y})</b>, reward <b>${f(u.r)}</b></div>
          <div style="color:var(--text-mute)">target = r + γ·max Q(s′) =
            ${f(u.r)} + ${agent.gamma.toFixed(2)}·${u.future.toFixed(3)} =
            <span style="color:var(--accent)">${u.target.toFixed(3)}</span></div>
          <div style="color:var(--text-mute)">TD error = target − Q(s,a) =
            ${u.target.toFixed(3)} − ${u.old.toFixed(3)} =
            <span style="color:${u.tdError >= 0 ? 'var(--good)' : 'var(--bad)'}">${f(u.tdError)}</span></div>
          <div>Q(s,a): ${u.old.toFixed(3)} → <b style="color:var(--text)">${u.next.toFixed(3)}</b>
            <span style="color:var(--text-mute)">(moved ${(agent.alpha * 100).toFixed(0)}% of the way)</span></div>
        </div>`;
    }

    /* ------------------------------- controls --------------------------- */
    let tool = WALL;
    pills(document.getElementById('gw-tools'), [
      { value: WALL, label: 'Wall' },
      { value: PIT, label: 'Pit (−1)' },
      { value: GOAL, label: 'Goal (+1)' },
      { value: START, label: 'Start' },
      { value: EMPTY, label: 'Erase' },
    ], WALL, (v) => { tool = v; });

    pills(document.getElementById('gw-presets'), [
      { value: 'maze', label: 'Maze' },
      { value: 'cliff', label: 'Cliff walk' },
      { value: 'rooms', label: 'Two rooms' },
      { value: 'open', label: 'Open field' },
      { value: 'random', label: 'Random' },
    ], 'maze', (v) => { grid.loadPreset(v); hardReset(); });

    let painting = false;
    const paint = (ev) => {
      const r = canvas.getBoundingClientRect();
      const p = ev.touches ? ev.touches[0] : ev;
      const x = Math.floor(((p.clientX - r.left) / r.width) * GW);
      const y = Math.floor(((p.clientY - r.top) / r.height) * GH);
      if (!grid.inside(x, y)) return;
      grid.set(x, y, tool);
      draw();
    };
    canvas.addEventListener('pointerdown', (e) => { painting = true; paint(e); });
    canvas.addEventListener('pointermove', (e) => { if (painting) paint(e); });
    window.addEventListener('pointerup', () => { painting = false; });

    const hpEl = document.getElementById('gw-hp');
    slider(hpEl, {
      label: 'learning rate α', min: 0.02, max: 1, step: 0.01, value: agent.alpha,
      format: (v) => v.toFixed(2), onInput: (v) => { agent.alpha = v; },
      desc: 'How far each update moves Q toward the new estimate. 1.0 means "believe the latest experience completely".',
    });
    slider(hpEl, {
      label: 'discount γ', min: 0.5, max: 0.999, step: 0.001, value: agent.gamma,
      format: (v) => v.toFixed(3), onInput: (v) => { agent.gamma = v; },
      desc: 'How much a reward one step later is worth. Drop it to 0.6 and distant goals stop being worth walking to.',
    });
    slider(hpEl, {
      label: 'exploration ε', min: 0, max: 1, step: 0.01, value: agent.eps,
      format: (v) => v.toFixed(2), onInput: (v) => { agent.eps = v; },
      desc: 'Chance of ignoring what it knows and trying a random move. Zero means it can never discover a better route than the first one it found.',
    });
    slider(hpEl, {
      label: 'ε decay per episode', min: 0.9, max: 1, step: 0.001, value: agent.epsDecay,
      format: (v) => v.toFixed(3), onInput: (v) => { agent.epsDecay = v; },
      desc: 'Explore early, exploit later. 1.000 keeps exploring forever.',
    });
    slider(hpEl, {
      label: 'step cost', min: -0.2, max: 0, step: 0.005, value: grid.stepReward,
      format: (v) => v.toFixed(3), onInput: (v) => { grid.stepReward = v; },
      desc: 'A small penalty for every move. Without it, dawdling is free and the agent has no reason to find a short route.',
    });
    slider(hpEl, {
      label: 'slippery floor', min: 0, max: 0.6, step: 0.02, value: 0,
      format: (v) => (v * 100).toFixed(0) + '%', onInput: (v) => { grid.slip = v; },
      desc: 'Chance the world ignores your action and moves you sideways instead — a stochastic environment.',
    });

    checkbox(hpEl, 'Show state values', true, (v) => { showValues = v; draw(); });
    checkbox(hpEl, 'Show greedy policy arrows', true, (v) => { showPolicy = v; draw(); });

    const btnRun = document.getElementById('btn-run');
    btnRun.addEventListener('click', () => {
      running = !running;
      btnRun.textContent = running ? '⏸ Pause' : '▶ Run';
      btnRun.classList.toggle('primary', !running);
    });
    document.getElementById('btn-step').addEventListener('click', () => { stepOnce(); animT = 1; draw(); drawUpdate(); updateStats(); });
    document.getElementById('btn-reset-q').addEventListener('click', () => hardReset());
    pills(document.getElementById('gw-speed'), [
      { value: 1, label: 'Step by step' },
      { value: 4, label: 'Normal' },
      { value: 40, label: 'Fast' },
      { value: 400, label: 'Instant' },
    ], 4, (v) => { stepsPerFrame = v; });

    function hardReset() {
      agent.reset();
      player = null; racing = false; raceMsg = '';
      updateRaceUI();
      agent.eps = 0.3;
      episode = 0; totalSteps = 0; bestRoute = Infinity;
      lastUpdate = null;
      chart.clear(); chartR.clear();
      newEpisode();
      draw(); drawUpdate(); updateStats();
    }

    function updateStats() {
      setStat('episode', episode);
      setStat('steps this ep', epSteps);
      setStat('reward this ep', epReward.toFixed(2), epReward > 0 ? 'good' : epReward < -0.5 ? 'bad' : '');
      setStat('epsilon', agent.eps.toFixed(3));
      setStat('total steps', totalSteps.toLocaleString());
      setStat('best route', bestRoute === Infinity ? '–' : bestRoute + ' steps');
    }

    /* ------------------------------- loop ------------------------------- */
    rafLoop((dt) => {
      if (running) {
        for (let i = 0; i < stepsPerFrame; i++) stepOnce();
        if (stepsPerFrame > 20) animT = 1;
        drawUpdate();
        updateStats();
        chart.draw(); chartR.draw();
      }
      animT = Math.min(1, animT + dt * 9);
      draw();
    }).start();

    newEpisode();
    draw(); drawUpdate(); updateStats();
  });
})();
