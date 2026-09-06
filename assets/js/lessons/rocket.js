/* ==========================================================================
   rocket.js — page wiring for the Rocket League reinforcement learning lesson:
   arena rendering, the interactive sandbox, the reward designer, the training
   loop and the policy inspector.
   ========================================================================== */

;(function () {
  'use strict';
  const {
    F, ACTIONS, ACTION_LABELS, OBS_SIZE, DEFAULT_REWARDS, REWARD_PRESETS,
    Arena, scriptedAction, Trainer, DEFAULT_HP,
    hidpi, fit, LineChart, diverging, heat, softmax, argmax, sampleFrom, clamp, MLP,
    chrome, nextLinks, slider, pills, checkbox, statGrid, rafLoop, achieve,
  } = window.ML;

  const COL = {
    blue: '#4da3ff', blueDark: '#1f5c9e',
    orange: '#ff9f45', orangeDark: '#a35c1c',
    ball: '#ffd166', pitch: '#0c1220', line: 'rgba(255,255,255,.13)',
  };

  /* ======================================================================
     Arena rendering
     ====================================================================== */

  function makeStage(canvas) {
    const resize = () => {
      const w = canvas.parentElement.clientWidth;
      const h = Math.round((w * (F.halfH * 2)) / (F.halfW * 2));
      canvas._ctx = hidpi(canvas, w, h);
      canvas._scale = w / (F.halfW * 2);
    };
    resize();
    window.addEventListener('resize', resize);
    return canvas;
  }

  const TX = (c, x) => (x + F.halfW) * c._scale;
  const TY = (c, y) => (y + F.halfH) * c._scale;

  function drawField(ctx, c) {
    const W = ctx._cssW, H = ctx._cssH, s = c._scale;
    ctx.clearRect(0, 0, W, H);
    const g = ctx.createLinearGradient(0, 0, W, H);
    g.addColorStop(0, '#0d1526'); g.addColorStop(0.5, '#0a1120'); g.addColorStop(1, '#0d1526');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

    // faint grid
    ctx.strokeStyle = 'rgba(255,255,255,.035)'; ctx.lineWidth = 1;
    for (let x = -80; x <= 80; x += 20) {
      ctx.beginPath(); ctx.moveTo(TX(c, x), 0); ctx.lineTo(TX(c, x), H); ctx.stroke();
    }
    for (let y = -60; y <= 60; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, TY(c, y)); ctx.lineTo(W, TY(c, y)); ctx.stroke();
    }

    // markings
    ctx.strokeStyle = COL.line; ctx.lineWidth = 1.5;
    ctx.strokeRect(1, 1, W - 2, H - 2);
    ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
    ctx.beginPath(); ctx.arc(W / 2, H / 2, 16 * s, 0, Math.PI * 2); ctx.stroke();

    // goals: left belongs to Blue (Orange scores there), right belongs to Orange
    const gh = F.goalHalf * s, depth = 7 * s;
    const goal = (x0, color) => {
      ctx.fillStyle = color + '22';
      ctx.fillRect(x0 < W / 2 ? 0 : W - depth, H / 2 - gh, depth, gh * 2);
      ctx.strokeStyle = color; ctx.lineWidth = 3;
      ctx.beginPath();
      const gx = x0 < W / 2 ? 1.5 : W - 1.5;
      ctx.moveTo(gx, H / 2 - gh); ctx.lineTo(gx, H / 2 + gh); ctx.stroke();
      // goal box
      ctx.strokeStyle = COL.line; ctx.lineWidth = 1.5;
      const bw = 22 * s;
      ctx.strokeRect(x0 < W / 2 ? 0 : W - bw, H / 2 - gh * 1.5, bw, gh * 3);
    };
    goal(0, COL.blue);
    goal(W, COL.orange);
  }

  function drawCar(ctx, c, car, color, dark, label) {
    const s = c._scale;
    ctx.save();
    ctx.translate(TX(c, car.x), TY(c, car.y));
    ctx.rotate(car.angle);
    const L = 11 * s, Wd = 7.2 * s;
    if (car.lastTouch > 0) {                       // contact glow
      ctx.shadowColor = color; ctx.shadowBlur = 18;
    }
    ctx.fillStyle = color;
    roundRect(ctx, -L / 2, -Wd / 2, L, Wd, 2.4 * s);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = dark;
    roundRect(ctx, -L / 2 + 1.2 * s, -Wd / 2 + 1.1 * s, L * 0.42, Wd - 2.2 * s, 1.2 * s);
    ctx.fill();
    // nose
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(L / 2, 0); ctx.lineTo(L / 2 - 2.6 * s, -1.8 * s); ctx.lineTo(L / 2 - 2.6 * s, 1.8 * s);
    ctx.closePath(); ctx.fill();
    ctx.restore();
    if (label) {
      ctx.fillStyle = 'rgba(255,255,255,.5)';
      ctx.font = `${Math.max(9, 3.4 * s)}px ui-monospace, monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(label, TX(c, car.x), TY(c, car.y) - 8 * s);
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawBall(ctx, c, ball, trail) {
    const s = c._scale;
    if (trail && trail.length > 1) {
      ctx.strokeStyle = 'rgba(255,209,102,.25)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      trail.forEach((p, i) => (i ? ctx.lineTo(TX(c, p.x), TY(c, p.y)) : ctx.moveTo(TX(c, p.x), TY(c, p.y))));
      ctx.stroke();
    }
    const x = TX(c, ball.x), y = TY(c, ball.y), r = F.ballR * s;
    ctx.save();
    ctx.shadowColor = COL.ball; ctx.shadowBlur = 14;
    const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.15, x, y, r);
    g.addColorStop(0, '#fff6d5'); g.addColorStop(1, COL.ball);
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }

  function drawOverlay(ctx, c, state) {
    const W = ctx._cssW, H = ctx._cssH;
    // clock
    const frac = 1 - state.arena.step_ / F.steps;
    ctx.fillStyle = 'rgba(255,255,255,.10)';
    ctx.fillRect(W * 0.25, 6, W * 0.5, 4);
    ctx.fillStyle = frac < 0.25 ? '#ffb547' : '#4da3ff';
    ctx.fillRect(W * 0.25, 6, W * 0.5 * frac, 4);
    // score
    ctx.font = 'bold 16px ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillStyle = COL.blue; ctx.fillText(String(state.score[0]), W * 0.5 - 22, 16);
    ctx.fillStyle = '#55627d'; ctx.fillText('–', W * 0.5, 16);
    ctx.fillStyle = COL.orange; ctx.fillText(String(state.score[1]), W * 0.5 + 22, 16);
    // goal flash
    if (state.flash > 0) {
      const a = Math.min(1, state.flash);
      ctx.fillStyle = `rgba(${state.flashTeam === 0 ? '77,163,255' : '255,159,69'},${a * 0.18})`;
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = `rgba(255,255,255,${a})`;
      ctx.font = 'bold 30px system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      ctx.fillText('GOAL', W / 2, H / 2);
    }
  }

  /* ======================================================================
     A live match: physics in real time, actions from whatever driver you pick
     ====================================================================== */

  class Match {
    constructor(canvas, opts = {}) {
      this.canvas = makeStage(canvas);
      this.arena = new Arena({ rewards: opts.rewards || DEFAULT_REWARDS, seed: (Math.random() * 1e9) | 0 });
      this.arena.spread = opts.spread ?? 1;
      this.mode = opts.mode || 'selfplay';
      this.trainer = opts.trainer || null;
      this.keys = {};
      this.score = [0, 0];
      this.flash = 0;
      this.flashTeam = 0;
      this.trail = [];
      this.acc = 0;
      this.tick = 0;
      this.actions = [4, 4];
      this.obs = [new Float32Array(OBS_SIZE), new Float32Array(OBS_SIZE)];
      this.lastProbs = null;
      this.lastValue = 0;
      this.onStep = opts.onStep || null;
      this.speed = 1;
    }

    humanAction() {
      const k = this.keys;
      const throttle = (k.up ? 1 : 0) + (k.down ? -1 : 0);
      const steer = (k.right ? 1 : 0) + (k.left ? -1 : 0);
      return ACTIONS.findIndex((a) => a.throttle === throttle && a.steer === steer);
    }

    chooseActions() {
      const a = this.arena;
      const T = this.trainer;
      const greedy = this.greedy ?? false;
      const pick = (i) => {
        if (!T) return 4;
        a.observe(i, this.obs[i]);
        const probs = softmax(T.policy.forward(this.obs[i]));
        if (i === 0) {
          this.lastProbs = Float32Array.from(probs);
          this.lastValue = T.value ? T.value.forward(this.obs[i])[0] : null;
          this.lastAct = T.policy.activations.map((x) => Float32Array.from(x));
        }
        return greedy ? argmax(probs) : sampleFrom(probs);
      };
      switch (this.mode) {
        case 'selfplay': this.actions = [pick(0), pick(1)]; break;
        case 'vs-scripted': this.actions = [pick(0), scriptedAction(a, 1)]; break;
        case 'human': this.actions = [pick(0), this.humanAction()]; break;
        case 'human-solo': this.actions = [this.humanAction(), 4]; break;
        case 'scripted': this.actions = [scriptedAction(a, 0), scriptedAction(a, 1)]; break;
        case 'solo': this.actions = [pick(0), 4]; break;
      }
    }

    update(dt) {
      const a = this.arena;
      a.solo = this.mode === 'solo' || this.mode === 'human-solo';
      this.acc += dt * this.speed;
      const repeat = this.trainer ? this.trainer.hp.actionRepeat : 2;
      let guard = 0;
      while (this.acc >= F.dt && guard++ < 8) {
        this.acc -= F.dt;
        if (this.tick % repeat === 0) this.chooseActions();
        if (this.mode === 'human' || this.mode === 'human-solo') {
          // The human's inputs are read every tick so steering feels responsive.
          const hi = this.mode === 'human' ? 1 : 0;
          this.actions[hi] = this.humanAction();
        }
        const res = a.step(this.actions);
        this.tick++;
        if (this.onStep) this.onStep(res, a);
        this.trail.push({ x: a.ball.x, y: a.ball.y });
        if (this.trail.length > 26) this.trail.shift();
        if (a.done) {
          if (a.scorer >= 0) {
            this.score[a.scorer]++;
            this.flash = 1.4;
            this.flashTeam = a.scorer;
            if (this.onGoal) this.onGoal(a.scorer, this.mode);
          }
          a.spread = this.spreadOverride ?? 1;
          a.reset();
          this.trail.length = 0;
          this.tick = 0;
        }
      }
      if (this.flash > 0) this.flash -= dt * 1.2;
    }

    render() {
      const c = this.canvas, ctx = c._ctx;
      drawField(ctx, c);
      drawBall(ctx, c, this.arena.ball, this.trail);
      drawCar(ctx, c, this.arena.cars[0], COL.blue, COL.blueDark,
              this.mode === 'human-solo' ? 'YOU' : 'AGENT');
      if (!this.arena.solo) {
        const label = this.mode === 'human' ? 'YOU'
          : this.mode === 'vs-scripted' ? 'SCRIPT'
          : this.mode === 'selfplay' ? 'SELF' : '';
        drawCar(ctx, c, this.arena.cars[1], COL.orange, COL.orangeDark, label);
      }
      drawOverlay(ctx, c, this);
    }
  }

  /* ======================================================================
     Keyboard
     ====================================================================== */
  const KEYMAP = {
    ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
    w: 'up', s: 'down', a: 'left', d: 'right',
    W: 'up', S: 'down', A: 'left', D: 'right',
  };
  function bindKeys(target) {
    const on = (e, down) => {
      const k = KEYMAP[e.key];
      if (!k) return;
      if (target.active) { e.preventDefault(); target.keys[k] = down; }
    };
    window.addEventListener('keydown', (e) => on(e, true));
    window.addEventListener('keyup', (e) => on(e, false));
  }

  /* ======================================================================
     Page setup
     ====================================================================== */

  document.addEventListener('DOMContentLoaded', () => {
    chrome('rocket', '../');
    nextLinks(document.getElementById('next-links'), 'racer', null, '../');

    /* --- shared reward weights, edited by the user, used everywhere --- */
    const rewards = { ...DEFAULT_REWARDS };
    const trainer = new Trainer({ rewards, opponent: 'self' });
    window.__trainer = trainer;   // handy if you want to poke at it in the console

    /* ==================================================================
       PANEL 1 — the sandbox: drive it yourself, see what the agent sees
       ================================================================== */
    const sandbox = new Match(document.getElementById('sandbox-canvas'), {
      mode: 'human-solo', rewards,
    });
    sandbox.active = true;
    sandbox.spreadOverride = 1;
    bindKeys(sandbox);

    const obsCanvas = document.getElementById('obs-canvas');
    const obsCtx = fit(obsCanvas, 132);
    const OBS_NAMES = [
      'my x', 'my y', 'speed fwd', 'speed side', 'facing cos', 'facing sin',
      'ball ahead', 'ball left', 'ball dist', 'ball near', 'ball vel fwd', 'ball vel side',
      'ball x', 'ball y', 'goal ahead', 'goal left', 'b→goal fwd', 'b→goal side',
      'angle cos', 'angle sin', 'opp ahead', 'opp left', 'opp dist', 'clock',
    ];
    function drawObs(obs) {
      const ctx = obsCtx, W = ctx._cssW, H = ctx._cssH;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0a0f19'; ctx.fillRect(0, 0, W, H);
      const cols = 6, rows = 4;
      const cw = W / cols, ch = H / rows;
      for (let i = 0; i < OBS_SIZE; i++) {
        const cx = (i % cols) * cw, cy = ((i / cols) | 0) * ch;
        const v = obs[i];
        ctx.fillStyle = diverging(v, 1);
        ctx.globalAlpha = 0.85;
        ctx.fillRect(cx + 2, cy + 2, cw - 4, ch - 15);
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#0a0f19';
        ctx.font = '9px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(v.toFixed(2), cx + cw / 2, cy + (ch - 13) / 2 + 2);
        ctx.fillStyle = '#6d7f9c';
        ctx.fillText(OBS_NAMES[i], cx + cw / 2, cy + ch - 6);
      }
    }

    // live reward read-out while you drive
    const rewardBars = document.getElementById('reward-bars');
    const TERMS = ['approach', 'push', 'touch', 'face', 'speed', 'time', 'goal'];
    const barEls = {};
    TERMS.forEach((t) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.innerHTML = `<span class="name">${t}</span>
        <span class="track"><i class="fill"></i></span><span class="num">0.00</span>`;
      rewardBars.appendChild(row);
      barEls[t] = { fill: row.querySelector('.fill'), num: row.querySelector('.num') };
    });
    const totalEl = document.getElementById('reward-total');
    let epReward = 0;
    sandbox.onStep = (res, arena) => {
      epReward += res.rewards[0];
      const terms = arena.lastTerms[0];
      for (const t of TERMS) {
        const v = terms[t] || 0;
        const el = barEls[t];
        const pct = Math.max(-1, Math.min(1, v / 0.6)) * 50;
        el.fill.style.background = v >= 0 ? 'var(--good)' : 'var(--bad)';
        el.fill.style.left = v >= 0 ? '50%' : (50 + pct) + '%';
        el.fill.style.width = Math.abs(pct) + '%';
        el.num.textContent = v.toFixed(3);
      }
      totalEl.textContent = epReward.toFixed(1);
      if (arena.done) epReward = 0;
    };

    const sandboxCtl = document.getElementById('sandbox-controls');
    const sandboxMode = pills(sandboxCtl, [
      { value: 'human-solo', label: 'Drive alone' },
      { value: 'human', label: 'vs scripted bot' },
      { value: 'scripted', label: 'Watch scripted bot' },
    ], 'human-solo', (v) => {
      if (v === 'human') { sandbox.mode = 'human'; }
      else sandbox.mode = v;
      sandbox.arena.reset();
    });
    // In "vs scripted" the human drives Orange, so swap the scripted car in.
    const origChoose = sandbox.chooseActions.bind(sandbox);
    sandbox.chooseActions = function () {
      if (this.mode === 'human') { this.actions = [scriptedAction(this.arena, 0), this.humanAction()]; return; }
      origChoose();
    };

    const sandboxLoop = rafLoop((dt) => {
      sandbox.update(dt);
      sandbox.render();
      sandbox.arena.observe(0, sandbox.obs[0]);
      drawObs(sandbox.obs[0]);
    });
    sandboxLoop.start();

    /* ==================================================================
       PANEL 2 — the reward designer
       ================================================================== */
    const rewardPanel = document.getElementById('reward-sliders');
    const REWARD_META = [
      ['goal', 0, 20, 'Scoring. The only thing that truly matters — and the hardest signal to learn from, because it arrives once every few hundred steps.'],
      ['push', 0, 3, 'Reward for moving the ball toward the opponent goal (and punishment for the reverse). The single most useful shaping term.'],
      ['touch', 0, 2, 'A bonus every time the car makes contact with the ball.'],
      ['approach', 0, 2, 'Reward for closing distance to the ball. Gets a beginner agent moving, but can teach it to hug the ball instead of scoring.'],
      ['face', 0, 0.4, 'Reward for pointing at the ball.'],
      ['speed', 0, 0.2, 'Reward for driving fast. Stops the agent from parking.'],
      ['time', 0, 0.02, 'A small penalty every step: hurry up.'],
    ];
    const rewardSliders = {};
    REWARD_META.forEach(([key, min, max, desc]) => {
      rewardSliders[key] = slider(rewardPanel, {
        label: key, min, max, step: (max - min) / 100, value: rewards[key],
        format: (v) => v.toFixed(3), desc,
        onInput: (v) => { rewards[key] = v; trainer.setRewards(rewards); },
      });
    });
    pills(document.getElementById('reward-presets'), [
      { value: 'recommended', label: 'Recommended' },
      { value: 'sparse', label: 'Goals only (hard)' },
      { value: 'chaser', label: 'Ball chaser' },
      { value: 'striker', label: 'Aggressive striker' },
    ], 'recommended', (name) => {
      const preset = REWARD_PRESETS[name];
      for (const k in preset) {
        rewards[k] = preset[k];
        if (rewardSliders[k]) rewardSliders[k].set(preset[k]);
      }
      trainer.setRewards(rewards);
    });

    /* ==================================================================
       PANEL 2.5 — the training timelapse.

       Live training is the honest experience, but it asks for patience before
       anything interesting happens. These are checkpoints from a full run
       trained offline by tools/train-rocket.cjs: drag through them and watch
       three thousand episodes of learning in a few seconds.
       ================================================================== */
    const stagesData = window.ML_ROCKET_STAGES;
    const tlBody = document.getElementById('timelapse-body');

    if (!stagesData) {
      tlBody.innerHTML =
        '<p class="muted">The pre-trained checkpoints are missing. Run ' +
        '<code>node tools/train-rocket.cjs</code> to generate them — or skip this and train ' +
        'an agent yourself in the next section.</p>';
    } else {
      const stages = stagesData.stages;
      const stageAgent = {
        policy: null, value: null,
        hp: { actionRepeat: (stagesData.hp && stagesData.hp.actionRepeat) || 2 },
      };
      const stageMatch = new Match(document.getElementById('stage-canvas'), {
        trainer: stageAgent, mode: 'selfplay', rewards,
      });
      stageMatch.active = true;
      stageMatch.greedy = false;

      // Two charts rather than one: a rate between 0 and 1 and a count that
      // reaches double figures do not share an axis usefully.
      const stageChart = new LineChart(document.getElementById('stage-chart'), {
        height: 120, xLabel: 'episodes', yMin: 0, yMax: 1,
        series: [{ name: 'episodes ending in a goal', color: '#38d39f' }],
      });
      const stageChart2 = new LineChart(document.getElementById('stage-chart2'), {
        height: 120, xLabel: 'episodes', yMin: 0,
        series: [{ name: 'ball touches per episode', color: '#ffb547' }],
      });
      for (const row of stagesData.curve || []) {
        stageChart.push(row.e, [row.g]);
        stageChart2.push(row.e, [row.t]);
      }

      const label = document.getElementById('stage-label');
      const detail = document.getElementById('stage-detail');
      const scrub = document.getElementById('stage-scrub');
      scrub.max = String(stages.length - 1);

      let stageIndex = 0;
      function showStage(i) {
        stageIndex = clamp(i, 0, stages.length - 1);
        const st = stages[stageIndex];
        stageAgent.policy = window.ML.MLP.fromJSON(st.policy);
        stageMatch.score = [0, 0];
        stageMatch.arena.reset();
        scrub.value = String(stageIndex);
        label.innerHTML =
          `<b>${st.label}</b> <span class="muted">— after ${st.episodes.toLocaleString()} episodes ` +
          `of self-play</span>`;
        detail.innerHTML =
          `<span class="chip">goal rate <b>${(st.goalRate * 100).toFixed(0)}%</b></span>` +
          `<span class="chip">touches/ep <b>${st.touches.toFixed(1)}</b></span>` +
          `<span class="chip">entropy <b>${st.entropy.toFixed(2)}</b></span>` +
          `<span class="chip">vs scripted bot <b>${st.vsScripted.scored}–${st.vsScripted.conceded}</b></span>` +
          `<span class="chip">empty net <b>${st.emptyNet}/20</b></span>`;
        stageChart.marker = st.episodes;
        stageChart2.marker = st.episodes;
        stageChart.draw();
        stageChart2.draw();
      }

      scrub.addEventListener('input', () => showStage(+scrub.value));
      document.getElementById('stage-prev').addEventListener('click', () => showStage(stageIndex - 1));
      document.getElementById('stage-next').addEventListener('click', () => showStage(stageIndex + 1));

      let playing = false, playTimer = 0;
      const btnPlay = document.getElementById('stage-play');
      btnPlay.addEventListener('click', () => {
        playing = !playing;
        btnPlay.textContent = playing ? '⏸ Pause timelapse' : '▶ Play the whole run';
        btnPlay.classList.toggle('primary', !playing);
        if (playing && stageIndex >= stages.length - 1) showStage(0);
      });

      pills(document.getElementById('stage-mode'), [
        { value: 'selfplay', label: 'Against itself' },
        { value: 'vs-scripted', label: 'Against the scripted bot' },
        { value: 'solo', label: 'Empty net' },
      ], 'selfplay', (v) => { stageMatch.mode = v; stageMatch.arena.reset(); });

      document.getElementById('stage-adopt').addEventListener('click', () => {
        trainer.policy = window.ML.MLP.fromJSON(stages[stageIndex].policy);
        trainer.frozen = window.ML.MLP.fromJSON(stages[stageIndex].policy);
        trainer.episodes = stages[stageIndex].episodes;
        trainer.spread = stages[stageIndex].spread ?? 1;
        // Badges are for agents you trained, so a borrowed policy disables them
        // until you reset and do it yourself.
        trainer.fromCheckpoint = true;
        refreshStats();
        flash(document.getElementById('stage-adopt-out'),
              `Loaded the ${stages[stageIndex].episodes.toLocaleString()}-episode agent into the trainer below.`);
      });

      showStage(stages.length - 1);          // open on the finished agent

      rafLoop((dt) => {
        stageMatch.update(dt);
        stageMatch.render();
        if (playing) {
          playTimer += dt;
          if (playTimer > 3.5) {
            playTimer = 0;
            if (stageIndex >= stages.length - 1) {
              playing = false;
              btnPlay.textContent = '▶ Play the whole run';
              btnPlay.classList.add('primary');
            } else showStage(stageIndex + 1);
          }
        }
      }).start();
    }

    /* ==================================================================
       PANEL 3 — training
       ================================================================== */
    const match = new Match(document.getElementById('train-canvas'), { rewards, trainer, mode: 'selfplay' });
    match.active = true;
    bindKeys(match);

    match.onGoal = (team, mode) => {
      if (mode === 'human' && team === 1) {
        achieve('rocket-human', 'You scored against an agent you trained');
      }
    };

    const setStat = statGrid(document.getElementById('train-stats'), [
      'episodes', 'updates', 'goal rate', 'touches/ep', 'reward/ep', 'value error', 'difficulty', 'eps/sec',
    ]);

    const chartReward = new LineChart(document.getElementById('chart-reward'), {
      height: 150, xLabel: 'episodes', zeroLine: true,
      series: [{ name: 'reward / episode', color: '#4da3ff' }],
    });
    const chartSkill = new LineChart(document.getElementById('chart-skill'), {
      height: 150, xLabel: 'episodes', yMin: 0,
      series: [
        { name: 'goal rate', color: '#38d39f' },
        { name: 'touches / ep', color: '#ffb547' },
      ],
    });
    // Entropy has a hard ceiling of ln(9) = 2.20 (a completely undecided policy),
    // so it gets a fixed axis: you can read "how much is it still exploring?" at a glance.
    const chartLearn = new LineChart(document.getElementById('chart-learn'), {
      height: 150, xLabel: 'episodes', yMin: 0, yMax: 2.3,
      series: [
        { name: 'entropy', color: '#7c5cff' },
        { name: 'policy change (KL×20)', color: '#ff6b6b' },
      ],
    });

    let training = false, budget = 14, epsAtLastSecond = 0, lastSecond = performance.now();
    const btnTrain = document.getElementById('btn-train');
    const btnReset = document.getElementById('btn-reset');
    const btnEval = document.getElementById('btn-eval');
    const evalOut = document.getElementById('eval-out');

    btnTrain.addEventListener('click', () => {
      training = !training;
      btnTrain.textContent = training ? '⏸ Pause training' : '▶ Start training';
      btnTrain.classList.toggle('primary', !training);
    });
    btnReset.addEventListener('click', () => {
      trainer.reset(true);
      trainer.fromCheckpoint = false;
      chartReward.clear(); chartSkill.clear(); chartLearn.clear();
      match.score = [0, 0];
      evalOut.textContent = '';
      refreshStats();
    });
    btnEval.addEventListener('click', () => {
      evalOut.textContent = 'playing 20 evaluation episodes…';
      setTimeout(() => {
        const vs = trainer.evaluate(20, 'scripted');
        const solo = trainer.evaluate(20, 'none');
        if (vs.scored > vs.conceded && !trainer.fromCheckpoint) {
          achieve('rocket-scores', `Beat the scripted bot ${vs.scored}–${vs.conceded} over 20 episodes`);
        }
        evalOut.innerHTML =
          `<b>vs scripted bot:</b> won ${vs.scored} – ${vs.conceded} over 20 episodes ` +
          `(${vs.touchesPerEp.toFixed(1)} touches/ep) &nbsp;·&nbsp; ` +
          `<b>empty net:</b> scored in ${solo.scored}/20`;
      }, 30);
    });

    const trainCtl = document.getElementById('train-controls-2');
    pills(trainCtl, [
      { value: 6, label: 'Slow (smooth UI)' },
      { value: 14, label: 'Normal' },
      { value: 34, label: 'Turbo' },
    ], 14, (v) => { budget = v; });

    pills(document.getElementById('opponent-mode'), [
      { value: 'self', label: 'Self-play (both cars learn)' },
      { value: 'past', label: 'vs frozen past self' },
      { value: 'scripted', label: 'vs scripted bot' },
      { value: 'none', label: 'Empty net (no opponent)' },
    ], 'self', (v) => {
      trainer.opponent = v;
      match.mode = v === 'none' ? 'solo' : v === 'scripted' ? 'vs-scripted' : 'selfplay';
    });

    const hp = document.getElementById('hp-controls');
    slider(hp, {
      label: 'learning rate', min: 0.0005, max: 0.02, step: 0.0005, value: trainer.hp.lr,
      format: (v) => v.toFixed(4), desc: 'How big a step to take on each update. Too high and the policy thrashes; too low and it crawls.',
      onInput: (v) => { trainer.hp.lr = v; },
    });
    slider(hp, {
      label: 'entropy bonus', min: 0, max: 0.05, step: 0.001, value: trainer.hp.entropyBonus,
      format: (v) => v.toFixed(3), desc: 'Pressure to keep trying different actions. Set it to zero and the agent may lock onto one habit forever.',
      onInput: (v) => { trainer.hp.entropyBonus = v; },
    });
    slider(hp, {
      label: 'discount γ', min: 0.8, max: 0.999, step: 0.001, value: trainer.hp.gamma,
      format: (v) => v.toFixed(3), desc: 'How much a reward one step in the future is worth. Low γ = short-sighted, high γ = plans ahead but learns slower.',
      onInput: (v) => { trainer.hp.gamma = v; },
    });
    slider(hp, {
      label: 'PPO clip ε', min: 0.05, max: 0.5, step: 0.01, value: trainer.hp.clip,
      format: (v) => v.toFixed(2), desc: 'The trust region: how far the new policy may move away from the one that gathered the data.',
      onInput: (v) => { trainer.hp.clip = v; },
    });
    slider(hp, {
      label: 'episodes per update', min: 2, max: 24, step: 1, value: trainer.hp.batchEpisodes,
      format: (v) => v.toFixed(0), desc: 'More episodes = a less noisy gradient but fewer updates per minute.',
      onInput: (v) => { trainer.hp.batchEpisodes = v; },
    });
    checkbox(hp, 'Curriculum: start with the ball nearby, widen as it improves', trainer.hp.curriculum,
      (v) => { trainer.hp.curriculum = v; if (!v) trainer.spread = 1; });
    checkbox(hp, 'Show the training match (uncheck for maximum speed)', true, (v) => { showMatch = v; });
    let showMatch = true;

    /* --- save / load --- */
    document.getElementById('btn-save').addEventListener('click', () => {
      localStorage.setItem('rl-rocket-agent', JSON.stringify(trainer.toJSON()));
      flash(document.getElementById('save-out'), 'Saved to this browser.');
    });
    document.getElementById('btn-load').addEventListener('click', () => {
      const raw = localStorage.getItem('rl-rocket-agent');
      if (!raw) return flash(document.getElementById('save-out'), 'Nothing saved yet.');
      trainer.loadJSON(JSON.parse(raw));
      refreshStats();
      flash(document.getElementById('save-out'), 'Loaded a saved agent.');
    });
    const btnPre = document.getElementById('btn-pretrained');
    if (btnPre) {
      if (!window.ML_ROCKET_STAGES) btnPre.disabled = true;
      btnPre.addEventListener('click', () => {
        const st = window.ML_ROCKET_STAGES.stages[window.ML_ROCKET_STAGES.stages.length - 1];
        trainer.policy = MLP.fromJSON(st.policy);
        trainer.frozen = MLP.fromJSON(st.policy);
        trainer.episodes = st.episodes;
        trainer.spread = 1;
        trainer.fromCheckpoint = true;
        refreshStats();
        flash(document.getElementById('save-out'), 'Loaded the fully trained agent.');
      });
    }

    document.getElementById('btn-export').addEventListener('click', () => {
      const blob = new Blob([JSON.stringify(trainer.toJSON())], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'rocket-agent.json';
      a.click();
      URL.revokeObjectURL(a.href);
    });
    function flash(el, msg) { el.textContent = msg; setTimeout(() => (el.textContent = ''), 2500); }

    /* ==================================================================
       PANEL 4 — inside the policy
       ================================================================== */
    const netCanvas = document.getElementById('net-canvas');
    const netCtx = fit(netCanvas, 190);
    const actionGrid = document.getElementById('action-grid');
    const actCells = ACTIONS.map((a, i) => {
      const d = document.createElement('div');
      d.className = 'act-cell';
      d.innerHTML = `<i></i><span>${ACTION_LABELS[i]}</span><b>0%</b>`;
      actionGrid.appendChild(d);
      return { fill: d.querySelector('i'), num: d.querySelector('b'), el: d };
    });
    const valueEl = document.getElementById('value-readout');

    function drawNet() {
      const ctx = netCtx, W = ctx._cssW, H = ctx._cssH;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0a0f19'; ctx.fillRect(0, 0, W, H);
      const acts = match.lastAct;
      if (!acts) return;
      const cols = [
        { a: acts[0], grid: [4, 6], label: 'observation (24)' },
        { a: acts[1], grid: [8, 8], label: 'hidden 1' },
        { a: acts[2], grid: [8, 8], label: 'hidden 2' },
        { a: acts[3], grid: [3, 3], label: 'action logits' },
      ];
      const pad = 16, colW = (W - pad * 2) / cols.length;
      cols.forEach((c, ci) => {
        const [rows, colsN] = c.grid;
        const size = Math.min((colW - 26) / colsN, (H - 46) / rows);
        const ox = pad + ci * colW + (colW - size * colsN) / 2;
        const oy = 24;
        for (let i = 0; i < c.a.length; i++) {
          const r = (i / colsN) | 0, cc = i % colsN;
          ctx.fillStyle = diverging(c.a[i], ci === 3 ? 1.2 : 1);
          ctx.fillRect(ox + cc * size, oy + r * size, size - 1.5, size - 1.5);
        }
        ctx.fillStyle = '#6d7f9c';
        ctx.font = '10px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(c.label, pad + ci * colW + colW / 2, 6);
        if (ci < cols.length - 1) {
          ctx.strokeStyle = '#26314a';
          ctx.beginPath();
          const x = pad + (ci + 1) * colW - 6;
          ctx.moveTo(x - 6, H / 2); ctx.lineTo(x + 2, H / 2);
          ctx.moveTo(x - 1, H / 2 - 4); ctx.lineTo(x + 2, H / 2); ctx.lineTo(x - 1, H / 2 + 4);
          ctx.stroke();
        }
      });
    }

    function drawPolicyReadout() {
      const probs = match.lastProbs;
      if (!probs) return;
      const best = argmax(probs);
      actCells.forEach((c, i) => {
        c.fill.style.width = (probs[i] * 100).toFixed(0) + '%';
        c.num.textContent = (probs[i] * 100).toFixed(0) + '%';
        c.el.classList.toggle('best', i === best);
      });
      const v = match.lastValue;
      valueEl.innerHTML = v === null ? '' :
        `V(s) = <b>${v.toFixed(2)}</b> <span class="muted">— the critic's guess at
        the total future reward from this exact moment</span>`;
    }

    /* ==================================================================
       PANEL 5 — play against your agent
       ================================================================== */
    pills(document.getElementById('play-mode'), [
      { value: 'selfplay', label: 'Agent vs agent' },
      { value: 'vs-scripted', label: 'Agent vs scripted bot' },
      { value: 'human', label: 'You vs agent' },
      { value: 'solo', label: 'Agent, empty net' },
    ], 'selfplay', (v) => { match.mode = v; match.arena.reset(); });
    checkbox(document.getElementById('play-mode'), 'Greedy (always take the most likely action)', false,
      (v) => { match.greedy = v; });
    slider(document.getElementById('play-mode'), {
      label: 'match speed', min: 0.25, max: 3, step: 0.25, value: 1,
      format: (v) => v.toFixed(2) + '×', onInput: (v) => { match.speed = v; },
    });

    /* ==================================================================
       main loop: train a slice, then draw everything
       ================================================================== */
    function refreshStats() {
      const h = trainer.history[trainer.history.length - 1];
      setStat('episodes', trainer.episodes.toLocaleString());
      setStat('updates', trainer.updates.toLocaleString());
      setStat('difficulty', (trainer.spread * 100).toFixed(0) + '%');
      if (!h) return;
      setStat('goal rate', h.scoreRate.toFixed(2), h.scoreRate > 0.5 ? 'good' : '');
      setStat('touches/ep', h.touches.toFixed(1));
      if (h.touches >= 3 && training && !trainer.fromCheckpoint) {
        achieve('rocket-touch', `${h.touches.toFixed(1)} touches per episode`);
      }
      setStat('reward/ep', h.reward.toFixed(1), h.reward > 0 ? 'good' : 'bad');
      setStat('value error', h.valueLoss.toFixed(1));
    }

    const loop = rafLoop((dt) => {
      if (training) {
        const rows = trainer.trainSlice(budget);
        for (const r of rows) {
          chartReward.push(r.episodes, [r.reward]);
          chartSkill.push(r.episodes, [r.scoreRate, r.touches]);
          chartLearn.push(r.episodes, [r.entropy, Math.min(r.kl * 20, 2.3)]);
        }
        if (rows.length) { chartReward.draw(); chartSkill.draw(); chartLearn.draw(); refreshStats(); }
      }
      const nowT = performance.now();
      if (nowT - lastSecond > 1000) {
        setStat('eps/sec', (((trainer.episodes - epsAtLastSecond) * 1000) / (nowT - lastSecond)).toFixed(1));
        epsAtLastSecond = trainer.episodes;
        lastSecond = nowT;
      }
      if (showMatch) {
        match.update(dt);
        match.render();
        drawNet();
        drawPolicyReadout();
      }
    });
    loop.start();
    refreshStats();
  });
})();
