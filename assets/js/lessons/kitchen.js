/* ==========================================================================
   kitchen.js — work a shift with an agent, then see who it can work with.
   ========================================================================== */

;(function () {
  'use strict';
  const { Kitchen, KITCHEN: K, KitchenTrainer, MLP, softmax, sampleFrom, mulberry32,
          hidpi, chrome, nextLinks, achieve, pills, slider, statGrid, rafLoop, dpad,
          LineChart } = window.ML;
  const arcade = window.ML.arcade;

  const COL = {
    floor: '#131c2e', counter: '#243250', onion: '#8d6b3f', plate: '#9fb0cc',
    pot: '#c0533f', hatch: '#38d39f', you: '#4da3ff', mate: '#ff9f45',
  };
  const TICK_MS = 140;                 // one kitchen tick; slow enough to play

  document.addEventListener('DOMContentLoaded', () => {
    chrome('kitchen', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.kitchen);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'kitchen');
    nextLinks(document.getElementById('next-links'), 'fighter', 'capstone', '../');

    const rand = mulberry32((Math.random() * 1e9) | 0);
    const fx = arcade.fx();
    const SHIPPED = window.ML_KITCHEN || null;

    /* ================================================================
       Partners
       ================================================================ */
    const agents = {};
    if (SHIPPED && SHIPPED.agents) {
      for (const a of SHIPPED.agents) agents['seed' + a.seed] = MLP.fromJSON(a.net);
    }
    const seedKeys = Object.keys(agents);
    let partnerKey = seedKeys[0] || 'scripted';

    const obs = new Float32Array(K.OBS);
    function agentAction(policy, kitchen, i) {
      kitchen.observe(i, obs);
      return sampleFrom(softmax(policy.forward(obs)), rand);
    }
    function partnerAction(kitchen) {
      if (partnerKey === 'scripted') return K.scriptedCook(kitchen, 1, rand);
      const p = agents[partnerKey];
      return p ? agentAction(p, kitchen, 1) : 4;
    }

    /* ================================================================
       The shift
       ================================================================ */
    const SHIFT_TICKS = 300;
    let kitchen = new Kitchen({ maxTicks: SHIFT_TICKS });
    let running = false, acc = 0, myAction = 4, pendingInteract = false;
    const keys = {};

    const hud = arcade.hud(document.getElementById('k-hud'), [
      { name: 'served', label: 'soups', value: '0' },
      { name: 'time', label: 'ticks left', value: String(SHIFT_TICKS) },
      { name: 'holding', label: 'you hold', value: 'nothing' },
      { name: 'best', label: 'best shift', value: '—' },
    ]);
    arcade.soundToggle(document.getElementById('k-sound'));
    const setShift = statGrid(document.getElementById('shift-stats'),
      ['soups', 'collisions', 'pot', 'partner holds']);

    const MEDALS = { gold: 8, silver: 5, bronze: 2 };
    function showBest() {
      const b = arcade.readBest('kitchen-shift');
      hud.set('best', b === null ? '—' : b);
      const t = document.getElementById('k-medals');
      if (!t) return;
      t.innerHTML = ['gold', 'silver', 'bronze'].map((m) =>
        `<span class="${b !== null && b >= MEDALS[m] ? 'won' : ''}">` +
        `${arcade.MEDALS[m].icon} <b>${MEDALS[m]} soups</b></span>`).join('') +
        '<span>the by-the-book cook is the easiest partner, and that is the point</span>';
    }

    function startShift() {
      kitchen = new Kitchen({ maxTicks: SHIFT_TICKS });
      kitchen.reset();
      running = true;
      acc = 0;
      document.getElementById('k-result').innerHTML = '';
      document.getElementById('shift-note').textContent = 'cooking';
      document.getElementById('btn-start').textContent = 'Restart the shift';
      arcade.sfx('start');
      draw();
    }

    function endShift() {
      running = false;
      const served = kitchen.served;
      const res = arcade.resultCard(served, {
        tiers: MEDALS, bestKey: 'kitchen-shift',
        format: (v) => `${v} soup${v === 1 ? '' : 's'}`,
        note: `${kitchen.collisions} collisions with your partner.`,
      });
      document.getElementById('k-result').innerHTML = res.html;
      document.getElementById('shift-note').textContent = 'shift over';
      showBest();
      arcade.sfx(res.tier === 'gold' ? 'win' : served > 0 ? 'score' : 'fail');
      if (served >= 1) achieve('kitchen-served', `${served} soups with the ${partnerLabel()}`);
      if (res.tier === 'gold') achieve('kitchen-gold', `${served} soups in one shift`);
    }

    const partnerLabel = () =>
      partnerKey === 'scripted' ? 'by-the-book cook' : `self-play agent (${partnerKey})`;

    /* ================================================================
       Input
       ================================================================ */
    const KEYMAP = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3,
                     w: 0, d: 1, s: 2, a: 3 };
    window.addEventListener('keydown', (e) => {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      if (e.key === ' ') { e.preventDefault(); pendingInteract = true; return; }
      const k = KEYMAP[e.key];
      if (k === undefined) return;
      e.preventDefault();
      keys[k] = true;
    });
    window.addEventListener('keyup', (e) => {
      const k = KEYMAP[e.key];
      if (k !== undefined) keys[k] = false;
    });
    dpad(document.getElementById('k-pad'), {
      discrete: true,
      onPress: (dir) => { keys[{ up: 0, right: 1, down: 2, left: 3 }[dir]] = true; },
    });
    // The shared dpad has no spare button, and space is not available on a
    // phone, so "use" gets its own control that works everywhere.
    document.getElementById('btn-use').addEventListener('click', () => { pendingInteract = true; });

    function readMyAction() {
      if (pendingInteract) { pendingInteract = false; return K.INTERACT; }
      for (const k of [0, 1, 2, 3]) if (keys[k]) return k;
      return 4;
    }

    /* ================================================================
       Drawing
       ================================================================ */
    const canvas = document.getElementById('kitchen');
    let ctx, cell = 40;
    function resize() {
      canvas.style.width = '';
      const w = canvas.clientWidth || canvas.parentElement.clientWidth || 420;
      ctx = hidpi(canvas, w, Math.round((w * K.H) / K.W));
      cell = ctx._cssW / K.W;
    }

    function draw() {
      if (!ctx) return;
      ctx.clearRect(0, 0, ctx._cssW, ctx._cssH);
      for (let y = 0; y < K.H; y++) {
        for (let x = 0; x < K.W; x++) {
          const t = kitchen.tileAt(x, y);
          ctx.fillStyle = t === K.FLOOR ? COL.floor
            : t === K.ONIONS ? COL.onion : t === K.PLATES ? COL.plate
            : t === K.POT ? COL.pot : t === K.HATCH ? COL.hatch : COL.counter;
          ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
          if (t === K.POT) {
            ctx.fillStyle = kitchen.potReady ? '#ffd166' : 'rgba(255,255,255,.75)';
            ctx.font = `${cell * 0.3}px ui-monospace, monospace`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(kitchen.potReady ? 'READY' : `${kitchen.potOnions}/${K.NEEDED}`,
                         x * cell + cell / 2, y * cell + cell / 2);
          }
        }
      }
      kitchen.cooks.forEach((c, i) => {
        ctx.beginPath();
        ctx.arc(c.x * cell + cell / 2, c.y * cell + cell / 2, cell * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? COL.you : COL.mate;
        ctx.fill();
        if (c.hold !== K.HOLD.NOTHING) {
          ctx.beginPath();
          ctx.arc(c.x * cell + cell * 0.72, c.y * cell + cell * 0.28, cell * 0.13, 0, Math.PI * 2);
          ctx.fillStyle = c.hold === K.HOLD.ONION ? COL.onion
            : c.hold === K.HOLD.PLATE ? COL.plate : '#ffd166';
          ctx.fill();
        }
      });
    }

    /* ================================================================
       The pairing table, from the shipped run
       ================================================================ */
    (function pairings() {
      const body = document.querySelector('#pairing-table tbody');
      const note = document.getElementById('pairing-note');
      if (!SHIPPED || !SHIPPED.pairings) {
        body.innerHTML = '<tr><td>Run <code>node tools/train-kitchen.cjs</code> to train the ' +
          'agents and fill this in.</td></tr>';
        return;
      }
      const p = SHIPPED.pairings;
      const seeds = SHIPPED.agents.map((a) => a.seed);
      const rows = [];
      for (const s of seeds) rows.push([`Agent ${s} with a copy of itself`, p['self_' + s], 'good']);
      if (p.cross !== undefined) rows.push([`Agent ${seeds[0]} with agent ${seeds[1]}`, p.cross, 'bad']);
      for (const s of seeds) rows.push([`Agent ${s} with a by-the-book cook`, p['scripted_' + s], 'bad']);
      rows.push(['Two by-the-book cooks', p.scripted_pair, '']);

      const max = Math.max(...rows.map((r) => r[1] || 0), 1);
      body.innerHTML = '<tr><th>pairing</th><th></th><th>soups per shift</th></tr>' +
        rows.map(([label, v, cls]) =>
          `<tr class="${cls}"><td>${label}</td>` +
          `<td style="width:38%"><span class="bar"><i style="width:${((v || 0) / max * 100).toFixed(0)}%"></i></span></td>` +
          `<td class="n">${(v || 0).toFixed(2)}</td></tr>`).join('');

      // State the advantage that vanishes, not just the raw drop: "25% worse"
      // undersells it, because the baseline they fall back to is the pair of
      // plain scripted cooks they were supposed to be better than.
      const best = Math.max(...seeds.map((s) => p['self_' + s]));
      const scriptAvg = seeds.reduce((a, s) => a + p['scripted_' + s], 0) / seeds.length;
      const base = p.scripted_pair || 1;
      const edgeAlone = best - base;
      const edgeStranger = scriptAvg - base;
      const kept = edgeAlone > 0 ? Math.max(0, (edgeStranger / edgeAlone) * 100) : 0;
      note.textContent =
        `Each row is the average of ${SHIPPED.evalRuns} shifts of ${SHIPPED.ticks} ticks, measured ` +
        `when the agents were trained. With a copy of itself the best agent is ` +
        `${edgeAlone.toFixed(2)} soups a shift better than two by-the-book cooks. With a partner it ` +
        `has never met it keeps ${kept.toFixed(0)}% of that edge` +
        (p.cross !== undefined ? `, and two of these agents together manage ${p.cross.toFixed(2)}.` : '.');
    })();

    /* ================================================================
       Train your own
       ================================================================ */
    const trainer = new KitchenTrainer({ seed: 21 });
    let training = false;
    const setStat = statGrid(document.getElementById('train-stats'),
      ['episodes', 'updates', 'soups per shift', 'with a stranger']);
    const chart = new LineChart(document.getElementById('k-chart'), {
      height: 170, xLabel: 'episodes', yMin: 0,
      series: [{ name: 'soups per shift, paired with itself', color: '#38d39f' }],
    });
    const runControls = window.ML.runBar(document.getElementById('run-bar'), {
      seed: trainer.seed,
      onSeed: (s) => { trainer.seed = s; trainer.reset(true); resetTrainUI(); },
      onPin: () => chart.pin(),
      onUnpin: () => chart.unpin(),
    });

    function resetTrainUI() {
      chart.clear(); chart.draw();
      setStat('episodes', 0); setStat('updates', 0);
      setStat('soups per shift', '—'); setStat('with a stranger', '—');
    }

    document.getElementById('btn-train').addEventListener('click', () => {
      training = !training;
      const b = document.getElementById('btn-train');
      b.textContent = training ? '⏸ Pause' : '▶ Start self-play';
      b.classList.toggle('primary', !training);
    });
    document.getElementById('btn-test').addEventListener('click', () => {
      const self = trainer.evaluate(trainer.policy, trainer.policy, 6).served;
      const stranger = trainer.evaluate(trainer.policy,
        (k, i, r) => K.scriptedCook(k, i, r), 6).served;
      setStat('soups per shift', self.toFixed(2));
      setStat('with a stranger', stranger.toFixed(2));
      if (self > 1 && stranger < self * 0.75) {
        achieve('kitchen-collapse', `${self.toFixed(1)} with itself, ${stranger.toFixed(1)} with a stranger`);
      }
    });
    document.getElementById('btn-train-reset').addEventListener('click', () => {
      trainer.reset(true);
      training = false;
      const b = document.getElementById('btn-train');
      b.textContent = '▶ Start self-play';
      b.classList.add('primary');
      resetTrainUI();
    });

    const hpHost = document.getElementById('k-hp');
    slider(hpHost, {
      label: 'entropy bonus', min: 0, max: 0.02, step: 0.001, value: trainer.hp.entropy,
      format: (v) => v.toFixed(3), onInput: (v) => { trainer.hp.entropy = v; },
      desc: 'How hard it is pushed to keep trying things. At zero it commits early and never serves a soup.',
    });
    slider(hpHost, {
      label: 'learning rate', min: 0.001, max: 0.03, step: 0.001, value: trainer.hp.lr,
      format: (v) => v.toFixed(3), onInput: (v) => { trainer.hp.lr = v; },
    });
    slider(hpHost, {
      label: 'shift length', min: 100, max: 400, step: 50, value: trainer.hp.ticks,
      onInput: (v) => { trainer.hp.ticks = v | 0; trainer.kitchen = new Kitchen({ maxTicks: v | 0 }); },
      desc: 'Longer shifts give more chances to complete the chain, and take proportionally longer to train.',
    });

    /* ================================================================
       Partner picker
       ================================================================ */
    const options = seedKeys.map((k) => ({ value: k, label: `Self-play agent (${k})` }))
      .concat([{ value: 'scripted', label: 'By-the-book cook' }]);
    pills(document.getElementById('partner-mode'), options, partnerKey, (v) => {
      partnerKey = v;
      document.getElementById('partner-note').textContent = v === 'scripted'
        ? 'Follows the recipe and routes around you. It does not adapt to you at all, which is '
          + 'exactly why it is the easier partner: there is no private convention to guess.'
        : 'Trained by playing thousands of shifts against a copy of itself. It is expecting a '
          + 'partner who moves the way it does, and you are not that.';
    });
    document.getElementById('partner-note').textContent =
      seedKeys.length
        ? 'Trained by playing thousands of shifts against a copy of itself. It is expecting a '
          + 'partner who moves the way it does, and you are not that.'
        : 'No trained agents are shipped yet — run tools/train-kitchen.cjs. The by-the-book cook '
          + 'works regardless.';

    document.getElementById('btn-start').addEventListener('click', startShift);

    /* ================================================================
       Loop
       ================================================================ */
    resize();
    window.addEventListener('resize', () => { resize(); draw(); });
    showBest();
    resetTrainUI();
    draw();

    let lastChart = 0;
    rafLoop((dt) => {
      if (running) {
        acc += dt * 1000;
        while (acc >= TICK_MS && running) {
          acc -= TICK_MS;
          const before = kitchen.served;
          kitchen.step([readMyAction(), partnerAction(kitchen)], rand);
          for (const k of [0, 1, 2, 3]) keys[k] = false;      // one square per press
          if (kitchen.served > before) {
            arcade.sfx('score');
            fx.burst(ctx._cssW * 0.9, ctx._cssH * 0.3,
                     { count: 24, speed: 120, colors: ['#38d39f', '#ffd166'] });
          }
          if (kitchen.tick >= kitchen.maxTicks) endShift();
        }
        hud.set('served', kitchen.served);
        hud.set('time', Math.max(0, kitchen.maxTicks - kitchen.tick));
        hud.set('holding', K.HOLD_NAME[kitchen.cooks[0].hold]);
        setShift('soups', kitchen.served);
        setShift('collisions', kitchen.collisions);
        setShift('pot', kitchen.potReady ? 'ready' : `${kitchen.potOnions}/${K.NEEDED}`);
        setShift('partner holds', K.HOLD_NAME[kitchen.cooks[1].hold]);
        draw();
      }
      if (training) {
        trainer.trainSlice(80);
        setStat('episodes', trainer.episodes.toLocaleString());
        setStat('updates', trainer.updates.toLocaleString());
        if (trainer.episodes - lastChart >= 80) {
          lastChart = trainer.episodes;
          const s = trainer.evaluate(trainer.policy, trainer.policy, 3).served;
          setStat('soups per shift', s.toFixed(2));
          chart.push(trainer.episodes, [s]);
          chart.draw();
        }
      }
      if (fx.busy && ctx) {
        fx.begin(ctx);
        fx.end(ctx, dt, ctx._cssW, ctx._cssH);
      }
    }).start();

    // handy from the console, and used by the tests
    window.__kitchen = {
      get kitchen() { return kitchen; }, trainer, agents, K,
      start: startShift, setPartner: (p) => { partnerKey = p; },
      step: (a) => kitchen.step([a, partnerAction(kitchen)], rand),
    };
  });
})();
