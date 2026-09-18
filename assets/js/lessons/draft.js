/* ==========================================================================
   draft.js — draft against a model that learned the rules from match results.
   ========================================================================== */

;(function () {
  'use strict';
  const { Roster, DRAFT, DraftModel, DRAFT_HP, MLP, mulberry32, hidpi,
          chrome, nextLinks, achieve, pills, slider, statGrid, runBar,
          LineChart } = window.ML;
  const arcade = window.ML.arcade;

  document.addEventListener('DOMContentLoaded', () => {
    chrome('draft', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.draft);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'draft');
    nextLinks(document.getElementById('next-links'), 'factory', 'capstone', '../');

    const SHIPPED = window.ML_DRAFT || null;
    const roster = new Roster({ seed: SHIPPED ? SHIPPED.rosterSeed : 99 });
    const rand = mulberry32((Math.random() * 1e9) | 0);
    const fx = arcade.fx();

    /* The two shipped models, rebuilt from their weights. Each one is wrapped
       in a DraftModel so it regenerates the identical held-out matches, which
       is what makes the scatter plot below comparable between them. */
    const shipped = {};
    if (SHIPPED) {
      for (const m of SHIPPED.models) {
        const dm = new DraftModel({ roster, kind: m.kind, hp: SHIPPED.hp, seed: SHIPPED.seed });
        dm.net = MLP.fromJSON(m.net);
        dm.meta = m;
        shipped[m.kind] = dm;
      }
    }
    let opponent = shipped.deep || shipped.linear || null;
    let opponentKind = opponent ? opponent.kind : 'none';

    /* ================================================================
       The draft
       ================================================================ */
    const teamA = [], teamB = [];
    const taken = new Set();
    let over = false, drafting = false;

    const elPool = document.getElementById('pool');
    const elA = document.getElementById('team-a');
    const elB = document.getElementById('team-b');
    const elBar = document.getElementById('odds-bar');
    const elOdds = document.getElementById('odds-text');
    const elNote = document.getElementById('draft-note');
    const elThink = document.getElementById('think-list');
    const elResult = document.getElementById('d-result');

    const hud = arcade.hud(document.getElementById('d-hud'), [
      { name: 'streak', label: 'win streak', value: '0' },
      { name: 'record', label: 'record', value: '0–0' },
      { name: 'edge', label: 'your edge', value: '–' },
    ]);
    arcade.soundToggle(document.getElementById('d-sound'));

    let wins = 0, losses = 0, streak = 0;

    const buttons = roster.champions.map((c) => {
      const b = document.createElement('button');
      b.className = 'champ';
      b.type = 'button';
      b.innerHTML = `<b>${c.name}</b><span>${c.role}</span>`;
      b.addEventListener('click', () => humanPick(c.id));
      elPool.appendChild(b);
      return b;
    });

    function renderTeam(el, ids, who) {
      el.innerHTML = '';
      for (let i = 0; i < DRAFT.TEAM_SIZE; i++) {
        const li = document.createElement('li');
        if (ids[i] === undefined) {
          li.className = 'empty';
          li.textContent = '·';
        } else {
          const c = roster.champions[ids[i]];
          li.textContent = `${c.name} — ${c.role}`;
        }
        el.appendChild(li);
      }
      el.dataset.count = String(ids.length);
      void who;
    }

    function renderOdds() {
      if (!teamA.length && !teamB.length) {
        elBar.style.width = '50%';
        elOdds.textContent = 'pick to begin';
        return;
      }
      const p = roster.winProb(teamA, teamB);
      elBar.style.width = (p * 100).toFixed(1) + '%';
      elOdds.textContent = `${(p * 100).toFixed(0)}% you  ·  ${(100 - p * 100).toFixed(0)}% it`;
      hud.set('edge', teamA.length + teamB.length < 4 ? '–'
        : (p >= 0.5 ? '+' : '') + ((p - 0.5) * 200).toFixed(0) + '%');
    }

    function renderThinking() {
      if (!opponent || over) { elThink.innerHTML = ''; return; }
      const rows = [];
      for (let i = 0; i < roster.n; i++) {
        if (taken.has(i)) continue;
        rows.push({ i, p: 1 - opponent.predict(teamA, teamB.concat([i])) });
      }
      rows.sort((a, b) => b.p - a.p);
      elThink.innerHTML = rows.slice(0, 5).map((r) => {
        const c = roster.champions[r.i];
        return `<div class="row-line"><span>${c.name} <span class="muted">${c.role}</span></span>` +
               `<b class="mono">${(r.p * 100).toFixed(1)}%</b></div>`;
      }).join('');
    }

    function renderPool() {
      buttons.forEach((b, i) => { b.disabled = taken.has(i) || over || drafting; });
    }

    function refresh() {
      renderTeam(elA, teamA, 'a');
      renderTeam(elB, teamB, 'b');
      renderOdds();
      renderThinking();
      renderPool();
    }

    function humanPick(id) {
      if (over || drafting || taken.has(id)) return;
      teamA.push(id); taken.add(id);
      arcade.sfx('click');
      refresh();
      if (teamA.length >= DRAFT.TEAM_SIZE) { elNote.textContent = 'it is picking'; }
      drafting = true;
      renderPool();
      setTimeout(modelPick, 380);
    }

    function modelPick() {
      if (teamB.length < DRAFT.TEAM_SIZE) {
        let id;
        if (opponent) {
          // It is drafting for itself, so it wants its own win probability up,
          // which is one minus the number the model reports for team A.
          let best = -1, bestP = -Infinity;
          for (let i = 0; i < roster.n; i++) {
            if (taken.has(i)) continue;
            const p = 1 - opponent.predict(teamA, teamB.concat([i]));
            if (p > bestP) { bestP = p; best = i; }
          }
          id = best;
        } else {
          const free = [];
          for (let i = 0; i < roster.n; i++) if (!taken.has(i)) free.push(i);
          id = free[(rand() * free.length) | 0];
        }
        teamB.push(id); taken.add(id);
        arcade.sfx('tick');
      }
      drafting = false;
      if (teamA.length >= DRAFT.TEAM_SIZE && teamB.length >= DRAFT.TEAM_SIZE) finish();
      else { elNote.textContent = 'your pick'; refresh(); }
    }

    function finish() {
      over = true;
      refresh();
      const p = roster.winProb(teamA, teamB);
      const won = rand() < p;
      if (won) { wins++; streak++; } else { losses++; streak = 0; }
      hud.set('record', `${wins}–${losses}`);
      hud.set('streak', String(streak), streak >= 3 ? 'good' : '');
      hud.flash('record', won ? 'good' : 'bad');
      arcade.sfx(won ? 'win' : 'lose');
      elNote.textContent = 'draft over';

      const best = arcade.submitBest('draft-streak', streak);
      elResult.innerHTML =
        `<b class="result-score ${won ? 'good' : 'bad'}">${won ? 'You win' : 'You lose'}</b> ` +
        `<span class="muted">the simulator rolled against ${(p * 100).toFixed(0)}%</span> ` +
        (best.isNew ? '<span class="result-new">longest streak yet</span>' : '');
      const medals = document.getElementById('d-medals');
      const tier = arcade.medalFor(streak, [1, 3, 5]);
      medals.innerHTML = tier ? arcade.medalHTML(tier) : '';
      if (streak >= 3) achieve('draft-streak');
      if (won && opponentKind === 'linear') achieve('draft-linear');
      achieve('draft-played');
    }

    function newDraft() {
      teamA.length = 0; teamB.length = 0; taken.clear();
      over = false; drafting = false;
      elResult.innerHTML = '';
      elNote.textContent = 'your pick';
      refresh();
    }

    document.getElementById('btn-new').addEventListener('click', newDraft);
    document.getElementById('btn-auto').addEventListener('click', () => {
      if (over) newDraft();
      // Draft the rest of your team the way the truth says you should, so you
      // can see what a perfect drafter would have taken.
      const step = () => {
        if (over || teamA.length >= DRAFT.TEAM_SIZE) return;
        const { pick } = roster.bestPick(teamA, teamB, taken);
        if (pick < 0) return;
        humanPick(pick);
        setTimeout(step, 520);
      };
      step();
    });

    /* ================================================================
       Which model you are drafting against
       ================================================================ */
    const modelNote = document.getElementById('model-note');
    const NOTES = {
      deep: 'A network with two hidden layers. It can represent "these two are good together", ' +
            'and the measurements below say it did.',
      linear: 'One weight per champion, added up. It drafts strong champions and is blind to who ' +
              'they are standing next to — beatable, once you spot it.',
      none: 'No model available, so it is picking at random.',
    };
    if (Object.keys(shipped).length) {
      pills(document.getElementById('model-mode'), [
        { label: 'Network', value: 'deep' },
        { label: 'One weight each', value: 'linear' },
      ], 'deep', (v) => {
        opponent = shipped[v] || opponent;
        opponentKind = v;
        modelNote.textContent = NOTES[v];
        refresh();
      });
      modelNote.textContent = NOTES.deep;
    } else {
      modelNote.textContent = NOTES.none;
    }

    /* ================================================================
       The comparison table
       ================================================================ */
    const tbody = document.querySelector('#model-table tbody');
    if (SHIPPED) {
      const rows = [['', 'test loss', 'accuracy', 'correlation with truth']];
      const head = document.createElement('tr');
      head.innerHTML = rows[0].map((h) => `<th>${h}</th>`).join('');
      tbody.appendChild(head);
      const order = ['linear', 'deep'];
      const label = { linear: 'One weight per champion', deep: 'Two hidden layers' };
      const bestCorr = Math.max(...SHIPPED.models.map((m) => m.correlation));
      for (const kind of order) {
        const m = SHIPPED.models.find((x) => x.kind === kind);
        if (!m) continue;
        const tr = document.createElement('tr');
        if (m.correlation === bestCorr) tr.className = 'better';
        tr.innerHTML = `<td>${label[kind]}</td>` +
          `<td class="n">${m.loss.toFixed(3)}</td>` +
          `<td class="n">${(m.accuracy * 100).toFixed(1)}%</td>` +
          `<td class="n">${m.correlation.toFixed(3)}</td>`;
        tbody.appendChild(tr);
      }
      const m0 = SHIPPED.models[0];
      document.getElementById('model-table-note').textContent =
        `Both trained on the same ${SHIPPED.hp.trainMatches.toLocaleString()} matches and scored on the same ` +
        `${SHIPPED.hp.testMatches.toLocaleString()} it never saw. Always guessing the more common outcome ` +
        `would score ${(m0.baseline * 100).toFixed(1)}%.`;
    }

    /* ================================================================
       Interaction readout
       ================================================================ */
    const syn = document.getElementById('synergy-readout');
    if (SHIPPED) {
      const fmt = (v) => (v >= 0 ? '+' : '') + v.toFixed(3);
      const html = ['linear', 'deep'].map((kind) => {
        const m = SHIPPED.models.find((x) => x.kind === kind);
        if (!m || !m.inter) return '';
        const zero = Math.abs(m.inter.strong) < 1e-3 && Math.abs(m.inter.plain) < 1e-3;
        return `<div class="inter-block">
            <h4>${kind === 'deep' ? 'Two hidden layers' : 'One weight per champion'}</h4>
            <div class="row-line"><span>real synergy pairs</span><b class="mono ${zero ? '' : 'good'}">${fmt(m.inter.strong)}</b></div>
            <div class="row-line"><span>ordinary pairs</span><b class="mono">${fmt(m.inter.plain)}</b></div>
            <div class="row-line"><span>tracks the true interaction</span><b class="mono">${
              zero ? '<span class="muted">nothing to track</span>' : m.inter.correlation.toFixed(2)}</b></div>
          </div>`;
      }).join('');
      const deep = SHIPPED.models.find((x) => x.kind === 'deep');
      syn.innerHTML = html + (deep && deep.inter
        ? `<p class="muted" style="font-size:.8rem;margin:10px 0 0">Every pair of champions in the
             roster measured the same way — ${deep.inter.compared} of them, of which
             ${deep.inter.pairs} genuinely work together.</p>` : '');
    }

    /* ================================================================
       Predicted against actual
       ================================================================ */
    const calib = document.getElementById('calib');
    let calibKind = 'deep';
    function drawCalib() {
      const m = shipped[calibKind];
      if (!calib || !m) return;
      const W = calib.clientWidth || 360, H = Math.round(Math.min(260, W * 0.75));
      const ctx = hidpi(calib, W, H);
      const pad = 34;
      ctx.clearRect(0, 0, W, H);
      const x = (v) => pad + v * (W - pad - 10);
      const y = (v) => H - pad - v * (H - pad - 10);

      ctx.strokeStyle = 'rgba(124,148,190,.25)';
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, 10, W - pad - 10, H - pad - 10);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(56,211,159,.55)';
      ctx.beginPath(); ctx.moveTo(x(0), y(0)); ctx.lineTo(x(1), y(1)); ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = calibKind === 'deep' ? 'rgba(77,163,255,.36)' : 'rgba(255,159,69,.36)';
      const step = Math.max(1, Math.floor(m.test.length / 900));
      for (let i = 0; i < m.test.length; i += step) {
        const t = m.test[i];
        ctx.beginPath();
        ctx.arc(x(m.predict(t.a, t.b)), y(t.p), 1.7, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = 'rgba(160,178,209,.9)';
      ctx.font = '11px ui-monospace, monospace';
      ctx.fillText('predicted', W / 2 - 24, H - 10);
      ctx.save();
      ctx.translate(12, H / 2 + 18); ctx.rotate(-Math.PI / 2);
      ctx.fillText('true', 0, 0);
      ctx.restore();
      const hint = document.getElementById('calib-hint');
      if (hint && m.meta) hint.textContent = `correlation ${m.meta.correlation.toFixed(3)}`;
    }
    if (Object.keys(shipped).length > 1) {
      pills(document.getElementById('calib-pills'), [
        { label: 'Network', value: 'deep' },
        { label: 'One weight each', value: 'linear' },
      ], 'deep', (v) => { calibKind = v; drawCalib(); });
    }
    window.addEventListener('resize', drawCalib);

    /* ================================================================
       Train one yourself
       ================================================================ */
    const hp = Object.assign({}, DRAFT_HP);
    let live = null, loop = null;
    const setStat = statGrid(document.getElementById('train-stats'),
      ['train loss', 'held-out loss', 'accuracy', 'epochs']);
    const chart = new LineChart(document.getElementById('draft-chart'), {
      series: [{ name: 'train', color: '#4da3ff' }, { name: 'held out', color: '#ff9f45' }],
      yLabel: 'loss', xLabel: 'steps', height: 190,
    });
    const bar = runBar(document.getElementById('run-bar'), {
      seed: 5, onSeed: () => rebuild(), charts: () => [chart],
    });

    let kind = 'deep';
    pills(document.getElementById('draft-hp'), [
      { label: 'Two hidden layers', value: 'deep' },
      { label: 'One weight each', value: 'linear' },
    ], 'deep', (v) => { kind = v; rebuild(); });
    slider(document.getElementById('draft-hp'), {
      label: 'training matches', min: 2000, max: 20000, step: 1000, value: hp.trainMatches,
      format: (v) => v.toLocaleString(),
      desc: 'Fewer matches than the network can pay for, and the held-out loss turns and climbs.',
      onInput: (v) => { hp.trainMatches = v; rebuild(); },
    });
    window.ML.checkbox(document.getElementById('draft-hp'),
      'Train on unfinished drafts too', true, (v) => { hp.prefixShare = v ? 0.5 : 0; rebuild(); });
    slider(document.getElementById('draft-hp'), {
      label: 'weight decay', min: 0, max: 5, step: 1, value: 3,
      format: (v) => (v === 0 ? 'off' : [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1][v].toExponential(0)),
      desc: 'Pulls every weight towards zero. Drop it to 1e-4 and the held-out curve turns.',
      onInput: (v) => { hp.wd = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1][v]; rebuild(); },
    });
    slider(document.getElementById('draft-hp'), {
      label: 'learning rate', min: 0.001, max: 0.03, step: 0.001, value: hp.lr,
      format: (v) => v.toFixed(3),
      onInput: (v) => { hp.lr = v; if (live) live.hp.lr = v; },
    });

    let bestHeldOut = Infinity, smoothTrain = null;
    function rebuild() {
      stop();
      live = new DraftModel({ roster, kind, hp, seed: bar ? bar.seed : 5 });
      bestHeldOut = Infinity;
      smoothTrain = null;
      chart.clear();
      setStat('train loss', '–');
      setStat('held-out loss', '–');
      setStat('accuracy', '–');
      setStat('epochs', '0.0');
    }

    function tick() {
      if (!live) return;
      live.trainSlice(70);
      const e = live.evaluate();
      // A single minibatch of 64 is far too noisy to read against the held-out
      // curve, and the whole point of the panel is comparing the two.
      smoothTrain = smoothTrain === null ? live.lastLoss : smoothTrain * 0.8 + live.lastLoss * 0.2;
      chart.push(live.steps, [smoothTrain, e.loss]);
      setStat('train loss', smoothTrain.toFixed(3));
      setStat('held-out loss', e.loss.toFixed(3), e.loss > 0.68 ? 'bad' : e.loss < 0.56 ? 'good' : '');
      setStat('accuracy', (e.accuracy * 100).toFixed(1) + '%');
      setStat('epochs', live.epochs.toFixed(1));
      // The gap that names itself: still improving on what it has seen, getting
      // worse on what it has not.
      if (e.loss < bestHeldOut) bestHeldOut = e.loss;
      else if (live.steps > 4000 && e.loss > bestHeldOut + 0.08 &&
               smoothTrain < bestHeldOut) achieve('draft-overfit');
    }

    const btnTrain = document.getElementById('btn-train');
    function stop() {
      if (loop) { loop.stop(); loop = null; }
      btnTrain.textContent = '▶ Train';
      btnTrain.classList.remove('toggle', 'on');
    }
    btnTrain.addEventListener('click', () => {
      if (loop) { stop(); return; }
      if (!live) rebuild();
      loop = window.ML.rafLoop(tick);
      loop.start();
      btnTrain.textContent = '⏸ Pause';
      btnTrain.classList.add('toggle', 'on');
    });
    document.getElementById('btn-train-reset').addEventListener('click', rebuild);

    rebuild();
    newDraft();
    drawCalib();
    void fx;

    window.__draft = {
      roster, shipped,
      get live() { return live; },
      get teams() { return { a: teamA.slice(), b: teamB.slice() }; },
      pick: humanPick,
      newDraft,
    };
  });
})();
