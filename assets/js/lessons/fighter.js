/* ==========================================================================
   fighter.js — the mixup game, the reader, and your own exploitability.
   ========================================================================== */

;(function () {
  'use strict';
  const { GT, hidpi, chrome, nextLinks, achieve, pills, rafLoop, mulberry32, LineChart } = window.ML;
  const arcade = window.ML.arcade;
  const { ACTIONS, PAYOFF, N } = GT;
  const KEYS = ['1', '2', '3', '4', '5'];

  document.addEventListener('DOMContentLoaded', () => {
    chrome('fighter', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.fighter);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'fighter');
    nextLinks(document.getElementById('next-links'), 'wordle', 'capstone', '../');

    const rand = mulberry32((Math.random() * 1e9) | 0);
    const fx = arcade.fx();

    /* ================================================================
       Match state
       ================================================================ */
    let mode = 'nash';                    // 'nash' | 'reader'
    /* The equilibrium mixture, worked out once by self-play. It is deliberately
       fixed: an agent that kept updating on YOUR moves would be best-responding
       to you, which is the reader's job and is exploitable. Playing equilibrium
       means playing the same thing forever no matter what you do — that is the
       whole guarantee, and an earlier version of this page broke it by letting
       the agent learn from the human, at which point it started taking two
       points a round off anyone predictable and quietly contradicted the
       matchup table further down. */
    const EQUILIBRIUM = GT.solve(120000, rand).mix;
    let reader = new GT.Reader({ rand });
    let myCounts = new Float64Array(N);
    let round = 0, myScore = 0, streakBest = 0;

    const hud = arcade.hud(document.getElementById('f-hud'), [
      { name: 'score', label: 'your score', value: '0' },
      { name: 'round', label: 'rounds', value: '0' },
      { name: 'rate', label: 'per round', value: '—' },
      { name: 'best', label: 'best run', value: '—' },
    ]);
    arcade.soundToggle(document.getElementById('f-sound'));

    const MEDALS = { gold: 20, silver: 10, bronze: 4 };
    function showBest() {
      const b = arcade.readBest('fighter-score');
      hud.set('best', b === null ? '—' : '+' + b);
      const t = document.getElementById('f-medals');
      if (!t) return;
      t.innerHTML = ['gold', 'silver', 'bronze'].map((m) =>
        `<span class="${b !== null && b >= MEDALS[m] ? 'won' : ''}">` +
        `${arcade.MEDALS[m].icon} <b>+${MEDALS[m]} ahead</b></span>`).join('') +
        '<span>only the reader can actually be beaten — the other one gives nothing away</span>';
    }

    /* ================================================================
       Mix bars
       ================================================================ */
    function mixBars(host, mix, cls, labelFn) {
      const el = typeof host === 'string' ? document.getElementById(host) : host;
      if (!el) return;
      if (!el.children.length) {
        el.innerHTML = ACTIONS.map((a) =>
          `<div class="mixrow"><span class="n">${a.label}</span>` +
          `<span class="track"><i class="${cls}"></i></span><span class="v"></span></div>`).join('');
      }
      [...el.children].forEach((row, i) => {
        row.querySelector('i').style.width = (mix[i] * 100).toFixed(1) + '%';
        row.querySelector('.v').textContent = labelFn ? labelFn(mix[i], i) : (mix[i] * 100).toFixed(0) + '%';
      });
    }

    /* ================================================================
       The exploitability meter — the centrepiece
       ================================================================ */
    function myMix() {
      let total = 0;
      for (let i = 0; i < N; i++) total += myCounts[i];
      const m = new Float64Array(N);
      if (total < 1) { m.fill(1 / N); return { mix: m, total: 0 }; }
      for (let i = 0; i < N; i++) m[i] = myCounts[i] / total;
      return { mix: m, total };
    }

    function refreshExploit() {
      const { mix, total } = myMix();
      mixBars('my-mix', mix, 'you');
      const exp = total >= 5 ? GT.exploitability(mix) : null;
      const val = document.getElementById('exp-val');
      const meter = document.getElementById('exp-meter');
      if (exp === null) {
        val.textContent = '—';
        meter.style.width = '0%';
        document.getElementById('exp-state').textContent = `${total} rounds played`;
        return;
      }
      val.textContent = '+' + exp.toFixed(2) + ' per round';
      // 1.5 per round is thoroughly readable; the scale tops out there.
      meter.style.width = Math.min(100, (exp / 1.5) * 100).toFixed(0) + '%';
      const br = GT.bestResponse(mix);
      document.getElementById('exp-state').textContent =
        exp < 0.15 ? 'hard to read' : `best answer: ${ACTIONS[br.action].label}`;
    }

    function refreshModel() {
      const note = document.getElementById('model-note');
      if (mode === 'reader') {
        const m = reader.model();
        mixBars('their-model', m, 'eq');
        const br = GT.bestResponse(m);
        note.textContent = `It believes you favour ${ACTIONS[indexOfMax(m)].label.toLowerCase()}, ` +
          `so it is leaning on ${ACTIONS[br.action].label.toLowerCase()}. Recent rounds count for ` +
          `more than old ones, so a change of habit shows up within a few exchanges.`;
      } else {
        mixBars('their-model', EQUILIBRIUM, 'eq');
        note.textContent = 'It builds no model of you at all. This is the mixture it plays, worked ' +
          'out by self-play before you arrived and fixed ever since. It is the same whatever you ' +
          'do, which is exactly why nothing you do can punish it.';
      }
    }

    const indexOfMax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

    /* ================================================================
       Playing a round
       ================================================================ */
    function play(mine) {
      const theirs = mode === 'nash' ? nashAct() : readerAct();
      const gain = PAYOFF[mine][theirs];

      myCounts[mine] += 1;
      round++;
      myScore += gain;

      // Only the reader learns. The equilibrium agent never updates on you.
      if (mode === 'reader') reader.observe(theirs, mine);

      document.getElementById('my-move').textContent = ACTIONS[mine].label;
      document.getElementById('their-move').textContent = ACTIONS[theirs].label;
      const clash = document.getElementById('clash');
      clash.classList.remove('win', 'lose');
      if (gain > 0) clash.classList.add('win');
      else if (gain < 0) clash.classList.add('lose');

      document.getElementById('round-note').textContent =
        gain > 0 ? `+${gain} to you` : gain < 0 ? `${gain} to you` : 'nothing in it';

      hud.set('score', (myScore > 0 ? '+' : '') + myScore);
      hud.set('round', round);
      hud.set('rate', (myScore / round >= 0 ? '+' : '') + (myScore / round).toFixed(2));
      if (gain > 0) { hud.flash('score'); arcade.sfx('hit'); }
      else if (gain < 0) { hud.flash('score', 'bad'); arcade.sfx('tick'); }

      if (myScore > streakBest) {
        streakBest = myScore;
        const rec = arcade.submitBest('fighter-score', streakBest);
        const tier = arcade.medalFor(streakBest, MEDALS);
        if (tier === 'gold') achieve('fighter-gold', `+${streakBest} against the reader`);
        if (streakBest >= MEDALS.bronze) achieve('fighter-ahead', `+${streakBest} in a match`);
        if (rec.isNew) showBest();
      }

      refreshExploit();
      refreshModel();
      updateResult();
    }

    function nashAct() {
      let r = rand();
      for (let i = 0; i < N; i++) { r -= EQUILIBRIUM[i]; if (r <= 0) return i; }
      return N - 1;
    }

    function readerAct() {
      const a = reader.act();
      return a.action;
    }

    function updateResult() {
      const el = document.getElementById('f-result');
      if (round < 8) { el.innerHTML = ''; return; }
      const rate = myScore / round;
      const { mix } = myMix();
      const exp = GT.exploitability(mix);
      el.innerHTML = `<span class="result-note">` +
        (rate > 0.15 ? `You are ahead by ${rate.toFixed(2)} a round. `
         : rate < -0.15 ? `It is ahead by ${(-rate).toFixed(2)} a round. `
         : 'Dead even, which is what equilibrium feels like. ') +
        (exp > 0.5 ? `A perfect reader would take ${exp.toFixed(2)} a round off you.`
         : exp > 0.2 ? `You are mildly readable (${exp.toFixed(2)} a round).`
         : `You are hard to read (${exp.toFixed(2)} a round).`) +
        `</span>`;
    }

    /* ================================================================
       Controls
       ================================================================ */
    const movesHost = document.getElementById('moves');
    ACTIONS.forEach((a, i) => {
      const b = document.createElement('button');
      b.className = 'move';
      b.type = 'button';
      b.innerHTML = `<span class="key">${KEYS[i]}</span><b>${a.label}</b><span>${a.hint}</span>`;
      b.addEventListener('click', () => play(i));
      movesHost.appendChild(b);
    });
    window.addEventListener('keydown', (e) => {
      const i = KEYS.indexOf(e.key);
      const tag = (e.target.tagName || '').toLowerCase();
      if (i < 0 || tag === 'input' || tag === 'textarea') return;
      e.preventDefault();
      play(i);
    });

    pills(document.getElementById('opponent-mode'), [
      { value: 'nash', label: 'Unexploitable (regret matching)' },
      { value: 'reader', label: 'The reader (best response)' },
    ], 'nash', (v) => {
      mode = v;
      document.getElementById('opponent-note').textContent = v === 'nash'
        ? 'Regret matching, playing from its running average. It never looks at you, cannot be '
          + 'punished for long, and will not punish you either. Expect to hover around even.'
        : 'It counts what you have played, weights recent rounds more heavily, and plays the single '
          + 'best answer to that. Beatable if you can spot what it has settled on — and it will '
          + 'spot you first if you have a habit.';
      refreshModel();
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
      reader = new GT.Reader({ rand });
      myCounts = new Float64Array(N);
      round = 0; myScore = 0; streakBest = 0;
      document.getElementById('my-move').textContent = '—';
      document.getElementById('their-move').textContent = '—';
      document.getElementById('clash').classList.remove('win', 'lose');
      document.getElementById('round-note').textContent = 'round 1';
      document.getElementById('f-result').innerHTML = '';
      hud.set('score', '0'); hud.set('round', '0'); hud.set('rate', '—');
      refreshExploit(); refreshModel(); showBest();
    });

    /* ================================================================
       The payoff table
       ================================================================ */
    (function drawPayoff() {
      const t = document.getElementById('payoff-table');
      const short = ACTIONS.map((a) => a.label.replace('Strike ', 'S-').replace('Block ', 'B-'));
      let html = '<tr><th></th>' + short.map((s) => `<th>${s}</th>`).join('') + '</tr>';
      for (let i = 0; i < N; i++) {
        html += `<tr><th class="row">${short[i]}</th>` + PAYOFF[i].map((v) => {
          const bg = v > 0 ? `rgba(56,211,159,${0.12 + 0.2 * (v / 3)})`
                   : v < 0 ? `rgba(255,107,107,${0.12 + 0.2 * (-v / 3)})` : 'transparent';
          const col = v > 0 ? 'var(--good)' : v < 0 ? 'var(--bad)' : 'var(--text-mute)';
          return `<td style="background:${bg};color:${col}">${v > 0 ? '+' : ''}${v}</td>`;
        }).join('') + '</tr>';
      }
      t.innerHTML = html;
    })();

    /* ================================================================
       Self-play convergence, run live
       ================================================================ */
    let convA = null, convB = null, convRunning = false, convRounds = 0;
    const convChart = new LineChart(document.getElementById('conv-chart'), {
      height: 190, xLabel: 'rounds of self-play', yMin: 0,
      series: [
        { name: 'current mixture', color: '#ff9f45' },
        { name: 'running average', color: '#38d39f' },
      ],
    });

    function convReset() {
      convA = new GT.RegretMatcher({ rand });
      convB = new GT.RegretMatcher({ rand });
      convRounds = 0;
      convChart.clear();
      convChart.draw();
      document.getElementById('conv-state').textContent = '';
      mixBars('eq-mix', new Float64Array(N).fill(1 / N), 'eq');
      document.getElementById('eq-note').textContent =
        'Run the self-play to the left and this fills in with what they agree on.';
    }

    function convStep(n) {
      for (let i = 0; i < n; i++) {
        const x = convA.act(), y = convB.act();
        convA.observe(x.action, y.action);
        convB.observe(y.action, x.action);
        convRounds++;
      }
      const cur = convA.strategy();
      const avg = convA.average();
      convChart.push(convRounds, [GT.exploitability(cur), GT.exploitability(avg)]);
      convChart.draw();
      mixBars('eq-mix', avg, 'eq');
      const e = GT.exploitability(avg);
      document.getElementById('conv-state').textContent = `${convRounds.toLocaleString()} rounds`;
      document.getElementById('eq-note').textContent =
        `After ${convRounds.toLocaleString()} rounds the average is exploitable for ` +
        `${e.toFixed(3)} a round. The current mixture, by contrast, is at ` +
        `${GT.exploitability(cur).toFixed(2)}.`;
      if (convRounds >= 20000 && e < 0.02) {
        achieve('fighter-equilibrium', `Converged to within ${e.toFixed(3)} of unexploitable`);
      }
    }

    document.getElementById('btn-converge').addEventListener('click', () => {
      convRunning = !convRunning;
      const b = document.getElementById('btn-converge');
      b.textContent = convRunning ? '⏸ Pause' : '▶ Run self-play';
      b.classList.toggle('primary', !convRunning);
    });
    document.getElementById('btn-conv-reset').addEventListener('click', () => {
      convRunning = false;
      const b = document.getElementById('btn-converge');
      b.textContent = '▶ Run self-play';
      b.classList.add('primary');
      convReset();
    });

    /* ================================================================
       The matchup table: what each style is worth against what
       ================================================================ */
    (function matchups() {
      const body = document.querySelector('#matchup-table tbody');
      const ROUNDS = 4000;
      const eq = GT.solve(60000, rand).mix;

      // A stand-in for a player with a habit: favours blocking high.
      const habit = () => (rand() < 0.45 ? 3 : (rand() * N) | 0);
      const coin = () => (rand() * N) | 0;

      function vsFixed(agentKind, humanFn) {
        const agent = agentKind === 'nash' ? null : new GT.Reader({ rand });
        let total = 0;
        for (let t = 0; t < ROUNDS; t++) {
          const h = humanFn();
          let a;
          if (agentKind === 'nash') {
            let r = rand(); a = N - 1;
            for (let i = 0; i < N; i++) { r -= eq[i]; if (r <= 0) { a = i; break; } }
          } else {
            a = agent.act().action;
            agent.observe(a, h);
          }
          total += PAYOFF[a][h];
        }
        return total / ROUNDS;
      }

      const rows = [
        ['Equilibrium agent', 'a player with a habit', vsFixed('nash', habit)],
        ['Equilibrium agent', 'a coin flip', vsFixed('nash', coin)],
        ['The reader', 'a player with a habit', vsFixed('reader', habit)],
        ['The reader', 'a coin flip', vsFixed('reader', coin)],
      ];
      body.innerHTML = '<tr><th>agent</th><th>against</th><th>points per round</th></tr>' +
        rows.map(([a, b, v]) =>
          `<tr><td>${a}</td><td>${b}</td><td class="num" style="color:${v > 0.05 ? 'var(--good)' : v < -0.05 ? 'var(--bad)' : 'var(--text-dim)'}">` +
          `${v >= 0 ? '+' : ''}${v.toFixed(2)}</td></tr>`).join('');
      document.getElementById('matchup-note').textContent =
        `${ROUNDS.toLocaleString()} rounds each, played here just now. The equilibrium agent earns ` +
        `nothing from either opponent — that is what unexploitable costs you. The reader takes real ` +
        `money off the habit and nothing off the coin flip, because there is nothing there to read.`;
    })();

    /* ================================================================
       Boot
       ================================================================ */
    convReset();
    refreshExploit();
    refreshModel();
    showBest();
    document.getElementById('opponent-note').textContent =
      'Regret matching, playing from its running average. It never looks at you, cannot be punished '
      + 'for long, and will not punish you either. Expect to hover around even.';

    rafLoop(() => {
      if (convRunning) convStep(400);
    }).start();

    // handy from the console, and used by the tests
    window.__fighter = {
      play, GT,
      get round() { return round; },
      get score() { return myScore; },
      get myMix() { return myMix().mix; },
      setMode: (m) => { mode = m; refreshModel(); },
      convStep, get convRounds() { return convRounds; },
      get convAverage() { return convA.average(); },
    };
  });
})();
