/* ==========================================================================
   wordle-lesson.js — race the solver, and see what a guess is worth.
   ========================================================================== */

;(function () {
  'use strict';
  const { wordle: W, WORDLE_WORDS: WORDS, hidpi, chrome, nextLinks, achieve,
          rafLoop, mulberry32 } = window.ML;
  const arcade = window.ML.arcade;

  const CLASSES = ['grey', 'yellow', 'green'];
  const ROWS = 6;

  document.addEventListener('DOMContentLoaded', () => {
    chrome('wordle', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.wordle);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'wordle');
    nextLinks(document.getElementById('next-links'), 'connect4', 'capstone', '../');

    const rand = mulberry32((Math.random() * 1e9) | 0);
    const fx = arcade.fx();

    /* ================================================================
       Game state
       ================================================================ */
    let answer = '';
    let typed = '';
    let myGuesses = [];            // { word, code }
    let solverGuesses = [];
    let solverCandidates = WORDS;
    let over = false, streak = 0, wins = 0, losses = 0;

    const hud = arcade.hud(document.getElementById('w-hud'), [
      { name: 'you', label: 'your guesses', value: '0' },
      { name: 'solver', label: 'solver', value: '0' },
      { name: 'streak', label: 'win streak', value: '0' },
      { name: 'best', label: 'best streak', value: '—' },
    ]);
    arcade.soundToggle(document.getElementById('w-sound'));

    const MEDALS = { gold: 5, silver: 3, bronze: 1 };
    function showBest() {
      const b = arcade.readBest('wordle-streak');
      hud.set('best', b === null ? '—' : b);
      const t = document.getElementById('w-medals');
      if (!t) return;
      t.innerHTML = ['gold', 'silver', 'bronze'].map((m) =>
        `<span class="${b !== null && b >= MEDALS[m] ? 'won' : ''}">` +
        `${arcade.MEDALS[m].icon} <b>beat it ${MEDALS[m]}×</b></span>`).join('') +
        '<span>a draw keeps your streak, a loss ends it</span>';
    }

    /* ================================================================
       Boards
       ================================================================ */
    function renderBoard(host, guesses, live) {
      host.innerHTML = '';
      for (let r = 0; r < ROWS; r++) {
        const row = document.createElement('div');
        row.className = 'grid5';
        const g = guesses[r];
        for (let i = 0; i < 5; i++) {
          const t = document.createElement('div');
          t.className = 'tile';
          if (g) {
            t.textContent = g.word[i];
            t.classList.add(CLASSES[W.patternDigits(g.code)[i]]);
          } else if (live && r === guesses.length) {
            t.textContent = typed[i] || '';
            if (typed[i]) t.classList.add('typing');
          }
          row.appendChild(t);
        }
        host.appendChild(row);
      }
    }

    const board = document.getElementById('board');
    const solverBoard = document.getElementById('solver-board');
    function draw() {
      renderBoard(board, myGuesses, !over);
      renderBoard(solverBoard, solverGuesses, false);
      drawKeyboard();
      hud.set('you', myGuesses.length);
      hud.set('solver', solverGuesses.length);
    }

    /* ================================================================
       Keyboard, coloured by the best news each letter has had
       ================================================================ */
    const ROWS_KB = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm'];
    function letterState() {
      const best = {};
      for (const g of myGuesses) {
        const d = W.patternDigits(g.code);
        for (let i = 0; i < 5; i++) {
          const c = g.word[i];
          best[c] = Math.max(best[c] ?? -1, d[i]);
        }
      }
      return best;
    }
    function drawKeyboard() {
      const host = document.getElementById('keyboard');
      const state = letterState();
      host.innerHTML = '';
      ROWS_KB.forEach((row, ri) => {
        const div = document.createElement('div');
        div.style.cssText = 'display:flex;gap:4px;width:100%';
        if (ri === 2) {
          div.appendChild(mkKey('enter', 'wide', () => submit()));
        }
        for (const ch of row) {
          const s = state[ch];
          div.appendChild(mkKey(ch, s === undefined ? '' : CLASSES[s], () => press(ch)));
        }
        if (ri === 2) div.appendChild(mkKey('⌫', 'wide', () => backspace()));
        host.appendChild(div);
      });
    }
    function mkKey(label, cls, onClick) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      if (cls) b.className = cls;
      b.addEventListener('click', onClick);
      return b;
    }

    /* ================================================================
       Playing
       ================================================================ */
    function press(ch) {
      if (over || typed.length >= 5) return;
      typed += ch;
      draw();
      refreshProbe();
    }
    function backspace() {
      if (over || !typed.length) return;
      typed = typed.slice(0, -1);
      draw();
      refreshProbe();
    }

    function note(msg) { document.getElementById('turn-note').textContent = msg; }

    function submit() {
      if (over) return;
      if (typed.length !== 5) { note('five letters'); return; }
      if (!WORDS.includes(typed)) {
        note(`"${typed}" is not in this word list`);
        arcade.sfx('fail');
        return;
      }
      const code = W.score(typed, answer);
      myGuesses.push({ word: typed, code });
      arcade.sfx(code === W.ALL_GREEN ? 'goal' : 'tick');
      typed = '';
      note('guess a five-letter word');

      // The solver takes its turn on the same word, knowing only its own colours.
      if (!solverDone()) solverTurn();

      draw();
      refreshProbe();
      checkOver();
    }

    function solverDone() {
      const last = solverGuesses[solverGuesses.length - 1];
      return !!last && last.code === W.ALL_GREEN;
    }

    function solverTurn() {
      const guess = solverGuesses.length === 0
        ? OPENER
        : W.suggest(solverCandidates, WORDS);
      const code = W.score(guess, answer);
      const before = solverCandidates.length;
      solverCandidates = W.filter(solverCandidates, guess, code);
      solverGuesses.push({ word: guess, code });
      const log = document.getElementById('solver-log');
      const line = document.createElement('div');
      line.className = 'solver-line';
      line.innerHTML = `${solverGuesses.length}. <b>${guess.toUpperCase()}</b> — ` +
        (code === W.ALL_GREEN
          ? '<span style="color:var(--good)">solved</span>'
          : `${before.toLocaleString()} → <b>${solverCandidates.length.toLocaleString()}</b> still possible`);
      log.appendChild(line);
      document.getElementById('solver-state').textContent =
        solverDone() ? `solved in ${solverGuesses.length}` : `${solverCandidates.length} possible`;
    }

    function checkOver() {
      const iWon = myGuesses.length && myGuesses[myGuesses.length - 1].code === W.ALL_GREEN;
      const itWon = solverDone();
      const iOut = myGuesses.length >= ROWS && !iWon;
      // The solver finishing first does not end your game: you play on until you
      // solve it or run out of rows, and the comparison happens then.
      if (!iWon && !iOut) return;

      over = true;
      const mine = iWon ? myGuesses.length : Infinity;
      const theirs = itWon ? solverGuesses.length : Infinity;
      const el = document.getElementById('w-result');

      let outcome;
      if (mine < theirs) outcome = 'win';
      else if (mine === theirs) outcome = 'draw';
      else outcome = 'loss';

      if (outcome === 'win') {
        streak++; wins++;
        const rec = arcade.submitBest('wordle-streak', streak);
        const tier = arcade.medalFor(streak, MEDALS);
        el.innerHTML = `<b class="result-score">You win.</b> ` +
          (tier ? arcade.medalHTML(tier) + ' ' : '') +
          (rec.isNew ? '<span class="result-new">new best</span> ' : '') +
          `<span class="result-note">${mine} guesses to its ${theirs === Infinity ? 'failure' : theirs}.</span>`;
        arcade.sfx(tier === 'gold' ? 'win' : 'goal');
        fx.burst(120, 60, { count: 40, speed: 170, colors: ['#2f9e70', '#ffd166'], gravity: 120 });
        achieve('wordle-beat', `Solved ${answer.toUpperCase()} in ${mine} to the solver's ${theirs}`);
        if (streak >= 5) achieve('wordle-streak', `${streak} wins in a row against the solver`);
      } else if (outcome === 'draw') {
        el.innerHTML = `<span class="result-note">A draw — ${mine} guesses each. ` +
          `The answer was <b>${answer.toUpperCase()}</b>.</span>`;
        arcade.sfx('tick');
      } else {
        streak = 0; losses++;
        el.innerHTML = `<span class="result-note">The solver got there in ${theirs}` +
          `${iWon ? `, you in ${mine}` : ''}. The answer was <b>${answer.toUpperCase()}</b>.</span>`;
        arcade.sfx('fail');
      }
      hud.set('streak', streak);
      showBest();
      note(iWon ? 'solved' : `it was ${answer.toUpperCase()}`);
    }

    function newGame() {
      answer = WORDS[(rand() * WORDS.length) | 0];
      typed = '';
      myGuesses = [];
      solverGuesses = [];
      solverCandidates = WORDS;
      over = false;
      document.getElementById('solver-log').innerHTML = '';
      document.getElementById('solver-state').textContent = `${WORDS.length} possible`;
      document.getElementById('w-result').innerHTML = '';
      note('guess a five-letter word');
      hud.set('streak', streak);
      showBest();
      draw();
      refreshProbe();
    }

    document.getElementById('btn-new').addEventListener('click', newGame);
    document.getElementById('btn-reveal').addEventListener('click', () => {
      if (over) return;
      over = true;
      streak = 0;
      hud.set('streak', 0);
      document.getElementById('w-result').innerHTML =
        `<span class="result-note">It was <b>${answer.toUpperCase()}</b>.</span>`;
      note('gave up');
      draw();
    });

    window.addEventListener('keydown', (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea') return;
      if (e.key === 'Enter') { e.preventDefault(); submit(); }
      else if (e.key === 'Backspace') { e.preventDefault(); backspace(); }
      else if (/^[a-zA-Z]$/.test(e.key)) { e.preventDefault(); press(e.key.toLowerCase()); }
    });

    /* ================================================================
       The probe: what would this word be worth?
       ================================================================ */
    let probeWord = '';
    const probeInput = document.createElement('input');
    probeInput.type = 'text';
    probeInput.maxLength = 5;
    probeInput.placeholder = 'type a word to weigh it';
    probeInput.setAttribute('aria-label', 'A word to measure the information of');
    probeInput.style.cssText =
      'width:100%;padding:10px 12px;font-family:var(--mono);font-size:1rem;text-transform:uppercase;' +
      'background:var(--panel-2);border:1px solid var(--line);border-radius:9px;color:var(--text)';
    document.getElementById('probe-controls').appendChild(probeInput);
    probeInput.addEventListener('input', () => {
      probeWord = probeInput.value.toLowerCase().replace(/[^a-z]/g, '');
      refreshProbe();
    });
    // While this box has focus the page's keyboard handler stands down, which
    // would otherwise leave the game silently unplayable. Enter hands control
    // back rather than making the reader work out why typing stopped working.
    probeInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); probeInput.blur(); }
    });

    /** What the player can still rule out, from their own colours only. */
    function myCandidates() {
      let c = WORDS;
      for (const g of myGuesses) c = W.filter(c, g.word, g.code);
      return c;
    }

    const bucketCanvas = document.getElementById('buckets');
    let bctx;
    function resizeBuckets() {
      bucketCanvas.style.width = '';
      const w = bucketCanvas.clientWidth || 340;
      bctx = hidpi(bucketCanvas, w, 170);
    }

    function refreshProbe() {
      const cands = myCandidates();
      const word = probeWord.length === 5 ? probeWord
        : (typed.length === 5 ? typed : '');
      drawBuckets(word, cands);
      drawRanked(cands);
    }

    function drawBuckets(word, cands) {
      if (!bctx) return;
      const Wd = bctx._cssW, H = bctx._cssH;
      bctx.clearRect(0, 0, Wd, H);
      bctx.fillStyle = '#0a0f19';
      bctx.fillRect(0, 0, Wd, H);
      const out = document.getElementById('probe-readout');

      if (!word || !WORDS.includes(word)) {
        bctx.fillStyle = 'rgba(159,176,204,.6)';
        bctx.font = '12px system-ui, sans-serif';
        bctx.textAlign = 'center';
        bctx.fillText(word ? `"${word.toUpperCase()}" is not in this list`
                           : 'type a five-letter word above', Wd / 2, H / 2);
        out.innerHTML = '';
        return;
      }

      const counts = new Int32Array(W.PATTERNS);
      for (const c of cands) counts[W.score(word, c)]++;
      const sizes = [];
      for (let i = 0; i < W.PATTERNS; i++) if (counts[i] > 0) sizes.push(counts[i]);
      sizes.sort((a, b) => b - a);

      const bits = W.entropy(word, cands);
      const worst = sizes[0] || 0;
      // The honest number to quote is the actual expected remainder, which is
      // the bucket-size average weighted by how likely each bucket is. It is
      // NOT candidates / 2^bits — that expression is the geometric mean, and it
      // always flatters the guess.
      let expected = 0;
      for (const n of sizes) expected += (n / cands.length) * n;
      const bw = Wd / Math.max(1, sizes.length);
      const max = worst || 1;
      for (let i = 0; i < sizes.length; i++) {
        const h = (sizes[i] / max) * (H - 26);
        bctx.fillStyle = i === 0 ? '#ff9f45' : 'rgba(77,163,255,.75)';
        bctx.fillRect(i * bw + Math.min(1, bw * 0.1), H - 16 - h, Math.max(1, bw * 0.8), h);
      }
      bctx.fillStyle = 'rgba(159,176,204,.85)';
      bctx.font = '11px ui-monospace, monospace';
      bctx.textAlign = 'left';
      bctx.fillText(`${sizes.length} different colourings`, 8, H - 4);
      bctx.textAlign = 'right';
      bctx.fillText(`biggest group ${worst}`, Wd - 8, H - 4);

      out.innerHTML = `<b class="result-score">${bits.toFixed(2)} bits</b> ` +
        `<span class="muted">from ${cands.length.toLocaleString()} possible answers — ` +
        `on average <b>${Math.max(1, Math.round(expected))}</b> would still be standing ` +
        `afterwards, and at worst ${worst}.</span>`;
    }

    function drawRanked(cands) {
      const body = document.querySelector('#rank-table tbody');
      if (!body) return;
      if (cands.length <= 1) {
        body.innerHTML = '<tr><td colspan="3">Only one word is still possible, so every guess is ' +
          'worth zero bits. There is nothing left to learn.</td></tr>';
        return;
      }
      const ranked = W.rankGuesses(cands, WORDS, 8);
      body.innerHTML = '<tr><th>word</th><th>bits</th><th></th></tr>' + ranked.map((r, i) =>
        `<tr class="${i === 0 ? 'top' : ''}"><td class="w">${r.word}</td>` +
        `<td class="n">${r.bits.toFixed(2)}</td>` +
        `<td class="n">${r.possible ? '' : "can't win"}</td></tr>`).join('');
    }

    /* ================================================================
       Openers, computed here rather than asserted
       ================================================================ */
    let OPENER = 'trace';
    (function rankOpeners() {
      const t0 = performance.now();
      const all = W.rankGuesses(WORDS, WORDS, WORDS.length);
      const ms = performance.now() - t0;
      OPENER = all[0].word;

      document.getElementById('opener-time').textContent =
        `${(WORDS.length * WORDS.length / 1000).toFixed(0)}k comparisons in ${ms.toFixed(0)} ms`;
      const body = document.querySelector('#opener-table tbody');
      const rows = all.slice(0, 6).concat([null]).concat(all.slice(-3));
      body.innerHTML = '<tr><th>word</th><th>bits</th></tr>' + rows.map((r, i) =>
        r === null
          ? '<tr><td class="n" colspan="2" style="color:var(--text-mute)">…</td></tr>'
          : `<tr class="${i === 0 ? 'top' : ''}"><td class="w">${r.word}</td>` +
            `<td class="n">${r.bits.toFixed(3)}</td></tr>`).join('');

      // the whole distribution, so the spread is visible rather than described
      const c = document.getElementById('bits-chart');
      const ctx = hidpi(c, c.clientWidth || 340, 170);
      const Wd = ctx._cssW, H = ctx._cssH;
      ctx.fillStyle = '#0a0f19';
      ctx.fillRect(0, 0, Wd, H);
      const hi = all[0].bits, lo = all[all.length - 1].bits;
      ctx.beginPath();
      all.forEach((r, i) => {
        const x = (i / (all.length - 1)) * Wd;
        const y = H - 18 - ((r.bits - lo) / (hi - lo)) * (H - 34);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.strokeStyle = '#4da3ff';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = 'rgba(159,176,204,.85)';
      ctx.font = '11px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${hi.toFixed(2)} bits`, 8, 16);
      ctx.textAlign = 'right';
      ctx.fillText(`${lo.toFixed(2)} bits`, Wd - 8, H - 4);

      document.getElementById('word-count').textContent = WORDS.length.toLocaleString();
      document.getElementById('answer-count').textContent = WORDS.length.toLocaleString();
      document.getElementById('total-bits').textContent = Math.log2(WORDS.length).toFixed(1);
    })();

    /* ================================================================
       How the greedy solver actually does, measured in this browser
       ================================================================ */
    (function measureSolver() {
      const body = document.querySelector('#results-table tbody');
      const note = document.getElementById('results-note');
      body.innerHTML = '<tr><td colspan="4">measuring…</td></tr>';

      // Run it off the main thread of the first paint, in slices, so the page
      // stays responsive while several hundred games are played.
      const openers = [OPENER, 'crane', 'vivid'];
      const results = [];
      let oi = 0, ai = 0;
      let acc = { total: 0, worst: 0, over: 0, n: 0 };

      const tick = () => {
        const t0 = performance.now();
        while (performance.now() - t0 < 25 && oi < openers.length) {
          const answer = WORDS[ai];
          const g = W.solve(answer, WORDS, openers[oi]);
          acc.total += g.length;
          acc.worst = Math.max(acc.worst, g.length);
          if (g.length > 6) acc.over++;
          acc.n++;
          ai++;
          if (ai >= WORDS.length) {
            results.push({ opener: openers[oi], mean: acc.total / acc.n, worst: acc.worst, over: acc.over });
            oi++; ai = 0; acc = { total: 0, worst: 0, over: 0, n: 0 };
            render();
          }
        }
        if (oi < openers.length) requestAnimationFrame(tick);
        else finish();
      };

      function render() {
        body.innerHTML = '<tr><th>opening word</th><th>mean guesses</th><th>worst case</th><th>failed</th></tr>' +
          results.map((r) =>
            `<tr><td class="w">${r.opener}</td><td class="n">${r.mean.toFixed(3)}</td>` +
            `<td class="n">${r.worst}</td><td class="n">${r.over}</td></tr>`).join('') +
          (oi < openers.length ? `<tr><td colspan="4" class="n">measuring ${openers[oi]}…</td></tr>` : '');
      }

      function finish() {
        render();
        const best = results.reduce((a, b) => (b.mean < a.mean ? b : a), results[0]);
        const safest = results.reduce((a, b) => (b.worst < a.worst ? b : a), results[0]);
        note.textContent =
          `Every one of the ${WORDS.length.toLocaleString()} possible answers, solved from scratch. ` +
          `Best average: ${best.opener.toUpperCase()} at ${best.mean.toFixed(3)}. ` +
          `Best worst case: ${safest.opener.toUpperCase()}, never worse than ${safest.worst}. ` +
          (best.opener === safest.opener
            ? 'On this list the same word happens to win both.'
            : 'Different words, which is the point.');
        if (results.every((r) => r.over === 0)) {
          achieve('wordle-measured', 'Measured the solver over every answer');
        }
      }
      requestAnimationFrame(tick);
    })();

    /* ================================================================
       Boot
       ================================================================ */
    const resizeAll = () => { resizeBuckets(); refreshProbe(); };
    resizeAll();
    window.addEventListener('resize', resizeAll);
    newGame();

    rafLoop((dt) => {
      if (fx.busy && bctx) {
        fx.begin(bctx);
        fx.end(bctx, dt, bctx._cssW, bctx._cssH);
      }
    }).start();

    // handy from the console, and used by the tests
    window.__wordle = {
      get answer() { return answer; },
      get myGuesses() { return myGuesses; },
      get solverGuesses() { return solverGuesses; },
      press, submit, newGame, W, WORDS,
    };
  });
})();
