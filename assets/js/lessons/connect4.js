/* ==========================================================================
   connect4.js — the page: play the agent, watch it search, train your own.
   ========================================================================== */

;(function () {
  'use strict';
  const { C4, C4Trainer, MCTS, pickMove, MLP, mulberry32, softmax, clamp, hidpi,
          LineChart, achieve, chrome, nextLinks, slider, pills, statGrid, rafLoop } = window.ML;
  const arcade = window.ML.arcade;

  const COL = { you: '#ffd166', bot: '#4da3ff', youDark: '#9c7c1f', botDark: '#1f5c9e' };

  document.addEventListener('DOMContentLoaded', () => {
    chrome('connect4', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.connect4);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'connect4');
    nextLinks(document.getElementById('next-links'), 'rocket', 'capstone', '../');

    /* =================================================================
       The shipped ladder of checkpoints
       ================================================================= */
    const STAGES = window.ML_C4_STAGES || null;
    const allStages = (STAGES && STAGES.stages) || [];

    /**
     * The model file already decided which checkpoints are worth keeping
     * weights for — it prunes the rest to save a megabyte — so the ladder is
     * simply those, ordered by how they actually did against the fixed bot.
     *
     * Ordering by measured result rather than by games trained matters here:
     * this run got dramatically *worse* for its first few hundred games, so a
     * ladder built on training time would hand you a "harder" opponent that is
     * in fact the easiest one in the file.
     */
    function buildLadder(stages) {
      return stages
        .filter((s) => s.net && s.eval)
        .sort((a, b) => (a.eval.wins + a.eval.draws * 0.5) - (b.eval.wins + b.eval.draws * 0.5));
    }

    const rungs = buildLadder(allStages).map((s) => ({
      label: s.label, games: s.games, net: s.net, eval: s.eval,
    }));
    // Without the model file there is still a game: an untrained network plus
    // the search, which is weak but perfectly playable.
    if (!rungs.length) rungs.push({ label: 'untrained', games: 0, net: null, eval: null });

    /* =================================================================
       Opponent: a network (or none) plus a search
       ================================================================= */
    const rand = mulberry32(1234);
    const scratch = new Float32Array(C4.N * 2);
    let oppNet = null;                       // null = uniform priors, pure search
    let sims = 120;
    let rungIdx = Math.min(rungs.length - 1, Math.max(0, rungs.length - 2));

    const GAME = {
      nActions: C4.W,
      legal: (b) => C4.legalMoves(b),
      apply: (b, a, p) => C4.play(b, a, p),
      undo: (b, a, row) => C4.undo(b, a, row),
      justWon: (b, a, row) => C4.winsAt(b, a, row),
      isDraw: (b) => C4.isFull(b),
      other: (p) => (p === C4.P1 ? C4.P2 : C4.P1),
    };

    /** The network's read on a position, or a blank opinion when there is none. */
    function evaluateWith(net, board, player) {
      if (!net) return { P: Float32Array.from({ length: C4.W }, () => 1 / C4.W), v: 0 };
      C4.encode(board, player, scratch);
      const out = net.forward(scratch);
      const logits = new Float32Array(C4.W);
      for (let a = 0; a < C4.W; a++) logits[a] = out[a];
      return { P: softmax(logits), v: Math.tanh(out[C4.W]) };
    }

    const search = new MCTS(GAME, (b, p) => evaluateWith(oppNet, b, p), { rand });

    function loadRung(i) {
      rungIdx = i;
      const r = rungs[i];
      oppNet = r.net ? MLP.fromJSON(r.net) : null;
      drawLadder();
    }

    function drawLadder() {
      const host = document.getElementById('ladder');
      host.innerHTML = '';
      rungs.forEach((r, i) => {
        const b = document.createElement('button');
        b.className = 'rung' + (i === rungIdx ? ' on' : '');
        b.type = 'button';
        const total = r.eval ? r.eval.wins + r.eval.losses + r.eval.draws : 0;
        b.innerHTML = `<b>${r.games === 0 ? 'Untrained' : r.games.toLocaleString() + ' games'}</b>` +
          `<span>${r.eval ? `won ${r.eval.wins}/${total} vs bot` : 'random weights'}</span>`;
        b.addEventListener('click', () => { loadRung(i); newGame(); });
        host.appendChild(b);
      });
    }

    /* =================================================================
       The game
       ================================================================= */
    const HUMAN = C4.P1, BOT = C4.P2;
    let board = C4.newBoard();
    let turn = HUMAN;
    let over = false, winLine = null, lastDrop = null;
    let history = [];                     // for take-back
    let thinking = false, lastVisits = null;
    let streak = 0;
    const fx = arcade.fx();

    const hud = arcade.hud(document.getElementById('c4-hud'), [
      { name: 'streak', label: 'win streak', value: '0' },
      { name: 'record', label: 'this session', value: '0–0' },
      { name: 'best', label: 'best streak', value: '—' },
      { name: 'level', label: 'opponent', value: '—' },
    ]);
    arcade.soundToggle(document.getElementById('c4-sound'));
    let sessionWins = 0, sessionLosses = 0;

    const MEDALS = { gold: 5, silver: 3, bronze: 1 };
    function showBest() {
      const b = arcade.readBest('connect4-streak');
      hud.set('best', b === null ? '—' : b);
      hud.set('level', rungs[rungIdx].games === 0 ? 'untrained' : rungs[rungIdx].games.toLocaleString());
      const track = document.getElementById('c4-medals');
      if (!track) return;
      track.innerHTML = ['gold', 'silver', 'bronze'].map((m) =>
        `<span class="${b !== null && b >= MEDALS[m] ? 'won' : ''}">` +
        `${arcade.MEDALS[m].icon} <b>${MEDALS[m]} in a row</b></span>`).join('') +
        '<span>beating a stronger rung is worth more than beating this one twice</span>';
    }

    function newGame() {
      board = C4.newBoard();
      turn = HUMAN;
      over = false; winLine = null; lastDrop = null; lastVisits = null;
      history = [];
      thinking = false;
      document.getElementById('c4-result').innerHTML = '';
      document.getElementById('last-think').textContent = '';
      setTurnNote('your move');
      showBest();
      draw();
      drawThink();
    }

    function setTurnNote(t) { document.getElementById('turn-note').textContent = t; }

    function finish(result) {
      over = true;
      const el = document.getElementById('c4-result');
      if (result === 'win') {
        streak++;
        sessionWins++;
        hud.set('streak', streak);
        hud.flash('streak');
        const rec = arcade.submitBest('connect4-streak', streak);
        const tier = arcade.medalFor(streak, MEDALS);
        el.innerHTML = `<b class="result-score">You win.</b> ` +
          (tier ? arcade.medalHTML(tier) + ' ' : '') +
          (rec.isNew ? '<span class="result-new">new best</span> ' : '') +
          `<span class="result-note">${streak} in a row against the ` +
          `${rungs[rungIdx].games.toLocaleString()}-game agent.</span>`;
        arcade.sfx(tier === 'gold' ? 'win' : 'goal');
        fx.burst(boardPx(lastDrop ? lastDrop.x : 3, lastDrop ? lastDrop.y : 0).x,
                 boardPx(lastDrop ? lastDrop.x : 3, lastDrop ? lastDrop.y : 0).y,
                 { count: 44, speed: 190, colors: [COL.you, '#38d39f'], gravity: 130 });
        fx.shake(6);
        achieve('c4-beat', `Beat the ${rungs[rungIdx].games.toLocaleString()}-game agent`);
        if (streak >= 5) achieve('c4-streak', `${streak} wins in a row`);
      } else if (result === 'loss') {
        streak = 0;
        sessionLosses++;
        hud.set('streak', '0');
        hud.flash('streak', 'bad');
        el.innerHTML = '<span class="result-note">It got there first. ' +
          'Try raising its thinking time and watching where the bars go.</span>';
        arcade.sfx('fail');
        fx.shake(5);
      } else {
        el.innerHTML = '<span class="result-note">A draw. The board filled up.</span>';
        arcade.sfx('tick');
      }
      hud.set('record', `${sessionWins}–${sessionLosses}`);
      showBest();
      setTurnNote(result === 'win' ? 'you won' : result === 'loss' ? 'it won' : 'drawn');
    }

    function afterMove(col, row, player) {
      lastDrop = { x: col, y: row };
      if (C4.winsAt(board, col, row)) {
        winLine = C4.winningLine(board, col, row);
        finish(player === HUMAN ? 'win' : 'loss');
        return true;
      }
      if (C4.isFull(board)) { finish('draw'); return true; }
      return false;
    }

    function humanDrop(col) {
      if (over || thinking || turn !== HUMAN) return;
      const row = C4.play(board, col, HUMAN);
      if (row < 0) return;
      history.push({ col, row, player: HUMAN });
      arcade.sfx('tick');
      if (afterMove(col, row, HUMAN)) { draw(); return; }
      turn = BOT;
      setTurnNote('it is thinking…');
      draw();
      // Let the browser paint the human's disc before the search blocks.
      setTimeout(botMove, 90);
    }

    function botMove() {
      if (over) return;
      thinking = true;
      const t0 = performance.now();
      const res = search.search(board, BOT, sims);
      const ms = performance.now() - t0;
      lastVisits = res.visits;
      const col = pickMove(res.visits, 0, rand);
      thinking = false;
      if (col < 0) { finish('draw'); return; }
      const row = C4.play(board, col, BOT);
      history.push({ col, row, player: BOT });
      document.getElementById('last-think').innerHTML =
        `<span class="muted">${sims} simulations in ${ms.toFixed(0)} ms · ` +
        `it rates its position ${res.value >= 0 ? '+' : ''}${res.value.toFixed(2)}</span>`;
      arcade.sfx('hit');
      if (!afterMove(col, row, BOT)) { turn = HUMAN; setTurnNote('your move'); }
      draw();
      drawThink();
    }

    document.getElementById('btn-new').addEventListener('click', newGame);
    document.getElementById('btn-undo').addEventListener('click', () => {
      if (thinking || history.length < 2) return;
      for (let i = 0; i < 2 && history.length; i++) {
        const h = history.pop();
        C4.undo(board, h.col, h.row);
      }
      over = false; winLine = null; lastDrop = null;
      turn = HUMAN;
      document.getElementById('c4-result').innerHTML = '';
      setTurnNote('your move');
      draw();
    });
    document.getElementById('btn-hint').addEventListener('click', () => {
      if (over || thinking || turn !== HUMAN) return;
      const res = search.search(board, HUMAN, sims);
      lastVisits = res.visits;
      const col = pickMove(res.visits, 0, rand);
      document.getElementById('c4-result').innerHTML =
        `<span class="result-note">In your position it would play column ${col + 1}.</span>`;
      drawThink();
    });

    /* =================================================================
       Drawing the board
       ================================================================= */
    const canvas = document.getElementById('board');
    let ctx, cell = 40, padX = 0, padY = 0;

    function resizeBoard() {
      canvas.style.width = '';
      const w = canvas.clientWidth || canvas.parentElement.clientWidth || 420;
      const h = Math.round((w * C4.H) / C4.W);
      ctx = hidpi(canvas, w, h);
      cell = ctx._cssW / C4.W;
      padX = 0; padY = 0;
    }

    const boardPx = (x, y) => ({ x: (x + 0.5) * cell, y: (y + 0.5) * cell });

    let hoverCol = -1;
    canvas.addEventListener('pointermove', (e) => {
      const r = canvas.getBoundingClientRect();
      hoverCol = Math.floor(((e.clientX - r.left) / r.width) * C4.W);
      draw();
    });
    canvas.addEventListener('pointerleave', () => { hoverCol = -1; draw(); });
    canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      const col = Math.floor(((e.clientX - r.left) / r.width) * C4.W);
      if (col >= 0 && col < C4.W) humanDrop(col);
    });

    function draw() {
      if (!ctx) return;
      const W = ctx._cssW, H = ctx._cssH;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#132038';
      ctx.fillRect(0, 0, W, H);

      // the column you are about to drop into
      if (!over && turn === HUMAN && hoverCol >= 0 && hoverCol < C4.W &&
          C4.dropRow(board, hoverCol) >= 0) {
        ctx.fillStyle = 'rgba(255,209,102,.09)';
        ctx.fillRect(hoverCol * cell, 0, cell, H);
      }

      const radius = cell * 0.38;
      for (let y = 0; y < C4.H; y++) {
        for (let x = 0; x < C4.W; x++) {
          const p = boardPx(x, y);
          const v = board[C4.idx(x, y)];
          ctx.beginPath();
          ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = v === C4.EMPTY ? '#0b1220'
            : v === HUMAN ? COL.you : COL.bot;
          ctx.fill();
          if (v !== C4.EMPTY) {
            ctx.lineWidth = 2;
            ctx.strokeStyle = v === HUMAN ? COL.youDark : COL.botDark;
            ctx.stroke();
          }
        }
      }

      // the ghost disc waiting to fall
      if (!over && turn === HUMAN && hoverCol >= 0 && hoverCol < C4.W) {
        const row = C4.dropRow(board, hoverCol);
        if (row >= 0) {
          const p = boardPx(hoverCol, row);
          ctx.beginPath();
          ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255,209,102,.28)';
          ctx.fill();
        }
      }

      if (lastDrop) {
        const p = boardPx(lastDrop.x, lastDrop.y);
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + 3, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255,255,255,.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      if (winLine) {
        const a = boardPx(winLine[0][0], winLine[0][1]);
        const b = boardPx(winLine[3][0], winLine[3][1]);
        ctx.strokeStyle = '#38d39f';
        ctx.lineWidth = 6;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    /* =================================================================
       "What it is considering"
       ================================================================= */
    const thinkCanvas = document.getElementById('think');
    let tctx;
    function resizeThink() {
      thinkCanvas.style.width = '';
      const w = thinkCanvas.clientWidth || 340;
      tctx = hidpi(thinkCanvas, w, 140);
    }
    function drawThink() {
      if (!tctx) return;
      const W = tctx._cssW, H = tctx._cssH;
      tctx.clearRect(0, 0, W, H);
      tctx.fillStyle = '#0a0f19';
      tctx.fillRect(0, 0, W, H);
      if (!lastVisits) {
        tctx.fillStyle = 'rgba(159,176,204,.6)';
        tctx.font = '12px system-ui, sans-serif';
        tctx.textAlign = 'center';
        tctx.fillText('play a move and this fills in', W / 2, H / 2);
        return;
      }
      let max = 1;
      for (let a = 0; a < C4.W; a++) max = Math.max(max, lastVisits[a]);
      const bw = W / C4.W;
      let bestA = 0;
      for (let a = 1; a < C4.W; a++) if (lastVisits[a] > lastVisits[bestA]) bestA = a;
      for (let a = 0; a < C4.W; a++) {
        const h = (lastVisits[a] / max) * (H - 26);
        tctx.fillStyle = a === bestA ? COL.bot : 'rgba(77,163,255,.35)';
        tctx.fillRect(a * bw + bw * 0.18, H - 18 - h, bw * 0.64, h);
        tctx.fillStyle = 'rgba(159,176,204,.85)';
        tctx.font = '11px ui-monospace, monospace';
        tctx.textAlign = 'center';
        tctx.fillText(String(a + 1), a * bw + bw / 2, H - 5);
        if (lastVisits[a] > 0) {
          tctx.fillStyle = a === bestA ? '#cfe6ff' : 'rgba(159,176,204,.7)';
          tctx.fillText(String(lastVisits[a] | 0), a * bw + bw / 2, H - 22 - h);
        }
      }
    }

    /* =================================================================
       Step the search one simulation at a time
       ================================================================= */
    const treeCanvas = document.getElementById('tree');
    let trctx, stepRoot = null, stepSims = 0;

    function resizeTree() {
      treeCanvas.style.width = '';
      const w = treeCanvas.clientWidth || 600;
      trctx = hidpi(treeCanvas, w, 190);
    }

    function stepSearch(n) {
      if (over) return;
      const player = turn;
      if (!stepRoot) {
        stepRoot = window.ML.makeNode(C4.W);
        search.simulate(stepRoot, board, player);
        stepSims = 0;
      }
      for (let i = 0; i < n; i++) { search.simulate(stepRoot, board, player); stepSims++; }
      drawTree();
    }

    function drawTree() {
      if (!trctx) return;
      const W = trctx._cssW, H = trctx._cssH;
      trctx.clearRect(0, 0, W, H);
      trctx.fillStyle = '#0a0f19';
      trctx.fillRect(0, 0, W, H);
      document.getElementById('sim-count').textContent =
        stepRoot ? `${stepSims} simulation${stepSims === 1 ? '' : 's'} so far` : '';
      if (!stepRoot) {
        trctx.fillStyle = 'rgba(159,176,204,.6)';
        trctx.font = '12px system-ui, sans-serif';
        trctx.textAlign = 'center';
        trctx.fillText('press "one simulation" to start building the tree', W / 2, H / 2);
        return;
      }

      const legal = C4.legalMoves(board);
      const bw = W / C4.W;
      trctx.font = '11px ui-monospace, monospace';
      trctx.textAlign = 'center';

      // root
      trctx.fillStyle = 'rgba(255,255,255,.75)';
      trctx.fillText(`root · ${stepRoot.N} visits`, W / 2, 16);

      const sqrtN = Math.sqrt(Math.max(1, stepRoot.N));
      for (const a of legal) {
        const n = stepRoot.childN[a];
        const q = n > 0 ? stepRoot.childW[a] / n : 0;
        const p = stepRoot.P ? stepRoot.P[a] : 0;
        const u = 1.4 * p * sqrtN / (1 + n);
        const x = a * bw + bw / 2;

        // the prior, as a faint bar, and the visits as a solid one
        const maxN = Math.max(1, ...Array.from(stepRoot.childN));
        const hv = (n / maxN) * 74;
        trctx.fillStyle = 'rgba(124,92,255,.35)';
        trctx.fillRect(x - bw * 0.3, 40, bw * 0.26, p * 74);
        trctx.fillStyle = 'rgba(77,163,255,.85)';
        trctx.fillRect(x + bw * 0.04, 40, bw * 0.26, hv);

        trctx.fillStyle = 'rgba(159,176,204,.85)';
        trctx.fillText(String(a + 1), x, H - 44);
        trctx.fillStyle = n > 0 ? (q > 0.05 ? '#38d39f' : q < -0.05 ? '#ff6b6b' : '#9fb0cc') : 'rgba(159,176,204,.4)';
        trctx.fillText(n > 0 ? `Q ${q.toFixed(2)}` : 'Q –', x, H - 28);
        trctx.fillStyle = 'rgba(159,176,204,.7)';
        trctx.fillText(`U ${u.toFixed(2)}`, x, H - 13);
      }

      trctx.textAlign = 'left';
      trctx.fillStyle = 'rgba(124,92,255,.8)';
      trctx.fillText('■ prior P(a)', 10, 30);
      trctx.fillStyle = 'rgba(77,163,255,.9)';
      trctx.fillText('■ visits N(a)', 92, 30);
    }

    document.getElementById('btn-sim1').addEventListener('click', () => stepSearch(1));
    document.getElementById('btn-sim25').addEventListener('click', () => stepSearch(25));
    document.getElementById('btn-simreset').addEventListener('click', () => {
      stepRoot = null; stepSims = 0; drawTree();
    });

    /* =================================================================
       Controls
       ================================================================= */
    slider(document.getElementById('sims-control'), {
      label: 'thinking time', min: 1, max: 600, step: 1, value: sims,
      format: (v) => `${v} sim${v === 1 ? '' : 's'}`,
      onInput: (v) => { sims = v | 0; },
      desc: 'How many games it plays out in its head before each move. At 1 you are playing the raw network with no search at all.',
    });

    /* =================================================================
       Train your own
       ================================================================= */
    const trainer = new C4Trainer({ seed: 11 });
    let training = false;
    const setStat = statGrid(document.getElementById('c4-stats'),
      ['self-play games', 'updates', 'policy loss', 'value loss', 'vs scripted bot', 'games/sec']);
    const chart = new LineChart(document.getElementById('c4-chart'), {
      height: 170, xLabel: 'self-play games',
      series: [{ name: 'policy loss', color: '#4da3ff' }, { name: 'value loss', color: '#ff9f45' }],
    });

    const runControls = window.ML.runBar(document.getElementById('run-bar'), {
      seed: trainer.seed,
      onSeed: (s) => { trainer.seed = s; trainer.reset(true); resetTrainUI(); },
      onPin: () => chart.pin(),
      onUnpin: () => chart.unpin(),
    });

    function resetTrainUI() {
      chart.clear();
      setStat('self-play games', 0);
      setStat('updates', 0);
      setStat('policy loss', '—');
      setStat('value loss', '—');
      setStat('vs scripted bot', '—');
      setStat('games/sec', '—');
    }

    document.getElementById('btn-train').addEventListener('click', () => {
      training = !training;
      const b = document.getElementById('btn-train');
      b.textContent = training ? '⏸ Pause' : '▶ Start self-play';
      b.classList.toggle('primary', !training);
    });
    document.getElementById('btn-eval').addEventListener('click', () => {
      const e = trainer.evaluate(20);
      setStat('vs scripted bot', `${e.wins}W ${e.losses}L ${e.draws}D`);
      if (e.wins >= 12) achieve('c4-trained', `${e.wins} of 20 against the scripted bot`);
    });
    document.getElementById('btn-reset-train').addEventListener('click', () => {
      trainer.reset(true);
      training = false;
      const b = document.getElementById('btn-train');
      b.textContent = '▶ Start self-play';
      b.classList.add('primary');
      resetTrainUI();
    });

    const hpHost = document.getElementById('c4-hp');
    slider(hpHost, {
      label: 'search per move (training)', min: 20, max: 200, step: 10, value: trainer.hp.sims,
      onInput: (v) => { trainer.hp.sims = v | 0; },
      desc: 'More search makes better training targets and slower games. This is the main quality-versus-speed dial.',
    });
    slider(hpHost, {
      label: 'learning rate', min: 0.0005, max: 0.01, step: 0.0005, value: trainer.hp.lr,
      format: (v) => v.toFixed(4), onInput: (v) => { trainer.hp.lr = v; },
      desc: 'Adam step size for both heads.',
    });
    slider(hpHost, {
      label: 'root noise', min: 0, max: 1, step: 0.05, value: trainer.hp.noiseFrac,
      format: (v) => (v * 100).toFixed(0) + '%',
      onInput: (v) => { trainer.hp.noiseFrac = v; },
      desc: 'How much Dirichlet noise is mixed into the priors at the root of each self-play search. At zero, every game is nearly identical and learning stalls.',
    });

    /* =================================================================
       What actually happened during the shipped run
       ================================================================= */
    (function drawStrengthCurve() {
      const el = document.getElementById('strength-chart');
      const note = document.getElementById('strength-note');
      if (!el) return;
      const pts = allStages.filter((s) => s.eval);
      if (pts.length < 2) {
        if (note) note.textContent = 'Run node tools/train-connect4.cjs to generate this.';
        return;
      }
      const total = pts[0].eval.wins + pts[0].eval.losses + pts[0].eval.draws;
      const c = new LineChart(el, {
        height: 190, xLabel: 'self-play games', yMin: 0, yMax: 100,
        series: [{ name: `% of ${total} games won vs the scripted bot`, color: '#38d39f' }],
      });
      let best = pts[0], worstAfterBest = null;
      for (const p of pts) {
        c.push(p.games, [(p.eval.wins / total) * 100]);
        if (p.eval.wins > best.eval.wins) { best = p; worstAfterBest = null; }
        else if (p.games > best.games && (!worstAfterBest || p.eval.wins < worstAfterBest.eval.wins)) {
          worstAfterBest = p;
        }
      }
      c.draw();
      const drops = [];
      for (let i = 1; i < pts.length; i++) {
        const d = pts[i - 1].eval.wins - pts[i].eval.wins;
        if (d >= 4) drops.push({ d, text: `${pts[i - 1].games.toLocaleString()} → ${pts[i].games.toLocaleString()} games (${pts[i - 1].eval.wins} → ${pts[i].eval.wins} wins)` });
      }
      drops.sort((a, b) => b.d - a.d);
      if (note) {
        note.textContent = `Best checkpoint: ${best.games.toLocaleString()} games, ` +
          `${best.eval.wins} of ${total}. ` +
          (drops.length
            ? `${drops.length} place${drops.length === 1 ? '' : 's'} where more training made it measurably worse, the largest being ${drops[0].text}.`
            : 'This particular run happened to climb without a serious setback.');
      }
    })();

    /* =================================================================
       The measured search ladder, straight from the model file
       ================================================================= */
    (function fillLadderTable() {
      const body = document.querySelector('#ladder-table tbody');
      const note = document.getElementById('ladder-note');
      if (!body) return;
      const L = STAGES && STAGES.searchLadder;
      if (!L) {
        body.innerHTML = '<tr><td class="num">—</td><td>Run <code>node tools/train-connect4.cjs</code> ' +
          'to generate the checkpoints and this table.</td></tr>';
        return;
      }
      const keys = Object.keys(L).map(Number).sort((a, b) => a - b);
      let bestKey = keys[0];
      for (const k of keys) if (L[k].wins > L[bestKey].wins) bestKey = k;
      body.innerHTML = '<tr><th>thinking time</th><th>won</th><th>lost</th><th>drawn</th></tr>' +
        keys.map((k) => {
          const r = L[k];
          return `<tr class="${k === bestKey ? 'best' : ''}"><td class="num">${k} sim${k === 1 ? '' : 's'}</td>` +
            `<td class="num">${r.wins}</td><td class="num">${r.losses}</td><td class="num">${r.draws}</td></tr>`;
        }).join('');
      note.textContent = `Same network in every row — only the amount of thinking changes. ` +
        `Trained for ${(STAGES.games || 0).toLocaleString()} self-play games.`;
    })();

    /* =================================================================
       Loop
       ================================================================= */
    let lastEvalGames = 0, lastTick = performance.now(), gamesAtTick = 0;

    rafLoop((dt) => {
      if (training) {
        trainer.trainSlice(90);
        setStat('self-play games', trainer.games.toLocaleString());
        setStat('updates', trainer.steps.toLocaleString());
        setStat('policy loss', trainer.lossP.toFixed(3));
        setStat('value loss', trainer.lossV.toFixed(3));
        const now = performance.now();
        if (now - lastTick > 1000) {
          setStat('games/sec', (((trainer.games - gamesAtTick) * 1000) / (now - lastTick)).toFixed(1));
          gamesAtTick = trainer.games;
          lastTick = now;
        }
        if (trainer.games - lastEvalGames >= 25) {
          lastEvalGames = trainer.games;
          chart.push(trainer.games, [trainer.lossP, trainer.lossV]);
          chart.draw();
        }
      }
      if (fx.busy) {
        fx.begin(ctx);
        fx.end(ctx, dt, ctx._cssW, ctx._cssH);
      }
    }).start();

    /* ---------------- boot ---------------- */
    const resizeAll = () => { resizeBoard(); resizeThink(); resizeTree(); draw(); drawThink(); drawTree(); };
    resizeAll();
    window.addEventListener('resize', resizeAll);
    loadRung(rungIdx);
    resetTrainUI();
    newGame();
    drawTree();

    // handy from the console, and used by the tests
    window.__c4 = { get board() { return board; }, search, trainer, drop: humanDrop, rungs };
  });
})();
