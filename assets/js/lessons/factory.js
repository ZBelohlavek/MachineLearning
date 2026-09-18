/* ==========================================================================
   factory.js — build a line, then watch a search beat it.
   ========================================================================== */

;(function () {
  'use strict';
  const { Factory, FACTORY: F, FactorySearch, randomSearch, handBuilt,
          hidpi, chrome, nextLinks, achieve, pills, slider, statGrid, rafLoop,
          LineChart } = window.ML;
  const arcade = window.ML.arcade;

  const COL = {
    empty: '#0f1626', belt: '#6f8bd0', smelter: '#c0533f', assembler: '#7c5cff',
    mine: '#8d6b3f', depot: '#38d39f', rock: '#3a4560',
  };
  const TICKS = 300;
  const BUDGET = 40;

  document.addEventListener('DOMContentLoaded', () => {
    chrome('factory', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.factory);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'factory');
    nextLinks(document.getElementById('next-links'), 'kitchen', 'draft', '../');

    const fx = arcade.fx();
    let mine = new Factory({ budget: BUDGET });
    let tool = F.BELT;

    const hud = arcade.hud(document.getElementById('f-hud'), [
      { name: 'rate', label: 'gears / 100 ticks', value: '—' },
      { name: 'gears', label: 'gears made', value: '—' },
      { name: 'spend', label: 'spent', value: '0' },
      { name: 'best', label: 'your best', value: '—' },
    ]);
    arcade.soundToggle(document.getElementById('f-sound'));
    const setRun = statGrid(document.getElementById('run-stats'),
      ['ore smelted', 'plates made', 'gears made', 'delivered']);

    const MEDALS = { gold: 16, silver: 12, bronze: 6 };
    function showBest() {
      const b = arcade.readBest('factory-throughput');
      hud.set('best', b === null ? '—' : b.toFixed(1));
      const t = document.getElementById('f-medals');
      if (!t) return;
      t.innerHTML = ['gold', 'silver', 'bronze'].map((m) =>
        `<span class="${b !== null && b >= MEDALS[m] ? 'won' : ''}">` +
        `${arcade.MEDALS[m].icon} <b>${MEDALS[m]} per 100</b></span>`).join('') +
        '<span>the search usually lands between 14 and 17</span>';
    }

    /* ---------------- costs table ---------------- */
    (function costs() {
      const host = document.getElementById('cost-table');
      const rows = [['Belt', F.COSTS[F.BELT], 'moves one item a square each tick'],
                    ['Smelter', F.COSTS[F.SMELTER], `ore to plate in ${F.SMELT_TICKS} ticks`],
                    ['Assembler', F.COSTS[F.ASSEMBLER], `plate to gear in ${F.ASSEMBLE_TICKS} ticks`]];
      host.innerHTML = rows.map(([n, c, d]) =>
        `<div class="mixrow" style="grid-template-columns:86px 42px 1fr;gap:10px;margin:5px 0">` +
        `<span class="n">${n}</span><span class="v">${c}</span>` +
        `<span class="n" style="font-size:.74rem;color:var(--text-mute)">${d}</span></div>`).join('');
    })();

    /* ---------------- drawing ---------------- */
    function drawFactory(canvas, factory, ctxRef) {
      const ctx = ctxRef;
      if (!ctx) return;
      const cell = ctx._cssW / F.W;
      ctx.clearRect(0, 0, ctx._cssW, ctx._cssH);
      for (let y = 0; y < F.H; y++) {
        for (let x = 0; x < F.W; x++) {
          const t = factory.at(x, y);
          ctx.fillStyle = t === F.BELT ? COL.belt : t === F.SMELTER ? COL.smelter
            : t === F.ASSEMBLER ? COL.assembler : t === F.MINE ? COL.mine
            : t === F.DEPOT ? COL.depot : t === F.ROCK ? COL.rock : COL.empty;
          ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
          if (t === F.BELT) {
            // an arrow, so the direction is readable at a glance
            const d = factory.dirAt(x, y);
            const cx = x * cell + cell / 2, cy = y * cell + cell / 2;
            const [dx, dy] = F.DIRS[d];
            ctx.strokeStyle = '#0b1220';
            ctx.lineWidth = Math.max(2, cell * 0.09);
            ctx.beginPath();
            ctx.moveTo(cx - dx * cell * 0.22, cy - dy * cell * 0.22);
            ctx.lineTo(cx + dx * cell * 0.22, cy + dy * cell * 0.22);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx + dx * cell * 0.22, cy + dy * cell * 0.22);
            ctx.lineTo(cx + dy * cell * 0.13 - dx * cell * 0.04, cy - dx * cell * 0.13 - dy * cell * 0.04);
            ctx.lineTo(cx - dy * cell * 0.13 - dx * cell * 0.04, cy + dx * cell * 0.13 - dy * cell * 0.04);
            ctx.closePath();
            ctx.fillStyle = '#0b1220';
            ctx.fill();
          } else if (t === F.MINE || t === F.DEPOT || t === F.SMELTER || t === F.ASSEMBLER) {
            ctx.fillStyle = 'rgba(0,0,0,.65)';
            ctx.font = `700 ${Math.max(9, cell * 0.26)}px system-ui, sans-serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(t === F.MINE ? 'ORE' : t === F.DEPOT ? 'OUT'
                         : t === F.SMELTER ? 'SMELT' : 'ASSY',
                         x * cell + cell / 2, y * cell + cell / 2);
          }
        }
      }
    }

    const gridCanvas = document.getElementById('grid');
    const bestCanvas = document.getElementById('best-grid');
    let gctx, bctx;
    function resize() {
      for (const [c, set] of [[gridCanvas, (v) => { gctx = v; }], [bestCanvas, (v) => { bctx = v; }]]) {
        c.style.width = '';
        const w = c.clientWidth || c.parentElement.clientWidth || 420;
        set(hidpi(c, w, Math.round((w * F.H) / F.W)));
      }
    }

    function redraw() {
      drawFactory(gridCanvas, mine, gctx);
      drawFactory(bestCanvas, search.best, bctx);
      const cost = mine.cost();
      document.getElementById('cost-label').textContent = `${cost} / ${BUDGET}`;
      const bar = document.getElementById('cost-bar');
      bar.style.width = Math.min(100, (cost / BUDGET) * 100).toFixed(0) + '%';
      bar.classList.toggle('over', cost > BUDGET);
      hud.set('spend', cost);
    }

    /* ---------------- building ---------------- */
    gridCanvas.addEventListener('click', (e) => {
      const r = gridCanvas.getBoundingClientRect();
      const x = Math.floor(((e.clientX - r.left) / r.width) * F.W);
      const y = Math.floor(((e.clientY - r.top) / r.height) * F.H);
      if (!F.inside(x, y)) return;
      const t = mine.at(x, y);
      if (tool === 'erase') {
        mine.remove(x, y);
      } else if (t === tool && tool === F.BELT) {
        mine.dirs[F.idx(x, y)] = (mine.dirAt(x, y) + 1) & 3;   // rotate on re-click
      } else {
        const wouldCost = mine.cost() - (F.COSTS[t] || 0) + F.COSTS[tool];
        if (wouldCost > BUDGET) {
          document.getElementById('build-note').textContent = 'over budget — clear something first';
          arcade.sfx('fail');
          return;
        }
        mine.place(x, y, tool, 1);
      }
      document.getElementById('build-note').textContent = 'click a belt again to turn it';
      redraw();
    });

    const TOOLS = [
      { value: F.BELT, label: 'Belt', hint: `cost ${F.COSTS[F.BELT]} · click again to turn` },
      { value: F.SMELTER, label: 'Smelter', hint: `cost ${F.COSTS[F.SMELTER]} · ore → plate` },
      { value: F.ASSEMBLER, label: 'Assembler', hint: `cost ${F.COSTS[F.ASSEMBLER]} · plate → gear` },
      { value: 'erase', label: 'Clear', hint: 'take it back' },
    ];
    (function drawTools() {
      const host = document.getElementById('tools');
      host.innerHTML = '';
      TOOLS.forEach((t) => {
        const b = document.createElement('button');
        b.className = 'tool' + (t.value === tool ? ' on' : '');
        b.type = 'button';
        b.innerHTML = `<b>${t.label}</b><span>${t.hint}</span>`;
        b.addEventListener('click', () => {
          tool = t.value;
          [...host.children].forEach((c, i) => c.classList.toggle('on', TOOLS[i].value === tool));
        });
        host.appendChild(b);
      });
    })();

    document.getElementById('btn-run').addEventListener('click', () => {
      const s = mine.score(TICKS);
      hud.set('rate', s.throughput.toFixed(2));
      hud.set('gears', s.produced);
      hud.flash('rate', s.produced > 0 ? 'good' : 'bad');
      setRun('ore smelted', s.oreSmelted || 0);
      setRun('plates made', s.platesMade || 0);
      setRun('gears made', s.gearsMade || 0);
      setRun('delivered', s.produced);
      const res = arcade.resultCard(s.throughput, {
        tiers: MEDALS, bestKey: 'factory-throughput',
        format: (v) => v.toFixed(2),
        note: s.produced === 0
          ? (s.platesMade ? 'Plates but no gears — the second half of the line is not connected.'
             : 'Nothing came out. Follow the ore and find where it falls off.')
          : `${s.produced} gears delivered for ${s.cost} of budget.`,
      });
      document.getElementById('f-result').innerHTML = res.html;
      showBest();
      arcade.sfx(s.produced > 0 ? 'score' : 'fail');
      if (s.produced > 0) {
        achieve('factory-built', `${s.throughput.toFixed(1)} gears per 100 ticks`);
        fx.burst(gctx._cssW * 0.92, gctx._cssH * 0.45,
                 { count: 26, speed: 130, colors: ['#38d39f', '#ffd166'] });
      }
      if (s.throughput > search.throughputOf(search.best) && search.steps > 5000) {
        achieve('factory-beat', `Beat the search at ${s.throughput.toFixed(1)}`);
      }
    });
    document.getElementById('btn-example').addEventListener('click', () => {
      mine = handBuilt(BUDGET);
      document.getElementById('build-note').textContent = 'a plain straight line — now improve it';
      redraw();
    });
    document.getElementById('btn-clear').addEventListener('click', () => {
      mine.clear();
      document.getElementById('f-result').innerHTML = '';
      hud.set('rate', '—'); hud.set('gears', '—');
      redraw();
    });

    /* ---------------- the search ---------------- */
    let search = new FactorySearch({ seed: 21, budget: BUDGET, ticks: TICKS });
    let searching = false;
    const setSearch = statGrid(document.getElementById('search-stats'),
      ['throughput', 'steps', 'restarts', 'temperature']);
    const chart = new LineChart(document.getElementById('search-chart'), {
      height: 175, xLabel: 'edits tried', yMin: 0,
      series: [{ name: 'best gears per 100 ticks', color: '#38d39f' }],
    });

    function refreshSearch() {
      setSearch('throughput', search.throughputOf(search.best).toFixed(2));
      setSearch('steps', search.steps.toLocaleString());
      setSearch('restarts', search.restarts);
      setSearch('temperature', search.temp.toFixed(3));
      document.getElementById('search-state').textContent =
        search.steps ? `${search.steps.toLocaleString()} edits tried` : '';
    }

    document.getElementById('btn-search').addEventListener('click', () => {
      searching = !searching;
      const b = document.getElementById('btn-search');
      b.textContent = searching ? '⏸ Pause' : '▶ Run the search';
      b.classList.toggle('primary', !searching);
    });
    document.getElementById('btn-search-reset').addEventListener('click', () => {
      searching = false;
      const b = document.getElementById('btn-search');
      b.textContent = '▶ Run the search';
      b.classList.add('primary');
      search.reset();
      chart.clear(); chart.draw();
      refreshSearch(); redraw();
    });
    document.getElementById('btn-copy').addEventListener('click', () => {
      mine = search.best.clone();
      document.getElementById('build-note').textContent = 'the search’s layout — see if you can trim it';
      redraw();
    });

    slider(document.getElementById('search-hp'), {
      label: 'starting temperature', min: 0, max: 6, step: 0.25, value: search.opts.temp0,
      format: (v) => v.toFixed(2),
      onInput: (v) => { search.opts.temp0 = v; },
      desc: 'How willing it is to accept a change that makes things worse. At zero it is plain hill climbing and gets stuck in the first decent layout it finds.',
    });
    slider(document.getElementById('search-hp'), {
      label: 'restart after', min: 200, max: 4000, step: 100, value: search.opts.restartAfter,
      format: (v) => v + ' idle edits',
      onInput: (v) => { search.opts.restartAfter = v | 0; },
      desc: 'How long it will polish a layout that is going nowhere before starting again.',
    });

    /* ---------------- the comparison, computed here ---------------- */
    (function compare() {
      const body = document.querySelector('#compare-table tbody');
      const note = document.getElementById('compare-note');
      body.innerHTML = '<tr><td colspan="3">measuring…</td></tr>';

      // Run it in slices so the page does not lock up while it works.
      const hand = handBuilt(BUDGET);
      const handScore = hand.score(TICKS).throughput;
      let tries = 0, randBest = 0;
      const TARGET = 20000;
      const probe = new FactorySearch({ seed: 5, budget: BUDGET, ticks: TICKS });

      const tick = () => {
        const t0 = performance.now();
        while (performance.now() - t0 < 20 && tries < TARGET) {
          const r = randomSearch(200, BUDGET, TICKS, 5 + tries);
          randBest = Math.max(randBest, r.best);
          tries += 200;
        }
        render();
        if (tries < TARGET) requestAnimationFrame(tick);
        else finish();
      };

      function render() {
        const anneal = search.throughputOf(search.best);
        const rows = [
          ['A plain hand-built line', handScore, hand.cost(), handScore > 0 ? 'win' : 'zero'],
          [`Random layouts (${tries.toLocaleString()} tried)`, randBest, '—', randBest > 0 ? '' : 'zero'],
          ['Simulated annealing', anneal, search.best.cost(), anneal > 0 ? 'win' : 'zero'],
        ];
        body.innerHTML = '<tr><th>approach</th><th>cost</th><th>gears / 100 ticks</th></tr>' +
          rows.map(([n, v, c, cls]) =>
            `<tr class="${cls}"><td>${n}</td><td class="n">${c}</td><td class="n">${v.toFixed(2)}</td></tr>`).join('');
      }

      function finish() {
        render();
        note.textContent =
          `Same simulator, same budget of ${BUDGET}, same ${TICKS}-tick run for every row. ` +
          `Random sampling tried ${TARGET.toLocaleString()} layouts and produced ` +
          (randBest > 0 ? `${randBest.toFixed(2)}.` : 'nothing at all — not one working factory.') +
          ' Run the search above and watch the third row move.';
        if (randBest === 0) achieve('factory-sparse', 'Saw random search find nothing in 20,000 tries');
      }
      requestAnimationFrame(tick);
    })();

    /* ---------------- loop ---------------- */
    resize();
    window.addEventListener('resize', () => { resize(); redraw(); });
    showBest();
    refreshSearch();
    redraw();

    let lastChart = 0;
    rafLoop((dt) => {
      if (searching) {
        search.run(60);
        refreshSearch();
        if (search.steps - lastChart >= 800) {
          lastChart = search.steps;
          chart.push(search.steps, [search.throughputOf(search.best)]);
          chart.draw();
          drawFactory(bestCanvas, search.best, bctx);
          if (search.throughputOf(search.best) >= 14) {
            achieve('factory-searched', `The search reached ${search.throughputOf(search.best).toFixed(1)}`);
          }
        }
      }
      if (fx.busy && gctx) {
        fx.begin(gctx);
        fx.end(gctx, dt, gctx._cssW, gctx._cssH);
      }
    }).start();

    // handy from the console, and used by the tests
    window.__factory = {
      get mine() { return mine; }, get search() { return search; },
      F, handBuilt, randomSearch,
      runSearch: (ms) => search.run(ms),
      setTool: (t) => { tool = t; },
    };
  });
})();
