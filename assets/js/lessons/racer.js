/* ==========================================================================
   racer.js — "Evolve a driver": neuroevolution on a racetrack.

   Sixty cars, each with its own tiny neural network, all driving at once.
   The ones that get furthest have their brains copied and mutated into the
   next generation. No gradients, no backpropagation, no reward shaping —
   just survival of the fastest, which turns out to be enough.

   The track is a bitmap mask, so you can draw your own with the mouse, and a
   breadth-first search over that mask gives every cell a distance-to-finish
   used both for fitness and for the progress bar.
   ========================================================================== */

;(function () {
  'use strict';
  const { MLP, mulberry32, randn, clamp, hidpi, LineChart, achieve, dpad,
          chrome, nextLinks, slider, pills, checkbox, statGrid, rafLoop } = window.ML;

  const GW = 240, GH = 150;          // track grid, in cells
  const SENSORS = [-1.35, -0.9, -0.45, 0, 0.45, 0.9, 1.35];   // radians, relative to heading
  const SENSOR_RANGE = 42;           // cells
  const CAR = { accel: 70, drag: 0.9, maxSpeed: 52, turn: 3.1, len: 5.2, wid: 3.0 };
  const DT = 1 / 60;
  const MAX_STEPS = 780;             // 13 seconds per generation

  /* ====================================================================
     Track: a bitmap of drivable cells plus a distance-to-finish field
     ==================================================================== */
  class Track {
    constructor() {
      this.mask = new Uint8Array(GW * GH);
      this.dist = new Int32Array(GW * GH);
      this.load('circuit');
    }

    stampDisc(cx, cy, r) {
      const r2 = r * r;
      for (let y = Math.max(0, (cy - r) | 0); y <= Math.min(GH - 1, (cy + r) | 0); y++) {
        for (let x = Math.max(0, (cx - r) | 0); x <= Math.min(GW - 1, (cx + r) | 0); x++) {
          const dx = x - cx, dy = y - cy;
          if (dx * dx + dy * dy <= r2) this.mask[y * GW + x] = 1;
        }
      }
    }

    stampPath(points, width) {
      for (let i = 0; i < points.length - 1; i++) {
        const [x0, y0] = points[i], [x1, y1] = points[i + 1];
        const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0));
        for (let s = 0; s <= steps; s++) {
          const t = s / Math.max(1, steps);
          this.stampDisc(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, width);
        }
      }
    }

    load(name) {
      this.mask.fill(0);
      /* Every course runs from a start to a finish that are far apart, and no
         two stretches of road pass within a road-width of each other. Both
         matter: if the finish sits next to the start, the distance field makes
         "reverse a little" a winning move and nothing ever learns to drive. */
      const COURSES = {
        circuit: [[30, 126], [196, 126], [216, 96], [196, 66], [58, 66], [34, 44], [60, 22], [204, 22]],
        snake: [[25, 28], [70, 28], [96, 58], [70, 88], [96, 118], [152, 118], [178, 90], [152, 60], [178, 30], [216, 30]],
        hairpin: [[25, 126], [190, 126], [215, 100], [190, 76], [58, 76], [34, 52], [60, 26], [216, 26]],
        grand: [[25, 126], [92, 126], [116, 100], [92, 74], [30, 74], [26, 44], [58, 22], [140, 22],
                [172, 48], [148, 78], [176, 104], [216, 104], [216, 60], [200, 34]],
      };
      const P = COURSES[name] || COURSES.circuit;
      this.path = P;
      this.stampPath(P, 11);
      this.start = { x: P[0][0], y: P[0][1] };
      this.startAngle = Math.atan2(P[1][1] - P[0][1], P[1][0] - P[0][0]);
      this.finish = { x: P[P.length - 1][0], y: P[P.length - 1][1] };
      this.computeDistance();
    }

    /** BFS from the finish over drivable cells: distance-to-finish, in cells. */
    computeDistance() {
      this.dist.fill(-1);
      const q = new Int32Array(GW * GH);
      let head = 0, tail = 0;
      const push = (x, y, d) => {
        if (x < 0 || y < 0 || x >= GW || y >= GH) return;
        const i = y * GW + x;
        if (!this.mask[i] || this.dist[i] >= 0) return;
        this.dist[i] = d;
        q[tail++] = i;
      };
      // seed from a disc around the finish so the goal has some width
      for (let dy = -6; dy <= 6; dy++) for (let dx = -6; dx <= 6; dx++) {
        if (dx * dx + dy * dy <= 36) push((this.finish.x + dx) | 0, (this.finish.y + dy) | 0, 0);
      }
      while (head < tail) {
        const i = q[head++];
        const x = i % GW, y = (i / GW) | 0, d = this.dist[i] + 1;
        push(x + 1, y, d); push(x - 1, y, d); push(x, y + 1, d); push(x, y - 1, d);
      }
      const si = ((this.start.y | 0) * GW + (this.start.x | 0));
      this.startDist = Math.max(1, this.dist[si] < 0 ? 400 : this.dist[si]);
    }

    onRoad(x, y) {
      const xi = x | 0, yi = y | 0;
      if (xi < 0 || yi < 0 || xi >= GW || yi >= GH) return false;
      return this.mask[yi * GW + xi] === 1;
    }

    distanceAt(x, y) {
      const xi = clamp(x | 0, 0, GW - 1), yi = clamp(y | 0, 0, GH - 1);
      const d = this.dist[yi * GW + xi];
      return d < 0 ? this.startDist : d;
    }
  }

  /* ====================================================================
     Cars
     ==================================================================== */
  class Car {
    constructor(brain) {
      this.brain = brain;
      this.sensors = new Float32Array(SENSORS.length + 1);
      this.reset();
    }
    reset(track, jitter = 0, rand = Math.random) {
      this.x = track ? track.start.x : 0;
      this.y = track ? track.start.y : 0;
      this.angle = (track ? track.startAngle : 0) + (jitter ? (rand() - 0.5) * 0.25 : 0);
      this.speed = 0;
      this.alive = true;
      this.finished = false;
      this.steps = 0;
      this.best = track ? track.startDist : 0;
      this.fitness = 0;
      this.trail = [];
    }

    /** Cast the whisker sensors and normalise them into the network's input. */
    sense(track) {
      const s = this.sensors;
      for (let i = 0; i < SENSORS.length; i++) {
        const a = this.angle + SENSORS[i];
        const cx = Math.cos(a), cy = Math.sin(a);
        let d = 0;
        while (d < SENSOR_RANGE) {
          d += 1.5;
          if (!track.onRoad(this.x + cx * d, this.y + cy * d)) break;
        }
        s[i] = Math.min(d, SENSOR_RANGE) / SENSOR_RANGE;
      }
      s[SENSORS.length] = this.speed / CAR.maxSpeed;
      return s;
    }

    step(track, controls) {
      if (!this.alive) return;
      const [steer, throttle] = controls;
      this.speed += (throttle * CAR.accel - CAR.drag * this.speed) * DT;
      this.speed = clamp(this.speed, -6, CAR.maxSpeed);
      this.angle += steer * CAR.turn * Math.min(1, Math.abs(this.speed) / 14) * DT;
      this.x += Math.cos(this.angle) * this.speed * DT;
      this.y += Math.sin(this.angle) * this.speed * DT;
      this.steps++;

      if (!track.onRoad(this.x, this.y)) { this.alive = false; return; }

      const d = track.distanceAt(this.x, this.y);
      if (d < this.best) this.best = d;
      if (d < 7) { this.finished = true; this.alive = false; }

      if (this.steps % 4 === 0) {
        this.trail.push(this.x, this.y);
        if (this.trail.length > 260) this.trail.splice(0, 2);
      }
    }

    /** Distance covered toward the finish, as a fraction of the whole course. */
    progress(track) {
      return clamp((track.startDist - this.best) / track.startDist, 0, 1);
    }

    score(track) {
      // Getting further is what matters; finishing quickly is the tie-break.
      const p = this.progress(track);
      return p * 1000 + (this.finished ? 600 - this.steps * 0.35 : 0);
    }
  }

  /* ====================================================================
     Page
     ==================================================================== */
  document.addEventListener('DOMContentLoaded', () => {
    chrome('racer', '../');
    window.ML.quiz(document.getElementById('quiz'), window.ML.QUIZZES.racer);
    window.ML.goalPanel(document.getElementById('lesson-goals'), 'racer');
    nextLinks(document.getElementById('next-links'), 'gridworld', 'rocket', '../');

    let seed = 20240607;
    let rand = mulberry32(seed);
    const track = new Track();
    window.__racer = { track };            // handy from the console, and used by the tests

    let popSize = 60, mutation = 0.12, elitePct = 0.2, speedMult = 3;
    let showAll = true, showSensors = true;
    let generation = 0, stepInGen = 0, bestEver = null, bestEverScore = -1;
    let championTime = null, mode = 'evolve';

    /* ---------------- population ---------------- */
    let cars = [];
    function newBrain() {
      return new MLP([SENSORS.length + 1, 8, 2], { hidden: 'tanh', out: 'tanh', rand });
    }
    function newPopulation() {
      cars = Array.from({ length: popSize }, () => new Car(newBrain()));
      generation = 1;
      stepInGen = 0;
      bestEver = null; bestEverScore = -1;
      championTime = null;
      chart.clear();
      resetCars();
      history.length = 0;
    }
    function resetCars() {
      for (const c of cars) c.reset(track, 1, rand);
      stepInGen = 0;
    }

    function mutate(src, rate) {
      const child = newBrain();
      src.layers.forEach((l, i) => {
        const t = child.layers[i];
        for (let k = 0; k < l.W.length; k++) t.W[k] = l.W[k] + randn(rand) * rate;
        for (let k = 0; k < l.b.length; k++) t.b[k] = l.b[k] + randn(rand) * rate;
      });
      return child;
    }

    const history = [];

    /** Rank the generation, keep the elites, refill with mutated copies. */
    function evolve() {
      const scored = cars.map((c) => ({ c, s: c.score(track) })).sort((a, b) => b.s - a.s);
      const best = scored[0];
      const meanProgress = cars.reduce((a, c) => a + c.progress(track), 0) / cars.length;
      const bestProgress = best.c.progress(track);
      const finishers = cars.filter((c) => c.finished).length;

      if (best.s > bestEverScore) {
        bestEverScore = best.s;
        bestEver = MLP.fromJSON(best.c.brain.toJSON());
        if (best.c.finished) {
          const t = best.c.steps * DT;
          if (championTime === null || t < championTime) championTime = t;
        }
      }

      if (finishers > 0) {
        achieve('racer-finish', `${finishers} of ${popSize} finished in generation ${generation}`);
      }
      history.push({ generation, bestProgress, meanProgress, finishers });
      chart.push(generation, [bestProgress, meanProgress]);
      chart.draw();

      const nElite = Math.max(1, Math.round(popSize * elitePct));
      const elites = scored.slice(0, nElite).map((e) => e.c.brain);
      const next = [];
      // Elites survive untouched: without this, a good brain can be lost to a
      // bad mutation and the population goes backwards.
      for (const b of elites) next.push(new Car(MLP.fromJSON(b.toJSON())));
      while (next.length < popSize) {
        const parent = elites[(rand() * elites.length) | 0];
        next.push(new Car(mutate(parent, mutation)));
      }
      cars = next;
      generation++;
      resetCars();
      updateStats(bestProgress, meanProgress, finishers);
    }

    /* ---------------- simulation ---------------- */

    /**
     * Drawing can break the course: erase the road under the start and every
     * car dies on its first tick; erase the finish and there is nothing to
     * reach. Either way the population would churn through generations with no
     * feedback, so say what is wrong instead.
     */
    function trackProblem() {
      if (!track.onRoad(track.start.x, track.start.y)) {
        return 'The start square is off the road. Draw some road under the blue S.';
      }
      const si = (track.start.y | 0) * GW + (track.start.x | 0);
      if (track.dist[si] < 0) {
        return 'There is no route from S to F any more. The road is broken somewhere between them.';
      }
      return '';
    }

    function showTrackProblem(msg) {
      const el = document.getElementById('track-warning');
      if (!el) return;
      el.textContent = msg;
      el.hidden = !msg;
    }

    function simStep() {
      let anyAlive = false;
      for (const car of cars) {
        if (!car.alive) continue;
        anyAlive = true;
        const s = car.sense(track);
        const out = car.brain.forward(s);
        car.step(track, [out[0], (out[1] + 1) / 2]);
      }
      stepInGen++;
      if (!anyAlive || stepInGen >= MAX_STEPS) evolve();
    }

    /* ---------------- rendering ---------------- */
    const canvas = document.getElementById('track-canvas');
    let ctx, scale;
    const trackCanvas = document.createElement('canvas');
    trackCanvas.width = GW; trackCanvas.height = GH;
    const tctx = trackCanvas.getContext('2d');

    function paintTrack() {
      const img = tctx.createImageData(GW, GH);
      for (let i = 0; i < GW * GH; i++) {
        const road = track.mask[i] === 1;
        const d = track.dist[i];
        const o = i * 4;
        if (road) {
          // tint the road by distance-to-finish, so progress is visible
          const t = d < 0 ? 0 : 1 - Math.min(1, d / track.startDist);
          img.data[o] = 42 + t * 26;
          img.data[o + 1] = 48 + t * 52;
          img.data[o + 2] = 66 + t * 40;
          img.data[o + 3] = 255;
        } else {
          img.data[o] = 12; img.data[o + 1] = 17; img.data[o + 2] = 28; img.data[o + 3] = 255;
        }
      }
      tctx.putImageData(img, 0, 0);
    }

    function resize() {
      canvas.style.width = '';
      const w = canvas.clientWidth || canvas.parentElement.clientWidth;
      ctx = hidpi(canvas, w, Math.round((w * GH) / GW));
      scale = ctx._cssW / GW;
    }

    function drawCar(car, colour, alpha, highlight) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(car.x * scale, car.y * scale);
      ctx.rotate(car.angle);
      ctx.fillStyle = colour;
      if (highlight) { ctx.shadowColor = colour; ctx.shadowBlur = 12; }
      ctx.fillRect(-CAR.len * scale / 2, -CAR.wid * scale / 2, CAR.len * scale, CAR.wid * scale);
      ctx.restore();
    }

    function drawSensors(car) {
      ctx.save();
      ctx.lineWidth = 1;
      for (let i = 0; i < SENSORS.length; i++) {
        const a = car.angle + SENSORS[i];
        const d = car.sensors[i] * SENSOR_RANGE;
        ctx.strokeStyle = `rgba(255,209,102,${0.15 + 0.5 * (1 - car.sensors[i])})`;
        ctx.beginPath();
        ctx.moveTo(car.x * scale, car.y * scale);
        ctx.lineTo((car.x + Math.cos(a) * d) * scale, (car.y + Math.sin(a) * d) * scale);
        ctx.stroke();
      }
      ctx.restore();
    }

    function render() {
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, ctx._cssW, ctx._cssH);
      ctx.drawImage(trackCanvas, 0, 0, ctx._cssW, ctx._cssH);

      // start and finish
      const mark = (p, colour, label) => {
        ctx.fillStyle = colour;
        ctx.beginPath();
        ctx.arc(p.x * scale, p.y * scale, 5.5 * scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#0a0f19';
        ctx.font = `700 ${Math.max(9, 3.4 * scale)}px system-ui`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(label, p.x * scale, p.y * scale);
      };
      mark(track.start, 'rgba(77,163,255,.85)', 'S');
      mark(track.finish, 'rgba(56,211,159,.9)', 'F');

      if (mode === 'race') {
        if (player) {
          if (player.trail.length > 3) {
            ctx.strokeStyle = 'rgba(77,163,255,.35)'; ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < player.trail.length; i += 2) {
              const x = player.trail[i] * scale, y = player.trail[i + 1] * scale;
              i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
            }
            ctx.stroke();
          }
          drawCar(player, '#4da3ff', 1, true);
        }
        if (rival) drawCar(rival, '#ff9f45', rival.alive ? 1 : 0.35, true);
        return;
      }

      // the pack, with the leader picked out
      let leader = null, leaderScore = -Infinity;
      for (const car of cars) {
        const s = car.score(track);
        if (s > leaderScore) { leaderScore = s; leader = car; }
      }
      if (showAll) {
        for (const car of cars) {
          if (car === leader) continue;
          drawCar(car, car.alive ? '#7c5cff' : '#3a4666', car.alive ? 0.5 : 0.22, false);
        }
      }
      if (leader) {
        if (leader.trail.length > 3) {
          ctx.strokeStyle = 'rgba(56,211,159,.45)'; ctx.lineWidth = 2;
          ctx.beginPath();
          for (let i = 0; i < leader.trail.length; i += 2) {
            const x = leader.trail[i] * scale, y = leader.trail[i + 1] * scale;
            i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          }
          ctx.stroke();
        }
        if (showSensors && leader.alive) drawSensors(leader);
        drawCar(leader, '#38d39f', 1, true);
      }

      // generation clock
      const frac = 1 - stepInGen / MAX_STEPS;
      ctx.fillStyle = 'rgba(255,255,255,.10)';
      ctx.fillRect(8, 8, ctx._cssW - 16, 3);
      ctx.fillStyle = '#7c5cff';
      ctx.fillRect(8, 8, (ctx._cssW - 16) * frac, 3);
    }

    /* ---------------- race mode ---------------- */
    let player = null, rival = null, raceTime = 0, raceResult = '';
    const keys = {};
    window.addEventListener('keydown', (e) => {
      const k = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
                  w: 'up', s: 'down', a: 'left', d: 'right' }[e.key];
      if (k && mode === 'race') { e.preventDefault(); keys[k] = true; }
    });
    window.addEventListener('keyup', (e) => {
      const k = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
                  w: 'up', s: 'down', a: 'left', d: 'right' }[e.key];
      if (k) keys[k] = false;
    });

    function startRace() {
      player = new Car(null);
      player.reset(track, 0, rand);
      rival = new Car(bestEver ? MLP.fromJSON(bestEver.toJSON()) : newBrain());
      rival.reset(track, 0, rand);
      raceTime = 0;
      raceResult = '';
      updateRaceUI();
    }

    function raceStep() {
      raceTime += DT;
      if (player.alive) {
        const steer = (keys.right ? 1 : 0) - (keys.left ? 1 : 0);
        const throttle = keys.up ? 1 : keys.down ? -0.4 : 0.05;
        player.step(track, [steer, throttle]);
        if (player.finished) {
          if (rival.finished) raceResult = 'You lost. The champion got there first.';
          else {
            raceResult = 'You win! You beat the evolved driver.';
            achieve('racer-beaten', `You finished in ${raceTime.toFixed(2)}s`);
          }
        }
        else if (!player.alive) raceResult = 'You crashed. The wall is undefeated.';
      }
      if (rival.alive) {
        const s = rival.sense(track);
        const out = rival.brain.forward(s);
        rival.step(track, [out[0], (out[1] + 1) / 2]);
        if (rival.finished && !player.finished && player.alive) raceResult = 'The champion finished first. Try again.';
        else if (!rival.alive && !rival.finished && player.alive) raceResult = 'The champion crashed. The race is yours to lose.';
      }
      updateRaceUI();
    }

    function updateRaceUI() {
      const el = document.getElementById('race-readout');
      if (!el || !player) return;
      el.innerHTML =
        `<span class="mono">${raceTime.toFixed(2)}s</span> · ` +
        `you <b style="color:var(--accent)">${(player.progress(track) * 100).toFixed(0)}%</b> · ` +
        `champion <b style="color:var(--orange, #ff9f45)">${(rival.progress(track) * 100).toFixed(0)}%</b>` +
        (raceResult ? ` — <b>${raceResult}</b>` : '');
    }

    /* ---------------- track drawing ---------------- */
    let drawingTrack = false, brushErase = false, lastCell = null;
    canvas.addEventListener('pointerdown', (e) => {
      if (!drawMode.get()) return;
      drawingTrack = true; lastCell = null; paintAt(e);
    });
    canvas.addEventListener('pointermove', (e) => { if (drawingTrack) paintAt(e); });
    window.addEventListener('pointerup', () => {
      if (!drawingTrack) return;
      drawingTrack = false;
      track.computeDistance();
      paintTrack();
    });
    function paintAt(ev) {
      const r = canvas.getBoundingClientRect();
      const x = ((ev.clientX - r.left) / r.width) * GW;
      const y = ((ev.clientY - r.top) / r.height) * GH;
      if (brushErase) {
        for (let dy = -9; dy <= 9; dy++) for (let dx = -9; dx <= 9; dx++) {
          if (dx * dx + dy * dy > 81) continue;
          const xi = (x + dx) | 0, yi = (y + dy) | 0;
          if (xi >= 0 && yi >= 0 && xi < GW && yi < GH) track.mask[yi * GW + xi] = 0;
        }
      } else {
        if (lastCell) track.stampPath([lastCell, [x, y]], 9);
        else track.stampDisc(x, y, 9);
      }
      lastCell = [x, y];
      paintTrack();
      render();
    }

    /* ---------------- controls ---------------- */
    const chart = new LineChart(document.getElementById('racer-chart'), {
      height: 150, xLabel: 'generation', yMin: 0, yMax: 1,
      series: [
        { name: 'best car', color: '#38d39f' },
        { name: 'population average', color: '#7c5cff' },
      ],
    });
    const setStat = statGrid(document.getElementById('racer-stats'),
      ['generation', 'best', 'average', 'finished', 'champion time', 'brain size']);

    const say = window.ML.announcer();

    function updateStats(bestProgress, meanProgress, finishers) {
      say(`Generation ${generation}. Best car reached ${(bestProgress * 100).toFixed(0)} percent ` +
          `of the course, ${finishers} of ${popSize} finished.`);
      setStat('generation', generation);
      setStat('best', (bestProgress * 100).toFixed(0) + '%', bestProgress > 0.99 ? 'good' : '');
      setStat('average', (meanProgress * 100).toFixed(0) + '%');
      setStat('finished', `${finishers}/${popSize}`, finishers ? 'good' : '');
      setStat('champion time', championTime === null ? '–' : championTime.toFixed(2) + 's');
      setStat('brain size', cars[0] ? cars[0].brain.numParams() + ' weights' : '–');
    }

    window.ML.runBar(document.getElementById('run-bar'), {
      seed,
      charts: () => [chart],
      onSeed: (v) => { seed = v; rand = mulberry32(seed); newPopulation(); updateStats(0, 0, 0); },
      note: 'Same seed, same starting population, so a settings change is the only difference.',
    });

    const btnRun = document.getElementById('btn-run');
    let running = false;
    btnRun.addEventListener('click', () => {
      running = !running;
      btnRun.textContent = running ? '⏸ Pause' : '▶ Evolve';
      btnRun.classList.toggle('primary', !running);
    });
    document.getElementById('btn-restart').addEventListener('click', () => {
      newPopulation(); updateStats(0, 0, 0);
    });

    pills(document.getElementById('track-presets'), [
      { value: 'circuit', label: 'Circuit' },
      { value: 'snake', label: 'Snake' },
      { value: 'hairpin', label: 'Hairpin' },
      { value: 'grand', label: 'Grand tour' },
    ], 'circuit', (v) => {
      track.load(v);
      paintTrack();
      newPopulation();
      updateStats(0, 0, 0);
    });

    const modePills = pills(document.getElementById('mode-pills'), [
      { value: 'evolve', label: 'Watch it evolve' },
      { value: 'race', label: 'Race the champion' },
    ], 'evolve', (v) => {
      mode = v;
      document.getElementById('race-panel').style.display = v === 'race' ? '' : 'none';
      if (v === 'race') { running = false; btnRun.textContent = '▶ Evolve'; btnRun.classList.add('primary'); startRace(); }
    });
    document.getElementById('btn-race-again').addEventListener('click', startRace);
    dpad(document.getElementById('race-pad'), { keys });

    const cfg = document.getElementById('racer-config');
    slider(cfg, {
      label: 'population', min: 10, max: 150, step: 5, value: popSize,
      format: (v) => v.toFixed(0), onInput: (v) => { popSize = v | 0; newPopulation(); },
      desc: 'How many cars try at once. More cars explore more of the space of possible drivers per generation, at a proportional cost in time.',
    });
    slider(cfg, {
      label: 'mutation strength', min: 0.01, max: 0.6, step: 0.01, value: mutation,
      format: (v) => v.toFixed(2), onInput: (v) => { mutation = v; },
      desc: 'How much random noise is added to a parent’s weights. Too little and the population stops improving; too much and good drivers get destroyed the moment they appear.',
    });
    slider(cfg, {
      label: 'survivors kept', min: 0.02, max: 0.6, step: 0.02, value: elitePct,
      format: (v) => (v * 100).toFixed(0) + '%', onInput: (v) => { elitePct = v; },
      desc: 'The fraction of each generation that breeds. Keep only the very best and you lose diversity; keep almost everyone and selection stops meaning anything.',
    });
    pills(cfg, [
      { value: 1, label: '1× speed' },
      { value: 3, label: '3×' },
      { value: 8, label: '8×' },
      { value: 20, label: '20× (blur)' },
    ], 3, (v) => { speedMult = v; });
    checkbox(cfg, 'Show the whole population', true, (v) => { showAll = v; });
    checkbox(cfg, 'Show the leader’s sensors', true, (v) => { showSensors = v; });

    // The brush lives beside the canvas, not in the settings panel below it —
    // you should not have to scroll away from the track to start drawing on it.
    const drawHost = document.getElementById('draw-controls');
    const drawMode = checkbox(drawHost, 'Draw on the track with the mouse', false, (v) => {
      brushPills.style.display = v ? '' : 'none';
      canvas.style.cursor = v ? 'crosshair' : 'default';
    });
    const brushPills = document.createElement('div');
    brushPills.style.display = 'none';
    drawHost.appendChild(brushPills);
    pills(brushPills, [
      { value: false, label: 'Brush: add road' },
      { value: true, label: 'Brush: erase' },
    ], false, (v) => { brushErase = v; });

    /* ---------------- go ---------------- */
    paintTrack();
    resize();
    window.addEventListener('resize', () => { resize(); render(); });
    newPopulation();
    updateStats(0, 0, 0);

    rafLoop(() => {
      const problem = trackProblem();
      showTrackProblem(problem);
      if (mode === 'race') {
        if (!problem && player && (player.alive || rival.alive)) raceStep();
      } else if (running && !problem) {
        for (let i = 0; i < speedMult; i++) simStep();
      }
      render();
    }).start();
  });
})();
