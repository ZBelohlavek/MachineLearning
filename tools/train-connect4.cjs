#!/usr/bin/env node
/* ==========================================================================
   train-connect4.cjs — offline self-play, saved as a strength ladder.

   Training AlphaZero from scratch in a browser tab works, but the first few
   hundred games are a bot that hangs pieces, and nobody wants to sit through
   that before their first game. So we run it here and save checkpoints at
   intervals: the page ships a ladder of opponents from "just initialised" to
   "actually quite hard", and the lesson can show the same agent getting
   better rather than describing it.

     node tools/train-connect4.cjs [--minutes 30] [--seed 7]
   ========================================================================== */

const fs = require('fs');
const path = require('path');
const { C4Trainer } = require('../assets/js/lessons/connect4-train.js');

const arg = (name, dflt) => {
  const i = process.argv.indexOf('--' + name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
};

const minutes = parseFloat(arg('minutes', 30));
const seed = parseInt(arg('seed', 7), 10);
// Strength during self-play training is NOT monotonic — a run can be clearly
// better at 100 games than at 300. So take plenty of checkpoints, measure each
// one properly, and let the page build its ladder from what was measured
// rather than from an assumption about how training goes.
const STAGES = [0, 40, 120, 260, 450, 700, 1000, 1400, 1900, 2500, 3200, 4000, 5000, 6200];
const EVAL_GAMES = 40;

const trainer = new C4Trainer({ seed });
const stages = [];
const round = (w) => Math.round(w * 1e4) / 1e4;

function snapshot(label) {
  const json = trainer.net.toJSON();
  return {
    label,
    games: trainer.games,
    steps: trainer.steps,
    net: {
      sizes: json.sizes,
      acts: json.acts,
      layers: json.layers.map((l) => ({ W: l.W.map(round), b: l.b.map(round) })),
    },
  };
}

console.log(`Connect 4 self-play · seed ${seed} · ${minutes} minute budget`);
console.log(`hyperparameters: ${JSON.stringify(trainer.hp)}`);

const deadline = Date.now() + minutes * 60000;
let nextStage = 0;
let lastLog = 0;

while (Date.now() < deadline) {
  while (nextStage < STAGES.length && trainer.games >= STAGES[nextStage]) {
    const evalAt = trainer.evaluate(EVAL_GAMES);
    const snap = snapshot(`${STAGES[nextStage]} games`);
    snap.eval = { wins: evalAt.wins, losses: evalAt.losses, draws: evalAt.draws };
    stages.push(snap);
    console.log(`  checkpoint ${snap.label.padEnd(11)} steps=${String(trainer.steps).padStart(6)} ` +
                `vs scripted W/L/D=${evalAt.wins}/${evalAt.losses}/${evalAt.draws}`);
    nextStage++;
  }
  if (nextStage >= STAGES.length) break;
  trainer.trainSlice(4000);
  const el = ((Date.now() - (deadline - minutes * 60000)) / 1000) | 0;
  if (el - lastLog >= 60) {
    lastLog = el;
    console.log(`  t=${el}s games=${trainer.games} steps=${trainer.steps} ` +
                `lossP=${trainer.lossP.toFixed(3)} lossV=${trainer.lossV.toFixed(3)}`);
  }
}

// Always keep the final position of the run, whatever the stage list says.
if (!stages.length || stages[stages.length - 1].games !== trainer.games) {
  const evalAt = trainer.evaluate(EVAL_GAMES);
  const snap = snapshot(`${trainer.games} games`);
  snap.eval = { wins: evalAt.wins, losses: evalAt.losses, draws: evalAt.draws };
  stages.push(snap);
  console.log(`  final      ${snap.label.padEnd(11)} vs scripted W/L/D=` +
              `${evalAt.wins}/${evalAt.losses}/${evalAt.draws}`);
}

// How much of the strength is the search rather than the network? This is the
// number the lesson quotes, so it is measured here rather than guessed.
const ladder = {};
for (const sims of [1, 25, 100, 400]) {
  const e = trainer.evaluate(EVAL_GAMES, sims);
  ladder[sims] = { wins: e.wins, losses: e.losses, draws: e.draws };
  console.log(`  search budget ${String(sims).padStart(3)} sims -> W/L/D ${e.wins}/${e.losses}/${e.draws}`);
}

/**
 * Weights are the only large thing here, so only the checkpoints that actually
 * become rungs on the ladder keep theirs. Every checkpoint keeps its measured
 * result, because the strength curve on the page needs all of them and the
 * numbers cost nothing.
 *
 * Rungs are chosen by rank on measured strength, evenly spaced, so the ladder
 * always gets harder as you climb it even though training itself does not
 * improve monotonically.
 */
function pruneStages(all, want = 5) {
  const scored = all
    .filter((s) => s.eval)
    .map((s) => ({ s, score: s.eval.wins + s.eval.draws * 0.5 }))
    .sort((a, b) => a.score - b.score);
  const keep = new Set();
  for (let i = 0; i < want; i++) {
    keep.add(scored[Math.round((i * (scored.length - 1)) / (want - 1))].s);
  }
  return all.map((s) => (keep.has(s) ? s : { label: s.label, games: s.games, steps: s.steps, eval: s.eval }));
}

const pruned = pruneStages(stages);
console.log(`  keeping weights for ${pruned.filter((s) => s.net).length} of ${pruned.length} checkpoints`);

const out = {
  version: 1,
  trainedAt: new Date().toISOString().slice(0, 10),
  seed,
  games: trainer.games,
  evalGames: EVAL_GAMES,
  hp: trainer.hp,
  searchLadder: ladder,
  stages: pruned,
};

const file = path.join(__dirname, '..', 'assets', 'models', 'connect4-stages.js');
fs.writeFileSync(file,
  '/* Generated by tools/train-connect4.cjs — self-play Connect 4 checkpoints. */\n' +
  'window.ML_C4_STAGES = ' + JSON.stringify(out) + ';\n' +
  'if (typeof module === "object" && module.exports) module.exports = window.ML_C4_STAGES;\n');
console.log(`\nwrote ${file} (${(fs.statSync(file).size / 1024).toFixed(0)} KB, ${stages.length} checkpoints)`);
