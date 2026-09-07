/* ==========================================================================
   tools/train-rocket.cjs — trains the Rocket League agent offline and writes
   a series of checkpoints to assets/models/rocket-stages.json.

       node tools/train-rocket.cjs [totalEpisodes]

   The site ships those checkpoints so the lesson can do two things without
   making anyone wait: play a fully trained agent immediately, and scrub
   through the whole training run as a timelapse.
   ========================================================================== */

const fs = require('fs');
const path = require('path');
const { Trainer } = require(path.join(__dirname, '../assets/js/lessons/rocket-train.js'));
const { REWARD_PRESETS } = require(path.join(__dirname, '../assets/js/lessons/rocket-env.js'));

const TOTAL = +(process.argv[2] || 12000);

/* Which reward function to train under. "recommended" produces the agent the
   lesson ships; "chaser" produces the cautionary one — heavy rewards for
   touching and approaching the ball, almost nothing for scoring — so the
   lesson can show a fully trained failure next to a fully trained success. */
const PRESET = process.argv[3] || 'recommended';
const REWARDS = REWARD_PRESETS[PRESET];
if (!REWARDS) {
  console.error(`unknown preset "${PRESET}". Options: ${Object.keys(REWARD_PRESETS).join(', ')}`);
  process.exit(1);
}

/* Where to snapshot. Dense early on, because that is where the behaviour
   changes fastest and where the timelapse is most interesting. */
const STAGES = [0, 100, 250, 500, 1000, 1800, 3000, 5000, 7500, 10000, 12000]
  .filter((e) => e <= TOTAL);

const LABELS_BY_PRESET = {
  chaser: {
    0: 'Untrained — random twitching',
    100: 'Noticing the ball exists',
    250: 'Turning toward the ball',
    500: 'Driving at the ball',
    1000: 'Staying near the ball',
    1800: 'Following it everywhere',
    3000: 'Nudging it, aimlessly',
    5000: 'Expert ball-follower',
    7500: 'Devoted ball-follower',
    10000: 'Perfectly optimised, useless',
    12000: 'Fully trained on the wrong thing',
  },
};

const LABELS = LABELS_BY_PRESET[PRESET] || {
  0: 'Untrained — random twitching',
  100: 'First accidental touches',
  250: 'Starting to face the ball',
  500: 'Driving at the ball',
  1000: 'Hitting it somewhere',
  1800: 'Hitting it forwards',
  3000: 'Aiming at the goal',
  5000: 'Approaching from the right side',
  7500: 'Recovering and re-attacking',
  10000: 'Consistent scoring',
  12000: 'Fully trained',
};

const round = (arr, dp = 4) => Array.from(arr, (v) => +v.toFixed(dp));
const compact = (net) => ({
  sizes: net.sizes,
  acts: net.layers.map((l) => l.act),
  layers: net.layers.map((l) => ({ W: round(l.W), b: round(l.b) })),
});

const trainer = new Trainer({ opponent: 'self', seed: 987654, rewards: REWARDS });
const stages = [];
const curve = [];

console.log(`training to ${TOTAL} episodes, snapshotting at ${STAGES.join(', ')}`);
const t0 = Date.now();
let nextStage = 0;

while (trainer.episodes < TOTAL) {
  if (nextStage < STAGES.length && trainer.episodes >= STAGES[nextStage]) {
    const target = STAGES[nextStage];
    const evalVsScripted = trainer.evaluate(20, 'scripted');
    const evalSolo = trainer.evaluate(20, 'none');
    const recent = trainer.history[trainer.history.length - 1] || {};
    stages.push({
      episodes: trainer.episodes,
      label: LABELS[target] || `${target} episodes`,
      goalRate: +(recent.scoreRate || 0).toFixed(3),
      touches: +(recent.touches || 0).toFixed(2),
      entropy: +(recent.entropy || Math.log(9)).toFixed(3),
      spread: +(trainer.spread || 1).toFixed(2),
      vsScripted: { scored: evalVsScripted.scored, conceded: evalVsScripted.conceded },
      emptyNet: evalSolo.scored,
      policy: compact(trainer.policy),
    });
    console.log(
      `  stage ${stages.length}/${STAGES.length}  ep=${trainer.episodes}  ` +
      `goalRate=${(recent.scoreRate || 0).toFixed(2)}  touches=${(recent.touches || 0).toFixed(1)}  ` +
      `vs scripted ${evalVsScripted.scored}-${evalVsScripted.conceded}  ` +
      `empty net ${evalSolo.scored}/20  (${((Date.now() - t0) / 1000).toFixed(0)}s elapsed)`
    );
    nextStage++;
  }
  const rows = trainer.trainSlice(900);
  for (const r of rows) {
    curve.push({ e: r.episodes, r: +r.reward.toFixed(2), g: +r.scoreRate.toFixed(3),
                 t: +r.touches.toFixed(2), h: +r.entropy.toFixed(3), d: +r.spread.toFixed(2) });
  }
}

// final stage
{
  const evalVsScripted = trainer.evaluate(30, 'scripted');
  const evalSolo = trainer.evaluate(30, 'none');
  const recent = trainer.history[trainer.history.length - 1] || {};
  stages.push({
    episodes: trainer.episodes,
    label: LABELS[TOTAL] || 'Fully trained',
    goalRate: +(recent.scoreRate || 0).toFixed(3),
    touches: +(recent.touches || 0).toFixed(2),
    entropy: +(recent.entropy || 0).toFixed(3),
    spread: +(trainer.spread || 1).toFixed(2),
    vsScripted: { scored: evalVsScripted.scored, conceded: evalVsScripted.conceded },
    emptyNet: evalSolo.scored,
    policy: compact(trainer.policy),
  });
  console.log(`  final  ep=${trainer.episodes}  vs scripted ${evalVsScripted.scored}-${evalVsScripted.conceded} ` +
              `over 30, empty net ${evalSolo.scored}/30`);
}

const out = {
  version: 1,
  preset: PRESET,
  trainedAt: new Date().toISOString().slice(0, 10),
  episodes: trainer.episodes,
  hp: trainer.hp,
  rewards: trainer.rewards,
  note: 'Trained by tools/train-rocket.cjs with PPO self-play. Policy weights only — ' +
        'the value network is not needed for playback.',
  stages,
  curve: curve.filter((_, i) => i % 2 === 0),
};

/* Written as a .js file that assigns a global rather than raw .json: browsers
   block fetch() on file:// URLs, and this site is meant to work when you just
   double-click index.html. A <script> tag has no such restriction. */
const suffix = PRESET === 'recommended' ? '' : '-' + PRESET;
const globalName = PRESET === 'recommended' ? 'ML_ROCKET_STAGES' : 'ML_ROCKET_STAGES_CHASER';
const file = path.join(__dirname, `../assets/models/rocket-stages${suffix}.js`);
fs.writeFileSync(file,
  `/* Generated by tools/train-rocket.cjs (${PRESET} rewards) — pre-trained policies. */\n` +
  `window.${globalName} = ` + JSON.stringify(out) + ';\n');
console.log(`\nwrote ${file}  (${(fs.statSync(file).size / 1024).toFixed(0)} KB, ${stages.length} stages, ` +
            `${((Date.now() - t0) / 60000).toFixed(1)} min)`);
