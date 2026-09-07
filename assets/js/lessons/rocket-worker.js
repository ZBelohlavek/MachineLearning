/* ==========================================================================
   rocket-worker.js — runs PPO training off the main thread.

   Training and rendering were competing for the same thread, so training got a
   ~14ms slice of every frame. In a worker it runs flat out while the arena
   animates smoothly, which roughly triples the episodes per second.

   The worker owns the trainer while it is running; the page keeps a mirror of
   the policy weights for rendering and inspection, refreshed after every slice.
   The page falls back to training inline when workers are unavailable — which
   includes opening the site straight from disk in some browsers.
   ========================================================================== */

/* eslint-env worker */
importScripts('../lib/nn.js', './rocket-env.js', './rocket-train.js');

const ML = self.ML;
let trainer = null;
let running = false;

function loop() {
  if (!running || !trainer) return;
  const rows = trainer.trainSlice(120);      // long slices: nothing here has to stay responsive
  sendSlice(rows);
  setTimeout(loop, 0);                       // yield so incoming messages are handled
}

self.onmessage = (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init': {
        trainer = new ML.Trainer({
          hp: msg.hp, rewards: msg.rewards, opponent: msg.opponent, seed: msg.seed,
        });
        if (msg.weights) trainer.policy.fromFlat(new Float32Array(msg.weights));
        if (msg.episodes) trainer.episodes = msg.episodes;
        if (msg.spread != null) trainer.spread = msg.spread;
        self.postMessage({ type: 'ready', params: trainer.policy.numParams() });
        break;
      }
      case 'config': {
        if (!trainer) return;
        if (msg.rewards) trainer.setRewards(msg.rewards);
        if (msg.hp) Object.assign(trainer.hp, msg.hp);
        if (msg.opponent) trainer.opponent = msg.opponent;
        break;
      }
      case 'seed': {
        if (!trainer) return;
        trainer.setSeed(msg.seed);
        sendSlice([]);
        break;
      }
      case 'weights': {                     // the page has swapped the policy in
        if (!trainer) return;
        trainer.policy.fromFlat(new Float32Array(msg.weights));
        trainer.frozen.fromFlat(new Float32Array(msg.weights));
        if (msg.episodes != null) trainer.episodes = msg.episodes;
        if (msg.spread != null) trainer.spread = msg.spread;
        break;
      }
      case 'train': {                     // one slice, on request
        if (!trainer) return;
        sendSlice(trainer.trainSlice(msg.budget || 40));
        break;
      }
      case 'run': {
        // Train continuously rather than a slice per request: waiting for the
        // page to ask again leaves the worker idle for part of every frame.
        running = true;
        loop();
        break;
      }
      case 'stop': {
        running = false;
        break;
      }
      case 'reset': {
        if (!trainer) return;
        trainer.reset(true);
        sendSlice([]);
        break;
      }
      default: break;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};

function sendSlice(rows) {
  const flat = trainer.policy.toFlat();
  self.postMessage({
    type: 'slice',
    rows,
    episodes: trainer.episodes,
    updates: trainer.updates,
    spread: trainer.spread,
    termSums: trainer.lastTermSums,
    weights: flat.buffer,
  }, [flat.buffer]);
}
