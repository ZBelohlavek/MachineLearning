/* ==========================================================================
   connect4-train.js — AlphaZero's training loop, at a size that fits in a tab.

   The loop has three parts and no human input anywhere in it:

     1. The current network plays itself, using tree search to choose moves.
     2. Every position is labelled with what the search decided there, and
        with who eventually won.
     3. The network is trained to predict both: the search's move preferences,
        and the game's result.

   Then the better network makes the search better, and it goes round again.
   Nobody ever tells it that the centre column is good.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const req = (name) => (typeof module === 'object' && module.exports
    ? require(name) : root.ML);
  const { MLP, mulberry32, softmax } = req('../lib/nn.js');
  const { MCTS, pickMove } = req('../lib/mcts.js');
  const { C4 } = req('./connect4-env.js');

  const IN = C4.N * 2;          // 84: two planes of the board
  const OUT = C4.W + 1;         // 7 move logits + 1 value

  const DEFAULT_HP = {
    sims: 100,                  // tree search simulations per move
    lr: 0.002,
    batch: 64,
    bufferSize: 12000,
    hidden: 96,
    cPuct: 1.4,
    dirichlet: 0.9,             // noise concentration at the self-play root
    noiseFrac: 0.25,
    openingMoves: 8,            // plies played at temperature 1, for variety
    valueWeight: 1,
    trainPerGame: 8,            // minibatches after each self-play game
  };

  /** The game, in the shape the search expects. */
  const GAME = {
    nActions: C4.W,
    legal: (b) => C4.legalMoves(b),
    apply: (b, a, p) => C4.play(b, a, p),
    undo: (b, a, row) => C4.undo(b, a, row),
    justWon: (b, a, row) => C4.winsAt(b, a, row),
    isDraw: (b) => C4.isFull(b),
    other: (p) => (p === C4.P1 ? C4.P2 : C4.P1),
  };

  class C4Trainer {
    constructor(opts = {}) {
      this.hp = Object.assign({}, DEFAULT_HP, opts.hp);
      this.seed = opts.seed ?? 7;
      this.reset(true);
    }

    reset(hard = false) {
      if (hard) {
        this.rand = mulberry32(this.seed);
        this.net = new MLP([IN, this.hp.hidden, this.hp.hidden, OUT],
                           { hidden: 'relu', out: 'linear', rand: this.rand, outScale: 0.1 });
      }
      this.buffer = [];
      this.bufAt = 0;
      this.games = 0;
      this.steps = 0;
      this.history = [];
      this.lossP = 0;
      this.lossV = 0;
      this.obs = new Float32Array(IN);
      this.mcts = new MCTS(GAME, (b, p) => this.policyValue(b, p),
                           { cPuct: this.hp.cPuct, rand: this.rand });
    }

    /** One network call: move priors and a guess at who is winning. */
    policyValue(board, player) {
      C4.encode(board, player, this.obs);
      const out = this.net.forward(this.obs);
      const logits = new Float32Array(C4.W);
      for (let a = 0; a < C4.W; a++) logits[a] = out[a];
      return { P: softmax(logits), v: Math.tanh(out[C4.W]) };
    }

    /** What the agent would play, with no noise: the move the search liked most. */
    bestMove(board, player, sims = this.hp.sims, temperature = 0) {
      const res = this.mcts.search(board, player, sims);
      return { move: pickMove(res.visits, temperature, this.rand), ...res };
    }

    push(sample) {
      if (this.buffer.length < this.hp.bufferSize) this.buffer.push(sample);
      else { this.buffer[this.bufAt] = sample; this.bufAt = (this.bufAt + 1) % this.hp.bufferSize; }
    }

    /**
     * One complete self-play game. Every position is stored with the search's
     * move distribution; the result is stitched on once the game ends.
     */
    selfPlayGame() {
      const board = C4.newBoard();
      let player = C4.P1;
      const trail = [];
      let winner = 0, plies = 0;

      for (;;) {
        const legal = C4.legalMoves(board);
        if (!legal.length) break;
        const res = this.mcts.search(board, player, this.hp.sims, {
          dirichlet: this.hp.dirichlet, noiseFrac: this.hp.noiseFrac,
        });
        const temp = plies < this.hp.openingMoves ? 1 : 0;
        const move = pickMove(res.visits, temp, this.rand);
        if (move < 0) break;

        trail.push({ x: C4.encode(board, player), pi: res.policy, player });
        const row = C4.play(board, move, player);
        plies++;
        if (C4.winsAt(board, move, row)) { winner = player; break; }
        if (C4.isFull(board)) break;
        player = GAME.other(player);
      }

      // z is +1 if the player to move at that position went on to win.
      for (const t of trail) {
        t.z = winner === 0 ? 0 : (winner === t.player ? 1 : -1);
        this.push(t);
      }
      this.games++;
      return { winner, plies };
    }

    /**
     * One minibatch. The policy head is trained towards the search's visit
     * distribution, the value head towards the game's actual result — the
     * network is learning to shortcut a search it already ran.
     */
    trainBatch() {
      const n = Math.min(this.hp.batch, this.buffer.length);
      if (n < 8) return null;
      this.net.zeroGrad();
      let lp = 0, lv = 0;
      const d = new Float32Array(OUT);

      for (let i = 0; i < n; i++) {
        const s = this.buffer[(this.rand() * this.buffer.length) | 0];
        const out = this.net.forward(s.x);

        const logits = new Float32Array(C4.W);
        for (let a = 0; a < C4.W; a++) logits[a] = out[a];
        const p = softmax(logits);
        const v = Math.tanh(out[C4.W]);

        // cross-entropy on the policy: the gradient is simply p - pi
        for (let a = 0; a < C4.W; a++) {
          d[a] = (p[a] - s.pi[a]) / n;
          if (s.pi[a] > 0) lp -= s.pi[a] * Math.log(Math.max(1e-9, p[a]));
        }
        // squared error on the value, through the tanh
        const diff = v - s.z;
        d[C4.W] = (this.hp.valueWeight * 2 * diff * (1 - v * v)) / n;
        lv += diff * diff;

        this.net.backward(d);
      }

      this.net.clipGradients(5);
      this.net.step(this.hp.lr);
      this.steps++;
      this.lossP = lp / n;
      this.lossV = lv / n;
      return { lossP: this.lossP, lossV: this.lossV };
    }

    /** Play games and train for roughly `ms` milliseconds. */
    trainSlice(ms = 120) {
      const t0 = Date.now();
      let played = 0;
      while (Date.now() - t0 < ms) {
        this.selfPlayGame();
        for (let i = 0; i < this.hp.trainPerGame; i++) this.trainBatch();
        played++;
      }
      return played;
    }

    /**
     * Measure strength against the fixed scripted opponent, alternating who
     * starts. Self-play scores tell you nothing, exactly as in Rocket League.
     */
    evaluate(games = 20, sims = this.hp.sims) {
      let wins = 0, losses = 0, draws = 0;
      for (let g = 0; g < games; g++) {
        const board = C4.newBoard();
        const agentIs = g % 2 === 0 ? C4.P1 : C4.P2;
        let player = C4.P1, result = 0;
        for (;;) {
          const legal = C4.legalMoves(board);
          if (!legal.length) break;
          const move = player === agentIs
            ? pickMove(this.mcts.search(board, player, sims).visits, 0, this.rand)
            : C4.scriptedMove(board, player, this.rand);
          if (move < 0) break;
          const row = C4.play(board, move, player);
          if (C4.winsAt(board, move, row)) { result = player === agentIs ? 1 : -1; break; }
          if (C4.isFull(board)) break;
          player = GAME.other(player);
        }
        if (result > 0) wins++; else if (result < 0) losses++; else draws++;
      }
      const rec = { games: this.games, steps: this.steps, wins, losses, draws,
                    winRate: wins / games, lossP: this.lossP, lossV: this.lossV };
      this.history.push(rec);
      return rec;
    }

    toJSON() { return { hp: this.hp, games: this.games, net: this.net.toJSON() }; }

    loadJSON(obj) {
      this.net = MLP.fromJSON(obj.net);
      this.games = obj.games || 0;
      this.mcts = new MCTS(GAME, (b, p) => this.policyValue(b, p),
                           { cPuct: this.hp.cPuct, rand: this.rand });
      return this;
    }
  }

  return { C4Trainer, C4_GAME: GAME, C4_DEFAULT_HP: DEFAULT_HP, C4_IN: IN, C4_OUT: OUT };
});
