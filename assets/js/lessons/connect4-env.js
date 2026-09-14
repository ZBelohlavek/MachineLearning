/* ==========================================================================
   connect4-env.js — the rules of Connect 4, and nothing else.

   Seven columns, six rows, drop a disc and it falls. Four in a row wins.
   The whole game fits in 42 bytes, which is what makes it a good place to
   watch a search algorithm think.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const W = 7, H = 6, N = W * H;
  const EMPTY = 0, P1 = 1, P2 = 2;

  /* Board cells are row-major from the top: idx(0, 0) is the top-left.
     Discs fall, so a move lands in the lowest empty row of its column. */
  const idx = (x, y) => y * W + x;

  function newBoard() { return new Int8Array(N); }

  /** The row a disc would land in, or -1 when the column is full. */
  function dropRow(board, col) {
    for (let y = H - 1; y >= 0; y--) if (board[idx(col, y)] === EMPTY) return y;
    return -1;
  }

  function legalMoves(board) {
    const out = [];
    for (let x = 0; x < W; x++) if (board[idx(x, 0)] === EMPTY) out.push(x);
    return out;
  }

  /** Drop a disc. Returns the row it landed in, or -1 if the move was illegal. */
  function play(board, col, player) {
    const y = dropRow(board, col);
    if (y < 0) return -1;
    board[idx(col, y)] = player;
    return y;
  }

  function undo(board, col, row) { board[idx(col, row)] = EMPTY; }

  const DIRS = [[1, 0], [0, 1], [1, 1], [1, -1]];

  /**
   * Did the disc at (x, y) complete a four? Checking only around the last move
   * is what keeps the search cheap — a full-board scan per node would dominate.
   */
  function winsAt(board, x, y) {
    const p = board[idx(x, y)];
    if (p === EMPTY) return false;
    for (const [dx, dy] of DIRS) {
      let run = 1;
      for (const s of [1, -1]) {
        let cx = x + dx * s, cy = y + dy * s;
        while (cx >= 0 && cx < W && cy >= 0 && cy < H && board[idx(cx, cy)] === p) {
          run++;
          if (run >= 4) return true;
          cx += dx * s; cy += dy * s;
        }
      }
    }
    return false;
  }

  /** The four cells of the winning line, for drawing. Null if there is none. */
  function winningLine(board, x, y) {
    const p = board[idx(x, y)];
    if (p === EMPTY) return null;
    for (const [dx, dy] of DIRS) {
      const cells = [[x, y]];
      for (const s of [1, -1]) {
        let cx = x + dx * s, cy = y + dy * s;
        while (cx >= 0 && cx < W && cy >= 0 && cy < H && board[idx(cx, cy)] === p) {
          cells.push([cx, cy]);
          cx += dx * s; cy += dy * s;
        }
      }
      if (cells.length >= 4) {
        cells.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
        return cells.slice(0, 4);
      }
    }
    return null;
  }

  function isFull(board) {
    for (let x = 0; x < W; x++) if (board[idx(x, 0)] === EMPTY) return false;
    return true;
  }

  /**
   * The network's view of the position: two 42-cell planes, "my discs" and
   * "their discs", always from the side to move. Because the board is encoded
   * from the mover's point of view, one network plays both colours — the same
   * trick the Rocket League agent uses to play both ends of the pitch.
   */
  function encode(board, player, out) {
    const buf = out && out.length === N * 2 ? out : new Float32Array(N * 2);
    const them = player === P1 ? P2 : P1;
    for (let i = 0; i < N; i++) {
      buf[i] = board[i] === player ? 1 : 0;
      buf[N + i] = board[i] === them ? 1 : 0;
    }
    return buf;
  }

  /** A plain-object snapshot, so positions can be stored and replayed. */
  function clone(board) { return Int8Array.from(board); }

  /**
   * A deliberately modest opponent: takes a win, blocks a loss, otherwise
   * prefers the middle with a bit of noise. It exists so there is something
   * fixed to measure progress against, the way the scripted bot does in the
   * Rocket League lesson.
   */
  function scriptedMove(board, player, rand = Math.random) {
    const them = player === P1 ? P2 : P1;
    const moves = legalMoves(board);
    if (!moves.length) return -1;
    for (const who of [player, them]) {          // win first, then block
      for (const c of moves) {
        const y = play(board, c, who);
        const win = winsAt(board, c, y);
        undo(board, c, y);
        if (win) return c;
      }
    }
    // Otherwise lean towards the centre, where more fours pass through.
    const weights = moves.map((c) => 1 / (1 + Math.abs(c - 3)) + rand() * 0.35);
    let best = 0;
    for (let i = 1; i < moves.length; i++) if (weights[i] > weights[best]) best = i;
    return moves[best];
  }

  return {
    C4: {
      W, H, N, EMPTY, P1, P2, idx,
      newBoard, dropRow, legalMoves, play, undo,
      winsAt, winningLine, isFull, encode, clone, scriptedMove,
    },
  };
});
