/* ==========================================================================
   wordle.js — scoring, filtering, and how much a guess is worth.

   There is no neural network anywhere in this file, which is the point. The
   best-known way to play Wordle is not learned from data at all: it is a
   direct calculation of how much each guess would tell you, taken from
   information theory. Sometimes the right model is arithmetic.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const L = 5;                         // letters per word
  const PATTERNS = 243;                // 3^5 possible colourings
  const GREY = 0, YELLOW = 1, GREEN = 2;

  /**
   * Colour a guess against an answer, as a base-3 number (digit i = position i).
   *
   * The duplicate-letter rule is the part everyone gets wrong, and it is worth
   * being exact about: greens are claimed first, and only the answer's leftover
   * letters can turn a later position yellow. So guessing "geese" against
   * "these" leaves the first e grey, because both of the answer's e's have
   * already been used by greens.
   */
  function score(guess, answer) {
    const used = [false, false, false, false, false];
    const out = [GREY, GREY, GREY, GREY, GREY];

    for (let i = 0; i < L; i++) {
      if (guess[i] === answer[i]) { out[i] = GREEN; used[i] = true; }
    }
    for (let i = 0; i < L; i++) {
      if (out[i] === GREEN) continue;
      for (let j = 0; j < L; j++) {
        if (!used[j] && guess[i] === answer[j]) { out[i] = YELLOW; used[j] = true; break; }
      }
    }
    let code = 0;
    for (let i = L - 1; i >= 0; i--) code = code * 3 + out[i];
    return code;
  }

  /** A pattern code back into an array of 0/1/2, position 0 first. */
  function patternDigits(code) {
    const d = new Array(L);
    for (let i = 0; i < L; i++) { d[i] = code % 3; code = (code / 3) | 0; }
    return d;
  }

  const ALL_GREEN = (() => { let c = 0; for (let i = L - 1; i >= 0; i--) c = c * 3 + GREEN; return c; })();

  /** Every candidate still consistent with the colours a guess came back with. */
  function filter(candidates, guess, code) {
    const out = [];
    for (const w of candidates) if (score(guess, w) === code) out.push(w);
    return out;
  }

  /**
   * Expected information gain of a guess, in bits.
   *
   * Each possible colouring splits the remaining candidates into a bucket. If a
   * bucket holds a fraction p of them, seeing it multiplies your uncertainty by
   * p, which is log2(1/p) bits of information. Average that over the buckets,
   * weighted by how likely each is, and you have the guess's worth before you
   * play it. A guess that always produces the same colours tells you nothing
   * and scores zero.
   */
  function entropy(guess, candidates, counts) {
    const buckets = counts && counts.length === PATTERNS ? counts : new Int32Array(PATTERNS);
    buckets.fill(0);
    for (const w of candidates) buckets[score(guess, w)]++;
    const n = candidates.length;
    let bits = 0;
    for (let c = 0; c < PATTERNS; c++) {
      const k = buckets[c];
      if (k > 0) { const p = k / n; bits -= p * Math.log2(p); }
    }
    return bits;
  }

  /** The size of the largest bucket: the worst case this guess leaves you in. */
  function worstCase(guess, candidates, counts) {
    const buckets = counts && counts.length === PATTERNS ? counts : new Int32Array(PATTERNS);
    buckets.fill(0);
    for (const w of candidates) buckets[score(guess, w)]++;
    let worst = 0;
    for (let c = 0; c < PATTERNS; c++) if (buckets[c] > worst) worst = buckets[c];
    return worst;
  }

  /**
   * Rank guesses by expected information. `pool` is what you are allowed to
   * play, `candidates` what could still be the answer — they differ, because a
   * word that cannot be the answer can still be the most informative thing to
   * say. Ties go to a word that could actually win.
   */
  function rankGuesses(candidates, pool, topN = 8) {
    const counts = new Int32Array(PATTERNS);
    const possible = new Set(candidates);
    const scored = [];
    for (const g of pool) {
      const bits = entropy(g, candidates, counts);
      scored.push({ word: g, bits, possible: possible.has(g) });
    }
    scored.sort((a, b) => (b.bits - a.bits) || (b.possible - a.possible) || a.word.localeCompare(b.word));
    return scored.slice(0, topN);
  }

  /**
   * The greedy solver: always play whatever reveals the most, except that when
   * only one or two candidates remain it plays one of them, since guessing a
   * word that cannot win to learn about a word that can is a wasted turn.
   */
  function suggest(candidates, pool) {
    if (candidates.length <= 2) return candidates[0];
    const ranked = rankGuesses(candidates, pool, 1);
    return ranked.length ? ranked[0].word : candidates[0];
  }

  /** Play a whole game with the greedy solver. Returns the guesses it made. */
  function solve(answer, words, opening) {
    let candidates = words;
    const guesses = [];
    for (let turn = 0; turn < 10; turn++) {
      const guess = turn === 0 && opening ? opening : suggest(candidates, words);
      const code = score(guess, answer);
      guesses.push({ word: guess, code, remaining: candidates.length });
      if (code === ALL_GREEN) break;
      candidates = filter(candidates, guess, code);
    }
    return guesses;
  }

  return {
    wordle: {
      L, PATTERNS, GREY, YELLOW, GREEN, ALL_GREEN,
      score, patternDigits, filter, entropy, worstCase, rankGuesses, suggest, solve,
    },
  };
});
