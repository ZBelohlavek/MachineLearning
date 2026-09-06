/* ==========================================================================
   digits.js — a handwriting-ish digit dataset generated in the browser.

   Instead of downloading MNIST, we draw digits with the canvas text API in
   random fonts, sizes, rotations and offsets. That keeps the site completely
   self-contained, and it makes the amount of variation a slider you can move,
   which turns "data augmentation" from a footnote into something you can see.
   ========================================================================== */

;(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ML = Object.assign(root.ML || {}, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, function (root) {
  'use strict';

  const FONTS = ['sans-serif', 'serif', 'monospace', 'Georgia', 'Verdana', 'Courier New', 'Arial'];
  let cv = null, ctx = null, cvSize = 0;

  function surface(size) {
    if (!cv || cvSize !== size) {
      cv = document.createElement('canvas');
      cv.width = cv.height = size;
      ctx = cv.getContext('2d', { willReadFrequently: true });
      cvSize = size;
    }
    return ctx;
  }

  /** One digit image as a Float32Array of size*size values in [0,1]. */
  function renderDigit(digit, rng, aug = 1, size = 20) {
    const g = surface(size);
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#000';
    g.fillRect(0, 0, size, size);
    const fontSize = size * 0.75 + (rng() - 0.5) * size * 0.3 * aug;
    const font = FONTS[(rng() * FONTS.length) | 0];
    const bold = rng() < 0.4 ? 'bold ' : '';
    g.fillStyle = '#fff';
    g.font = `${bold}${fontSize.toFixed(1)}px ${font}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.translate(size / 2 + (rng() - 0.5) * size * 0.2 * aug,
                size / 2 + (rng() - 0.5) * size * 0.2 * aug);
    g.rotate((rng() - 0.5) * 0.5 * aug);
    g.fillText(String(digit), 0, 0);

    const d = g.getImageData(0, 0, size, size).data;
    const img = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
      let v = d[i * 4] / 255;
      if (aug > 0) v = Math.max(0, Math.min(1, v + (rng() - 0.5) * 0.14 * aug));
      img[i] = v;
    }
    return img;
  }

  /** n images, cycling through the ten digits so classes stay balanced. */
  function makeDigitSet(n, rng, aug = 1, size = 20) {
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      const d = i % 10;
      xs.push(renderDigit(d, rng, aug, size));
      ys.push(d);
    }
    return { xs, ys, n };
  }

  /** Downscale a drawing canvas into a size×size Float32Array. */
  function imageFromCanvas(source, size = 20) {
    const g = surface(size);
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#000';
    g.fillRect(0, 0, size, size);
    g.drawImage(source, 0, 0, size, size);
    const d = g.getImageData(0, 0, size, size).data;
    const img = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) img[i] = d[i * 4] / 255;
    return img;
  }

  return { renderDigit, makeDigitSet, imageFromCanvas, DIGIT_FONTS: FONTS };
});
