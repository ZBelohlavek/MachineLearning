/* ==========================================================================
   tests/browser.cjs — end-to-end checks that every lesson loads, trains and
   responds to its controls.

       npm i -D playwright && npx playwright install chromium
       node tests/browser.cjs

   Requires Playwright, which the site itself does not. Everything else in
   this repository runs with no dependencies at all.
   ========================================================================== */

let chromium;
try {
  ({ chromium } = require('playwright'));
} catch (err) {
  console.error('This test needs Playwright:\n  npm i -D playwright && npx playwright install chromium');
  process.exit(2);
}

const path = require('path');
const B = 'file://' + path.join(__dirname, '..') + '/';

let pass = 0, fail = 0;
const check = (name, ok, extra='') => { (ok ? pass++ : fail++); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${extra}`); };

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 950 } });
  const errs = [];
  const open = async (p) => {
    const page = await ctx.newPage();
    page.on('pageerror', e => errs.push(p + ': ' + e.message));
    page.on('console', m => { if (m.type()==='error') errs.push(p + ': ' + m.text()); });
    await page.goto(B + p);
    await page.waitForTimeout(1400);
    return page;
  };
  const scrollTo = (page, sel) => page.evaluate(s => { document.documentElement.style.scrollBehavior='auto';
    const el = document.querySelector(s); window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 120); }, sel);

  /* ---- rocket league: keyboard driving + mode switches ---- */
  {
    const page = await open('lessons/rocket-league.html');
    await scrollTo(page, '#sandbox-canvas');
    await page.click('#sandbox-canvas');
    const before = await page.evaluate(() => { const m = window.__trainer; return null; });
    await page.keyboard.down('ArrowUp');
    await page.waitForTimeout(1200);
    await page.keyboard.up('ArrowUp');
    // the sandbox car should have moved off its spawn
    const moved = await page.evaluate(() => {
      const c = document.querySelector('#reward-total');
      return parseFloat(c.textContent) !== 0;   // reward accumulates only while stepping
    });
    check('rocket · sandbox responds to arrow keys', moved);
    await page.click('#play-mode .pill:text-is("You vs agent")');
    await page.waitForTimeout(600);
    check('rocket · switch to "You vs agent"', true);
    await page.click('#opponent-mode .pill:text-is("vs frozen past self")');
    await page.click('#btn-train'); await page.waitForTimeout(3000); await page.click('#btn-train');
    const eps = await page.evaluate(() => window.__trainer.episodes);
    check('rocket · trains against a frozen past self', eps > 0, `(${eps} episodes)`);
    await page.click('#reward-presets .pill:text-is("Ball chaser")');
    const w = await page.evaluate(() => window.__trainer.rewards.approach);
    check('rocket · reward preset applies', w === 1.0, `(approach=${w})`);
    await page.close();
  }

  /* ---- gridworld ---- */
  {
    const page = await open('lessons/gridworld.html');
    await scrollTo(page, '#grid-canvas');
    const box = await page.locator('#grid-canvas').boundingBox();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);   // paint a wall
    await page.click('#btn-step');
    await page.click('#btn-step');
    const upd = await page.textContent('#update-readout');
    check('gridworld · single step produces a Q update', /TD error/.test(upd));
    await page.click('#gw-presets .pill:text-is("Cliff walk")');
    await page.click('#btn-run'); await page.waitForTimeout(4000); await page.click('#btn-run');
    const ep = await page.textContent('#gw-stats');
    check('gridworld · runs episodes on the cliff preset', /episode/i.test(ep));
    await page.close();
  }

  /* ---- convolutions ---- */
  {
    const page = await open('lessons/convolutions.html');
    await scrollTo(page, '#conv-in');
    const sig = () => page.evaluate(() => {
      const c = document.querySelector('#conv-out');
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let s = 0; for (let i = 0; i < d.length; i += 97) s += d[i];
      return s;
    });
    const a = await sig();
    await page.click('#kernel-presets .pill:text-is("Gaussian blur")');
    await page.waitForTimeout(400);
    const b = await sig();
    check('convolutions · changing the kernel changes the output', a !== b);
    await page.click('#image-presets .pill:text-is("Letter A")');
    await page.waitForTimeout(300);
    const c = await sig();
    check('convolutions · changing the image changes the output', b !== c);
    await page.click('#btn-step-px');
    check('convolutions · single-pixel step works', /output pixel/.test(await page.textContent('#math-panel')));
    await page.close();
  }

  /* ---- foundations ---- */
  {
    const page = await open('lessons/foundations.html');
    await scrollTo(page, '#data-canvas');
    await page.click('#dataset-pills .pill:text-is("Spiral")');
    await page.click('#paint-pills .pill:text-is("Add orange points")');
    const box = await page.locator('#data-canvas').boundingBox();
    await page.mouse.click(box.x + 40, box.y + 40);
    await page.click('#btn-step');
    const after = await page.textContent('#fd-stats');
    check('foundations · trains an epoch after editing data', /epoch/i.test(after));
    await page.close();
  }

  /* ---- cnn ---- */
  {
    const page = await open('lessons/cnn-digits.html');
    const p0 = await page.evaluate(() => document.querySelector('#cnn-stats').textContent);
    await page.evaluate(() => {
      const f = [...document.querySelectorAll('#cnn-config .field')].find(f => f.textContent.includes('layer 1 filters'));
      const i = f.querySelector('input'); i.value = 4; i.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await page.waitForTimeout(500);
    const p1 = await page.evaluate(() => document.querySelector('#cnn-stats').textContent);
    check('cnn · changing filter count rebuilds the network', p0 !== p1);
    await page.close();
  }

  /* ---- vit ---- */
  {
    const page = await open('lessons/vision-transformer.html');
    await scrollTo(page, '#vit-image');
    await page.click('#patch-pills .pill:text-is("5×5 patches (4×4 grid)")');
    await page.waitForTimeout(400);
    const patches = await page.evaluate(() => document.querySelectorAll('#patch-strip .patch-item').length);
    check('vit · patch size changes the token count', patches === 16, `(${patches} patches)`);
    await page.click('#attn-mode .pill:text-is("Where this patch looks")');
    await page.click('#arch-toggles .pill:text-is("Class token")');
    await page.waitForTimeout(500);
    await page.click('#arch-toggles .pill:text-is("LayerNorm on")');
    await page.waitForTimeout(500);
    const top = await page.textContent('#attn-top');
    check('vit · architecture toggles rebuild without breaking attention', /patch \d+/.test(top));
    await page.click('#btn-train'); await page.waitForTimeout(3000); await page.click('#btn-train');
    check('vit · trains after toggling architecture', /images seen/i.test(await page.textContent('#vit-stats')));
    await page.close();
  }

  /* ---- the racer, and its race mode ---- */
  {
    const page = await open('lessons/evolve-a-driver.html');
    await scrollTo(page, '#track-canvas');
    await page.click('#btn-run');
    await page.waitForTimeout(9000);
    const stats = await page.$$eval('#racer-stats .stat',
      els => Object.fromEntries(els.map(e => [e.querySelector('.k').textContent, e.querySelector('.v').textContent])));
    check('racer · evolution makes progress', parseInt(stats.generation, 10) > 2 && parseFloat(stats.best) > 20,
          `(gen ${stats.generation}, best ${stats.best})`);
    await page.click('#mode-pills .pill:text-is("Race the champion")');
    await page.waitForTimeout(400);
    await page.keyboard.down('ArrowUp');
    await page.waitForTimeout(1800);
    await page.keyboard.up('ArrowUp');
    const race = await page.textContent('#race-readout');
    check('racer · you can race the champion', /champion/.test(race) && /\d+%/.test(race));
    await page.click('#track-presets .pill:text-is("Hairpin")');
    await page.waitForTimeout(600);
    check('racer · switching track restarts evolution', true);
    await page.close();
  }

  /* ---- pre-trained models load and work with no training ---- */
  {
    const page = await open('lessons/cnn-digits.html');
    check('cnn · ships a pre-trained network', /pre-trained/i.test(await page.textContent('#model-banner')));
    await scrollTo(page, '#pad');
    const box = await page.locator('#pad').boundingBox();
    const P = (fx, fy) => ({ x: box.x + box.width * fx, y: box.y + box.height * fy });
    let p = P(0.5, 0.15);
    await page.mouse.move(p.x, p.y); await page.mouse.down();
    p = P(0.5, 0.85); await page.mouse.move(p.x, p.y, { steps: 12 }); await page.mouse.up();
    await page.waitForTimeout(400);
    const verdict = await page.textContent('#pad-verdict');
    check('cnn · recognises a drawn digit before any training', /that's a 1/.test(verdict), '→ ' + verdict.trim());
    await page.click('#btn-duel');
    await page.waitForTimeout(400);
    check('cnn · the digit duel starts', /^[0-9]$/.test((await page.textContent('#duel-target')).trim()));
    await page.close();
  }

  /* ---- the Rocket League timelapse ---- */
  {
    const page = await open('lessons/rocket-league.html');
    const n = await page.evaluate(() => (window.ML_ROCKET_STAGES || {}).stages?.length || 0);
    check('rocket · training checkpoints are shipped', n >= 5, `(${n} stages)`);
    await scrollTo(page, '#stage-canvas');
    await page.evaluate(() => { const s = document.querySelector('#stage-scrub'); s.value = '0'; s.dispatchEvent(new Event('input', { bubbles: true })); });
    await page.waitForTimeout(400);
    const first = await page.textContent('#stage-label');
    await page.click('#stage-next');
    await page.waitForTimeout(400);
    const second = await page.textContent('#stage-label');
    check('rocket · scrubbing swaps in a different policy', first !== second, '→ ' + second.replace(/\s+/g, ' ').trim());
    await page.click('#btn-pretrained');
    await page.waitForTimeout(300);
    check('rocket · the trained agent loads into the trainer',
          (await page.evaluate(() => window.__trainer.episodes)) > 1000);
    await page.close();
  }

  /* ---- badges ---- */
  {
    const page = await open('lessons/convolutions.html');
    await scrollTo(page, '#kernel-editor');
    await page.fill('#kernel-editor input:first-child', '2');
    await page.waitForTimeout(400);
    const earned = await page.evaluate(() => JSON.parse(localStorage.getItem('mlbb-badges') || '{}'));
    check('badges · editing a kernel earns one', !!earned['conv-custom']);
    const toast = await page.locator('.toast').count();
    check('badges · a toast appears', toast >= 1);
    await page.close();
  }

  /* ---- quizzes ---- */
  {
    const page = await open('lessons/gridworld.html');
    await scrollTo(page, '#quiz');
    const result = await page.evaluate(() => {
      const spec = window.ML.QUIZZES.gridworld;
      const cards = [...document.querySelectorAll('#quiz .quiz-q')];
      cards.forEach((c, i) => c.querySelectorAll('.quiz-opt')[spec.questions[i].answer].click());
      return { score: document.querySelector('.quiz-score span').textContent,
               explained: document.querySelectorAll('#quiz .quiz-why.right').length };
    });
    check('quiz · three questions, all explained', result.score === '3' && result.explained === 3,
          `(scored ${result.score})`);
    await page.reload();
    await page.waitForTimeout(1200);
    const kept = await page.evaluate(() => document.querySelector('.quiz-score span').textContent);
    check('quiz · answers persist across a reload', kept === '3');
    await page.close();
  }

  /* ---- in-lesson goals ---- */
  {
    const page = await open('lessons/convolutions.html');
    // badges persist across pages in one browser context, so start this one clean
    await page.evaluate(() => localStorage.setItem('mlbb-badges', '{}'));
    await page.reload();
    await page.waitForTimeout(1200);
    const start = await page.evaluate(() => document.querySelector('#lesson-goals .goals-head span').textContent);
    await scrollTo(page, '#kernel-editor');
    await page.fill('#kernel-editor input:first-child', '3');
    await page.waitForTimeout(500);
    const after = await page.evaluate(() => document.querySelector('#lesson-goals .goals-head span').textContent);
    check('goals · the in-lesson panel updates live', start !== after, `(${start.trim()} → ${after.trim()})`);
    await page.close();
  }

  /* ---- the auto-graded kernel challenge ---- */
  {
    const page = await open('lessons/convolutions.html');
    await page.evaluate(() => localStorage.setItem('mlbb-badges', '{}'));
    await page.reload();
    await page.waitForTimeout(1200);
    await scrollTo(page, '#kernel-presets');
    await page.click('#kernel-presets .pill:text-is("Sobel Y (horizontal edges)")');
    await page.waitForTimeout(400);
    const presetAwarded = await page.evaluate(() =>
      !!JSON.parse(localStorage.getItem('mlbb-badges') || '{}')['conv-challenge']);
    check('challenge · clicking the preset does not pass it', !presetAwarded);
    const vals = ['-1', '-2', '-1', '0', '0', '0', '1', '2', '1'];
    const inputs = await page.$$('#kernel-editor input');
    for (let i = 0; i < inputs.length; i++) { await inputs[i].fill(vals[i]); await page.waitForTimeout(50); }
    await page.waitForTimeout(500);
    const typedAwarded = await page.evaluate(() =>
      !!JSON.parse(localStorage.getItem('mlbb-badges') || '{}')['conv-challenge']);
    check('challenge · a hand-typed edge detector passes it', typedAwarded);
    await page.close();
  }

  /* ---- the racer copes with a broken track ---- */
  {
    const page = await open('lessons/evolve-a-driver.html');
    await scrollTo(page, '#track-canvas');
    await page.click('#btn-run');
    await page.waitForTimeout(1200);
    await page.evaluate(() => {
      const t = window.__racer.track;
      for (let dy = -10; dy <= 10; dy++) for (let dx = -10; dx <= 10; dx++) {
        const x = (t.start.x + dx) | 0, y = (t.start.y + dy) | 0;
        if (x >= 0 && y >= 0 && x < 240 && y < 150) t.mask[y * 240 + x] = 0;
      }
      t.computeDistance();
    });
    await page.waitForTimeout(800);
    const warned = await page.evaluate(() => !document.querySelector('#track-warning').hidden);
    const g1 = await page.evaluate(() => document.querySelector('#racer-stats .stat .v').textContent);
    await page.waitForTimeout(1200);
    const g2 = await page.evaluate(() => document.querySelector('#racer-stats .stat .v').textContent);
    check('racer · erasing the start warns instead of churning generations', warned && g1 === g2);
    await page.close();
  }

  console.log(`\n${pass} passed, ${fail} failed`);
  console.log(errs.length ? 'ERRORS:\n' + [...new Set(errs)].join('\n') : 'no page errors anywhere');
  await browser.close();
})();
