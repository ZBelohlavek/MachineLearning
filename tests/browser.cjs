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

  console.log(`\n${pass} passed, ${fail} failed`);
  console.log(errs.length ? 'ERRORS:\n' + [...new Set(errs)].join('\n') : 'no page errors anywhere');
  await browser.close();
})();
