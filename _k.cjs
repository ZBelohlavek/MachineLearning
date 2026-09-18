const { chromium } = require('playwright');
const SP = process.env.SP;
(async () => {
  const b = await chromium.launch({ executablePath: process.env.CHROME_PATH });
  const page = await b.newPage({ viewport: { width: 1320, height: 1000 } });
  const errs = [];
  page.on('pageerror', e => errs.push('JS: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await page.goto('file://' + __dirname + '/lessons/kitchen.html');
  await page.waitForTimeout(1500);

  console.log('pairing rows:', await page.locator('#pairing-table tr').count());
  console.log('pairing note:', (await page.textContent('#pairing-note')).replace(/\s+/g,' ').slice(0,180));
  console.log('partners:', await page.locator('#partner-mode .pill').count());

  // Run a shift with the scripted partner while standing still.
  await page.click('#btn-start');
  await page.waitForTimeout(200);
  await page.evaluate(() => window.__kitchen.setPartner('scripted'));
  await page.waitForTimeout(6000);
  const s1 = await page.evaluate(() => ({
    tick: window.__kitchen.kitchen.tick,
    served: window.__kitchen.kitchen.served,
    collisions: window.__kitchen.kitchen.collisions,
  }));
  console.log('6s into a shift (player idle):', JSON.stringify(s1));

  // The scripted cook alone should still make progress on the pot.
  const pot = await page.evaluate(() => window.__kitchen.kitchen.potOnions + (window.__kitchen.kitchen.potReady?3:0));
  console.log('pot progress while player idle:', pot);

  await page.locator('#game-panel').locator('..').screenshot({ path: SP + '/kitchen.png' });
  console.log(errs.length ? 'ERRORS: ' + errs.slice(0,3).join(' | ') : 'no page errors');
  await b.close();
})();
