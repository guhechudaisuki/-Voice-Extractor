// Checks retry behavior, then leaves the real native desktop open for the user.
const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_EXE});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => { window.close = () => {}; });
  const endpoint = '**/api/switch_to_desktop';
  let release;
  const waiting = new Promise(resolve => { release = resolve; });
  await page.route(endpoint, async route => {
    await waiting;
    await route.fulfill({status:500, contentType:'application/json',
      body:JSON.stringify({detail:'桌面窗口启动超时，网页仍可使用'})});
  });
  try {
    await page.goto(process.env.CLUSTER_UI_URL || 'http://127.0.0.1:7860');
    await page.click('#switchDesktopBtn');
    assert.equal(await page.locator('#switchDesktopBtn').isDisabled(), true);
    assert.equal(await page.locator('#goDesktopBtn').isDisabled(), true);
    assert.equal(await page.getByText('桌面版已打开', {exact:true}).count(), 0);
    release();
    await page.locator('.toast.err').filter({hasText:'网页仍可使用'}).waitFor();
    assert.equal(await page.locator('#switchDesktopBtn').isEnabled(), true);
    assert.equal(await page.locator('#goDesktopBtn').isEnabled(), true);
    await page.unroute(endpoint);
    const responsePromise = page.waitForResponse(r => r.url().endsWith('/api/switch_to_desktop'), {timeout:25000});
    await page.click('#switchDesktopBtn');
    const response = await responsePromise;
    const ready = await response.json();
    assert.equal(response.status(), 200, JSON.stringify(ready));
    assert.equal(ready.visible, true);
    assert.ok(ready.pid > 0);
    await page.getByText('桌面版已打开', {exact:true}).waitFor();
    assert.deepEqual(errors, []);
    const report = {passed:true, checks:['waits_for_window', 'disables_both_buttons',
      'failure_keeps_page_and_allows_retry', 'real_native_handoff'], ready, page_errors:errors};
    fs.writeFileSync(path.resolve(__dirname, '../work/cluster_p0p2_logs/desktop_switch_ui_smoke.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
