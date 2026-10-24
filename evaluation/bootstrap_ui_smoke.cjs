// Exercise the real form with one input and no exclusions; intercept submission
// so this UI check cannot launch a model job or modify the user's output.
const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const artifacts = path.resolve(__dirname, '../work/cluster_p0p2_logs');
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_EXE});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/extract', async route => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({status:200, contentType:'application/json', body:'{}'});
  });
  await page.route('**/api/extract/status?*', route => route.fulfill({status:200,
    contentType:'application/json', body:JSON.stringify({state:'stopped', cursor:0, lines:[]})}));
  try {
    await page.goto(process.env.CLUSTER_UI_URL || 'http://127.0.0.1:18782');
    await page.click('[data-panel="p-extract"]');
    await page.fill('#targetsInput', 'C:\\example\\one-input.wav');
    await page.fill('#refsInput', 'C:\\example\\one-reference.wav');
    await page.fill('#excludeInput', '');
    for (const enabled of [false, true]) {
      await page.locator('#useBootstrap').setChecked(enabled);
      const sent = page.waitForResponse(r => r.url().endsWith('/api/extract') && r.request().method() === 'POST');
      await page.click('#runBtn');
      await sent;
      const body = requests.at(-1);
      assert.equal(body.options.cluster_bootstrap, enabled);
      assert.equal(body.options.cluster_identity_mode, enabled ? 'enforce' : 'audit');
      assert.equal(body.targets.length, 1);
      assert.equal(body.references.length, 1);
      assert.deepEqual(body.negative_groups, []);
      assert.deepEqual(body.subtitles, {});
    }
    await page.screenshot({path:path.join(artifacts, 'bootstrap_ui_smoke.png'), fullPage:true});
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(artifacts, 'bootstrap_ui_smoke.json'), JSON.stringify({
      passed:true, checks:['single_file_form', 'bootstrap_off_audit', 'bootstrap_on_enforce', 'no_exclusions_required'],
      requests, page_errors:errors, screenshot:'bootstrap_ui_smoke.png'
    }, null, 2));
    console.log('Bootstrap UI smoke passed');
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
