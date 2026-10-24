const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '..');
const base = process.env.CLUSTER_UI_URL || 'http://127.0.0.1:7861';
const output = path.join(root, 'work', 'cluster_p0p2_logs');

async function checkViewport(browser, width, height, mobile) {
  const page = await browser.newPage({ viewport: { width, height }, isMobile: mobile, hasTouch: mobile });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('#batchSelect option').first().waitFor({ state: 'attached' });
  assert.equal(await page.locator('#includeMaterial').isDisabled(), true);
  assert.equal(await page.locator('#exportBtn').isEnabled(), true);
  await page.locator('[data-panel="p-training"]').click();
  assert.equal(await page.locator('#trainingFiles').getAttribute('accept'), '.zip');
  assert.equal(await page.locator('#exportModelBtn').count(), 0);
  const layout = await page.evaluate(() => ({
    pageWidth: document.documentElement.scrollWidth,
    viewport: innerWidth,
    reportWidth: document.getElementById('trainingReport').getBoundingClientRect().width,
    trainWidth: document.getElementById('trainBtn').getBoundingClientRect().width,
  }));
  await page.screenshot({ path: path.join(output, `training_ui_${width}.png`), fullPage: true });
  assert.ok(layout.pageWidth <= layout.viewport + 2, JSON.stringify(layout));
  assert.ok(layout.trainWidth >= 70 && layout.reportWidth > 100);
  assert.deepEqual(errors, []);
  await page.close();
  return layout;
}

async function checkCompleteReview(browser) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const segments = [
    { ordinal: 1, filename: 'a.wav', accepted: true, cluster_removed: false, start: 0, end: 1, duration: 1, review: {} },
    { ordinal: 2, filename: 'b.wav', accepted: false, cluster_removed: true, start: 2, end: 3, duration: 1, review: {} },
  ];
  await page.route('**/api/batches', route => route.fulfill({ json: [{ id: 'fake', accepted_count: 1, labeled_count: 0 }] }));
  await page.route('**/api/batches/fake', route => route.fulfill({ json: { batch_id: 'fake', total_segments: 2, segments } }));
  await page.route('**/api/batches/fake/clusters**', route => route.fulfill({ json: { available: false } }));
  await page.route('**/api/batches/fake/stage_turns', route => route.fulfill({ json: { available: false } }));
  await page.route('**/api/audio/fake/**', route => route.fulfill({ status: 404 }));
  await page.route('**/api/review/fake', route => route.fulfill({ json: { status: 'ok' } }));
  let exportPayload = null;
  await page.route('**/api/export_verified/fake', async route => {
    exportPayload = route.request().postDataJSON();
    await route.fulfill({ json: { verified_count: 1, includes_material: true, zip_name: 'fake_with_material.zip', download_url: '/unused' } });
  });
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.locator('#segmentCounter').waitFor();
  assert.equal(await page.locator('#includeMaterial').isDisabled(), true);
  await page.locator('[data-label="correct"]').click();
  assert.equal(await page.locator('#includeMaterial').isDisabled(), true);
  await page.locator('[data-filter="removed"]').click();
  await page.locator('#segList').getByText('b.wav').click();
  await page.locator('[data-label="wrong"]').click();
  assert.equal(await page.locator('#includeMaterial').isEnabled(), true);
  await page.locator('#includeMaterial').check();
  assert.equal(await page.locator('#materialIdentity, #customRole, .role').count(), 0);
  await page.locator('#exportBtn').click();
  await page.waitForFunction(() => document.querySelector('.toast.ok')?.textContent.includes('已附带训练原料'));
  // GPT-SoVITS train.list 勾选框默认选中，导出请求会附带该开关。
  assert.deepEqual(exportPayload, { include_material: true, gpt_sovits_list: true });
  await page.close();
}

async function main() {
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_EXE });
  try {
    const desktop = await checkViewport(browser, 1440, 900, false);
    const mobile = await checkViewport(browser, 390, 844, true);
    await checkCompleteReview(browser);
    const report = { passed: true, desktop, mobile, checks: ['page_errors', 'optional_raw_control', 'complete_review_gate', 'zip_payload', 'responsive_training_panel'] };
    fs.writeFileSync(path.join(output, 'training_ui_smoke.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
