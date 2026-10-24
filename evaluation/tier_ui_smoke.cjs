// Browser smoke for the Tier A/B filter views and the trust-main-output button.
const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '..');
const out = path.join(root, 'output');
const id = `tier_ui_smoke_${Date.now()}`;
const fixture = path.join(out, id);
const artifacts = path.join(root, 'work', 'cluster_p0p2_logs');

async function main() {
  fs.mkdirSync(path.join(fixture, 'episode', 'audio'), {recursive: true});
  const original = path.join(out, '20260928_214330_batch_5bd1f3');
  const manifest = JSON.parse(fs.readFileSync(path.join(original, 'batch_manifest.json'), 'utf8'));
  const source = path.join(original, manifest.sentences.find(s => s.accepted).audio_file);
  // 2 段 Tier A（主输出）、2 段 Tier B（边缘捡漏）、1 段已撤下。
  const rows = [];
  const kinds = [
    {accepted: true, tier: ''},
    {accepted: true, tier: ''},
    {accepted: false, tier: 'B'},
    {accepted: false, tier: 'B'},
    {accepted: false, tier: '', removed: true},
  ];
  kinds.forEach((kind, i) => {
    const name = `${i}.wav`;
    fs.copyFileSync(source, path.join(fixture, 'episode', 'audio', name));
    const diagnostics = kind.tier ? {cluster_tier: kind.tier} : {};
    if (kind.removed) diagnostics.cluster_removed = true;
    rows.push({
      target: 'source.wav', accepted: kind.accepted, start: i * 4, end: i * 4 + 3,
      text: `片段${i}`, audio_file: `episode/audio/${name}`,
      diagnostics,
    });
  });
  fs.writeFileSync(path.join(fixture, 'batch_manifest.json'),
    JSON.stringify({sentences: rows, targets: [{target: 'source.wav', output_dir: 'episode'}]}));

  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXE});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  let exportPayload = null;
  await page.route('**/api/export_verified/**', async route => {
    exportPayload = route.request().postDataJSON();
    await route.fulfill({json: {verified_count: 2, includes_material: false,
      zip_name: 'fake.zip', download_url: '/unused'}});
  });
  try {
    await page.goto(process.env.CLUSTER_UI_URL || 'http://127.0.0.1:18783');
    await page.locator(`#batchSelect option[value="${id}"]`).waitFor({state: 'attached'});
    await page.selectOption('#batchSelect', id);
    await page.waitForFunction(() => app?.segments?.length === 5);

    // 全部视图：主输出 + Tier B（撤下段走“撤下”筛选）。
    assert.equal(await page.locator('#segList .seg').count(), 4);

    // ✅ 主输出：2 段。
    await page.click('[data-filter="tierA"]');
    assert.equal(await page.locator('#segList .seg').count(), 2);

    // ⚡ 边缘捡漏：2 段。
    await page.click('[data-filter="tierB"]');
    assert.equal(await page.locator('#segList .seg').count(), 2);

    // 🚫 撤下：1 段（无 cluster_tier B 的未接受段）。
    await page.click('[data-filter="removed"]');
    assert.equal(await page.locator('#segList .seg').count(), 1);

    // 一键信任主输出：未标注的 Tier A 全部标 correct。
    await page.click('[data-filter="tierA"]');
    await page.on('dialog', dialog => dialog.accept());
    await page.click('#trustMainBtn');
    await page.waitForFunction(() => Object.values(app.reviews)
      .filter(r => r.label === 'correct').length === 2);
    // 导出请求带上 train.list 开关。
    await page.click('#exportBtn');
    await new Promise((resolve, reject) => {
      const timer = setInterval(() => {
        if (exportPayload) { clearInterval(timer); resolve(); }
      }, 100);
      setTimeout(() => { clearInterval(timer); reject(new Error('导出请求未发出')); }, 15000);
    });
    assert.equal(exportPayload.include_material, false);
    assert.equal(exportPayload.gpt_sovits_list, true);
    assert.deepEqual(errors, []);
    fs.mkdirSync(artifacts, {recursive: true});
    fs.writeFileSync(path.join(artifacts, 'tier_ui_smoke.json'), JSON.stringify({
      passed: true, checks: ['tier_filter_counts', 'trust_main_output_labels',
        'gpt_sovits_payload', 'page_errors'], page_errors: errors}, null, 2));
    console.log('Tier UI smoke passed');
  } finally {
    await browser.close();
    fs.rmSync(fixture, {recursive: true, force: true});
    const annotation = path.join(root, 'annotations', `${id}.json`);
    if (fs.existsSync(annotation)) fs.unlinkSync(annotation);
  }
}

main().catch(e => {console.error(e); process.exitCode = 1;});
