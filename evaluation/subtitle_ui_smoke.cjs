// Real form interactions with submission intercepted; no extraction is started.
const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const artifacts = path.resolve(__dirname, '../work/cluster_p0p2_logs');
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_EXE});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors = [], requests = [];
  const first = 'C:\\素材\\片段一.wav', second = 'C:\\素材\\片段二.mkv';
  const firstSubtitle = 'C:\\字幕\\片段一 中文.srt', secondSubtitle = 'C:\\字幕\\片段二.ass';
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/extract', async route => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({status:200, contentType:'application/json', body:'{}'});
  });
  await page.route('**/api/extract/status?*', route => route.fulfill({status:200,
    contentType:'application/json', body:JSON.stringify({state:'stopped', cursor:0, lines:[]})}));
  const submit = async () => {
    const sent = page.waitForResponse(r => r.url().endsWith('/api/extract') && r.request().method() === 'POST');
    await page.click('#runBtn');
    await sent;
    return requests.at(-1);
  };
  try {
    await page.goto(process.env.CLUSTER_UI_URL || 'http://127.0.0.1:18782');
    await page.click('[data-panel="p-extract"]');
    await page.click('#subtitleBindingsPanel summary');
    assert.match(await page.locator('#subtitleBindings').textContent(), /先填写待提取文件/);
    await page.fill('#refsInput', 'C:\\素材\\参考.wav');
    await page.fill('#targetsInput', first);
    assert.equal(await page.locator('.subtitle-path').count(), 1);
    assert.deepEqual((await submit()).subtitles, {});
    await page.locator('.subtitle-path').fill(firstSubtitle);
    assert.deepEqual((await submit()).subtitles, {[first]:firstSubtitle});

    await page.fill('#targetsInput', `${first}\n${second}`);
    assert.equal(await page.locator('.subtitle-path').nth(0).inputValue(), firstSubtitle);
    assert.equal(await page.locator('.subtitle-path').nth(1).inputValue(), '');
    assert.deepEqual((await submit()).subtitles, {[first]:firstSubtitle});
    await page.locator('.subtitle-path').nth(1).fill(secondSubtitle);
    await page.fill('#targetsInput', `${second}\n${first}`);
    assert.equal(await page.locator('.subtitle-path').nth(0).inputValue(), secondSubtitle);
    assert.equal(await page.locator('.subtitle-path').nth(1).inputValue(), firstSubtitle);
    assert.match(await page.locator('#subtitleBindingCount').textContent(), /2\/2/);
    await page.fill('#excludeInput', 'C:\\素材\\排除甲.wav, C:\\素材\\排除甲2.wav');
    await page.fill('#silenceMin', '0.25');
    await page.fill('#silenceMax', '1.05');
    await page.check('#useBootstrap');
    const together = await submit();
    assert.deepEqual(together.subtitles, {[first]:firstSubtitle, [second]:secondSubtitle});
    assert.deepEqual(together.negative_groups, [['C:\\素材\\排除甲.wav', 'C:\\素材\\排除甲2.wav']]);
    assert.equal(together.options.silence_min_seconds, .25);
    assert.equal(together.options.silence_max_seconds, 1.05);
    assert.equal(together.options.cluster_bootstrap, true);
    assert.equal(together.options.cluster_identity_mode, 'enforce');

    const screenshots = [];
    for (const viewport of [{width:1440,height:1000}, {width:1040,height:700}, {width:390,height:844}]) {
      await page.setViewportSize(viewport);
      assert.equal((await submit()).options.cluster_bootstrap, true);
      await page.locator('#subtitleBindingsPanel').scrollIntoViewIfNeeded();
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      for (const element of await page.locator('.subtitle-path').all()) {
        const box = await element.boundingBox();
        assert.ok(box.width >= 180 && box.x >= 0 && box.x + box.width <= viewport.width);
      }
      const name = `subtitle_ui_${viewport.width}.png`;
      await page.screenshot({path:path.join(artifacts, name), fullPage:true});
      screenshots.push(name);
    }
    await page.setViewportSize({width:1440,height:1000});
    await page.fill('#targetsInput', first);
    assert.equal(await page.locator('.subtitle-path').inputValue(), firstSubtitle);
    assert.deepEqual((await submit()).subtitles, {[first]:firstSubtitle});
    await page.locator('#subtitleBindings .mini-btn').click();
    assert.deepEqual((await submit()).subtitles, {});
    await page.locator('.subtitle-path').fill('C:\\字幕\\错误.txt');
    const beforeInvalid = requests.length;
    await page.click('#runBtn');
    await page.locator('.toast.err').filter({hasText:'字幕仅支持'}).waitFor();
    assert.equal(requests.length, beforeInvalid);
    await page.fill('#targetsInput', '');
    assert.equal(await page.locator('.subtitle-path').count(), 0);
    await page.fill('#targetsInput', first);
    assert.equal(await page.locator('.subtitle-path').inputValue(), '');
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(artifacts, 'subtitle_ui_smoke.json'), JSON.stringify({
      passed:true, checks:['empty_state', 'subtitles_optional', 'single_file_binding', 'partial_batch_binding',
        'reorder_keeps_binding', 'remove_target_removes_binding', 'clear_binding', 'invalid_extension_blocked',
        'subtitle_exclusion_silence_bootstrap_together', 'three_viewports'],
      screenshots, combined_request:together, page_errors:errors
    }, null, 2));
    console.log('Subtitle UI smoke passed');
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
