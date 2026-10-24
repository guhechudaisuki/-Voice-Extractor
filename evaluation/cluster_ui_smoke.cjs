// Browser smoke against a local QA server; writes only its own temporary batches.
const fs = require('fs');
const path = require('path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'output');
const id = `cluster_ui_smoke_${Date.now()}`;
const fixture = path.join(out, id);
const legacy = path.join(out, id + '_legacy');
const artifacts = path.join(root, 'work', 'cluster_p0p2_logs');
const save = (p, obj) => fs.writeFileSync(p, JSON.stringify(obj, null, 2));

async function main() {
  fs.mkdirSync(path.join(fixture, 'episode', 'audio'), {recursive:true});
  fs.mkdirSync(legacy);
  const original = path.join(out, '20260928_214330_batch_5bd1f3');
  const manifest = JSON.parse(fs.readFileSync(path.join(original,'batch_manifest.json'),'utf8'));
  const source = path.join(original,manifest.sentences.find(s=>s.accepted).audio_file);
  const sentences = [0,1,2].map(i => ({target:'source.wav', accepted:true, start:i*4, end:i*4+3,
    text:'界面验证片段', audio_file:`episode/audio/${i}.wav`}));
  sentences.forEach(s=>fs.copyFileSync(source,path.join(fixture,s.audio_file)));
  const vetoDir = path.join(fixture,'episode','cluster_outliers');
  fs.mkdirSync(vetoDir);
  fs.copyFileSync(source,path.join(vetoDir,'0001.wav'));
  fs.writeFileSync(path.join(vetoDir,'0001.txt'),'12.000–15.000\n簇级撤下');
  sentences.push({target:'source.wav',accepted:false,start:12,end:15,audio_file:'',
    text:'撤下片段试听验证',diagnostics:{cluster_removed:true}});
  save(path.join(fixture,'batch_manifest.json'), {sentences, targets:[{target:'source.wav',output_dir:'episode'}]});
  save(path.join(legacy,'batch_manifest.json'), {sentences:[]});
  save(path.join(fixture,'episode','cluster_report.json'), {
    target_cluster_id:0, mode:'audit', reason:'target_cluster_found', clusters:[0,1].map(i=>({
      cluster_id:i, is_target:i===0, member_count:2, total_speech_seconds:6,
      primary_cohesion:.8, target_score:i===0?.9:.5, target_margin:.2,
      members:sentences.filter((_,n)=>i===0?n<2:n>=2).map((s,n)=>({index:i===0?n:n+2,span:[s.start,s.end]}))
    }))
  });
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_EXE});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  try {
    await page.goto(process.env.CLUSTER_UI_URL || 'http://127.0.0.1:18781');
    await page.locator(`#batchSelect option[value="${id}"]`).waitFor({state:'attached'});
    await page.selectOption('#batchSelect',id);
    await page.waitForFunction(()=>document.querySelector('#clusterSummary')?.textContent.includes('2 个簇'));
    await page.click('#clusterViewBtn');
    assert.equal(await page.locator('.cluster-row').count(),2);
    await page.locator('.cluster-row').first().click();
    assert.equal(await page.locator('#segList .seg').count(),2);
    await page.selectOption('#clusterOverride','1');
    const saved=page.waitForResponse(r=>r.url().endsWith('/cluster_override')&&r.request().method()==='POST');
    await page.click('#saveClusterOverride');
    assert.equal((await saved).status(),200);
    assert.equal(JSON.parse(fs.readFileSync(path.join(root,'annotations',id+'.json'))).cluster_override.episode.cluster_id,1);
    await page.waitForFunction(()=>app.clusters?.override?.cluster_id===1);
    await page.click('#clusterViewBtn');
    await page.locator('.cluster-row').first().click();
    assert.equal(await page.locator('#segList .seg').count(),2);
    await page.click('#playBtn');
    await page.waitForFunction(()=>!audio.paused&&audio.currentTime>0);
    await page.click('#nextBtn');
    await page.click('#nextBtn');
    assert.equal(await page.evaluate(()=>app.segments[app.index].ordinal),2);
    await page.locator('.cluster-row').nth(1).click();
    assert.equal(await page.locator('#segList .seg').count(),2);
    await page.locator('#segList .seg').last().click();
    await page.waitForFunction(()=>audio.src.endsWith('/4')&&!audio.paused&&audio.currentTime>0);
    assert.equal(JSON.parse(fs.readFileSync(path.join(fixture,'batch_manifest.json'))).sentences[3].audio_file,'');
    await page.screenshot({path:path.join(artifacts,'cluster_ui_smoke.png'),fullPage:true});
    await page.selectOption('#batchSelect',id+'_legacy');
    await page.waitForFunction(()=>document.querySelector('#clusterViewBtn').disabled);
    assert.deepEqual(errors,[]);
    save(path.join(artifacts,'cluster_ui_smoke.json'),{passed:true,checks:['summary','filter','override_on_disk','audio_playback','next_clip_within_filter','removed_clip_playback','legacy_degradation'],page_errors:errors});
    console.log('Cluster UI smoke passed');
  } finally {
    await browser.close();
  }
}
main().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>{
  // The two literal, per-run fixtures are the only directories this script owns.
  for(const dir of [fixture,legacy]) {
    assert.equal(path.dirname(path.resolve(dir)),out);
    assert.ok(path.basename(dir).startsWith('cluster_ui_smoke_'));
    fs.rmSync(dir,{recursive:true,force:true});
    fs.rmSync(path.join(root,'annotations',path.basename(dir)+'.json'),{force:true});
  }
});
