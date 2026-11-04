/* Review revisions, cached role re-evaluation, corpus compilation and local picker. */
(() => {
  const transcript = $('sttLine');
  transcript.addEventListener('input', () => {
    const seg = currentSegment(), review = seg && app.reviews[seg.ordinal];
    if (!review || review.label !== 'correct') return;
    review.revised_text = transcript.innerText.replace(/\r?\n/g, ' ').trim();
    save(false);
  });
  transcript.addEventListener('paste', event => {
    event.preventDefault();
    document.execCommand('insertText', false, event.clipboardData.getData('text/plain'));
  });

  async function request(url, options = {}) {
    const response = await fetch(url, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    return result;
  }
  function download(data) {
    const link = document.createElement('a');
    link.href = data.download_url; link.download = data.zip_name;
    link.click();
  }
  const rescore = $('rescoreModel');
  rescore.addEventListener('click', async () => {
    if (!app.batchId || !await save(false)) return;
    const batchId = app.batchId, url = `/api/batches/${encodeURIComponent(batchId)}/rescore`;
    rescore.disabled = true;
    try {
      let result = await request(url, {method:'POST'});
      while (result.status === 'running') {
        rescore.textContent = result.message || '正在重评…';
        await new Promise(resolve => setTimeout(resolve, 1000));
        result = await request(url);
      }
      if (result.status !== 'completed') throw new Error(result.message || '重评失败');
      if (app.batchId === batchId) await loadBatch(batchId);
      toast(`重评完成：晋升 ${result.promoted} 段，降级 ${result.demoted} 段，用时 ${result.seconds} 秒`, 'ok');
    } catch (error) { toast(error.message, 'err'); }
    finally { rescore.disabled = false; rescore.textContent = '用最新强化模型重评本批次'; }
  });
  // Highlight acoustic promotions separately from the user's review verdict.
  const baseRender = renderList;
  renderList = function(...args) {
    const result = baseRender(...args);
    document.querySelectorAll('.seg').forEach(node => {
      const ordinal = Number(node.querySelector('.seg-no')?.textContent);
      if (app.segments.find(s => s.ordinal === ordinal)?.role_rescore?.promoted) {
        node.classList.add('role-promoted'); node.title = '最新强化模型救回 · 仍可人工复核';
      }
    });
    return result;
  };

  function dialog(title) {
    const element = document.createElement('dialog');
    element.className = 'wave3-dialog';
    const head = document.createElement('div'); head.className = 'rail-head';
    const label = document.createElement('strong'); label.textContent = title;
    const close = document.createElement('button'); close.className = 'mini-btn'; close.textContent = '关闭';
    close.onclick = () => element.close(); head.append(label, close); element.append(head);
    element.addEventListener('close', () => element.remove());
    document.body.append(element); element.showModal(); return element;
  }
  $('mergeDatasets').addEventListener('click', async () => {
    if (app.batchId && !await save(false)) return;
    const modal = dialog('多集数据集总装'), list = document.createElement('div');
    list.className = 'wave3-file-list'; modal.append(list);
    const loudness = document.createElement('input'); loudness.type = 'number'; loudness.value = '-23';
    loudness.min = '-36'; loudness.max = '-12'; loudness.setAttribute('aria-label', '目标 LUFS');
    const peak = document.createElement('input'); peak.type = 'number'; peak.value = '-1';
    peak.min = '-6'; peak.max = '-1'; peak.step = '.1'; peak.setAttribute('aria-label', '真峰值 dBTP');
    const action = document.createElement('button'); action.className = 'mini-btn solid'; action.textContent = '去重、统一响度并导出';
    const status = document.createElement('p'); status.setAttribute('role','status');
    modal.append('目标 LUFS ', loudness, ' 真峰值 dBTP ', peak, action, status);
    try {
      const batches = await request('/api/batches');
      for (const batch of batches.filter(b => b.labeled_count > 0 || b.is_reviewed)) {
        const row = document.createElement('label'); row.className = 'wave3-file-row';
        const check = document.createElement('input'); check.type = 'checkbox'; check.value = batch.id;
        check.checked = batch.id === app.batchId; row.append(check, document.createTextNode(batch.id)); list.append(row);
      }
    } catch (error) { status.textContent = error.message; }
    action.onclick = async () => {
      const ids = [...list.querySelectorAll('input:checked')].map(c => c.value);
      if (!ids.length) { status.textContent = '请勾选至少一个批次'; return; }
      action.disabled = true; status.textContent = '正在裁切、去重和校准响度…';
      try {
        const data = await request('/api/datasets/merge', {method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({batch_ids:ids,speaker:$('exportSpeaker').value,language:$('exportLanguage').value,
            target_lufs:Number(loudness.value),true_peak:Number(peak.value),min_quality_grade:$('minQuality').value||null})});
        download(data); status.textContent = `已导出 ${data.valid_count} 段，去重 ${data.duplicate_count} 段`;
      } catch (error) { status.textContent = error.message; }
      finally { action.disabled = false; }
    };
  });

  document.querySelectorAll('[data-browse]').forEach(button => button.addEventListener('click', () => {
    const target = $(button.dataset.browse), modal = dialog('选择本地文件');
    const kind = button.dataset.kind || 'media';
    const allowDirectories = target.id === 'targetsInput' || kind === 'directory';
    const navigation = document.createElement('div'), up = document.createElement('button'), address = document.createElement('input');
    up.textContent = '上一级'; address.className = 'field'; address.placeholder = '目录路径';
    navigation.append(up, address); modal.append(navigation);
    const list = document.createElement('div'); list.className = 'wave3-file-list'; modal.append(list);
    const status = document.createElement('p'), add = document.createElement('button'), more = document.createElement('button');
    add.textContent = allowDirectories ? '添加选中路径' : '添加选中文件'; more.textContent = '下一页'; add.className = 'mini-btn solid';
    modal.append(status, more, add);
    const selection = new Set(); let parent = '', next = null;
    async function load(path = '', offset = 0) {
      try {
        const data = await request(`/api/fs/browse?path=${encodeURIComponent(path)}&offset=${offset}&kind=${encodeURIComponent(kind)}`);
        address.value = data.path; parent = data.parent; next = data.next_offset; more.hidden = next == null;
        list.replaceChildren();
        for (const entry of data.entries) {
          const row = document.createElement('label'); row.className = 'wave3-file-row';
          if (!entry.directory || allowDirectories) {
            const check = document.createElement('input'); check.type = 'checkbox'; check.checked = selection.has(entry.path);
            check.setAttribute('aria-label', `选择 ${entry.name}`);
            check.onchange = () => { check.checked ? selection.add(entry.path) : selection.delete(entry.path); status.textContent = `已选 ${selection.size} 个路径`; };
            row.append(check);
          }
          if (entry.directory) {
            const folder = document.createElement('button'); folder.textContent = `📁 ${entry.name}`;
            folder.type = 'button'; folder.onclick = event => { event.preventDefault(); load(entry.path); }; row.append(folder);
          } else {
            row.append(document.createTextNode(entry.name));
          }
          list.append(row);
        }
      } catch (error) { status.textContent = error.message; }
    }
    up.onclick = () => load(parent || ''); more.onclick = () => load(address.value, next);
    address.onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); load(address.value); } };
    add.onclick = () => {
      target.value = [...new Set([...target.value.split(/\r?\n/).filter(Boolean), ...selection])].join('\n');
      target.dispatchEvent(new Event('input', {bubbles:true})); modal.close();
    };
    load();
  }));
})();
