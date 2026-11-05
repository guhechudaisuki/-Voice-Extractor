/* Acoustic constellation selection and authentic word-time trimming. */
(() => {
  function retainedPreview(seg, fullText, intervals) {
    const full=String(fullText || '').replace(/\|/g,'／').trim();
    const characters=Array.from(full), normalized=characters.filter(c=>!(/\s/u.test(c))).join('');
    const words=seg.word_timestamps || [];
    const valid=words.length && words.every(w=>w.timing_estimated===false && !w.diagnostics?.timing_estimated &&
      String(w.text || '').trim() && Number.isFinite(w.start) && Number.isFinite(w.end) &&
      seg.start<=w.start && w.start<w.end && w.end<=seg.end);
    const ordered=[...words].sort((a,b)=>a.start-b.start);
    if(!valid || ordered.map(w=>w.text.replace(/\s/gu,'')).join('')!==normalized ||
       ordered.some((w,i)=>i && ordered[i-1].end>w.start+1e-6))
      return '需手工补写：真实字词时间戳不完整，或台词已修改而未重新对齐。';
    const positions=characters.flatMap((c,i)=>/\s/u.test(c)?[]:[i]);
    const keeps=[];let cursor=0;
    for(const [a,b] of [...intervals].sort((a,b)=>a[0]-b[0])){
      if(a>cursor)keeps.push([cursor,a]);cursor=Math.max(cursor,b);
    }
    if(cursor<seg.duration)keeps.push([cursor,seg.duration]);
    if(!keeps.length)return '没有保留区间。';
    return '保留文本预览 · 来源：真实字词时间戳\n'+keeps.map(([a,b],i)=>{
      const prefix=`${i+1}. ${a.toFixed(3)}–${b.toFixed(3)} 秒：`;
      if(b-a<.35-1e-9)return prefix+'短于 0.35 秒，不进入训练导出';
      let offset=0,first=null,last=0;
      for(const w of ordered){
        const length=Array.from(w.text.replace(/\s/gu,'')).length;
        if(Math.min(seg.start+b,w.end)>Math.max(seg.start+a,w.start)+1e-9){
          if(w.start<seg.start+a-1e-6 || w.end>seg.start+b+1e-6)return prefix+'需手工补写（边界切入词内）';
          first??=offset;last=offset+length;
        }
        offset+=length;
      }
      return prefix+(first==null?'无完整字词，需手工补写':characters.slice(positions[first],positions[last]??characters.length).join('').trim());
    }).join('\n');
  }
  function refreshPreview() {
    const holder=$('retainedTextPreview'), seg=currentSegment();
    if(!seg || $('trimBox').hidden)return;
    let text;
    try { text=retainedPreview(seg,app.reviews[seg.ordinal]?.revised_text ?? seg.stt_text,readTrimIntervals(seg)); }
    catch(error){text=error.message;}
    if(holder.textContent!==text)holder.textContent=text;
  }
  // All trim paths redraw the waveform; keep the read-only view synchronized
  // without assigning automatic text to the manually authored partial_text.
  const baseDraw=drawWave;
  drawWave=function(...args){const result=baseDraw(...args);refreshPreview();return result;};
  $('trimRanges').addEventListener('input',refreshPreview);
  function realWords(seg) {
    return (seg?.word_timestamps || []).filter(w => w.timing_estimated === false &&
      Number.isFinite(w.start) && Number.isFinite(w.end) && w.end>w.start &&
      w.start>=seg.start-.001 && w.end<=seg.end+.001);
  }
  function renderWords() {
    const holder = $('wordBricks'), seg = currentSegment();
    holder.replaceChildren();
    holder.hidden = !seg || $('trimBox').hidden;
    if (holder.hidden) return;
    refreshPreview();
    const words = realWords(seg);
    const note = document.createElement('span'); note.className = 'word-note';
    note.textContent = words.length ? '点词试听；Shift 点两个词标记他人声音，再按 3 保存。' :
      '没有可靠的字词时间，请在波形上标记，并填写保留段台词。';
    holder.append(note);
    let anchor = null;
    words.forEach((word,index) => {
      const button = document.createElement('button'); button.type = 'button';
      button.className = 'word-brick'; button.textContent = word.text;
      button.title = `${(word.start-seg.start).toFixed(3)}–${(word.end-seg.start).toFixed(3)} 秒`;
      button.onclick = event => {
        if (!event.shiftKey) { audio.currentTime = word.start-seg.start; audio.play(); return; }
        if (anchor == null) { anchor=index; button.classList.add('is-selected'); return; }
        const a = Math.min(anchor,index), b = Math.max(anchor,index);
        const start = words[a].start-seg.start, end = words[b].end-seg.start;
        $('trimStart').value=start; $('trimEnd').value=end;
        $('trimRanges').value=`${start}-${end}`;
        holder.querySelectorAll('.word-brick').forEach((node,i) => node.classList.toggle('is-selected',a<=i&&i<=b));
        app.trimDirty=true; anchor=null; drawWave(app.peaks || []);
      };
      holder.append(button);
    });
  }
  const baseReflect = reflectReview;
  reflectReview = function(...args) { const result=baseReflect(...args); renderWords(); return result; };
  const baseDecide = decide;
  decide = function(...args) { const result=baseDecide(...args); renderWords(); return result; };

  $('openStarMap').onclick = async () => {
    const batchId=app.batchId;
    const modal=document.createElement('dialog'); modal.className='star-dialog';
    const heading=document.createElement('h2'); heading.textContent='声纹星图';
    const note=document.createElement('p'); note.textContent='拖动框选相近声线；Shift 拖动套索。点选单句可试听。';
    const canvas=document.createElement('canvas'); canvas.width=900; canvas.height=480;
    canvas.tabIndex=0; canvas.setAttribute('aria-label','声纹星图，拖动选择；也可按说话人簇选择');
    const status=document.createElement('p'); status.setAttribute('role','status');
    const toolbar=document.createElement('div'); toolbar.className='star-toolbar';
    const group=document.createElement('select'); group.className='field'; group.setAttribute('aria-label','按说话人簇选择');
    const close=document.createElement('button'); close.className='mini-btn'; close.textContent='关闭'; close.onclick=()=>modal.close();
    modal.append(heading,note,canvas,status,toolbar); toolbar.append(group,close);
    modal.addEventListener('close',()=>modal.remove()); document.body.append(modal); modal.showModal();
    let points=[], selection=new Set(), drag=null;
    const ctx=canvas.getContext('2d');
    const pos=e=>{const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*900/r.width,(e.clientY-r.top)*480/r.height];};
    const pixel=p=>[(p.x*.44+.5)*900,(.5-p.y*.42)*480];
    function draw() {
      ctx.clearRect(0,0,900,480);
      points.forEach(p=>{
        const [x,y]=pixel(p), label=app.reviews[p.ordinal]?.label;
        ctx.beginPath(); ctx.arc(x,y,selection.has(p.ordinal)?6:4,0,2*Math.PI);
        ctx.fillStyle=label==='correct'?'#38e8a0':label==='wrong'?'#ff6b81':`hsl(${185+(Number(p.cluster_id)||0)*47%150} 68% 70%)`;
        ctx.fill(); if(selection.has(p.ordinal)){ctx.strokeStyle='#ffc247';ctx.lineWidth=2;ctx.stroke();}
      });
      if(drag){ctx.strokeStyle='#ffc247';ctx.beginPath();drag.path.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();}
      status.textContent=`${points.length} 句 · 已选 ${selection.size} 句。标注只应用于未标句，Ctrl+Z 可撤销。`;
    }
    function inside(point,polygon){let yes=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
      const [x,y]=point,[a,b]=polygon[i],[c,d]=polygon[j];
      if((b>y)!==(d>y)&&x<(c-a)*(y-b)/(d-b)+a)yes=!yes;
    }return yes;}
    canvas.onpointerdown=e=>{drag={first:pos(e),path:[pos(e)],lasso:e.shiftKey};canvas.setPointerCapture(e.pointerId);};
    canvas.onpointermove=e=>{if(!drag)return;const p=pos(e),a=drag.first;
      if(drag.lasso)drag.path.push(p);else drag.path=[a,[p[0],a[1]],p,[a[0],p[1]],a];draw();};
    canvas.onpointerup=e=>{if(!drag)return;const p=pos(e),distance=Math.max(...drag.path.map(v=>Math.hypot(v[0]-drag.first[0],v[1]-drag.first[1])),Math.hypot(p[0]-drag.first[0],p[1]-drag.first[1]));
      if(distance<6){const nearest=[...points].sort((a,b)=>Math.hypot(...pixel(a).map((v,i)=>v-p[i]))-Math.hypot(...pixel(b).map((v,i)=>v-p[i])))[0];
        if(nearest){selection=new Set([nearest.ordinal]);select(app.segments.findIndex(s=>s.ordinal===nearest.ordinal),true);}}
      else selection=new Set(points.filter(q=>inside(pixel(q),drag.path)).map(q=>q.ordinal));
      drag=null;canvas.releasePointerCapture(e.pointerId);draw();};
    canvas.onpointercancel=()=>{drag=null;draw();};
    for(const [label,title] of [['correct','所选未标句 → 目标匹配'],['wrong','所选未标句 → 非目标']]){
      const button=document.createElement('button');button.className='mini-btn';button.textContent=title;
      button.onclick=async()=>{
        if(app.batchId!==batchId){status.textContent='批次已切换，请重新打开星图。';return;}
        const chosen=app.segments.filter(s=>selection.has(s.ordinal)&&!app.reviews[s.ordinal]?.label);
        if(!chosen.length)return;
        app.history.push(JSON.stringify(app.reviews));if(app.history.length>60)app.history.shift();
        chosen.forEach(s=>{app.reviews[s.ordinal]={ordinal:s.ordinal,filename:s.filename,label};});
        renderList();refreshMeta();if(currentSegment())reflectReview(currentSegment().ordinal);
        await save(false);draw();
      };toolbar.insertBefore(button,close);
    }
    try{
      const response=await fetch(`/api/batches/${encodeURIComponent(batchId)}/star_map`);
      const result=await response.json();if(!response.ok)throw new Error(result.detail||'星图读取失败');
      if(app.batchId!==batchId||!modal.isConnected)return;
      points=result.points;
      group.add(new Option('按说话人簇选择',''));
      const groups=new Map();
      points.forEach(p=>{if(p.cluster_key!=null)groups.set(p.cluster_key,p.cluster_label || p.cluster_key);});
      [...groups].sort((a,b)=>a[1].localeCompare(b[1])).forEach(([key,label])=>group.add(new Option(label,key)));
      group.onchange=()=>{selection=new Set(points.filter(p=>p.cluster_key===group.value).map(p=>p.ordinal));draw();};
      draw();
    }catch(error){status.textContent=error.message;}
  };
})();
