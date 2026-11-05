"use strict";

/* Voice Extractor Studio — 试听标注与提取任务 */

const $ = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

const LABELS = {
  correct: "目标匹配",
  wrong: "整句非目标",
  partial: "部分非目标",
  singing: "歌声伴奏",
  overlap: "重叠讲话",
  skip: "存疑跳过",
};

const app = {
  batchId: "",
  segments: [],
  index: -1,
  reviews: {},
  filter: "all",
  blind: true,
  autoNext: true,
  history: [],
  peaks: null,
  peaksFor: "",
  decodeToken: 0,
  extractCursor: 0,
  extractTimer: null,
  subtitleBindings: new Map(),
  // 簇视图
  clusters: null,        // /api/batches/<id>/clusters 的返回
  turns: [],            // 说话轮次时间段
  segCluster: {},        // 输出片段序号 -> 簇 id
  activeCluster: null,   // 当前筛选中的簇 id
  view: "player",        // player | cluster
};

const audio = new Audio();
audio.preload = "auto";

/* ── 小工具 ─────────────────────────────────────────────── */

function toast(message, kind) {
  const node = document.createElement("div");
  node.className = "toast" + (kind ? " " + kind : "");
  node.textContent = message;
  $("toasts").appendChild(node);
  setTimeout(() => {
    node.style.opacity = "0";
    setTimeout(() => node.remove(), 200);
  }, 2600);
}

function setSaveState(text, isError) {
  const node = $("saveState");
  node.textContent = text;
  node.classList.toggle("is-err", Boolean(isError));
}

function fmtTime(seconds) {
  if (!Number.isFinite(seconds)) return "0.00";
  return seconds.toFixed(2);
}

function currentSegment() {
  return app.segments[app.index] || null;
}

function tierRank(seg) {
  return [seg.tier_b_rank_score, seg.calibrated_fusion_score, seg.score]
    .find(value => typeof value === "number" && Number.isFinite(value)) ?? -Infinity;
}

function qualityGrade(seg) {
  const grade = seg.acoustic_qc?.quality_grade;
  return ["Gold", "Silver", "Review"].includes(grade) ? grade : "";
}

function qualityBadge(seg) {
  const grade = qualityGrade(seg);
  return grade ? `<span class="quality-badge quality-${grade.toLowerCase()}">${{Gold:"🏅", Silver:"🥈", Review:"⚠️"}[grade]} ${grade}</span>` : "";
}

/* ── 面板与标签页 ───────────────────────────────────────── */

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-on"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("is-on"));
    tab.classList.add("is-on");
    $(tab.dataset.panel).classList.add("is-on");
    if (tab.dataset.panel === "p-settings") loadSettings();
  });
});

/* ── 盲听 ───────────────────────────────────────────────── */

function applyBlind() {
  $("blindBtn").classList.toggle("is-on", app.blind);
  $("readout").hidden = app.blind;
}

$("blindBtn").addEventListener("click", () => {
  app.blind = !app.blind;
  if (!app.blind) prefillTrimProposal();
  else if (app.trimSuggested && !app.trimDirty) { $('trimBox').hidden = true; $('trimRanges').value = ''; }
  drawWave(app.peaks || []);
  applyBlind();
  renderList();
  toast(app.blind ? "盲听：已隐藏分数" : "已显示分数与判定路线");
});

/* ── 批次 ───────────────────────────────────────────────── */

async function loadBatches() {
  try {
    const response = await fetch("/api/batches");
    if (!response.ok) throw new Error("HTTP " + response.status);
    const batches = await response.json();
    const select = $("batchSelect");
    select.innerHTML = "";

    if (!batches.length) {
      select.innerHTML = '<option value="">output 目录里没有批次</option>';
      $("segList").innerHTML = '<p class="empty">先跑一次提取任务。</p>';
      return;
    }

    renderBatches(batches);
  } catch (error) {
    toast("读取批次失败：" + error.message, "err");
  }
}

function renderBatches(batches) {
  const select = $("batchSelect");
  batches.forEach((batch) => {
    const option = document.createElement("option");
    option.value = batch.id;
    const done = batch.labeled_count || 0;
    option.textContent = `${batch.id} · ${batch.accepted_count} 段 · 已标 ${done}`;
    select.appendChild(option);
  });
  select.value = batches[0].id;
  loadBatch(batches[0].id);
}

$("batchSelect").addEventListener("change", (event) => {
  if (event.target.value) loadBatch(event.target.value);
});

$("reloadBatchesBtn").addEventListener("click", loadBatches);

async function loadBatch(batchId) {
  app.batchId = batchId;
  app.segments = [];
  app.reviews = {};
  app.index = -1;
  app.activeCluster = null;
  resetPlayer();
  refreshMeta();
  try {
    const response = await fetch(`/api/batches/${encodeURIComponent(batchId)}`);
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    if (app.batchId !== batchId) return;

    // Keep legacy API display fields, but all word/trim arithmetic uses the
    // original analysis coordinates. Round only human-readable time labels.
    app.segments = (data.segments || []).map(seg => ({...seg,
      start: seg.exact_start ?? seg.start, end: seg.exact_end ?? seg.end,
      duration: seg.exact_duration ?? seg.duration})).filter(
      (seg) => seg.accepted || seg.remaining_sentence_export || seg.cluster_removed || seg.cluster_tier === "B");
    $('openStarMap').hidden = !app.segments.some(s => s.all_sentence_export);
    $('trustMainBtn').hidden = app.segments.some(s => s.all_sentence_export);
    app.reviews = {};
    (data.segments || []).forEach((seg) => {
      if (seg.review && seg.review.label) app.reviews[seg.ordinal] = seg.review;
    });
    app.history = [];
    app.index = -1;

    $("clusterSummary")?.remove();
    refreshMeta();

    renderList();
    const first = visibleSegments()[0];
    if (first) select(app.segments.indexOf(first), false);
    else {
      $("segList").innerHTML = '<p class="empty">这个批次没有保留的片段。</p>';
      resetPlayer();
    }
    // 簇报告是可选资源，失败不影响试听
    await loadClusters(batchId);
    renderList();
    if (app.clusters) renderClusters();
  } catch (error) {
    toast("加载批次失败：" + error.message, "err");
  }
}

/* ── 簇视图 ─────────────────────────────────────────────── */

async function loadClusters(batchId, targetDir = null) {
  app.clusters = null;
  app.turns = [];
  app.segCluster = {};
  app.activeCluster = null;
  $("clusterViewBtn").disabled = true;
  showClusterRail(false);

  try {
    const [clusterResponse, turnResponse] = await Promise.all([
      fetch(`/api/batches/${encodeURIComponent(batchId)}/clusters` +
        (targetDir ? `?target_dir=${encodeURIComponent(targetDir)}` : "")),
      fetch(`/api/batches/${encodeURIComponent(batchId)}/stage_turns`),
    ]);
    const clusterData = await clusterResponse.json();
    const turnData = await turnResponse.json();
    if (app.batchId !== batchId) return;

    if (!clusterData.available) {
      $("clusterViewBtn").disabled = true;
      $("clusterViewBtn").title = clusterData.reason || "没有簇报告";
      return;
    }
    app.clusters = clusterData;
    const targets = $("clusterTarget");
    targets.replaceChildren();
    (clusterData.targets || []).forEach((target) => {
      const option = document.createElement("option");
      option.value = target; option.textContent = target;
      targets.appendChild(option);
    });
    targets.value = clusterData.target_dir;
    let summary = $("clusterSummary");
    if (!summary) {
      summary = document.createElement("span"); summary.id = "clusterSummary";
      $("batchMeta").appendChild(summary);
    }
    const chosen = clusterData.clusters.find((c) => c.is_target);
    summary.textContent = ` · ${clusterData.cluster_count} 个簇 · ` +
      (chosen ? `程序目标 #${chosen.cluster_id}（${chosen.member_count} 轮）` : "未找到目标簇");
    if (turnData.available) app.turns = turnData.turns || [];
    app.segCluster = mapSegmentsToClusters(app.segments, app.clusters, app.turns);
    $("clusterViewBtn").disabled = false;
    $("clusterViewBtn").title = `共 ${clusterData.cluster_count} 个簇`;
  } catch (error) {
    $("clusterViewBtn").disabled = true;
    $("clusterViewBtn").title = "簇报告读取失败";
  }
}

// 轮次序号与输出片段序号不同，按最大时间重叠匹配所属簇。
function mapSegmentsToClusters(segments, clusters, turns) {
  const mapping = {};
  if (!clusters) return mapping;

  const indexToCluster = {};
  clusters.clusters.forEach((cluster) => {
    (cluster.member_indexes || []).forEach((index) => {
      if (index != null) indexToCluster[index] = cluster.cluster_id;
    });
  });

  segments.forEach((segment) => {
    if (clusters.target_dir && clusters.target_dir !== "." && segment.target_dir !== clusters.target_dir) return;
    let best = null;
    let bestOverlap = 0;
    const members = clusters.clusters.flatMap((c) => (c.members || []).map((m) => ({...m, cluster_id: c.cluster_id})));
    const evidence = members.length ? members : turns.map((span,index) => ({span, cluster_id:indexToCluster[index]}));
    evidence.forEach(({span, cluster_id}) => {
      const overlap = Math.min(segment.end, span[1]) - Math.max(segment.start, span[0]);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = cluster_id;
      }
    });
    if (best != null && bestOverlap > 0) mapping[segment.ordinal] = best;
  });
  return mapping;
}

function showClusterRail(show) {
  app.view = show ? "cluster" : "player";
  $("clusterRail").hidden = !show;
  document.querySelector(".rail-play").hidden = show;
  document.querySelectorAll('[data-view]').forEach((chip) => {
    chip.classList.toggle("is-on", chip.dataset.view === app.view);
  });
}

document.querySelectorAll("[data-view]").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (chip.disabled) return;
    showClusterRail(chip.dataset.view === "cluster");
    if (chip.dataset.view === "cluster") renderClusters();
  });
});

function renderClusters() {
  const data = app.clusters;
  if (!data) return;

  $("clusterCount").textContent = String(data.cluster_count);
  $("clusterCheck").innerHTML = renderClusterCheck(data);
  const override = $("clusterOverride");
  override.replaceChildren();
  const automatic = document.createElement("option");
  automatic.value = ""; automatic.textContent = "使用程序判定";
  override.appendChild(automatic);
  data.clusters.forEach((cluster) => {
    const option = document.createElement("option");
    option.value = String(cluster.cluster_id);
    option.textContent = `簇 #${cluster.cluster_id} · ${cluster.member_count} 轮`;
    override.appendChild(option);
  });
  override.value = data.override ? String(data.override.cluster_id) : "";

  const list = $("clusterList");
  list.innerHTML = "";
  const maxSeconds = Math.max(
    ...data.clusters.map((c) => c.total_speech_seconds || 0), 1
  );

  data.clusters.forEach((cluster) => {
    const row = document.createElement("button");
    row.className = "cluster-row";
    if (cluster.is_target) row.classList.add("is-target");
    if (app.activeCluster === cluster.cluster_id) row.classList.add("is-on");

    const ratio = (cluster.total_speech_seconds || 0) / maxSeconds;
    const margin = Number(cluster.target_margin);
    const hasMargin = Number.isFinite(margin);
    const marginClass = margin >= 0 ? "margin-pos" : "margin-neg";
    const inThisCluster = Object.values(app.segCluster)
      .filter((id) => id === cluster.cluster_id).length;

    row.innerHTML =
      `<span class="cluster-id">#${cluster.cluster_id}` +
        `${cluster.is_target ? "★" : ""}</span>` +
      `<span class="cluster-bar-wrap">` +
        `<span class="cluster-name">${cluster.member_count} 个轮次 · ` +
          `${(cluster.total_speech_seconds || 0).toFixed(0)}s` +
          `${inThisCluster ? ` · 可试听 ${inThisCluster} 段` : ""}</span>` +
        `<span class="cluster-bar"><i style="width:${(ratio * 100).toFixed(1)}%"></i></span>` +
      `</span>` +
      `<span class="cluster-stats">` +
        `相似 ${Number(cluster.target_score).toFixed(3)}<br>` +
        (hasMargin
          ? `<span class="${marginClass}">余量 ${margin >= 0 ? "+" : ""}` +
            `${margin.toFixed(3)}</span>`
          : "余量 —") +
      `</span>`;

    row.addEventListener("click", () => {
      app.activeCluster =
        app.activeCluster === cluster.cluster_id ? null : cluster.cluster_id;
      renderClusters();
      applyClusterFilter();
      showClusterRail(false);
    });
    list.appendChild(row);
  });
}

function renderClusterCheck(data) {
  const flagged = data.flagged || {};
  const lines = [];
  lines.push(`目标簇 <b>#${data.target_cluster_id ?? "无"}</b> · ${data.cluster_count} 个簇`);
  if (data.override) lines.push(`人工标注目标簇 #${data.override.cluster_id}`);
  if (flagged.whole_wrong) {
    lines.push(
      `整句非目标 ${flagged.whole_wrong} 段：` +
      `<span class="bad">错进目标簇 ${flagged.inside_target}</span> · ` +
      `<span class="good">已被挡住 ${flagged.elsewhere + flagged.caught}</span>`
    );
  }
  if (flagged.mixed_total) {
    lines.push(
      `部分非目标 ${flagged.mixed_total} 段 · 仍保留 ${flagged.mixed_in_target || 0} 段`
    );
  }
  if (flagged.withheld) lines.push(`未找到目标簇，整集暂扣 ${flagged.withheld} 条已标错误`);
  if (flagged.verdict) lines.push(`结论：${verdictText(flagged.verdict)}`);
  return lines.join("<br>");
}

function verdictText(code) {
  return {
    solved: "已解决",
    review_errors_blocked: "已标错误均未保留（开发集）",
    no_target_cluster: "未找到目标簇，不能算验证通过",
    incomplete_review_coverage: "试听区间覆盖不足",
    identity_blocked_boundary_remaining: "整句错误已挡下，局部错误仍存在",
    partially_solved: "部分解决",
    not_solved: "未解决",
    no_review_overlap: "无试听数据可比对",
  }[code] || "尚无评估结论";
}

$("clusterTarget").addEventListener("change", async (event) => {
  await loadClusters(app.batchId, event.target.value);
  renderClusters(); renderList();
});
$("saveClusterOverride").addEventListener("click", async () => {
  if (!app.clusters) return;
  const value = $("clusterOverride").value;
  try {
    const response = await fetch(`/api/batches/${encodeURIComponent(app.batchId)}/cluster_override`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({cluster_id: value === "" ? null : Number(value),
        target_dir: app.clusters.target_dir, report_sha256: app.clusters.report_sha256}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "保存失败");
    await loadClusters(app.batchId, app.clusters.target_dir);
    renderClusters(); toast("目标簇标注已保存");
  } catch (error) { toast(error.message, "err"); }
});

function applyClusterFilter() {
  const chip = $("clusterFilterChip");
  if (app.activeCluster != null) {
    app.filter = "cluster";
    chip.hidden = false;
    chip.textContent = `簇 #${app.activeCluster}`;
    document.querySelectorAll("[data-filter]").forEach((node) => {
      node.classList.remove("is-on");
    });
    chip.classList.add("is-on");
  } else {
    chip.hidden = true;
    if (app.filter === "cluster") app.filter = "all";
    document.querySelectorAll("[data-filter]").forEach((node) => {
      node.classList.toggle("is-on", node.dataset.filter === "all");
    });
  }
  renderList();
  const first = visibleSegments()[0];
  if (first) select(app.segments.indexOf(first), false);
  else { app.index = -1; resetPlayer(); }
}

/* ── 片段列表 ───────────────────────────────────────────── */

$("filters").addEventListener("click", (event) => {
  const chip = event.target.closest("[data-filter]");
  if (!chip) return;
  const wasClusterChip = chip.id === "clusterFilterChip";
  // 切换筛选时清除簇选择。
  app.activeCluster = null;
  app.filter = wasClusterChip ? "all" : chip.dataset.filter;
  $("clusterFilterChip").hidden = true;

  document.querySelectorAll("[data-filter]").forEach((c) => c.classList.remove("is-on"));
  (wasClusterChip ? document.querySelector('[data-filter="all"]') : chip)
    .classList.add("is-on");
  renderList();
  const first = visibleSegments()[0];
  if (first) select(app.segments.indexOf(first), false);
  else { app.index = -1; resetPlayer(); }
});

function visibleSegments() {
  const rows = app.segments.filter((seg) => {
    // 主输出、其余单句、已保存候选与撤下片段分别复核。
    if (app.filter === "tierA") return seg.accepted;
    if (app.filter === "remaining") return seg.remaining_sentence_export;
    if (app.filter === "tierB") return !seg.accepted && seg.cluster_tier === "B";
    if (app.filter === "removed") return !seg.accepted && !seg.remaining_sentence_export && seg.cluster_tier !== "B";
    if (!seg.accepted && !seg.remaining_sentence_export && seg.cluster_tier !== "B" && !["cluster", "active"].includes(app.filter)) return false;
    const done = Boolean(app.reviews[seg.ordinal]);
    if (app.filter === "unreviewed" || app.filter === "active") return !done;
    if (app.filter === "done") return done;
    if (app.filter === "cluster" && app.activeCluster != null) {
      return app.segCluster[seg.ordinal] === app.activeCluster;
    }
    return true;
  });
  if (app.filter === "active") return activeLearningOrder(rows);
  if (["tierB","removed"].includes(app.filter) && rows.some(s => s.live_information != null))
    return rows.sort((a,b) => (b.live_information ?? -1)-(a.live_information ?? -1));
  return app.filter === "tierB" ? rows.sort((a, b) => tierRank(b) - tierRank(a)) : rows;
}

function activeLearningOrder(rows) {
  if (rows.some(s => Number.isFinite(s.acquisition_rank)))
    return [...rows].sort((a,b) => (a.acquisition_rank ?? Infinity)-(b.acquisition_rank ?? Infinity));
  // Greedy information + farthest-signature sampling for a diverse first 15.
  const remaining = [...rows], selected = [];
  const distance = (a, b) => {
    const pairs = (a.identity_signature || []).map((v,i) => [v,b.identity_signature?.[i]])
      .filter(pair => pair.every(v => typeof v === "number" && Number.isFinite(v)));
    return pairs.length ? Math.sqrt(pairs.reduce((s,[x,y]) => s+(x-y)**2,0)/pairs.length) : 0;
  };
  while (remaining.length && selected.length < 15) {
    const score = seg => (seg.live_information ?? (Number(seg.information_score) || 0)) +
      (selected.length ? Math.min(...selected.map(other => distance(seg,other))) : 0);
    remaining.sort((a,b) => score(b)-score(a) || a.ordinal-b.ordinal);
    selected.push(remaining.shift());
  }
  return [...selected,...remaining.sort((a,b) => (Number(b.information_score)||0)-(Number(a.information_score)||0))];
}

function renderList() {
  const list = $("segList");
  const rows = visibleSegments();
  list.innerHTML = "";
  $("segmentCounter").textContent = String(rows.length);

  if (!rows.length) {
    list.innerHTML = '<p class="empty">没有符合筛选的片段。</p>';
    return;
  }

  const targetId = app.clusters
    ? (app.clusters.clusters.find((c) => c.is_target) || {}).cluster_id
    : undefined;

  rows.forEach((seg) => {
    const review = app.reviews[seg.ordinal];
    const node = document.createElement("div");
    node.className = "seg";
    if (seg.ordinal === (currentSegment() || {}).ordinal) node.classList.add("is-on");

    const tag = review
      ? `<span class="seg-tag tag-${review.label}">${LABELS[review.label] || review.label}</span>`
      : "";

    const clusterId = app.segCluster[seg.ordinal];
    const clusterTag = clusterId == null
      ? ""
      : `<span class="seg-tag cluster-tag">簇${clusterId}` +
        `${clusterId === targetId ? "★" : ""}</span>`;

    node.innerHTML =
      `<span class="seg-no">${String(seg.ordinal).padStart(3, "0")}</span>` +
      `<span class="seg-body">` +
        `<span class="seg-text">${escapeHtml(seg.stt_text) || escapeHtml(seg.filename)}</span>` +
        `<span class="seg-sub">${fmtTime(seg.start)}–${fmtTime(seg.end)}s · ${seg.duration.toFixed(2)}s${seg.accepted ? "" : seg.remaining_sentence_export ? " · 其余单句待复核" : " · 已撤下"}</span>` +
      `</span>` + (!app.blind ? qualityBadge(seg) : "") + clusterTag + tag;

    node.addEventListener("click", () => {
      const position = app.segments.indexOf(seg);
      if (position >= 0) select(position, true);
    });
    list.appendChild(node);
  });
}

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* ── 选中与播放 ─────────────────────────────────────────── */

function select(position, autoplay) {
  if (position < 0 || position >= app.segments.length) return;
  clearKeepPlayback();
  app.index = position;
  const seg = app.segments[position];

  document.querySelectorAll(".seg").forEach((node) => {
    const ordinal = Number(node.querySelector(".seg-no").textContent);
    node.classList.toggle("is-on", ordinal === seg.ordinal);
  });

  $("nowIndex").textContent = String(seg.ordinal).padStart(3, "0");
  $("nowFile").textContent = seg.filename;
  $("nowDuration").textContent = seg.duration.toFixed(2) + "s";
  $("srcRange").textContent = `源 ${fmtTime(seg.start)} – ${fmtTime(seg.end)}s`;
  $("sttLine").textContent = app.reviews[seg.ordinal]?.revised_text ?? seg.stt_text ?? "";
  $("scopeEmpty").hidden = true;

  const readout = [
    `质量 ${qualityGrade(seg) || "—"} · SNR ${seg.acoustic_qc?.snr_db ?? "—"} dB`,
    `融合分 ${seg.calibrated_fusion_score ?? "—"}`,
    `层级 ${seg.tier || "—"}`,
    `路线 ${seg.route || "—"}`,
    `相似度 ${seg.score == null ? "—" : Number(seg.score).toFixed(4)}`,
  ];
  if (seg.exclusion_margin != null) {
    readout.push(`排除裕度 ${Number(seg.exclusion_margin).toFixed(3)}`);
  }
  $("readout").textContent = readout.join("   ");
  applyBlind();

  reflectReview(seg.ordinal);
  loadAudio(seg, autoplay);
}

function resetPlayer() {
  app.identityHeatmap = null;
  clearKeepPlayback();
  audio.pause();
  audio.removeAttribute("src");
  ++app.decodeToken;
  $("nowIndex").textContent = "—";
  $("nowFile").textContent = "未选择片段";
  $("nowDuration").textContent = "0.00s";
  $("curTime").textContent = "0.00";
  $("totTime").textContent = "0.00";
  $("srcRange").textContent = "";
  $("sttLine").textContent = "（无识别文本）";
  $("readout").textContent = "";
  $("scopeEmpty").hidden = false;
  drawWave([]);
}

async function loadAudio(seg, autoplay) {
  app.identityHeatmap = null;
  audio.src = `/api/audio/${encodeURIComponent(app.batchId)}/${seg.ordinal}`;
  audio.load();
  $("curTime").textContent = "0.00";
  $("totTime").textContent = seg.duration.toFixed(2);
  drawWave([]);
  await drawWaveFor(seg, autoplay);
}

async function drawWaveFor(seg, autoplay) {
  const key = `${app.batchId}:${seg.ordinal}`;
  const token = ++app.decodeToken;
  app.identityHeatmap = null;
  {
    fetch(`/api/identity_heatmap/${encodeURIComponent(app.batchId)}/${seg.ordinal}`)
      .then(response => response.ok ? response.json() : null).then(data => {
        if (token !== app.decodeToken) return;
        app.identityHeatmap = data;
        prefillTrimProposal();
        drawWave(app.peaks || []);
      }).catch(() => {});
  }
  try {
    const response = await fetch(`/api/audio/${encodeURIComponent(app.batchId)}/${seg.ordinal}`);
    const buffer = await response.arrayBuffer();
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await context.decodeAudioData(buffer);
    context.close();
    if (token !== app.decodeToken) return;
    app.peaks = computePeaks(decoded.getChannelData(0), 900);
    app.waveSamples = decoded.getChannelData(0);
    app.waveRate = decoded.sampleRate;
    app.peaksFor = key;
    drawWave(app.peaks);
  } catch (error) {
    if (token !== app.decodeToken) return;
    app.peaks = null;
    app.peaksFor = "";
    drawWave([]);
  }
  if (autoplay) play();
}

function computePeaks(samples, buckets) {
  const step = Math.max(1, Math.floor(samples.length / buckets));
  const peaks = [];
  for (let i = 0; i < samples.length; i += step) {
    let peak = 0;
    const end = Math.min(samples.length, i + step);
    for (let j = i; j < end; j += 1) {
      const value = Math.abs(samples[j]);
      if (value > peak) peak = value;
    }
    peaks.push(peak);
  }
  const max = peaks.reduce((a, b) => Math.max(a, b), 0) || 1;
  return peaks.map((value) => value / max);
}

function drawWave(peaks) {
  const canvas = $("waveCanvas");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.clientHeight || 168;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const mid = height / 2;

  // 中线与横向刻度
  ctx.strokeStyle = "rgba(126, 148, 210, 0.16)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(width, mid);
  ctx.stroke();

  ctx.strokeStyle = "rgba(126, 148, 210, 0.07)";
  ctx.beginPath();
  for (let x = 0; x <= width; x += width / 10) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  ctx.stroke();

  drawIdentityHeat(ctx, width, height);
  if (!peaks.length) return;

  const barWidth = width / peaks.length;
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#4de3ff");
  gradient.addColorStop(0.52, "#6d8dff");
  gradient.addColorStop(1, "#a06bff");

  ctx.shadowColor = "rgba(77, 227, 255, 0.55)";
  ctx.shadowBlur = 9;

  // 旧版 WebView 不支持 roundRect。
  const rounded = typeof ctx.roundRect === "function";
  ctx.beginPath();
  for (let i = 0; i < peaks.length; i += 1) {
    const x = i * barWidth;
    const half = Math.max(0.6, peaks[i] * (mid - 8));
    const bar = Math.max(1, barWidth - 0.6);
    if (rounded) {
      ctx.roundRect(x, mid - half, bar, half * 2, Math.min(bar / 2, 2));
    } else {
      ctx.rect(x, mid - half, bar, half * 2);
    }
  }
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.shadowBlur = 0;

  // 部分非目标的区间标记
  const segment = currentSegment();
  if (segment && !$('trimBox').hidden) {
    let ranges = [];
    try { ranges = readTrimIntervals(segment); } catch (_) { /* In-progress numeric input. */ }
    for (const bounds of ranges) {
      const from = clamp((Number(bounds[0]) || 0) / segment.duration, 0, 1);
      const to = clamp((Number(bounds[1]) || segment.duration) / segment.duration, 0, 1);
      const left = from * width;
      const span = Math.max(2, (to - from) * width);
      const mark = ctx.createLinearGradient(left, 0, left + span, 0);
      mark.addColorStop(0, "rgba(255, 107, 129, 0.30)");
      mark.addColorStop(1, "rgba(255, 107, 129, 0.14)");
      ctx.fillStyle = mark;
      ctx.fillRect(left, 0, span, height);
      ctx.strokeStyle = "rgba(255, 107, 129, 0.75)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, 0); ctx.lineTo(left, height);
      ctx.moveTo(left + span, 0); ctx.lineTo(left + span, height);
      ctx.stroke();
      ctx.fillStyle = '#ff6b81';
      ctx.fillRect(left-4,height/2-14,8,28);
      ctx.fillRect(left+span-4,height/2-14,8,28);
    }
  }
}

function prefillTrimProposal() {
  const seg = currentSegment(), heat = app.identityHeatmap;
  if (!seg || !heat || app.blind || app.trimDirty || $('trimBox').hidden || app.reviews[seg.ordinal]?.label) return;
  const ranges = [];
  for (const frame of heat.frames || []) {
    if (!(Number.isFinite(frame.evidence) && frame.evidence < -.5)) continue;
    const a = clamp(frame.offset,0,seg.duration), b = clamp(frame.end,0,seg.duration);
    if (b <= a) continue;
    if (ranges.length && a <= ranges.at(-1)[1]+.021) ranges.at(-1)[1] = Math.max(b,ranges.at(-1)[1]);
    else ranges.push([a,b]);
  }
  const span = heat.classifier_trim_proposal?.span;
  if (!ranges.length && span?.length === 2 && span.every(Number.isFinite)) {
    const a = clamp(span[0]-seg.start,0,seg.duration), b = clamp(span[1]-seg.start,0,seg.duration);
    if (a > 0) ranges.push([0,a]); if (b < seg.duration) ranges.push([b,seg.duration]);
  }
  const useful = ranges.filter(([a,b]) => b-a >= .1);
  if (!useful.length || useful.reduce((s,[a,b]) => s+b-a,0) >= seg.duration-.1) return;
  $('trimRanges').value = useful.map(pair => pair.join('-')).join('; ');
  $('trimStart').value = useful[0][0]; $('trimEnd').value = useful[0][1];
  app.trimSuggested = true;
}

function snapTrimTime(value, seg) {
  const words = (seg.word_timestamps || []).filter(w => w.timing_estimated === false &&
    Number.isFinite(w.start) && Number.isFinite(w.end));
  const wordEdges = words.flatMap(w => [w.start-seg.start,w.end-seg.start]);
  const wordEdge = wordEdges.sort((a,b) => Math.abs(a-value)-Math.abs(b-value))[0];
  // Preserve the authentic timestamp exactly. Moving it inward to a waveform
  // zero crossing could turn a complete retained word into a partial word.
  if (wordEdge != null && Math.abs(wordEdge-value)<.06) return clamp(wordEdge,0,seg.duration);
  const boundaries = (app.identityHeatmap?.frames || []).filter((f,i,a) => i &&
    Number.isFinite(f.evidence) && Number.isFinite(a[i-1].evidence) &&
    Math.sign(f.evidence) !== Math.sign(a[i-1].evidence)).map(f => f.offset);
  const nearest = boundaries.sort((a,b) => Math.abs(a-value)-Math.abs(b-value))[0];
  if (nearest != null && Math.abs(nearest-value)<.025) value = nearest;
  const samples = app.waveSamples, rate = app.waveRate;
  if (samples && rate && app.peaksFor === `${app.batchId}:${seg.ordinal}`) {
    const center = Math.round(value*rate), radius = Math.round(.003*rate);
    let best = center, distance = Infinity;
    for (let i=Math.max(1,center-radius); i<Math.min(samples.length,center+radius); i++) {
      if (samples[i]*samples[i-1] <= 0 && Math.abs(i-center)<distance) { best=i; distance=Math.abs(i-center); }
    }
    value = best/rate;
  }
  return clamp(value,0,seg.duration);
}
const trimCanvas = $('waveCanvas');
trimCanvas.style.touchAction = 'none';
trimCanvas.addEventListener('pointerdown',event => {
  const seg = currentSegment(); if (!seg || $('trimBox').hidden) return;
  let ranges; try { ranges = readTrimIntervals(seg); } catch (_) { return; }
  const rect = trimCanvas.getBoundingClientRect(), x = event.clientX-rect.left;
  let handle = null, distance = 12;
  ranges.forEach((range,i) => range.forEach((time,edge) => {
    const d = Math.abs(time/seg.duration*rect.width-x);
    if (d <= distance) { distance=d; handle={index:i,edge,ranges}; }
  }));
  if (!handle) return;
  app.trimDrag = handle; app.trimDragged = true; app.trimDirty = true;
  app.trimRangeIndex = handle.index;
  trimCanvas.setPointerCapture(event.pointerId); event.preventDefault();
});
trimCanvas.addEventListener('pointermove',event => {
  const drag = app.trimDrag, seg = currentSegment(); if (!drag || !seg) return;
  const rect = trimCanvas.getBoundingClientRect();
  const value = snapTrimTime((event.clientX-rect.left)/rect.width*seg.duration,seg);
  const pair = drag.ranges[drag.index];
  pair[drag.edge] = drag.edge ? clamp(value,pair[0]+.01,seg.duration) : clamp(value,0,pair[1]-.01);
  $('trimRanges').value = drag.ranges.map(p => p.join('-')).join('; ');
  $('trimStart').value = pair[0]; $('trimEnd').value = pair[1];
  drawWave(app.peaks || []);
});
for (const type of ['pointerup','pointercancel']) trimCanvas.addEventListener(type,event => {
  app.trimDrag = null;
  if (trimCanvas.hasPointerCapture(event.pointerId)) trimCanvas.releasePointerCapture(event.pointerId);
});
window.addEventListener("resize", () => drawWave(app.peaks || []));

function drawIdentityHeat(ctx, width, height) {
  const heat = app.identityHeatmap, seg = currentSegment();
  const available = Boolean(!app.blind && heat?.frames?.length && seg?.duration);
  $("identityHeatLegend").hidden = !available;
  if (!available) return;
  const channels = heat.channels || [];
  for (const frame of heat.frames) {
    const x = clamp(frame.offset/seg.duration,0,1)*width;
    const w = Math.max(1,(clamp(frame.end/seg.duration,0,1)*width)-x);
    const color = value => value == null || !Number.isFinite(value) ? "rgba(150,150,150,.18)" :
      value > 0 ? `rgba(50,210,150,${.15+.5*Math.tanh(Math.abs(value))})` :
      `rgba(255,80,105,${.2+.5*Math.tanh(Math.abs(value))})`;
    ctx.fillStyle = frame.blocked ? "rgba(255,80,105,.30)" : color(frame.evidence);
    ctx.fillRect(x,0,w,height);
    channels.forEach((name,i) => {
      ctx.fillStyle = color(frame.channels?.[name]);
      ctx.fillRect(x,height-(channels.length-i)*5,w,4);
    });
  }
}

$("waveCanvas").addEventListener("click", (event) => {
  if (app.trimDragged) { app.trimDragged = false; return; }
  if (!audio.duration) return;
  const rect = event.target.getBoundingClientRect();
  audio.currentTime = clamp((event.clientX - rect.left) / rect.width, 0, 1) * audio.duration;
  $("scopeHead").style.left = ((audio.currentTime / audio.duration) * 100) + "%";
});

audio.addEventListener("timeupdate", () => {
  if (!audio.duration) return;
  $("curTime").textContent = fmtTime(audio.currentTime);
  $("scopeHead").style.left = ((audio.currentTime / audio.duration) * 100) + "%";
});

audio.addEventListener("ended", () => {
  $("playLabel").textContent = "播放";
});

audio.addEventListener("error", () => {
  if (audio.src) toast("这段音频读不出来", "err");
});

function play() {
  audio.play().then(() => {
    $("playLabel").textContent = "暂停";
  }).catch(() => {
    $("playLabel").textContent = "播放";
  });
}

$("playBtn").addEventListener("click", () => {
  if (audio.paused) play();
  else { audio.pause(); $("playLabel").textContent = "播放"; }
});

$("replayBtn").addEventListener("click", () => {
  if (app.reviews[currentSegment()?.ordinal]?.label === "partial") $("keepPlayBtn").click();
  else { clearKeepPlayback(); audio.currentTime = 0; play(); }
});
$("prevBtn").addEventListener("click", () => step(-1));
$("nextBtn").addEventListener("click", () => step(1));

function step(delta) {
  const visible = visibleSegments();
  const position = visible.indexOf(currentSegment());
  const next = position >= 0 ? visible[position + delta] : visible[delta > 0 ? 0 : visible.length - 1];
  if (next) select(app.segments.indexOf(next), true);
}

/* ── 裁决 ───────────────────────────────────────────────── */

document.querySelectorAll(".pad").forEach((pad) => {
  pad.addEventListener("click", () => decide(pad.dataset.label));
});

function readTrimIntervals(seg) {
  const text = $("trimRanges").value.trim();
  const intervals = text ? text.split(/[;；]+/).map(part => {
    const match = part.trim().match(/^(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*-\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)$/i);
    if (!match) throw new Error("请填写区间，例如 0.5-1.2; 2.8-3.4");
    return [Number(match[1]), Number(match[2])];
  }) : [[parseFloat($("trimStart").value), parseFloat($("trimEnd").value)]];
  if (!intervals.every(([a, b]) => Number.isFinite(a) && Number.isFinite(b) && a >= 0 && b > a && b <= seg.duration + .02))
    throw new Error("污染区间必须位于片段内");
  return intervals;
}

function decide(label) {
  const seg = currentSegment();
  if (!seg) return;

  if (label === "partial" && $("trimBox").hidden) {
    $("trimBox").hidden = false;
    prefillTrimProposal();
    $("trimStart").focus();
    return;
  }
  const visibleBefore = visibleSegments();
  const next = visibleBefore[visibleBefore.indexOf(seg) + 1];
  let intervals = null;
  if (label === "partial") {
    try { intervals = readTrimIntervals(seg); }
    catch (error) { toast(error.message, "err"); return; }
  }

  app.history.push(JSON.stringify(app.reviews));
  if (app.history.length > 60) app.history.shift();

  app.reviews[seg.ordinal] = {
    revised_text: app.reviews[seg.ordinal]?.revised_text ?? null,
    ordinal: seg.ordinal,
    filename: seg.filename,
    label: label,
    partial_start: intervals ? intervals[0][0] : null,
    partial_intervals: intervals,
    partial_end: intervals ? intervals[0][1] : null,
    partial_text: label === "partial" ? (($("partialText") || {}).value || "").trim() || null : null,
  };

  reflectReview(seg.ordinal);
  renderList();
  refreshMeta();
  save(false);

  if ($("autoNext").checked && label !== "partial" && next) {
    select(app.segments.indexOf(next), true);
  }
}

function reflectReview(ordinal) {
  app.trimRangeIndex = 0;
  app.trimDirty = false; app.trimSuggested = false;
  const review = app.reviews[ordinal] || {};
  $('sttLine').contentEditable = review.label === 'correct' ? 'true' : 'false';
  document.querySelectorAll(".pad").forEach((pad) => {
    pad.classList.toggle("is-set", pad.dataset.label === review.label);
  });

  const verdict = $("verdictTag");
  verdict.textContent = review.label ? LABELS[review.label] : "待审";
  verdict.classList.toggle("is-set", Boolean(review.label));

  const seg = currentSegment();
  if (review.label === "partial" && seg) {
    $("trimBox").hidden = false;
    const partialInput = $("partialText");
    if (partialInput) partialInput.value = review.partial_text || "";
    $("trimRanges").value = (review.partial_intervals || []).map(pair => pair.join("-")).join("; ");
    $("trimStart").value = review.partial_intervals?.[0]?.[0] ?? review.partial_start ?? 0;
    $("trimEnd").value = review.partial_intervals?.[0]?.[1] ?? review.partial_end ?? seg.duration;
  } else {
    $("trimBox").hidden = true;
    $("partialText").value = "";
    $("trimRanges").value = "";
    if (seg) {
      $("trimStart").value = "0.00";
      $("trimEnd").value = seg.duration;
    }
  }

  drawWave(app.peaks || []);
}

$("trimFromPlayhead").addEventListener("click", () => {
  const seg = currentSegment();
  if (!seg) return;
  $("trimStart").value = clamp(audio.currentTime, 0, seg.duration);
  syncTrimNumbers();
});

function syncTrimNumbers() {
  app.trimDirty = true;
  const parts = $("trimRanges").value.split(/[;；]+/).filter(v => v.trim());
  parts[app.trimRangeIndex || 0] = `${$("trimStart").value}-${$("trimEnd").value}`;
  $("trimRanges").value = parts.join("; ");
  drawWave(app.peaks || []);
}
for (const id of ["trimStart", "trimEnd"]) $(id).addEventListener("input", syncTrimNumbers);
$("trimRanges").addEventListener("input", () => {
  app.trimDirty = true;
  try {
    const ranges = readTrimIntervals(currentSegment());
    app.trimRangeIndex = 0;
    $("trimStart").value = ranges[0][0];
    $("trimEnd").value = ranges[0][1];
    drawWave(app.peaks || []);
  } catch (_) { /* Keep incomplete text editable. */ }
});

/* 试听保留段：按顺序播放全部干净区间。 */
let keepPlayStop = null;
function clearKeepPlayback() {
  if (keepPlayStop) audio.removeEventListener("timeupdate", keepPlayStop);
  keepPlayStop = null;
}
$("keepPlayBtn").addEventListener("click", () => {
  const seg = currentSegment();
  if (!seg) return;
  let intervals;
  try { intervals = readTrimIntervals(seg).sort((a,b) => a[0]-b[0]); }
  catch (error) { toast(error.message, "err"); return; }
  const pieces = [];
  let cursor = 0;
  for (const [a,b] of intervals) {
    if (a - cursor >= .2) pieces.push([cursor,a]);
    cursor = Math.max(cursor,b);
  }
  if (seg.duration-cursor >= .2) pieces.push([cursor,seg.duration]);
  if (!pieces.length) { toast("没有可保留的干净段", "err"); return; }
  if (keepPlayStop) audio.removeEventListener("timeupdate", keepPlayStop);
  let index = 0;
  audio.currentTime = pieces[0][0];
  play();
  toast(`依次试听 ${pieces.length} 个干净区间`, "ok");
  keepPlayStop = () => {
    if (audio.currentTime >= pieces[index][1] - .02) {
      index += 1;
      if (index < pieces.length) { audio.currentTime = pieces[index][0]; play(); }
      else {
        audio.pause(); audio.removeEventListener("timeupdate", keepPlayStop); keepPlayStop = null;
      }
    }
  };
  audio.addEventListener("timeupdate", keepPlayStop);
});

$("undoBtn").addEventListener("click", () => {
  const previous = app.history.pop();
  if (!previous) { toast("没有可撤销的动作"); return; }
  app.reviews = JSON.parse(previous);
  reflectReview(currentSegment() ? currentSegment().ordinal : -1);
  renderList();
  refreshMeta();
  save(false);
  toast("已撤销");
});

function refreshMeta() {
  const clusterSummary = $("clusterSummary");
  const acceptedCount = app.segments.filter((seg) => seg.accepted).length;
  const remainingCount = app.segments.filter((seg) => seg.remaining_sentence_export).length;
  const removedCount = app.segments.length - acceptedCount - remainingCount;
  const hasBorderline = app.segments.some((seg) => !seg.accepted && seg.cluster_tier === "B");
  const hasRemoved = app.segments.some((seg) => !seg.accepted && !seg.remaining_sentence_export && seg.cluster_tier !== "B");
  for (const [filter, available] of [["tierA", acceptedCount > 0 && removedCount + remainingCount > 0],
                                   ["remaining", remainingCount > 0],
                                   ["tierB", hasBorderline], ["removed", hasRemoved]]) {
    document.querySelector(`[data-filter="${filter}"]`).hidden = !available;
    if (!available && app.filter === filter) app.filter = "all";
  }
  document.querySelectorAll("[data-filter]").forEach((chip) => {
    chip.classList.toggle("is-on", chip.dataset.filter === app.filter);
  });
  $("trustBorderlineBtn").hidden = !hasBorderline;
  $("batchMeta").innerHTML =
    `<span><b>${acceptedCount}</b> 输出</span>` +
    (remainingCount ? `<span>· ${remainingCount} 其余单句待复核</span>` : "") +
    (removedCount ? `<span>· ${removedCount} 撤下可试听</span>` : "") +
    `<span>·</span>` +
    `<span><b>${Object.keys(app.reviews).length}</b> 已标</span>`;
  if (clusterSummary) $("batchMeta").appendChild(clusterSummary);
  const complete = app.segments.filter(seg => validMaterialReview(seg)).length;
  $("includeMaterial").disabled = complete === 0;
  if ($("includeMaterial").disabled) $("includeMaterial").checked = false;
  $("materialState").textContent = complete > 0
    ? `可增量导出 ${complete}/${app.segments.length} 段已标注原料`
    : "标注任意片段后即可导出训练原料";
}

function validMaterialReview(seg) {
  const review = app.reviews[seg.ordinal];
  if (!review || !Object.hasOwn(LABELS, review.label) || review.ordinal !== seg.ordinal || review.filename !== seg.filename) return false;
  return review.label !== "partial" || (review.partial_intervals || [[review.partial_start, review.partial_end]])
    .every(([a, b]) => Number.isFinite(a) && Number.isFinite(b) && a >= 0 && b > a && b <= seg.duration + .02);

}

$("autoNext").addEventListener("change", () => {
  app.autoNext = $("autoNext").checked;
});

/* ── 保存与导出 ─────────────────────────────────────────── */

let saveQueue = Promise.resolve();
function save(notify) {
  const batchId = app.batchId;
  const reviews = JSON.parse(JSON.stringify(app.reviews));
  saveQueue = saveQueue.then(() => saveSnapshot(batchId, reviews, notify));
  return saveQueue;
}

async function saveSnapshot(batchId, reviews, notify) {
  if (!batchId) return false;
  try {
    const response = await fetch(`/api/review/${encodeURIComponent(batchId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch_id: batchId, reviews }),
    });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const saved = await response.json();
    if (app.batchId === batchId) {
      setSaveState(`已保存 ${Object.keys(reviews).length} 条标注至 annotations 目录`);
      $("saveState").title = saved.annotation_path || "";
    }
    if (notify) toast("标注已保存至 annotations 目录", "ok");
    return true;
  } catch (error) {
    setSaveState("保存失败：" + error.message, true);
    if (notify) toast("保存失败：" + error.message, "err");
    return false;
  }
}

$("exportBtn").addEventListener("click", async () => {
  if (!app.batchId) return;
  const batchId = app.batchId;
  const includeMaterial = $("includeMaterial").checked;
  const includeGptList = ($("includeGptList") || {}).checked;
  if (!await save(false)) { toast("标注保存失败，请重试", "err"); return; }
  try {
    const response = await fetch(`/api/export_verified/${encodeURIComponent(batchId)}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker: $("exportSpeaker").value, language: $("exportLanguage").value, include_material: includeMaterial, gpt_sovits_list: includeGptList, min_quality_grade: $("minQuality").value || null }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "HTTP " + response.status);
    const link = document.createElement("a");
    link.href = data.download_url;
    link.download = data.zip_name;
    document.body.append(link);
    link.click();
    link.remove();
    toast(data.includes_material
      ? "已附带训练原料：已标注子集与音频 ZIP 已保存到 annotations 目录，可直接用于模型强化"
      : `ZIP 已交给浏览器下载：${data.verified_count} 段合格音频`, "ok");
  } catch (error) {
    toast("导出失败：" + error.message, "err");
  }
});

/* 批量采纳未标注的主输出。 */
$("trustMainBtn").addEventListener("click", async () => {
  if (!app.batchId) return;
  const pending = app.segments.filter((seg) => seg.accepted && !seg.all_sentence_export && !app.reviews[seg.ordinal]);
  if (!pending.length) { toast("没有待标注的 Tier A 主输出片段", "err"); return; }
  if (!window.confirm(`将把 ${pending.length} 段未标注的 Tier A 主输出全部标为“目标匹配”，继续？`)) return;
  app.history.push(JSON.stringify(app.reviews));
  pending.forEach((seg) => {
    app.reviews[seg.ordinal] = { ordinal: seg.ordinal, filename: seg.filename, label: "correct" };
  });
  renderList();
  refreshMeta();
  if (await save(false)) {
    toast(`已将 ${pending.length} 段输出标为目标匹配`, "ok");
  }
});

/* ── 快捷键 ─────────────────────────────────────────────── */
document.addEventListener("keydown", (event) => {
  if (!$("p-review").classList.contains("is-on") || $("speakerPanel").open || event.repeat || event.isComposing
      || event.target.closest("input, textarea, select, [contenteditable]:not([contenteditable='false'])")) return;
  const key = event.key.toLowerCase();
  if (event.ctrlKey || event.metaKey || event.altKey) {
    if (key === "z" && !event.altKey && (event.ctrlKey || event.metaKey)) {
      event.preventDefault(); $("undoBtn").click();
    }
    return;
  }
  const labels = {1:"correct", 2:"wrong", 3:"partial", 4:"singing", 5:"overlap", 6:"skip"};
  const actions = {" ":"playBtn", r:"replayBtn", arrowup:"prevBtn", w:"prevBtn", arrowdown:"nextBtn", s:"nextBtn", arrowleft:"prevBtn", arrowright:"nextBtn", b:"blindBtn"};
  if (labels[key]) { event.preventDefault(); decide(labels[key]); }
  else if (actions[key]) { event.preventDefault(); $(actions[key]).click(); }
});

$("trustBorderlineBtn").addEventListener("click", async () => {
  if (!app.batchId) return;
  const pending = app.segments.filter(
    (seg) => seg.high_confidence_borderline && !app.reviews[seg.ordinal]?.label,
  );
  if (!pending.length) {
    toast("没有待采纳的高分边缘句");
    return;
  }
  app.history.push(JSON.stringify(app.reviews));
  if (app.history.length > 60) app.history.shift();
  pending.forEach((seg) => {
    app.reviews[seg.ordinal] = {
      ordinal: seg.ordinal,
      filename: seg.filename,
      label: "correct",
    };
  });
  renderList();
  refreshMeta();
  reflectReview(currentSegment()?.ordinal);
  if (await save(false)) {
    const duration = pending.reduce((sum, seg) => sum + seg.duration, 0);
    toast(`已采纳 ${pending.length} 句，共 ${duration.toFixed(2)} 秒`, "ok");
  }
});

$("closeSpeakersBtn").addEventListener("click", () => $("speakerPanel").close());
$("discoverSpeakersBtn").addEventListener("click", async () => {
  if (!app.batchId) return;
  const batchId = app.batchId;
  $("speakerPanel").showModal();
  $("speakerStatus").textContent = "正在扫描角色簇…";
  $("speakerList").replaceChildren();
  try {
    const response = await fetch(`/api/batches/${encodeURIComponent(batchId)}/speakers`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "角色探索失败");
    if (app.batchId !== batchId) return;
    $("speakerStatus").textContent = `发现 ${data.speakers.length} 个角色簇 · 每簇导出最多 4 段参考 WAV`;
    data.speakers.forEach(speaker => {
      const card = document.createElement("article"); card.className = "speaker-card";
      const title = document.createElement("strong");
      title.textContent = `${speaker.episode} · 簇 #${speaker.cluster_id}`;
      const info = document.createElement("p");
      info.textContent = `${Number(speaker.total_speech_seconds).toFixed(1)}s · ${speaker.member_count} 段 · 纯度 ${Number(speaker.purity_score).toFixed(3)}`;
      const preview = document.createElement("p"); preview.textContent = (speaker.sample_texts || []).join(" / ");
      const spans = document.createElement("p");
      spans.textContent = "推荐种子：" + (speaker.recommended_seeds || []).map(seed => `${fmtTime(seed.start ?? seed.span?.[0])}–${fmtTime(seed.end ?? seed.span?.[1])}s`).join(" · ");
      const button = document.createElement("button"); button.className = "mini-btn";
      button.textContent = "导出 Top-4 并设为目标参考音频";
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          const result = await fetch(`/api/batches/${encodeURIComponent(batchId)}/export_seeds`, {
            method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({cluster_ids:[speaker.key]})});
          const exported = await result.json();
          if (!result.ok) throw new Error(exported.detail || "种子导出失败");
          if (!exported.paths.length) throw new Error("没有可导出的种子，原轨及对应切片可能已删除");
          $("refsInput").value = exported.paths.join("\n");
          $("speakerStatus").textContent = exported.paths.join("\n");
          toast(`已导出 ${exported.paths.length} 段，已填入提取页参考音频`, "ok");
        } catch (error) { toast(error.message, "err"); }
        finally { button.disabled = false; }
      });
      card.append(title, info, preview, spans, button); $("speakerList").append(card);
    });
  } catch (error) { $("speakerStatus").textContent = error.message; }
});

let rollbackBusy = false;
async function refreshRollbackAvailability() {
  try {
    const response = await fetch('/api/training/rollback');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '无法读取模型备份');
    $('rollbackModelBtn').disabled = rollbackBusy || !data.available;
    $('rollbackModelBtn').title = data.message || '';
  } catch (error) {
    $('rollbackModelBtn').disabled = true;
    $('rollbackModelBtn').title = error.message;
  }
}
$("rollbackModelBtn").addEventListener("click", async () => {
  rollbackBusy = true;
  $('rollbackModelBtn').disabled = true;
  try {
    const response = await fetch("/api/training/rollback", {method:"POST"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "回滚失败");
    toast("已回滚至上一次模型备份，下次提取生效", "ok");
  } catch (error) { toast(error.message, "err"); }
  finally { rollbackBusy = false; await refreshRollbackAvailability(); }
});
refreshRollbackAvailability();

$('liveRecalibrate').addEventListener('click',async () => {
  if (!app.batchId) return;
  if (!await save(false)) return;
  const batchId = app.batchId;
  try {
    const response = await fetch(`/api/batches/${encodeURIComponent(app.batchId)}/live_recalibrate`,{method:'POST'});
    const data = await response.json();
    if (app.batchId !== batchId) return;
    if (!response.ok) throw new Error(data.detail || '重排失败');
    if (data.status !== 'recalibrated') return toast(data.message,'err');
    const scores = new Map(data.scores.map(s => [s.ordinal,s]));
    app.segments.forEach(s => { s.live_information = scores.get(s.ordinal)?.information;
      s.acquisition_rank = scores.get(s.ordinal)?.acquisition_rank;
      s.live_probability = scores.get(s.ordinal)?.probability; });
    document.querySelector('[data-filter="active"]').click();
    toast(`已根据 ${data.labels} 条标注刷新待审顺序`,'ok');
  } catch(error) { toast(error.message,'err'); }
});

let trainingJob = null;
let trainingTimer = null;
async function pollTraining() {
  if (!trainingJob) return;
  const jobId = trainingJob;
  try {
    const response = await fetch(`/api/training/status/${jobId}`);
    const data = await response.json();
    if (trainingJob !== jobId) return;
    if (!response.ok) throw new Error(data.detail || "无法读取训练状态");
    $("trainingState").textContent = data.message;
    $("trainingStorage").textContent = `临时磁盘占用 ${(data.temporary_bytes / 1024 / 1024).toFixed(1)} MB · ${data.temporary_path}`;
    $("trainBtn").disabled = true;
    $("discardModelBtn").disabled = false;
    if (data.state === 'running') $('rollbackModelBtn').disabled = true;
    if (data.report) {
      const report = data.report;
      const state = {already_applied:"重复 ZIP，模型未改动", validation_unavailable:"独立留出集不足，未更新", rejected_holdout:"留出验证未通过，未更新", updated_in_place:"全局模型已更新"}[report.status] || report.status;
      $("trainingReport").textContent = `ZIP：新增 ${report.new_packages} 份，跳过重复 ${report.duplicate_packages} 份\n新 ZIP 来源：${report.source_count} 集／文件，${report.reference_sets} 组参考\n参与训练 ${report.training_clips} 段，留出验证 ${report.held_out_clips} 段\n状态：${state}\n现用模型：${report.model_path}\n\n` + JSON.stringify(report.metrics, null, 2);
    }
    if (data.state === "running") trainingTimer = setTimeout(pollTraining, 1500);
    else await refreshRollbackAvailability();
  } catch (error) { if (trainingJob === jobId) $("trainingState").textContent = error.message; }
}

$("trainBtn").addEventListener("click", async () => {
  const files = $("trainingFiles").files;
  if (!files.length) { toast("先选择训练 ZIP", "err"); return; }
  if (files.length > 30 || [...files].some(file => !file.name.toLowerCase().endsWith(".zip"))) {
    toast("请选择 1 至 30 个含标注和音频的 ZIP", "err"); return;
  }
  if ([...files].some(file => file.size > 4 * 1024 ** 3) || [...files].reduce((sum, file) => sum + file.size, 0) > 8 * 1024 ** 3) {
    toast("单个原料最多 4 GB，本次合计最多 8 GB", "err"); return;
  }
  const form = new FormData();
  for (const file of files) form.append("files", file);
  $("trainBtn").disabled = true;
  $("trainingState").textContent = "正在上传原料文件到本机临时磁盘…";
  try {
    const response = await fetch("/api/training/start", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "启动失败");
    trainingJob = data.job_id;
    $("discardModelBtn").disabled = false;
    clearTimeout(trainingTimer);
    $("trainingReport").textContent = "正在训练…";
    pollTraining();
  } catch (error) { $("trainBtn").disabled = false; $("trainingState").textContent = error.message; }
});

$("discardModelBtn").addEventListener("click", async () => {
  if (!trainingJob) return;
  try {
    const response = await fetch(`/api/training/${trainingJob}`, { method: "DELETE" });
    if (!response.ok) throw new Error((await response.json()).detail);
    clearTimeout(trainingTimer);
    trainingJob = null;
    $("trainBtn").disabled = false;
    $("discardModelBtn").disabled = true;
    $("trainingState").textContent = "本次训练临时文件已清理";
    $("trainingStorage").textContent = "已强化的现用模型和已导出的原料会保留。";
    $("trainingReport").textContent = "等待原料文件";
    await refreshRollbackAvailability();
  } catch (error) { toast(error.message, "err"); }
});

async function resumeTraining() {
  try {
    const data = await (await fetch("/api/training/current")).json();
    trainingJob = data.job_id;
    if (trainingJob) pollTraining();
  } catch (error) { /* Service might be stopping. */ }
}

/* ── 提取任务 ───────────────────────────────────────────── */

$("threshold").addEventListener("input", () => {
  $("thresholdVal").textContent = Number($("threshold").value).toFixed(2);
});

function linesOf(id) {
  return $(id).value.split("\n").map((line) => line.trim()).filter(Boolean);
}

function updateSubtitleCount() {
  const count = [...app.subtitleBindings.values()].filter(value => value.trim()).length;
  $("subtitleBindingCount").textContent = count ? `已绑定 ${count}/${app.subtitleBindings.size}` : "未绑定";
}

function renderSubtitleBindings() {
  const targets = [...new Set(linesOf("targetsInput"))];
  app.subtitleBindings = new Map(targets.map(target => [target, app.subtitleBindings.get(target) || ""]));
  const container = $("subtitleBindings");
  container.replaceChildren();
  if (!targets.length) {
    const hint = document.createElement("p");
    hint.className = "form-help";
    hint.textContent = "先填写待提取文件，再绑定对应字幕。";
    container.append(hint);
  }
  targets.forEach((target, index) => {
    const row = document.createElement("div");
    row.className = "form-row";
    const label = document.createElement("label");
    label.className = "form-label subtitle-target";
    label.textContent = `${index + 1} · ${target.split(/[\\/]/).pop() || target}`;
    label.title = target;
    label.htmlFor = `subtitlePath${index}`;
    const controls = document.createElement("div");
    controls.className = "subtitle-path-row";
    const input = document.createElement("input");
    input.type = "text";
    input.id = label.htmlFor;
    input.className = "field subtitle-path";
    input.dataset.target = target;
    input.placeholder = "对应字幕的完整路径";
    input.value = app.subtitleBindings.get(target);
    input.addEventListener("input", () => {
      app.subtitleBindings.set(target, input.value);
      updateSubtitleCount();
    });
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "mini-btn";
    clear.textContent = "清除";
    clear.setAttribute("aria-label", `清除 ${target} 的字幕绑定`);
    clear.addEventListener("click", () => {
      input.value = "";
      app.subtitleBindings.set(target, "");
      updateSubtitleCount();
      input.focus();
    });
    controls.append(input, clear);
    row.append(label, controls);
    container.append(row);
  });
  updateSubtitleCount();
}

$("targetsInput").addEventListener("input", renderSubtitleBindings);
renderSubtitleBindings();

$("runBtn").addEventListener("click", async () => {
  const targets = linesOf("targetsInput");
  const references = linesOf("refsInput");
  if (!targets.length) { toast("请填写待提取的路径", "err"); return; }
  if (!references.length) { toast("请填写参考音频", "err"); return; }
  const subtitles = Object.fromEntries(targets.map(target =>
    [target, (app.subtitleBindings.get(target) || "").trim()]).filter(([, value]) => value));
  const invalidSubtitle = Object.values(subtitles).find(path => !/\.(ass|ssa|srt|vtt)$/i.test(path));
  if (invalidSubtitle) {
    $("subtitleBindingsPanel").open = true;
    toast("字幕仅支持 ASS、SSA、SRT、VTT，请检查字幕路径", "err");
    return;
  }

  const negativeGroups = linesOf("excludeInput").map((line) =>
    line.split(/[,，]/).map((part) => part.trim()).filter(Boolean)
  );

  $("term").innerHTML = '<p class="term-line dim">正在启动…</p>';
  $("extractState").textContent = "运行中";
  $("progressPct").textContent = "0%";
  $("progressBar").style.width = "0%";
  app.extractCursor = 0;

  try {
    const response = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        targets,
        references,
        negative_groups: negativeGroups,
        subtitles,
        options: {
          speaker_threshold: parseFloat($("threshold").value),
          silence_min_seconds: parseFloat($("silenceMin").value),
          silence_max_seconds: parseFloat($("silenceMax").value),
          use_overlap_detector: $("useOverlap").checked,
          use_singing_detector: $("useSinging").checked,
          export_video_clips: $("useVideo").checked,
          branch_validation: $("useBranchValidation").checked,
          export_all_sentences: $("exportAllSentences").checked,
          cluster_bootstrap: $("useBootstrap").checked,
          cluster_identity_mode: "enforce",
          export_sample_rate: Number($('exportSampleRate').value),
        },
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "HTTP " + response.status);
    pollExtract();
  } catch (error) {
    $("extractState").textContent = "启动失败";
    $("term").innerHTML = `<p class="term-line err">${escapeHtml(error.message)}</p>`;
    toast("启动失败：" + error.message, "err");
  }
});

$("stopBtn").addEventListener("click", async () => {
  await fetch("/api/extract/stop", { method: "POST" });
  toast("已请求停止");
});

function pollExtract() {
  if (app.extractTimer) clearInterval(app.extractTimer);
  app.extractTimer = setInterval(async () => {
    try {
      const response = await fetch(`/api/extract/status?cursor=${app.extractCursor}`);
      const data = await response.json();
      app.extractCursor = data.cursor;

      data.lines.forEach((event) => appendTerm(event));
      if (data.state !== "running") {
        clearInterval(app.extractTimer);
        app.extractTimer = null;
        $("extractState").textContent =
          data.state === "done" ? "完成" : data.state === "failed" ? "失败" : "已停止";
        if (data.state === "done" && data.batch_id) {
          toast("提取完成：" + data.batch_id, "ok");
          loadBatches();
        }
      }
    } catch (error) {
      /* 服务未就绪时静默重试 */
    }
  }, 900);
}

function appendTerm(event) {
  const term = $("term");
  const line = document.createElement("p");
  line.className = "term-line";

  if (event.type === "progress") {
    const percent = Math.round((event.value || 0) * 100);
    $("progressPct").textContent = percent + "%";
    $("progressBar").style.width = percent + "%";
    line.textContent = `[${percent}%] ${event.message || ""}`;
  } else if (event.type === "error") {
    line.classList.add("err");
    line.textContent = "错误：" + (event.message || "未知错误");
  } else if (event.type === "complete") {
    line.classList.add("ok");
    const remaining = Number(event.remaining_count) || 0;
    const discarded = Math.max(0, event.rejected_count - remaining);
    line.textContent =
      `完成：保留 ${event.accepted_count} 段` +
      (remaining ? `，另存 ${remaining} 段其余单句待复核` : "") +
      `，舍弃 ${discarded} 段 → ${event.batch_id}`;
    $("progressPct").textContent = "100%";
    $("progressBar").style.width = "100%";
  } else {
    line.classList.add("dim");
    line.textContent = event.message || JSON.stringify(event);
  }

  term.appendChild(line);
  term.scrollTop = term.scrollHeight;
}

/* ── 设置 ───────────────────────────────────────────────── */

async function loadSettings(restoreExtraction = false) {
  try {
    const response = await fetch("/api/settings");
    const settings = await response.json();
    $("portInput").value = settings.webui_port || 7860;
    $("blindDefault").checked = Boolean(settings.blind_mode);
    $("autoNextDefault").checked = settings.auto_next_on_rate !== false;
    // Opening Settings must retain the current extraction choices for saving.
    if (restoreExtraction) {
      $("useBranchValidation").checked = settings.branch_validation === true;
      $("exportAllSentences").checked = settings.export_all_sentences === true;
      if (settings.speaker_threshold != null) {
        $("threshold").value = settings.speaker_threshold;
        $("thresholdVal").textContent = Number(settings.speaker_threshold).toFixed(2);
      }
    }
  } catch (error) {
    toast("读取设置失败：" + error.message, "err");
  }
}

$("saveSettingsBtn").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        webui_port: parseInt($("portInput").value, 10) || 7860,
        blind_mode: $("blindDefault").checked,
        auto_next_on_rate: $("autoNextDefault").checked,
        speaker_threshold: parseFloat($("threshold").value),
        branch_validation: $("useBranchValidation").checked,
        export_all_sentences: $("exportAllSentences").checked,
      }),
    });
    if (!response.ok) throw new Error("HTTP " + response.status);
    $("settingsState").textContent = "已保存";
    toast("设置已保存", "ok");
  } catch (error) {
    $("settingsState").textContent = "保存失败";
    toast("保存设置失败：" + error.message, "err");
  }
});

/* ── 两个界面互切 ───────────────────────────────────────── */

async function switchToDesktop() {
  $("switchDesktopBtn").disabled = true;
  $("goDesktopBtn").disabled = true;
  toast("正在启动桌面窗口…");
  try {
    const response = await fetch("/api/switch_to_desktop", { method: "POST" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "HTTP " + response.status);
    document.body.innerHTML =
      '<div style="display:flex;height:100vh;align-items:center;justify-content:center;' +
      'background:#0c0d10;color:#e8eaf0;font-family:sans-serif;text-align:center">' +
      '<div><p style="font-size:16px;margin-bottom:6px">桌面版已打开</p>' +
      '<p style="color:#6b7183;font-size:13px">可以关闭这个标签页，桌面窗口继续使用本机服务。</p></div></div>';
    setTimeout(() => window.close(), 700);
  } catch (error) {
    $("switchDesktopBtn").disabled = false;
    $("goDesktopBtn").disabled = false;
    toast("切换失败：" + error.message, "err");
  }
}

$("switchDesktopBtn").addEventListener("click", switchToDesktop);
$("goDesktopBtn").addEventListener("click", switchToDesktop);

const desktopMode = new URLSearchParams(location.search).get("desktop") === "1";
if (desktopMode) {
  $("switchDesktopBtn").hidden = true;
  $("goDesktopBtn").disabled = true;
  $("goDesktopBtn").textContent = "桌面版（当前）";
  $("goWebBtn").disabled = false;
  $("goWebBtn").textContent = "打开网页版";
}
$("goWebBtn").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/switch_to_webui", { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).detail);
    toast("网页版已打开，可关闭此桌面窗口", "ok");
  } catch (error) { toast(error.message, "err"); }
});
$("closeAppBtn").addEventListener("click", async () => {
  $("closeAppBtn").disabled = true;
  $("closeAppBtn").textContent = "正在清理…";
  try {
    const response = await fetch("/api/close", { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).detail);
    document.body.textContent = "临时文件已清理，程序即将关闭。已导出的文件会保留。";
    setTimeout(() => window.close(), 1000);
  } catch (error) { $("closeAppBtn").disabled = false; $("closeAppBtn").textContent = "关闭并清理"; toast(error.message, "err"); }
});

/* ── 启动 ───────────────────────────────────────────────── */

async function loadVersion() {
  try {
    const response = await fetch("/api/settings");
    const settings = await response.json();
    if (settings.version) $("versionTag").textContent = "v" + settings.version;
  } catch (error) {
    /* 版本号只是显示，读不到就保留 HTML 里的默认值 */
  }
}

applyBlind();
loadBatches();
loadSettings(true);
loadVersion();
resumeTraining();
