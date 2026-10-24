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
  turns: [],             // stage_audit 里的说话轮次时间段
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
  applyBlind();
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
  refreshMeta();
  try {
    const response = await fetch(`/api/batches/${encodeURIComponent(batchId)}`);
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    if (app.batchId !== batchId) return;

    app.segments = (data.segments || []).filter(
      (seg) => seg.accepted || seg.cluster_removed || seg.cluster_tier === "B");
    app.reviews = {};
    (data.segments || []).forEach((seg) => {
      if (seg.review && seg.review.label) app.reviews[seg.ordinal] = seg.review;
    });
    app.history = [];
    app.index = -1;

    $("clusterSummary")?.remove();
    refreshMeta();

    renderList();
    const firstAccepted = app.segments.findIndex((seg) => seg.accepted);
    if (app.segments.length) select(Math.max(0, firstAccepted), false);
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

/**
 * 把输出片段对到簇上。
 *
 * 报告里的成员是**说话轮次**序号（一集 254 个），而界面试听的是**最终输出
 * 片段**（一集 28 段）。一个输出段可能由多个轮次拼成，也可能只覆盖某个轮次的
 * 一部分，所以只能按时间段重叠匹配，并且取重叠最多的那个轮次所属的簇。
 */
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
}

/* ── 片段列表 ───────────────────────────────────────────── */

$("filters").addEventListener("click", (event) => {
  const chip = event.target.closest("[data-filter]");
  if (!chip) return;
  const wasClusterChip = chip.id === "clusterFilterChip";
  // 再点一次同一类筛选，或点别的筛选条件，都退出"按簇看"的状态
  app.activeCluster = null;
  app.filter = wasClusterChip ? "all" : chip.dataset.filter;
  $("clusterFilterChip").hidden = true;

  document.querySelectorAll("[data-filter]").forEach((c) => c.classList.remove("is-on"));
  (wasClusterChip ? document.querySelector('[data-filter="all"]') : chip)
    .classList.add("is-on");
  renderList();
});

function visibleSegments() {
  return app.segments.filter((seg) => {
    // Tier 视图：主输出 / 边缘捡漏 / 已撤下
    if (app.filter === "tierA") return seg.accepted;
    if (app.filter === "tierB") return !seg.accepted && seg.cluster_tier === "B";
    if (app.filter === "removed") return !seg.accepted && seg.cluster_tier !== "B";
    if (!seg.accepted && seg.cluster_tier !== "B" && app.filter !== "cluster") return false;
    const done = Boolean(app.reviews[seg.ordinal]);
    if (app.filter === "todo") return !done;
    if (app.filter === "done") return done;
    if (app.filter === "cluster" && app.activeCluster != null) {
      return app.segCluster[seg.ordinal] === app.activeCluster;
    }
    return true;
  });
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
        `<span class="seg-text">${escapeHtml(seg.stt_text) || seg.filename}</span>` +
        `<span class="seg-sub">${fmtTime(seg.start)}–${fmtTime(seg.end)}s · ${seg.duration.toFixed(2)}s${seg.accepted ? "" : " · 已撤下"}</span>` +
      `</span>` + clusterTag + tag;

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
  $("sttLine").textContent = seg.stt_text || "（无识别文本）";
  $("scopeEmpty").hidden = true;

  const readout = [
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
  audio.removeAttribute("src");
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
  try {
    const response = await fetch(`/api/audio/${encodeURIComponent(app.batchId)}/${seg.ordinal}`);
    const buffer = await response.arrayBuffer();
    const context = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await context.decodeAudioData(buffer);
    context.close();
    if (token !== app.decodeToken) return;
    app.peaks = computePeaks(decoded.getChannelData(0), 900);
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

  if (!peaks.length) return;

  const barWidth = width / peaks.length;
  // 频谱渐变：左青右紫，和界面主色一致
  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#4de3ff");
  gradient.addColorStop(0.52, "#6d8dff");
  gradient.addColorStop(1, "#a06bff");

  // 上下各一层光晕，让波形看起来在发光
  ctx.shadowColor = "rgba(77, 227, 255, 0.55)";
  ctx.shadowBlur = 9;

  // roundRect 是较新的 Canvas API，旧内核上没有，退化用矩形
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
  const rev = segment ? app.reviews[segment.ordinal] : null;
  if (rev && rev.label === "partial") {
    const from = clamp((Number(rev.partial_start) || 0) / segment.duration, 0, 1);
    const to = clamp((Number(rev.partial_end) || segment.duration) / segment.duration, 0, 1);
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
  }
}

window.addEventListener("resize", () => drawWave(app.peaks || []));

$("waveCanvas").addEventListener("click", (event) => {
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

$("replayBtn").addEventListener("click", () => { audio.currentTime = 0; play(); });
$("prevBtn").addEventListener("click", () => step(-1));
$("nextBtn").addEventListener("click", () => step(1));

function step(delta) {
  const visible = new Set(visibleSegments().map((seg) => seg.ordinal));
  for (let position = app.index + delta; position >= 0 && position < app.segments.length; position += delta) {
    if (visible.has(app.segments[position].ordinal)) {
      select(position, true);
      return;
    }
  }
}

/* ── 裁决 ───────────────────────────────────────────────── */

document.querySelectorAll(".pad").forEach((pad) => {
  pad.addEventListener("click", () => decide(pad.dataset.label));
});

function decide(label) {
  const seg = currentSegment();
  if (!seg) return;

  if (label === "partial") {
    const duration = seg.duration;
    const from = parseFloat($("trimStart").value);
    const to = parseFloat($("trimEnd").value);
    if (!Number.isFinite(from) || !Number.isFinite(to) || from < 0 || to <= from || to > duration) {
      $("trimBox").hidden = false;
      toast("请先填好他人声音的起止秒数");
      return;
    }
  }

  app.history.push(JSON.stringify(app.reviews));
  if (app.history.length > 60) app.history.shift();

  app.reviews[seg.ordinal] = {
    ordinal: seg.ordinal,
    filename: seg.filename,
    label: label,
    partial_start: label === "partial" ? parseFloat($("trimStart").value) : null,
    partial_end: label === "partial" ? parseFloat($("trimEnd").value) : null,
    partial_text: label === "partial" ? (($("partialText") || {}).value || "").trim() || null : null,
  };

  reflectReview(seg.ordinal);
  renderList();
  refreshMeta();
  save(false);

  if ($("autoNext").checked && app.index < app.segments.length - 1) {
    setTimeout(() => step(1), 260);
  }
}

function reflectReview(ordinal) {
  const review = app.reviews[ordinal] || {};
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
    $("trimStart").value = (review.partial_start ?? 0).toFixed(2);
    $("trimEnd").value = (review.partial_end ?? seg.duration).toFixed(2);
  } else {
    $("trimBox").hidden = true;
    if (seg) {
      $("trimStart").value = "0.00";
      $("trimEnd").value = seg.duration.toFixed(2);
    }
  }

  drawWave(app.peaks || []);
}

$("trimFromPlayhead").addEventListener("click", () => {
  const seg = currentSegment();
  if (!seg) return;
  $("trimStart").value = clamp(audio.currentTime, 0, seg.duration).toFixed(2);
});

/* 试听保留段：播错误区间之外最长的一段（范围播放，到点自动停）。 */
let keepPlayStop = null;
$("keepPlayBtn").addEventListener("click", () => {
  const seg = currentSegment();
  if (!seg) return;
  const from = parseFloat($("trimStart").value);
  const to = parseFloat($("trimEnd").value);
  if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) {
    toast("请先填好他人声音的起止秒数", "err");
    return;
  }
  const pieces = [[0, from], [to, seg.duration]].filter(([a, b]) => b - a > 0.2);
  if (!pieces.length) { toast("错人区间覆盖了整句，没有可保留的干净段", "err"); return; }
  const [keepStart, keepEnd] = pieces.sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]))[0];
  if (keepPlayStop) { audio.removeEventListener("timeupdate", keepPlayStop); keepPlayStop = null; }
  audio.currentTime = keepStart;
  audio.play();
  toast(`试听保留段 ${keepStart.toFixed(2)}s – ${keepEnd.toFixed(2)}s`, "ok");
  keepPlayStop = () => {
    if (audio.currentTime >= keepEnd - 0.02) {
      audio.pause();
      audio.removeEventListener("timeupdate", keepPlayStop);
      keepPlayStop = null;
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
  const removedCount = app.segments.length - acceptedCount;
  $("batchMeta").innerHTML =
    `<span><b>${acceptedCount}</b> 输出</span>` +
    (removedCount ? `<span>· ${removedCount} 撤下可试听</span>` : "") +
    `<span>·</span>` +
    `<span><b>${Object.keys(app.reviews).length}</b> 已标</span>`;
  if (clusterSummary) $("batchMeta").appendChild(clusterSummary);
  const complete = app.segments.filter(seg => validMaterialReview(seg)).length;
  $("includeMaterial").disabled = !app.segments.length || complete !== app.segments.length;
  if ($("includeMaterial").disabled) $("includeMaterial").checked = false;
  $("materialState").textContent = app.segments.length && complete === app.segments.length
    ? "已全部打标；勾选后保存到 annotations/批次名.zip，可直接用于模型强化"
    : `原料待完成：${complete}/${app.segments.length} 段已打标`;
}

function validMaterialReview(seg) {
  const review = app.reviews[seg.ordinal];
  if (!review || !Object.hasOwn(LABELS, review.label) || review.ordinal !== seg.ordinal || review.filename !== seg.filename) return false;
  return review.label !== "partial" || (Number.isFinite(review.partial_start) && Number.isFinite(review.partial_end)
    && review.partial_start >= 0 && review.partial_end > review.partial_start && review.partial_end <= seg.duration + .02);
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
      body: JSON.stringify({ include_material: includeMaterial, gpt_sovits_list: includeGptList }),
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
      ? "已附带训练原料：标注与全部音频 ZIP 已保存到 annotations 目录，可直接用于模型强化"
      : `ZIP 已交给浏览器下载：${data.verified_count} 段合格音频`, "ok");
  } catch (error) {
    toast("导出失败：" + error.message, "err");
  }
});

/* 一键信任主输出：未标注的 Tier A 全部标为 correct，再切到 Tier B 捡漏。 */
$("trustMainBtn").addEventListener("click", async () => {
  if (!app.batchId) return;
  const pending = app.segments.filter((seg) => seg.accepted && !app.reviews[seg.ordinal]);
  if (!pending.length) { toast("没有待标注的 Tier A 主输出片段", "err"); return; }
  if (!window.confirm(`将把 ${pending.length} 段未标注的 Tier A 主输出全部标为“目标匹配”，继续？`)) return;
  app.history.push(JSON.stringify(app.reviews));
  pending.forEach((seg) => {
    app.reviews[seg.ordinal] = { ordinal: seg.ordinal, filename: seg.filename, label: "correct" };
  });
  renderList();
  refreshMeta();
  if (await save(false)) {
    toast(`已信任 ${pending.length} 段主输出；请切到「⚡ 边缘捡漏」继续捡漏`, "ok");
  }
});

/* ── 快捷键 ─────────────────────────────────────────────── */
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
    if (data.report) {
      const report = data.report;
      const state = report.status === "already_applied" ? "全部 ZIP 已处理，模型未改动" : "已强化并自动保存";
      $("trainingReport").textContent = `ZIP：新增 ${report.new_packages} 份，跳过重复 ${report.duplicate_packages} 份\n新 ZIP 来源：${report.source_count} 集／文件，${report.reference_sets} 组参考\n参与训练 ${report.training_clips} 段，留出验证 ${report.held_out_clips} 段\n状态：${state}\n现用模型：${report.model_path}\n\n` + JSON.stringify(report.metrics, null, 2);
    }
    if (data.state === "running") trainingTimer = setTimeout(pollTraining, 1500);
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
  } catch (error) { toast(error.message, "err"); }
});

async function resumeTraining() {
  try {
    const data = await (await fetch("/api/training/current")).json();
    trainingJob = data.job_id;
    if (trainingJob) pollTraining();
  } catch (error) { /* Service might be stopping. */ }
}

document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
  if (typing) return;
  if (!$("p-review").classList.contains("is-on")) return;

  const keys = { "1": "correct", "2": "wrong", "3": "partial", "4": "singing", "5": "overlap", "6": "skip" };
  if (keys[event.key]) { event.preventDefault(); decide(keys[event.key]); return; }

  if (event.key === " ") { event.preventDefault(); $("playBtn").click(); return; }
  if (event.key === "ArrowLeft") { event.preventDefault(); step(-1); return; }
  if (event.key === "ArrowRight") { event.preventDefault(); step(1); return; }
  if (event.key.toLowerCase() === "r") { event.preventDefault(); $("replayBtn").click(); return; }
  if (event.key.toLowerCase() === "b") { event.preventDefault(); $("blindBtn").click(); return; }
  if (event.key.toLowerCase() === "z" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    $("undoBtn").click();
  }
});

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
          cluster_bootstrap: $("useBootstrap").checked,
          cluster_identity_mode: $("useBootstrap").checked ? "enforce" : "audit",
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
    line.textContent =
      `完成：保留 ${event.accepted_count} 段，舍弃 ${event.rejected_count} 段 → ${event.batch_id}`;
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

async function loadSettings() {
  try {
    const response = await fetch("/api/settings");
    const settings = await response.json();
    $("portInput").value = settings.webui_port || 7860;
    $("blindDefault").checked = Boolean(settings.blind_mode);
    $("autoNextDefault").checked = settings.auto_next_on_rate !== false;
    if (settings.speaker_threshold != null) {
      $("threshold").value = settings.speaker_threshold;
      $("thresholdVal").textContent = Number(settings.speaker_threshold).toFixed(2);
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
loadSettings();
loadVersion();
resumeTraining();
