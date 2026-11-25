"""Client-side renderer for the dashboard.

Kept separate from the HTML template so the markup stays readable and the
rendering logic can be reviewed on its own.
"""

JS = r"""
const ICONS = {
  logo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v1H7a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-1V6a4 4 0 0 0-4-4Z"/><path d="M9 13h6"/><path d="M12 11v4"/></svg>',
  video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="m16 11 6-3v8l-6-3z"/></svg>',
  chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  grid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
  seats: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="5" rx="1.5"/><rect x="4" y="12" width="5" height="6" rx="1.5"/><rect x="10" y="12" width="5" height="6" rx="1.5"/><rect x="16" y="12" width="5" height="6" rx="1.5"/></svg>',
  bell: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-3 8-3 8h18s-3-1-3-8"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>',
  trend: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 17 6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>',
  chart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/></svg>',
  matrix: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 12h18"/><path d="M12 3v18"/></svg>',
  table: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/></svg>',
  shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/></svg>',
  target: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/></svg>',
  close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
  info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/></svg>',
};

const BEHAVIOR_COLORS = {
  'Reading / Writing': '#34d399',
  'Oriented / Listening': '#38bdf8',
  'Hand Raising': '#a78bfa',
  'Peer Interaction': '#fbbf24',
  'Using Phone': '#fb7185',
  'Sleeping / Fatigue': '#f472b6',
};

const ALERT_TONE = {
  'Using Phone': { tone: 'var(--bad)', soft: 'var(--bad-soft)' },
  'Peer Interaction': { tone: 'var(--warn)', soft: 'var(--warn-soft)' },
  'Sleeping / Fatigue': { tone: 'var(--violet)', soft: 'var(--violet-soft)' },
  'Reading / Writing': { tone: 'var(--brand-2)', soft: 'rgba(34,211,238,.13)' },
  'Oriented / Listening': { tone: 'var(--ok)', soft: 'var(--ok-soft)' },
  'Hand Raising': { tone: 'var(--brand)', soft: 'var(--brand-soft)' },
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};

let currentKey = 'test_6';
let currentFilter = 'all';
let activeSeat = null;

/* ------------------------------------------------------------------ *
 * Tooltip
 * ------------------------------------------------------------------ */
const tip = el('div', 'tip');
document.addEventListener('DOMContentLoaded', () => document.body.appendChild(tip));

function showTip(evt, html) {
  tip.innerHTML = html;
  tip.classList.add('is-on');
  moveTip(evt);
}
function moveTip(evt) {
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = Math.max(8, x) + 'px';
  tip.style.top = Math.max(8, y) + 'px';
}
function hideTip() { tip.classList.remove('is-on'); }

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */
function boot() {
  const sel = $('videoSelect');
  Object.entries(DATABASE).forEach(([key, v]) => {
    const o = el('option');
    o.value = key;
    o.textContent = `${v.title} — ${v.subtitle}`;
    sel.appendChild(o);
  });
  sel.value = currentKey;
  sel.addEventListener('change', (e) => switchVideo(e.target.value));
  $('searchInput').addEventListener('input', searchTable);
  $('studentModal').addEventListener('click', (e) => {
    if (e.target.id === 'studentModal') closeModal();
  });
  $('modalClose').addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
  });
  window.addEventListener('resize', () => {
    renderSeating(DATABASE[currentKey]);
    renderTimeline(DATABASE[currentKey]);
  });

  // Inspect buttons are delegated so the table can re-render freely.
  $('tableBody').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-inspect]');
    if (!btn) return;
    const sid = Number(btn.dataset.inspect);
    const student = (DATABASE[currentKey].students || []).find(s => s.id === sid);
    if (student) openStudent(student);
  });

  renderCurrentVideo();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

function switchVideo(key) {
  currentKey = key;
  activeSeat = null;
  currentFilter = 'all';
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('is-on'));
  document.querySelector('.chip[data-filter="all"]').classList.add('is-on');
  $('searchInput').value = '';
  renderCurrentVideo();
}

function renderCurrentVideo() {
  const data = DATABASE[currentKey];
  if (!data) return;

  renderHeader(data);
  renderKpis(data);
  renderSeating(data);
  renderAlerts(data);
  renderTimeline(data);
  renderBehaviors(data);
  renderConfusion(data);
  renderTable(data);
  renderFooter(data);
}

/* ------------------------------------------------------------------ *
 * Header / provenance
 * ------------------------------------------------------------------ */
function renderHeader(data) {
  const ev = data.evaluation || {};
  const human = ev.kind === 'human';
  const badge = $('gtBadge');
  badge.className = 'badge ' + (human ? 'badge-human' : 'badge-heuristic');
  badge.innerHTML = (human ? ICONS.shield : ICONS.info) +
    (human ? '<span>Human ground truth</span>' : '<span>Heuristic agreement</span>');

  $('videoTitle').textContent = data.title;
  $('videoMeta').textContent =
    `${data.resolution} · ${data.fps} fps · ${data.duration_s}s analysed`;
  $('provenanceNote').textContent = ev.source || 'Not evaluated';
  $('provenanceDetail').textContent = ev.source_detail || '';
  $('provenanceNote').title = ev.provenance || '';
}

/* ------------------------------------------------------------------ *
 * KPIs
 * ------------------------------------------------------------------ */
function renderKpis(data) {
  const m = data.metrics;
  const ev = data.evaluation || {};
  const mean = m.class_mean_score;
  const scoreTone = mean >= 65 ? 'kpi-accent-ok' : mean >= 50 ? 'kpi-accent-warn' : 'kpi-accent-bad';

  $('kpiScore').textContent = mean.toFixed(1);
  $('kpiScoreCard').className = 'kpi ' + scoreTone;
  $('kpiScoreNote').textContent = mean >= 50 ? 'Above decision threshold' : 'Below decision threshold';

  const acc = ev.accuracy;
  const accCard = $('kpiAccuracyCard');
  accCard.className = 'kpi ' + (acc === null ? 'kpi-accent-brand'
    : acc >= 90 ? 'kpi-accent-ok' : acc >= 75 ? 'kpi-accent-warn' : 'kpi-accent-bad');
  $('kpiAccuracy').textContent = acc === null || acc === undefined ? 'n/a' : acc.toFixed(1) + '%';
  $('kpiAccuracyLabel').textContent = ev.kind === 'human' ? 'Ground-truth accuracy' : 'Rubric agreement';
  $('kpiMacroF1').textContent = ev.macro_f1 === null || ev.macro_f1 === undefined
    ? 'n/a' : ev.macro_f1.toFixed(1) + '%';
  $('kpiBalanced').textContent = ev.balanced_accuracy === null || ev.balanced_accuracy === undefined
    ? 'n/a' : ev.balanced_accuracy.toFixed(1) + '%';
  $('kpiSampleSize').textContent = ev.n_samples ? ev.n_samples.toLocaleString() : 'n/a';

  $('kpiStudents').textContent = m.total_students;
  $('kpiAttentive').textContent = `${m.attentive_students} of ${m.total_students}`;
  $('kpiRate').textContent = m.attentive_rate_pct.toFixed(1) + '%';

  $('kpiDrops').textContent = (data.drop_events || []).length;
  $('kpiDropsCard').className = 'kpi ' +
    ((data.drop_events || []).length ? 'kpi-accent-warn' : 'kpi-accent-ok');
  $('kpiDropNote').textContent = (data.drop_events || []).length
    ? 'Flagged for teacher review'
    : 'No sustained drops detected';
}

/* ------------------------------------------------------------------ *
 * Seating map — aspect-correct projection
 * ------------------------------------------------------------------ */
function renderSeating(data) {
  const box = $('seating');
  const stage = $('seatingStage');
  const students = data.students || [];
  if (!students.length) {
    stage.innerHTML = '<div class="empty" style="position:absolute;inset:auto 12px 12px">No students detected</div>';
    return;
  }

  // These are perspective views, not top-down plans: each student is drawn
  // as their actual camera-space bounding box, uniformly scaled from the
  // source resolution. That keeps every desk inside the frame and lets the
  // natural depth occlusion (near students overlapping far students) read
  // correctly.
  const [vw, vh] = data.resolution.split('x').map(Number);
  const boxW = box.clientWidth || 900;
  const boxH = box.clientHeight || 506;
  const scale = Math.min(boxW / vw, boxH / vh);
  const stageW = vw * scale;
  const stageH = vh * scale;

  stage.style.width = stageW + 'px';
  stage.style.height = stageH + 'px';
  stage.style.left = Math.max(0, (boxW - stageW) / 2) + 'px';
  stage.style.top = Math.max(0, (boxH - stageH) / 2) + 'px';

  stage.innerHTML = '<div class="podium">Front of Classroom</div>';

  // Depth order: paint the furthest students first so nearer ones occlude
  // them, matching the camera's own occlusion.
  const ordered = students.slice().sort((a, b) => b.y - a.y);

  ordered.forEach(s => {
    const high = s.mean_score >= 50;
    const node = el('div', 'seat');
    const w = Math.max(20, s.w * scale);
    const h = Math.max(16, s.h * scale);
    node.style.left = s.x * scale + 'px';
    node.style.top = s.y * scale + 'px';
    node.style.width = w + 'px';
    node.style.height = h + 'px';
    node.style.setProperty('--seat', high ? 'var(--ok)' : 'var(--bad)');
    node.dataset.id = s.id;
    node.innerHTML =
      `<span class="seat-id">S${String(s.id).padStart(2, '0')}</span>` +
      `<span class="seat-score">${s.mean_score.toFixed(0)}</span>`;

    node.addEventListener('click', () => {
      activeSeat = activeSeat === s.id ? null : s.id;
      document.querySelectorAll('.seat').forEach(n =>
        n.classList.toggle('is-active', Number(n.dataset.id) === activeSeat));
      if (activeSeat === s.id) openStudent(s);
    });
    node.addEventListener('mouseenter', (e) => showTip(e, seatTip(s)));
    node.addEventListener('mousemove', moveTip);
    node.addEventListener('mouseleave', hideTip);
    stage.appendChild(node);
  });

  $('seatingCaption').textContent =
    `${students.length} students · ${data.resolution} camera space`;
}

function seatTip(s) {
  const high = s.mean_score >= 50;
  return `<b>Student ${String(s.id).padStart(2, '0')}</b>` +
    `<div class="row"><span>Mean score</span><span>${s.mean_score.toFixed(1)}</span></div>` +
    `<div class="row"><span>Attentive</span><span>${s.attentive_pct.toFixed(1)}%</span></div>` +
    `<div class="row"><span>Behaviour</span><span>${s.top_cue}</span></div>` +
    `<div class="row"><span>Status</span><span style="color:${high ? 'var(--ok)' : 'var(--bad)'}">` +
    `${high ? 'Attentive' : 'Distracted'}</span></div>`;
}

/* ------------------------------------------------------------------ *
 * Alerts
 * ------------------------------------------------------------------ */
function renderAlerts(data) {
  const list = $('alertList');
  list.innerHTML = '';
  const events = data.drop_events || [];
  $('alertCount').textContent = events.length + (events.length === 1 ? ' event' : ' events');

  if (!events.length) {
    list.appendChild(el('div', 'empty',
      '<strong>No attention drops flagged</strong>Every tracked student held a sustained score above the decision threshold.'));
    return;
  }

  events.forEach(e => {
    const tone = ALERT_TONE[e.behavior] || { tone: 'var(--warn)', soft: 'var(--warn-soft)' };
    const card = el('div', 'alert');
    card.style.setProperty('--tone', tone.tone);
    card.style.setProperty('--tone-soft', tone.soft);
    card.innerHTML =
      `<div class="alert-top">
         <span class="alert-who">Student ${String(e.student_id).padStart(2, '0')}</span>
         <span class="alert-when">t = ${Number(e.timestamp).toFixed(1)}s</span>
       </div>
       <div class="alert-tags">
         <span class="tag">${e.behavior || e.cue}</span>
         <span class="tag tag-mute">${e.kind === 'sudden' ? 'Sudden drop' : 'Sustained drop'}</span>
         <span class="tag tag-mute">min ${Number(e.score).toFixed(0)}/100</span>
       </div>
       <div class="alert-move"><b>Move:</b> ${e.recommendation}</div>
       <div class="alert-move" style="color:var(--text-low)"><b>Delivery:</b> ${e.suggestion}</div>`;
    list.appendChild(card);
  });
}

/* ------------------------------------------------------------------ *
 * Timeline chart
 * ------------------------------------------------------------------ */
function renderTimeline(data) {
  const canvas = $('timelineChart');
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const timeline = data.timeline || [];
  if (timeline.length < 2) return;

  const W = rect.width, H = rect.height;
  const pad = { t: 16, r: 16, b: 30, l: 38 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  const t0 = timeline[0].t;
  const t1 = timeline[timeline.length - 1].t;
  const span = Math.max(0.001, t1 - t0);

  const X = (t) => pad.l + ((t - t0) / span) * plotW;
  const Y = (v) => pad.t + plotH - (Math.max(0, Math.min(100, v)) / 100) * plotH;

  // grid + y axis
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = 'middle';
  for (let v = 0; v <= 100; v += 25) {
    const y = Y(v);
    ctx.strokeStyle = v === 50 ? 'rgba(148,163,184,.16)' : 'rgba(148,163,184,.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();
    ctx.fillStyle = '#6B7A93';
    ctx.textAlign = 'right';
    ctx.fillText(String(v), pad.l - 8, y);
  }

  // x axis ticks
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const tickCount = Math.min(6, Math.max(2, Math.floor(plotW / 90)));
  for (let i = 0; i <= tickCount; i++) {
    const t = t0 + (span * i) / tickCount;
    const x = X(t);
    ctx.fillStyle = '#6B7A93';
    ctx.fillText(t.toFixed(1) + 's', x, H - pad.b + 9);
    ctx.strokeStyle = 'rgba(148,163,184,.07)';
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + plotH);
    ctx.stroke();
  }

  // decision threshold
  const th = Y(50);
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = 'rgba(251,113,133,.5)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(pad.l, th);
  ctx.lineTo(W - pad.r, th);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = 'rgba(251,113,133,.9)';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'bottom';
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.fillText('threshold 50', W - pad.r, th - 4);

  const pts = timeline.map(p => ({ x: X(p.t), y: Y(p.mean_score), s: p.mean_score, t: p.t, high: p.high, low: p.low }));

  // area fill
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + plotH);
  grad.addColorStop(0, 'rgba(109,139,255,.34)');
  grad.addColorStop(0.5, 'rgba(34,211,238,.12)');
  grad.addColorStop(1, 'rgba(34,211,238,0)');
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pad.t + plotH);
  pts.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(pts[pts.length - 1].x, pad.t + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // line
  ctx.beginPath();
  ctx.strokeStyle = '#8AA4FF';
  ctx.lineWidth = 2.2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.stroke();

  // below-threshold markers
  pts.forEach(p => {
    if (p.s < 50) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = '#FB7185';
      ctx.fill();
    }
  });

  // hover interaction
  const hit = { pts, x: 0, y: 0 };
  canvas._hit = hit;
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    let best = null, bd = 1e9;
    hit.pts.forEach(p => {
      const d = Math.abs(p.x - mx);
      if (d < bd) { bd = d; best = p; }
    });
    if (!best || bd > 26) { hideTip(); return; }
    showTip(e,
      `<b>t = ${best.t.toFixed(2)}s</b>` +
      `<div class="row"><span>Mean score</span><span>${best.s.toFixed(1)}</span></div>` +
      `<div class="row"><span>Attentive</span><span>${best.high}</span></div>` +
      `<div class="row"><span>Distracted</span><span>${best.low}</span></div>`);
  };
  canvas.onmouseleave = hideTip;
}

/* ------------------------------------------------------------------ *
 * Behaviour breakdown
 * ------------------------------------------------------------------ */
function renderBehaviors(data) {
  const list = $('behaviorList');
  list.innerHTML = '';
  const dist = data.behavior_distribution || {};
  const entries = Object.entries(dist);
  if (!entries.length) {
    list.appendChild(el('div', 'empty', 'No behaviour observations recorded'));
    return;
  }
  const total = entries.reduce((a, [, v]) => a + v, 0) || 1;

  entries
    .slice()
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, count]) => {
      const pct = (count / total) * 100;
      const color = BEHAVIOR_COLORS[name] || '#94A3B8';
      const row = el('div', 'behavior-row');
      row.innerHTML =
        `<div class="behavior-meta">
           <span><i style="background:${color}"></i>${name}</span>
           <span><b>${pct.toFixed(1)}%</b> <em>${count.toLocaleString()}</em></span>
         </div>
         <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>`;
      list.appendChild(row);
    });
}

/* ------------------------------------------------------------------ *
 * Confusion matrix
 * ------------------------------------------------------------------ */
function renderConfusion(data) {
  const ev = data.evaluation || {};
  const c = ev.confusion || {};
  const box = $('confusionBox');
  box.innerHTML = '';

  if (!c || (c.tn === undefined && c.precision_high === undefined)) {
    box.appendChild(el('div', 'empty', 'No confusion matrix recorded for this clip'));
    $('confusionNote').textContent = '';
    return;
  }

  const isHuman = ev.kind === 'human';
  if (isHuman && c.tn !== undefined) {
    const cells = [
      ['cm-head', 'Predicted LOW', 'Actual ground truth'],
      ['cm-tn', 'True Negative', 'Correctly flagged disengaged'],
      ['cm-fp', 'False Positive', 'Engaged, flagged disengaged'],
      ['cm-head', 'Predicted HIGH', ''],
      ['cm-fn', 'False Negative', 'Disengaged, missed'],
      ['cm-tp', 'True Positive', 'Correctly flagged engaged'],
    ];
    const grid = el('div', 'cm');
    cells.forEach(([cls, k, n]) => {
      const cell = el('div', 'cm-cell ' + cls);
      if (cls === 'cm-head') {
        cell.className = 'cm-head';
        cell.innerHTML = `<span class="k">${k}</span><b>${n}</b>`;
      } else {
        cell.innerHTML =
          `<span class="k">${k}</span>` +
          `<span class="v">${({ 'cm-tn': c.tn, 'cm-fp': c.fp, 'cm-fn': c.fn, 'cm-tp': c.tp })[cls]}</span>` +
          `<span class="n">${n}</span>`;
      }
      grid.appendChild(cell);
    });
    box.appendChild(grid);
    $('confusionNote').textContent =
      `True Macro-F1 = mean(F1_LOW, F1_HIGH) = ${ev.macro_f1.toFixed(2)}%. ` +
      `F1_LOW ${c.f1_low.toFixed(2)}% · F1_HIGH ${c.f1_high.toFixed(2)}%. ` +
      `Class imbalance means raw accuracy alone overstates performance.`;
  } else {
    const grid = el('div', 'cm');
    const mk = (cls, k, v, n) => {
      const cell = el('div', 'cm-cell ' + cls);
      cell.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span><span class="n">${n}</span>`;
      return cell;
    };
    grid.appendChild(mk('cm-tp', 'HIGH precision', (c.precision_high ?? 0).toFixed(1) + '%', 'precision'));
    grid.appendChild(mk('cm-tn', 'HIGH recall', (c.recall_high ?? 0).toFixed(1) + '%', 'recall'));
    grid.appendChild(mk('cm-fp', 'LOW precision', (c.precision_low ?? 0).toFixed(1) + '%', 'precision'));
    grid.appendChild(mk('cm-fn', 'LOW recall', (c.recall_low ?? 0).toFixed(1) + '%', 'recall'));
    box.appendChild(grid);
    $('confusionNote').textContent =
      'No human ground truth exists for this clip. These figures are agreement between the ' +
      'model and the internal heuristic rubric, so they measure internal consistency, not ' +
      'supervised accuracy.';
  }
}

/* ------------------------------------------------------------------ *
 * Student table
 * ------------------------------------------------------------------ */
function renderTable(data) {
  const body = $('tableBody');
  body.innerHTML = '';
  const q = ($('searchInput').value || '').trim().toLowerCase();

  const rows = (data.students || []).filter(s => {
    const high = s.mean_score >= 50;
    if (currentFilter === 'high' && !high) return false;
    if (currentFilter === 'low' && high) return false;
    if (currentFilter === 'phone' && s.behavior !== 'Using Phone') return false;
    if (currentFilter === 'risk' && s.behavior !== 'Sleeping / Fatigue' && s.behavior !== 'Peer Interaction') return false;
    if (currentFilter === 'study' && s.behavior !== 'Reading / Writing' && s.behavior !== 'Hand Raising') return false;
    if (q) {
      const hay = `${s.id} ${s.top_cue} ${s.behavior}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  if (!rows.length) {
    const tr = el('tr');
    const td = el('td');
    td.colSpan = 7;
    td.innerHTML = '<div class="empty" style="margin:0">No students match the current filter</div>';
    tr.appendChild(td);
    body.appendChild(tr);
    $('tableFoot').textContent = '0 students shown';
    return;
  }

  rows.forEach(s => {
    const high = s.mean_score >= 50;
    const color = high ? 'var(--ok)' : 'var(--bad)';
    const tr = el('tr');
    tr.innerHTML =
      `<td class="cell-id">S${String(s.id).padStart(2, '0')}</td>
       <td>
         <div class="cell-score">
           <b style="color:${color}">${s.mean_score.toFixed(1)}</b>
           <span class="mini-bar"><i style="width:${Math.max(0, Math.min(100, s.mean_score))}%;background:${color}"></i></span>
         </div>
       </td>
       <td class="mono">${s.attentive_pct.toFixed(1)}%</td>
       <td><span class="pill ${high ? 'pill-high' : 'pill-low'}">${high ? 'Attentive' : 'Distracted'}</span></td>
       <td>${s.top_cue}</td>
       <td class="mono" style="color:var(--text-low)">${s.min_score.toFixed(1)} / ${s.max_score.toFixed(1)}</td>
       <td><button class="btn-ghost" data-inspect="${s.id}">Inspect</button></td>`;
    body.appendChild(tr);
  });

  $('tableFoot').textContent =
    `${rows.length} of ${(data.students || []).length} students shown`;
}

function filterTable(filter, btn) {
  currentFilter = filter;
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('is-on'));
  if (btn) btn.classList.add('is-on');
  renderTable(DATABASE[currentKey]);
}

function searchTable() {
  renderTable(DATABASE[currentKey]);
}

/* ------------------------------------------------------------------ *
 * Student modal
 * ------------------------------------------------------------------ */
function openStudent(s) {
  const high = s.mean_score >= 50;
  $('modalTitle').textContent = `Student ${String(s.id).padStart(2, '0')}`;
  const pill = $('modalPill');
  pill.className = 'pill ' + (high ? 'pill-high' : 'pill-low');
  pill.textContent = high ? 'Attentive' : 'Distracted';

  $('modalMean').textContent = s.mean_score.toFixed(1);
  $('modalMean').style.color = high ? 'var(--ok)' : 'var(--bad)';
  $('modalAttentive').textContent = s.attentive_pct.toFixed(1) + '%';
  $('modalRange').textContent = `${s.min_score.toFixed(1)} – ${s.max_score.toFixed(1)}`;
  $('modalBehavior').textContent = s.top_cue;
  $('modalObs').textContent = s.observations.toLocaleString();

  const tone = ALERT_TONE[s.behavior] || { tone: 'var(--brand)', soft: 'var(--brand-soft)' };
  const advice = $('modalAdvice');
  advice.style.background = tone.soft;
  advice.style.borderColor = 'color-mix(in srgb, ' + tone.tone + ' 30%, transparent)';
  advice.innerHTML = adviceFor(s);

  renderSparkline(s.timeline || []);
  $('studentModal').classList.add('is-open');
  $('modalClose').focus();
}

function adviceFor(s) {
  const b = s.behavior;
  if (b === 'Using Phone') {
    return '<b>Phone use detected.</b> Apply non-verbal proximity control — circulate calmly down the aisle near the flagged desk without halting instruction. Avoid public callouts; quiet physical presence usually prompts the device to be stowed.';
  }
  if (b === 'Peer Interaction') {
    return '<b>Peer conversation detected.</b> Pivot into a structured 30-second Think-Pair-Share to channel conversational energy into academic discussion, rather than singling the student out.';
  }
  if (b === 'Sleeping / Fatigue') {
    return '<b>Fatigue or posture collapse detected.</b> Trigger a whole-class energy reset — a stand-and-stretch or a rapid live poll — and check in privately with the student afterwards.';
  }
  if (b === 'Reading / Writing' || b === 'Hand Raising') {
    return '<b>Active study behaviour.</b> This student is engaged in task-relevant work. Downward head pitch here reflects productive desk study, not disengagement. Continue with positive visual acknowledgment.';
  }
  return '<b>Sustained orientation.</b> The student is tracking the lesson. Reinforce with positive acknowledgment and occasional open retrieval questions.';
}

function renderSparkline(timeline) {
  const canvas = $('modalSparkline');
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (!timeline || timeline.length < 2) {
    ctx.fillStyle = '#6B7A93';
    ctx.font = '12px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No trajectory recorded', rect.width / 2, rect.height / 2);
    return;
  }

  const W = rect.width, H = rect.height;
  const pad = 10;
  const plotW = W - pad * 2;
  const plotH = H - pad * 2;
  const Y = (v) => pad + plotH - (Math.max(0, Math.min(100, v)) / 100) * plotH;

  ctx.strokeStyle = 'rgba(148,163,184,.12)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(pad, Y(50));
  ctx.lineTo(W - pad, Y(50));
  ctx.stroke();
  ctx.setLineDash([]);

  const pts = timeline.map((p, i) => ({
    x: pad + (i / (timeline.length - 1)) * plotW,
    y: Y(p.score),
  }));

  const grad = ctx.createLinearGradient(0, pad, 0, pad + plotH);
  grad.addColorStop(0, 'rgba(109,139,255,.3)');
  grad.addColorStop(1, 'rgba(109,139,255,0)');
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pad + plotH);
  pts.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(pts[pts.length - 1].x, pad + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  ctx.strokeStyle = '#8AA4FF';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.stroke();

  const last = pts[pts.length - 1];
  ctx.beginPath();
  ctx.arc(last.x, last.y, 3.5, 0, Math.PI * 2);
  ctx.fillStyle = '#8AA4FF';
  ctx.fill();
}

function closeModal() {
  $('studentModal').classList.remove('is-open');
}

/* ------------------------------------------------------------------ *
 * Footer
 * ------------------------------------------------------------------ */
function renderFooter(data) {
  const ev = data.evaluation || {};
  $('footSource').textContent = ev.provenance || 'runs/';
}
"""
