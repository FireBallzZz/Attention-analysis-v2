"""Generates the modern, self-contained, defense-grade Classroom Attention
Analysis Dashboard HTML.

The dashboard is a static artifact: the dataset is embedded so the file can
be opened directly from disk with no server or network access.

All numbers rendered in the UI come from ``dashboard/dashboard_dataset.json``,
which is built by ``scripts/build_dashboard_dataset.py`` from the real run
artifacts under ``runs/``. Nothing is hardcoded in this file.
"""

import json
from pathlib import Path

from dashboard_js import JS
from dashboard_theme import CSS

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE_ROOT / "dashboard" / "dashboard_dataset.json"
OUT_DASHBOARD = WORKSPACE_ROOT / "dashboard" / "index.html"

ICON = {
    "logo": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v1H7a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-1V6a4 4 0 0 0-4-4Z"/><path d="M9 13h6"/><path d="M12 11v4"/></svg>',
    "video": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="m16 11 6-3v8l-6-3z"/></svg>',
    "chevron": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>',
    "search": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    "grid": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
    "seats": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="5" rx="1.5"/><rect x="4" y="12" width="5" height="6" rx="1.5"/><rect x="10" y="12" width="5" height="6" rx="1.5"/><rect x="16" y="12" width="5" height="6" rx="1.5"/></svg>',
    "bell": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-3 8-3 8h18s-3-1-3-8"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>',
    "trend": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 17 6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/></svg>',
    "matrix": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 12h18"/><path d="M12 3v18"/></svg>',
    "table": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/></svg>',
    "target": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/></svg>',
    "users": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/></svg>',
    "close": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
}


def build() -> str:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    data_json = json.dumps(dataset, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Classroom Multimodal Attention Intelligence</title>
<meta name="description" content="Multimodal student attention analytics: spatial seating map, attention dynamics, and pedagogical interventions.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
{CSS}
</style>
</head>
<body>

<header class="masthead">
  <div class="brand">
    <div class="brand-mark">{ICON["logo"]}</div>
    <div class="brand-copy">
      <div class="brand-name">NEUROCLASS <span>Multimodal Attention Intelligence</span></div>
      <div class="brand-sub">Student attention analytics &amp; pedagogical intervention</div>
    </div>
  </div>

  <div class="masthead-tools">
    <span id="gtBadge" class="badge badge-neutral"></span>
    <div class="field">
      {ICON["video"]}
      <label class="sr-only" for="videoSelect">Select benchmark video</label>
      <select id="videoSelect"></select>
      <span class="chevron">{ICON["chevron"]}</span>
    </div>
  </div>
</header>

<main class="shell">

  <!-- ================= Video banner ================= -->
  <section class="panel" style="margin-top:22px;padding:16px 20px">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap">
      <div>
        <h1 id="videoTitle" style="font-size:1.22rem;font-weight:800">—</h1>
        <div id="videoMeta" class="panel-sub" style="margin-top:4px">—</div>
      </div>
      <div style="display:flex;gap:26px;flex-wrap:wrap">
        <div>
          <div style="font-size:.63rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--text-low)">Metric source</div>
          <div id="provenanceNote" style="font-size:.82rem;font-weight:600;color:var(--text-mid);margin-top:3px">—</div>
          <div id="provenanceDetail" style="font-size:.72rem;color:var(--text-low);margin-top:2px">—</div>
        </div>
      </div>
    </div>
  </section>

  <!-- ================= KPI row ================= -->
  <section class="kpi-grid">
    <div class="kpi kpi-accent-ok" id="kpiScoreCard">
      <div class="kpi-label">{ICON["trend"]} Class attention score</div>
      <div class="kpi-value" id="kpiScore">—</div>
      <div class="kpi-meta"><span id="kpiScoreNote">—</span> <b>scale 0–100</b></div>
    </div>

    <div class="kpi kpi-accent-brand" id="kpiAccuracyCard">
      <div class="kpi-label">{ICON["target"]} <span id="kpiAccuracyLabel">Ground-truth accuracy</span></div>
      <div class="kpi-value" id="kpiAccuracy">—</div>
      <div class="kpi-meta">
        <span>Macro-F1 <b id="kpiMacroF1">—</b></span>
        <span>Bal-acc <b id="kpiBalanced">—</b></span>
        <span>N <b id="kpiSampleSize">—</b></span>
      </div>
    </div>

    <div class="kpi kpi-accent-cyan">
      <div class="kpi-label">{ICON["users"]} Student population</div>
      <div class="kpi-value" id="kpiStudents">—</div>
      <div class="kpi-meta">
        <span><b id="kpiAttentive">—</b> attentive</span>
        <span><b id="kpiRate">—</b> rate</span>
      </div>
    </div>

    <div class="kpi kpi-accent-warn" id="kpiDropsCard">
      <div class="kpi-label">{ICON["bell"]} Attention-drop alerts</div>
      <div class="kpi-value" id="kpiDrops">—</div>
      <div class="kpi-meta"><span id="kpiDropNote">—</span></div>
    </div>
  </section>

  <!-- ================= Seating + alerts ================= -->
  <div class="section-head"><h2>Spatial &amp; temporal analysis</h2><div class="rule"></div></div>

  <section class="grid-2">
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title"><span class="ico">{ICON["seats"]}</span> Classroom seating &amp; attention map</div>
          <div class="panel-sub">Desk positions projected from camera space; colour reflects the student's mean attention score.</div>
        </div>
        <span class="badge badge-brand">Interactive</span>
      </div>

      <div class="seating" id="seating">
        <div class="seating-stage" id="seatingStage"></div>
      </div>

      <div class="seat-legend">
        <span><i style="background:var(--ok)"></i>Attentive (≥ 50)</span>
        <span><i style="background:var(--bad)"></i>Distracted (&lt; 50)</span>
        <span><i style="background:var(--brand)"></i>Front of classroom</span>
        <span id="seatingCaption" style="margin-left:auto;color:var(--text-low)">—</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title"><span class="ico">{ICON["bell"]}</span> Pedagogical alerts</div>
          <div class="panel-sub">Sustained attention drops translated into in-the-moment teacher moves.</div>
        </div>
        <span class="badge badge-neutral" id="alertCount">0 events</span>
      </div>
      <div class="alert-list" id="alertList"></div>
    </div>
  </section>

  <!-- ================= Timeline + behaviours ================= -->
  <section class="grid-2" style="margin-top:16px">
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title"><span class="ico">{ICON["trend"]}</span> Attention dynamics over time</div>
          <div class="panel-sub">Mean attention score per frame across all tracked students. Decision threshold at 50.</div>
        </div>
      </div>
      <div class="chart"><canvas id="timelineChart"></canvas></div>
      <div class="chart-foot">
        <span><i style="background:#8AA4FF"></i>Mean attention score</span>
        <span><i style="background:#FB7185"></i>Below threshold</span>
        <span><i style="background:rgba(251,113,133,.6)"></i>Threshold (τ* = 50)</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title"><span class="ico">{ICON["chart"]}</span> Behavioural modality share</div>
          <div class="panel-sub">Distribution of observed classroom behaviours across scored frames.</div>
        </div>
      </div>
      <div class="behavior-list" id="behaviorList"></div>
    </div>
  </section>

  <!-- ================= Confusion matrix ================= -->
  <div class="section-head"><h2>Error analysis</h2><div class="rule"></div></div>

  <section class="grid-2 equal">
    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title"><span class="ico">{ICON["matrix"]}</span> Confusion matrix</div>
          <div class="panel-sub">Outcome counts for the current benchmark.</div>
        </div>
      </div>
      <div id="confusionBox"></div>
      <div class="cm-note" id="confusionNote"></div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <div>
          <div class="panel-title"><span class="ico">{ICON["grid"]}</span> How to read these numbers</div>
          <div class="panel-sub">Metric definitions used throughout this dashboard.</div>
        </div>
      </div>
      <div class="cm-note" style="font-size:.79rem;line-height:1.65">
        <p><b>Accuracy</b> — share of all scored observations classified correctly. On an
        engagement-dominated classroom this is inflated by the majority class and should
        never be quoted alone.</p>
        <p style="margin-top:9px"><b>True Macro-F1</b> — mean of the attentive-class and
        distracted-class F1 scores. This is the headline metric: it weights the rare
        disengagement class equally with the common one.</p>
        <p style="margin-top:9px"><b>Balanced accuracy</b> — mean of per-class recall.
        A score near 50% means the model is barely better than always predicting the
        majority class.</p>
        <p style="margin-top:9px"><b>Ground truth</b> — clips marked
        <em>Human ground truth</em> are compared against independent CVAT annotation.
        Clips marked <em>Heuristic agreement</em> have no human annotation and report
        internal consistency against the rubric instead.</p>
      </div>
    </div>
  </section>

  <!-- ================= Student table ================= -->
  <div class="section-head"><h2>Student registry</h2><div class="rule"></div></div>

  <section class="panel">
    <div class="panel-head">
      <div>
        <div class="panel-title"><span class="ico">{ICON["table"]}</span> Per-student attention registry</div>
        <div class="panel-sub">Multimodal attention metrics and observed behaviour for every tracked student.</div>
      </div>
    </div>

    <div class="filters">
      <button class="chip is-on" data-filter="all" onclick="filterTable('all', this)">All students</button>
      <button class="chip" data-filter="high" onclick="filterTable('high', this)">Attentive</button>
      <button class="chip" data-filter="low" onclick="filterTable('low', this)">Distracted</button>
      <button class="chip" data-filter="study" onclick="filterTable('study', this)">Active study</button>
      <button class="chip" data-filter="phone" onclick="filterTable('phone', this)">Phone use</button>
      <button class="chip" data-filter="risk" onclick="filterTable('risk', this)">Fatigue &amp; peer</button>

      <div class="search">
        {ICON["search"]}
        <label class="sr-only" for="searchInput">Search students</label>
        <input type="search" id="searchInput" placeholder="Search by ID or behaviour…" autocomplete="off">
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Student</th>
            <th>Mean score</th>
            <th>Attentive %</th>
            <th>Status</th>
            <th>Primary behaviour</th>
            <th>Min / max</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
    <div class="table-foot" id="tableFoot">—</div>
  </section>

  <footer class="foot">
    <div>Metrics sourced from <code id="footSource">—</code></div>
    <div>Attention is an <em>observable behavioural surrogate</em>, not a measure of internal cognitive state. Use for pedagogical reflection only.</div>
  </footer>

</main>

<!-- ================= Student inspector ================= -->
<div class="modal" id="studentModal" role="dialog" aria-modal="true" aria-labelledby="modalTitle">
  <div class="modal-card">
    <button class="modal-close" id="modalClose" onclick="closeModal()" aria-label="Close student details">{ICON["close"]}</button>

    <div class="modal-head">
      <h3 id="modalTitle">Student</h3>
      <span class="pill pill-high" id="modalPill">Attentive</span>
    </div>

    <div class="modal-stats">
      <div class="stat"><span class="k">Mean score</span><span class="v" id="modalMean">—</span></div>
      <div class="stat"><span class="k">Attentive rate</span><span class="v" id="modalAttentive">—</span></div>
      <div class="stat"><span class="k">Score range</span><span class="v" id="modalRange">—</span></div>
      <div class="stat"><span class="k">Primary behaviour</span><span class="v" id="modalBehavior" style="font-size:.92rem">—</span></div>
      <div class="stat"><span class="k">Observations</span><span class="v" id="modalObs">—</span></div>
    </div>

    <div>
      <div class="modal-label">Individual attention trajectory</div>
      <div class="spark"><canvas id="modalSparkline"></canvas></div>
    </div>

    <div>
      <div class="modal-label">Suggested pedagogical move</div>
      <div class="advice" id="modalAdvice"></div>
    </div>
  </div>
</div>

<script>
const DATABASE = {data_json};
{JS}
</script>
</body>
</html>
"""


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}. Run scripts/build_dashboard_dataset.py first."
        )
    html = build()
    OUT_DASHBOARD.parent.mkdir(parents=True, exist_ok=True)
    OUT_DASHBOARD.write_text(html, encoding="utf-8")
    print(f"Generated dashboard HTML: {OUT_DASHBOARD.resolve()} ({OUT_DASHBOARD.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
