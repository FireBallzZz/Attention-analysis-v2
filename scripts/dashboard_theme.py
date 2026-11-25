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

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE_ROOT / "dashboard" / "dashboard_dataset.json"
OUT_DASHBOARD = WORKSPACE_ROOT / "dashboard" / "index.html"


CSS = r"""
:root {
  color-scheme: dark;

  --bg-base: #070A12;
  --bg-elevated: #0D1220;
  --surface-1: rgba(19, 25, 40, 0.72);
  --surface-2: rgba(28, 36, 56, 0.66);
  --surface-solid: #121827;

  --stroke-soft: rgba(148, 163, 184, 0.12);
  --stroke-strong: rgba(148, 163, 184, 0.22);

  --text-hi: #EEF2F8;
  --text-mid: #A7B3C7;
  --text-low: #6B7A93;

  --brand: #6D8BFF;
  --brand-2: #22D3EE;
  --brand-soft: rgba(109, 139, 255, 0.14);

  --ok: #34D399;
  --ok-soft: rgba(52, 211, 153, 0.13);
  --warn: #FBBF24;
  --warn-soft: rgba(251, 191, 36, 0.13);
  --bad: #FB7185;
  --bad-soft: rgba(251, 113, 133, 0.13);
  --violet: #A78BFA;
  --violet-soft: rgba(167, 139, 250, 0.13);

  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 18px;
  --r-xl: 26px;

  --shadow-card: 0 1px 2px rgba(0,0,0,.34), 0 12px 32px -12px rgba(0,0,0,.6);
  --shadow-pop: 0 24px 64px -18px rgba(0,0,0,.82);

  --ease: cubic-bezier(.22,1,.36,1);
  --font-ui: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { scroll-behavior: smooth; }

body {
  min-height: 100vh;
  background: var(--bg-base);
  background-image:
    radial-gradient(1100px 620px at 8% -12%, rgba(109,139,255,.16), transparent 62%),
    radial-gradient(900px 520px at 96% 4%, rgba(34,211,238,.11), transparent 60%),
    radial-gradient(700px 700px at 50% 108%, rgba(167,139,250,.09), transparent 62%);
  background-attachment: fixed;
  color: var(--text-hi);
  font-family: var(--font-ui);
  font-size: 15px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

h1, h2, h3 { line-height: 1.2; letter-spacing: -.022em; font-weight: 700; }

::selection { background: rgba(109,139,255,.34); color: #fff; }

:focus-visible {
  outline: 2px solid var(--brand);
  outline-offset: 2px;
  border-radius: 6px;
}

.sr-only {
  position: absolute; width: 1px; height: 1px;
  padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}

/* ---------- Shell ---------- */
.shell { max-width: 1560px; margin: 0 auto; padding: 0 28px 72px; }

/* ---------- Header ---------- */
.masthead {
  position: sticky; top: 0; z-index: 60;
  display: flex; align-items: center; justify-content: space-between;
  gap: 20px; padding: 14px 28px;
  background: rgba(7,10,18,.72);
  backdrop-filter: blur(22px) saturate(160%);
  -webkit-backdrop-filter: blur(22px) saturate(160%);
  border-bottom: 1px solid var(--stroke-soft);
}
.brand { display: flex; align-items: center; gap: 13px; min-width: 0; }
.brand-mark {
  width: 38px; height: 38px; flex: none;
  display: grid; place-items: center;
  border-radius: 11px;
  background: linear-gradient(145deg, rgba(109,139,255,.28), rgba(34,211,238,.16));
  border: 1px solid rgba(109,139,255,.36);
  box-shadow: 0 0 0 1px rgba(109,139,255,.08), 0 8px 22px -8px rgba(109,139,255,.6);
}
.brand-mark svg { width: 19px; height: 19px; }
.brand-copy { min-width: 0; }
.brand-name {
  font-size: .96rem; font-weight: 800; letter-spacing: -.01em;
  display: flex; align-items: baseline; gap: 7px;
}
.brand-name span { font-weight: 500; color: var(--text-low); font-size: .82rem; }
.brand-sub {
  font-size: .7rem; color: var(--text-low);
  text-transform: uppercase; letter-spacing: .1em; font-weight: 600;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.masthead-tools { display: flex; align-items: center; gap: 12px; flex: none; }

.field {
  display: flex; align-items: center; gap: 9px;
  background: var(--surface-1);
  border: 1px solid var(--stroke-soft);
  border-radius: 11px; padding: 0 11px;
  transition: border-color .18s var(--ease), box-shadow .18s var(--ease);
}
.field:focus-within {
  border-color: rgba(109,139,255,.55);
  box-shadow: 0 0 0 3px rgba(109,139,255,.14);
}
.field svg { width: 15px; height: 15px; color: var(--text-low); flex: none; }
.field select {
  appearance: none; -webkit-appearance: none;
  background: transparent; border: 0; outline: 0;
  color: var(--text-hi); font-family: var(--font-ui);
  font-size: .84rem; font-weight: 600;
  padding: 9px 24px 9px 0; cursor: pointer;
  max-width: 320px; text-overflow: ellipsis;
}
.field select option { background: var(--surface-solid); color: var(--text-hi); }
.field .chevron {
  width: 12px; height: 12px; margin-left: -22px; pointer-events: none; color: var(--text-low);
}

/* ---------- Badges ---------- */
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 999px;
  font-size: .68rem; font-weight: 700;
  letter-spacing: .045em; text-transform: uppercase;
  border: 1px solid transparent; white-space: nowrap;
}
.badge svg { width: 12px; height: 12px; }
.badge-human { background: var(--ok-soft); color: var(--ok); border-color: rgba(52,211,153,.28); }
.badge-heuristic { background: var(--warn-soft); color: var(--warn); border-color: rgba(251,191,36,.28); }
.badge-neutral { background: var(--surface-2); color: var(--text-mid); border-color: var(--stroke-soft); }
.badge-brand { background: var(--brand-soft); color: #9FB3FF; border-color: rgba(109,139,255,.3); }

/* ---------- Panels ---------- */
.panel {
  background: var(--surface-1);
  backdrop-filter: blur(18px) saturate(140%);
  -webkit-backdrop-filter: blur(18px) saturate(140%);
  border: 1px solid var(--stroke-soft);
  border-radius: var(--r-lg);
  box-shadow: var(--shadow-card);
  padding: 20px 22px 22px;
  display: flex; flex-direction: column; gap: 16px;
  min-width: 0;
}
.panel-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 16px; padding-bottom: 14px;
  border-bottom: 1px solid var(--stroke-soft);
}
.panel-title {
  display: flex; align-items: center; gap: 10px;
  font-size: .98rem; font-weight: 700;
}
.panel-title .ico {
  width: 30px; height: 30px; flex: none;
  display: grid; place-items: center;
  border-radius: 9px;
  background: var(--surface-2);
  border: 1px solid var(--stroke-soft);
  color: var(--brand);
}
.panel-title .ico svg { width: 15px; height: 15px; }
.panel-sub { font-size: .77rem; color: var(--text-low); margin-top: 3px; }

/* ---------- Section heading ---------- */
.section-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 16px; margin: 30px 0 14px; flex-wrap: wrap;
}
.section-head h2 {
  font-size: .74rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .13em; color: var(--text-low);
}
.section-head .rule { flex: 1; height: 1px; background: var(--stroke-soft); min-width: 20px; }

/* ---------- KPI ---------- */
.kpi-grid {
  display: grid; gap: 16px; margin-top: 22px;
  grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
}
.kpi {
  position: relative; overflow: hidden;
  background: var(--surface-1);
  backdrop-filter: blur(16px);
  border: 1px solid var(--stroke-soft);
  border-radius: var(--r-md);
  padding: 17px 18px 16px;
  box-shadow: var(--shadow-card);
  transition: transform .22s var(--ease), border-color .22s var(--ease);
}
.kpi::after {
  content: ''; position: absolute; inset: 0 auto 0 0; width: 3px;
  background: var(--accent, var(--brand));
  opacity: .85;
}
.kpi:hover { transform: translateY(-2px); border-color: var(--stroke-strong); }
.kpi-label {
  display: flex; align-items: center; gap: 7px;
  font-size: .68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .085em; color: var(--text-low);
}
.kpi-label svg { width: 13px; height: 13px; }
.kpi-value {
  font-family: var(--font-mono); font-size: 2.05rem; font-weight: 700;
  letter-spacing: -.03em; margin: 9px 0 4px; line-height: 1;
  color: var(--accent, var(--text-hi));
}
.kpi-meta { font-size: .74rem; color: var(--text-low); display: flex; flex-wrap: wrap; gap: 5px 9px; }
.kpi-meta b { color: var(--text-mid); font-weight: 600; }
.kpi-accent-ok { --accent: var(--ok); }
.kpi-accent-warn { --accent: var(--warn); }
.kpi-accent-bad { --accent: var(--bad); }
.kpi-accent-brand { --accent: var(--brand); }
.kpi-accent-cyan { --accent: var(--brand-2); }

/* ---------- Layout ---------- */
.grid-2 { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 1080px) {
  .grid-2 { grid-template-columns: minmax(0, 1.72fr) minmax(0, 1fr); }
  .grid-2.equal { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.stack { display: flex; flex-direction: column; gap: 16px; }

/* ---------- Heatmap ---------- */
.seating {
  position: relative; width: 100%;
  aspect-ratio: 16 / 9;
  border-radius: var(--r-md);
  overflow: hidden;
  border: 1px solid var(--stroke-soft);
  background:
    radial-gradient(ellipse 70% 46% at 50% 0%, rgba(109,139,255,.13), transparent 70%),
    linear-gradient(180deg, #0C111E, #080B14);
}
.seating-stage { position: absolute; inset: 0; transform-origin: 0 0; }
.podium {
  position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
  padding: 5px 14px; border-radius: 999px;
  background: rgba(109,139,255,.12);
  border: 1px dashed rgba(109,139,255,.4);
  color: #9FB3FF; font-size: .62rem; font-weight: 700;
  letter-spacing: .13em; text-transform: uppercase; white-space: nowrap;
  z-index: 4; pointer-events: none;
}
.seat {
  position: absolute; transform: translate(-50%, -50%);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 0; padding: 2px; border-radius: 7px;
  background: rgba(12,17,30,.72);
  border: 1.5px solid var(--seat, var(--ok));
  box-shadow: 0 3px 10px rgba(0,0,0,.45);
  cursor: pointer; user-select: none;
  overflow: hidden;
  transition: transform .2s var(--ease), box-shadow .2s var(--ease), border-color .2s var(--ease), background .2s var(--ease);
  z-index: 2;
}
.seat:hover { transform: translate(-50%, -50%) scale(1.22); z-index: 60; background: rgba(20,28,48,.92); box-shadow: 0 0 0 2px rgba(109,139,255,.55), 0 8px 22px rgba(0,0,0,.6); }
.seat.is-active { box-shadow: 0 0 0 2px var(--brand), 0 8px 22px rgba(0,0,0,.6); background: rgba(20,28,48,.92); z-index: 70; }
.seat-id { font-size: .6rem; font-weight: 800; color: #fff; font-family: var(--font-mono); letter-spacing: -.03em; line-height: 1.1; }
.seat-score { font-size: .62rem; font-weight: 700; font-family: var(--font-mono); color: var(--seat, var(--ok)); line-height: 1.15; }
.seat-legend {
  display: flex; flex-wrap: wrap; gap: 8px 16px;
  font-size: .72rem; color: var(--text-low);
}
.seat-legend span { display: inline-flex; align-items: center; gap: 6px; }
.seat-legend i { width: 9px; height: 9px; border-radius: 3px; display: inline-block; }

/* ---------- Alerts ---------- */
.alert-list { display: flex; flex-direction: column; gap: 10px; max-height: 520px; overflow-y: auto; padding-right: 6px; }
.alert-list::-webkit-scrollbar { width: 6px; }
.alert-list::-webkit-scrollbar-track { background: transparent; }
.alert-list::-webkit-scrollbar-thumb { background: rgba(148,163,184,.24); border-radius: 99px; }
.alert {
  background: var(--surface-2);
  border: 1px solid var(--stroke-soft);
  border-left: 3px solid var(--tone, var(--warn));
  border-radius: var(--r-sm);
  padding: 12px 14px;
  display: flex; flex-direction: column; gap: 7px;
  transition: background .18s var(--ease), transform .18s var(--ease);
}
.alert:hover { background: rgba(38,48,72,.7); transform: translateX(2px); }
.alert-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.alert-who { font-size: .82rem; font-weight: 700; }
.alert-when { font-size: .7rem; font-family: var(--font-mono); color: var(--text-low); }
.alert-tags { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.tag {
  font-size: .63rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
  padding: 3px 8px; border-radius: 5px;
  background: var(--tone-soft, var(--warn-soft)); color: var(--tone, var(--warn));
  border: 1px solid color-mix(in srgb, var(--tone, var(--warn)) 26%, transparent);
}
.tag-mute { background: var(--surface-2); color: var(--text-low); border-color: var(--stroke-soft); }
.alert-move { font-size: .79rem; color: var(--text-mid); line-height: 1.5; }
.alert-move b { color: var(--text-hi); font-weight: 700; }
.empty {
  padding: 34px 18px; text-align: center;
  font-size: .82rem; color: var(--text-low);
  border: 1px dashed var(--stroke-strong); border-radius: var(--r-md);
  background: rgba(148,163,184,.03);
}
.empty strong { display: block; color: var(--text-mid); margin-bottom: 3px; font-size: .88rem; }

/* ---------- Charts ---------- */
.chart { width: 100%; height: 268px; position: relative; }
.chart canvas { width: 100% !important; height: 100% !important; display: block; }
.chart-foot {
  display: flex; flex-wrap: wrap; gap: 8px 18px;
  font-size: .72rem; color: var(--text-low);
}
.chart-foot span { display: inline-flex; align-items: center; gap: 6px; }
.chart-foot i { width: 14px; height: 3px; border-radius: 2px; }

/* ---------- Behaviors ---------- */
.behavior-list { display: flex; flex-direction: column; gap: 13px; }
.behavior-row { display: flex; flex-direction: column; gap: 6px; }
.behavior-meta { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; font-size: .8rem; }
.behavior-meta span:first-child { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }
.behavior-meta i { width: 9px; height: 9px; border-radius: 3px; flex: none; }
.behavior-meta b { font-family: var(--font-mono); font-weight: 700; color: var(--text-hi); }
.behavior-meta em { font-style: normal; font-family: var(--font-mono); color: var(--text-low); font-size: .74rem; }
.bar-track { height: 7px; border-radius: 99px; background: rgba(148,163,184,.13); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; transition: width .7s var(--ease); }

/* ---------- Confusion matrix ---------- */
.cm { display: grid; grid-template-columns: auto repeat(2, minmax(0,1fr)); gap: 8px; align-items: stretch; }
.cm-cell {
  padding: 12px 14px; border-radius: var(--r-sm);
  border: 1px solid var(--stroke-soft);
  background: var(--surface-2);
  display: flex; flex-direction: column; gap: 3px;
}
.cm-cell .k { font-size: .63rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--text-low); }
.cm-cell .v { font-family: var(--font-mono); font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em; }
.cm-cell .n { font-size: .69rem; color: var(--text-low); font-family: var(--font-mono); }
.cm-head { display: flex; flex-direction: column; justify-content: center; gap: 2px; }
.cm-head .k { font-size: .63rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--text-low); }
.cm-head b { font-size: .78rem; }
.cm-tn { border-color: rgba(52,211,153,.3); background: var(--ok-soft); } .cm-tn .v { color: var(--ok); }
.cm-fp { border-color: rgba(251,191,36,.3); background: var(--warn-soft); } .cm-fp .v { color: var(--warn); }
.cm-fn { border-color: rgba(251,113,133,.3); background: var(--bad-soft); } .cm-fn .v { color: var(--bad); }
.cm-tp { border-color: rgba(109,139,255,.3); background: var(--brand-soft); } .cm-tp .v { color: #9FB3FF; }
.cm-note { font-size: .72rem; color: var(--text-low); line-height: 1.5; }

/* ---------- Table ---------- */
.filters { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.chip {
  background: var(--surface-2); color: var(--text-mid);
  border: 1px solid var(--stroke-soft); border-radius: 999px;
  padding: 6px 13px; font-size: .78rem; font-weight: 600;
  font-family: var(--font-ui); cursor: pointer;
  transition: all .18s var(--ease);
}
.chip:hover { color: var(--text-hi); border-color: var(--stroke-strong); }
.chip.is-on { background: var(--brand-soft); color: #B9C8FF; border-color: rgba(109,139,255,.45); }
.search {
  margin-left: auto; display: flex; align-items: center; gap: 8px;
  background: var(--surface-2); border: 1px solid var(--stroke-soft);
  border-radius: 999px; padding: 0 13px; min-width: 210px;
  transition: border-color .18s var(--ease), box-shadow .18s var(--ease);
}
.search:focus-within { border-color: rgba(109,139,255,.55); box-shadow: 0 0 0 3px rgba(109,139,255,.13); }
.search svg { width: 14px; height: 14px; color: var(--text-low); flex: none; }
.search input {
  background: transparent; border: 0; outline: 0; color: var(--text-hi);
  font-family: var(--font-ui); font-size: .82rem; padding: 8px 0; width: 100%;
}
.search input::placeholder { color: var(--text-low); }

.table-wrap { overflow-x: auto; border-radius: var(--r-sm); }
table { width: 100%; border-collapse: collapse; font-size: .84rem; }
thead th {
  position: sticky; top: 0; z-index: 2;
  padding: 11px 14px; text-align: left; white-space: nowrap;
  background: rgba(13,18,32,.94); backdrop-filter: blur(8px);
  color: var(--text-low); font-size: .67rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .075em;
  border-bottom: 1px solid var(--stroke-soft);
}
tbody td { padding: 11px 14px; border-bottom: 1px solid rgba(148,163,184,.07); color: var(--text-mid); vertical-align: middle; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr { transition: background .14s var(--ease); }
tbody tr:hover td { background: rgba(148,163,184,.045); color: var(--text-hi); }
.cell-id { font-weight: 700; color: var(--text-hi); font-family: var(--font-mono); font-size: .82rem; }
.cell-score { display: flex; align-items: center; gap: 9px; }
.cell-score b { font-family: var(--font-mono); font-weight: 700; font-size: .84rem; }
.mini-bar { width: 58px; height: 5px; border-radius: 99px; background: rgba(148,163,184,.16); overflow: hidden; flex: none; }
.mini-bar i { display: block; height: 100%; border-radius: 99px; }
.mono { font-family: var(--font-mono); }
.pill {
  display: inline-block; padding: 3px 9px; border-radius: 6px;
  font-size: .65rem; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; font-family: var(--font-mono);
  border: 1px solid transparent;
}
.pill-high { background: var(--ok-soft); color: var(--ok); border-color: rgba(52,211,153,.28); }
.pill-low { background: var(--bad-soft); color: var(--bad); border-color: rgba(251,113,133,.28); }
.btn-ghost {
  background: transparent; border: 1px solid var(--stroke-soft);
  color: #9FB3FF; border-radius: 7px; padding: 5px 11px;
  font-size: .73rem; font-weight: 600; font-family: var(--font-ui);
  cursor: pointer; transition: all .18s var(--ease);
}
.btn-ghost:hover { background: var(--brand-soft); border-color: rgba(109,139,255,.45); color: #fff; }
.table-foot { font-size: .74rem; color: var(--text-low); }

/* ---------- Modal ---------- */
.modal {
  position: fixed; inset: 0; z-index: 100;
  display: none; align-items: center; justify-content: center; padding: 24px;
  background: rgba(4,6,12,.76); backdrop-filter: blur(10px);
}
.modal.is-open { display: flex; }
.modal-card {
  position: relative; width: 100%; max-width: 660px; max-height: 88vh; overflow-y: auto;
  background: linear-gradient(180deg, #141B2B, #0F1521);
  border: 1px solid var(--stroke-strong); border-radius: var(--r-xl);
  box-shadow: var(--shadow-pop);
  padding: 26px 28px 28px;
  display: flex; flex-direction: column; gap: 18px;
  animation: rise .26s var(--ease);
}
@keyframes rise { from { opacity: 0; transform: translateY(14px) scale(.985); } to { opacity: 1; transform: none; } }
.modal-close {
  position: absolute; top: 16px; right: 16px;
  width: 32px; height: 32px; display: grid; place-items: center;
  background: var(--surface-2); border: 1px solid var(--stroke-soft);
  border-radius: 9px; color: var(--text-mid); cursor: pointer;
  transition: all .18s var(--ease);
}
.modal-close:hover { background: rgba(251,113,133,.16); color: var(--bad); border-color: rgba(251,113,133,.4); }
.modal-close svg { width: 15px; height: 15px; }
.modal-head { display: flex; align-items: center; gap: 13px; flex-wrap: wrap; padding-right: 40px; }
.modal-head h3 { font-size: 1.16rem; font-weight: 800; }
.modal-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 11px; }
.stat {
  background: var(--surface-2); border: 1px solid var(--stroke-soft);
  border-radius: var(--r-sm); padding: 13px 15px;
  display: flex; flex-direction: column; gap: 4px;
}
.stat .k { font-size: .63rem; font-weight: 700; text-transform: uppercase; letter-spacing: .075em; color: var(--text-low); }
.stat .v { font-family: var(--font-mono); font-size: 1.28rem; font-weight: 700; letter-spacing: -.02em; }
.modal-label { font-size: .73rem; font-weight: 700; color: var(--text-mid); margin-bottom: 8px; }
.advice {
  background: var(--brand-soft); border: 1px solid rgba(109,139,255,.26);
  border-radius: var(--r-sm); padding: 14px 16px;
  font-size: .83rem; line-height: 1.6; color: var(--text-mid);
}
.advice b { color: var(--text-hi); }
.spark { width: 100%; height: 130px; }
.spark canvas { width: 100% !important; height: 100% !important; display: block; }

/* ---------- Tooltip ---------- */
.tip {
  position: fixed; z-index: 120; pointer-events: none;
  background: rgba(10,14,24,.97); border: 1px solid var(--stroke-strong);
  border-radius: 9px; padding: 8px 11px; font-size: .75rem;
  box-shadow: 0 12px 32px rgba(0,0,0,.6);
  opacity: 0; transition: opacity .14s var(--ease);
  max-width: 250px;
}
.tip.is-on { opacity: 1; }
.tip b { font-family: var(--font-mono); }
.tip .row { display: flex; justify-content: space-between; gap: 14px; color: var(--text-low); }
.tip .row span:last-child { color: var(--text-hi); font-family: var(--font-mono); }

/* ---------- Footer ---------- */
.foot {
  margin-top: 44px; padding-top: 20px;
  border-top: 1px solid var(--stroke-soft);
  display: flex; justify-content: space-between; gap: 18px; flex-wrap: wrap;
  font-size: .73rem; color: var(--text-low);
}
.foot code { font-family: var(--font-mono); color: var(--text-mid); }

@media (max-width: 780px) {
  .shell { padding: 0 16px 52px; }
  .masthead { flex-direction: column; align-items: stretch; gap: 12px; padding: 12px 16px; }
  .masthead-tools { justify-content: space-between; }
  .field select { max-width: 190px; }
  .panel { padding: 17px 16px 18px; }
  .kpi-value { font-size: 1.75rem; }
  .search { margin-left: 0; width: 100%; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
  html { scroll-behavior: auto; }
}
"""
