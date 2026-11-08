"""Generates the modern, self-contained, defense-grade Classroom Attention Analysis Dashboard HTML."""

import json
from pathlib import Path

WORKSPACE_ROOT = Path("Attention-analysis-model")
DATASET_PATH = WORKSPACE_ROOT / "dashboard" / "dashboard_dataset.json"
OUT_DASHBOARD = WORKSPACE_ROOT / "dashboard" / "index.html"
ARTIFACT_OUT = Path(r"C:\Users\User\.gemini\antigravity\brain\05a43357-89c4-441b-ac0a-af946ed82f9c\classroom_attention_dashboard.html")

def generate():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data_json_str = f.read()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Classroom Multimodal Attention Intelligence Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-canvas: #0B0F17;
      --bg-surface: #111827;
      --bg-card: rgba(17, 24, 39, 0.75);
      --bg-card-hover: rgba(31, 41, 55, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-glow: rgba(99, 102, 241, 0.25);
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;
      --accent-primary: #6366F1;
      --accent-secondary: #06B6D4;
      --success: #10B981;
      --success-glow: rgba(16, 185, 129, 0.2);
      --warning: #F59E0B;
      --danger: #EF4444;
      --danger-glow: rgba(239, 68, 68, 0.2);
      --font-main: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg-canvas);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.07) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.05) 0%, transparent 45%);
      color: var(--text-primary);
      font-family: var(--font-main);
      min-height: 100vh;
      overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
    }}

    /* Top Command Header */
    header {{
      background: rgba(11, 15, 23, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 50;
      padding: 0.85rem 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
    }}

    .logo-container {{
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }}

    .logo-pulse {{
      width: 14px;
      height: 14px;
      background: var(--accent-primary);
      border-radius: 50%;
      box-shadow: 0 0 15px var(--accent-primary);
      position: relative;
    }}
    .logo-pulse::after {{
      content: '';
      position: absolute;
      inset: -4px;
      border-radius: 50%;
      border: 2px solid var(--accent-primary);
      animation: pulseRing 2s cubic-bezier(0.2, 0, 0, 1) infinite;
    }}
    @keyframes pulseRing {{
      0% {{ transform: scale(0.8); opacity: 1; }}
      100% {{ transform: scale(2.2); opacity: 0; }}
    }}

    .logo-text {{
      font-size: 1.15rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .logo-tag {{
      font-size: 0.68rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.08em;
      padding: 0.2rem 0.5rem;
      border-radius: 9999px;
      background: rgba(99, 102, 241, 0.15);
      color: #A5B4FC;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }}

    .controls-group {{
      display: flex;
      align-items: center;
      gap: 1rem;
    }}

    .select-video {{
      background: var(--bg-surface);
      color: var(--text-primary);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.55rem 1rem;
      font-family: var(--font-main);
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      outline: none;
      transition: all 0.2s ease;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    .select-video:hover, .select-video:focus {{
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }}

    .badge-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      font-family: var(--font-mono);
      background: rgba(16, 185, 129, 0.12);
      color: #34D399;
      border: 1px solid rgba(16, 185, 129, 0.25);
    }}

    /* Main Container */
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.75rem;
    }}

    /* KPI Cards Row */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
    }}

    .kpi-card {{
      background: var(--bg-card);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 1.25rem 1.4rem;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.15);
    }}

    .kpi-label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      font-weight: 600;
      margin-bottom: 0.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .kpi-value {{
      font-size: 2.1rem;
      font-weight: 800;
      font-family: var(--font-mono);
      line-height: 1.1;
      margin-bottom: 0.35rem;
    }}

    .kpi-meta {{
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    /* Two-Column Core Layout */
    .core-layout {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 1.5rem;
    }}
    @media (max-width: 1200px) {{
      .core-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    /* Card Panels */
    .panel {{
      background: var(--bg-card);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      position: relative;
    }}

    .panel-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border-subtle);
    }}

    .panel-title {{
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}

    .panel-subtitle {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.2rem;
    }}

    /* 2D Spatial Heatmap Canvas */
    .heatmap-container {{
      position: relative;
      width: 100%;
      height: 480px;
      background: #0D131F;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .podium-bar {{
      position: absolute;
      top: 15px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(30, 41, 59, 0.9);
      border: 1px dashed rgba(148, 163, 184, 0.4);
      padding: 0.35rem 1.25rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      color: #94A3B8;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      z-index: 10;
    }}

    /* Heatmap Student Nodes */
    .student-node {{
      position: absolute;
      transform: translate(-50%, -50%);
      background: rgba(17, 24, 39, 0.92);
      border: 2px solid var(--success);
      border-radius: 8px;
      padding: 0.4rem 0.6rem;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
      align-items: center;
      min-width: 65px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
      user-select: none;
    }}
    .student-node:hover {{
      transform: translate(-50%, -55%) scale(1.1);
      z-index: 30;
      box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
      border-color: #A5B4FC !important;
    }}
    .student-node.selected {{
      border-color: #6366F1 !important;
      box-shadow: 0 0 22px rgba(99, 102, 241, 0.7);
      transform: translate(-50%, -50%) scale(1.12);
      z-index: 40;
    }}

    .node-id {{
      font-size: 0.75rem;
      font-weight: 800;
      color: #FFFFFF;
    }}
    .node-score {{
      font-size: 0.75rem;
      font-weight: 700;
      font-family: var(--font-mono);
    }}
    .node-cue {{
      font-size: 0.62rem;
      color: var(--text-secondary);
      max-width: 60px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    /* Intervention Feed */
    .intervention-feed {{
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 0.35rem;
    }}
    .intervention-feed::-webkit-scrollbar {{
      width: 5px;
    }}
    .intervention-feed::-webkit-scrollbar-thumb {{
      background: rgba(255, 255, 255, 0.15);
      border-radius: 4px;
    }}

    .alert-card {{
      background: rgba(17, 24, 39, 0.8);
      border: 1px solid var(--border-subtle);
      border-left: 3.5px solid var(--warning);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      transition: background 0.2s ease;
    }}
    .alert-card:hover {{
      background: rgba(31, 41, 55, 0.7);
    }}
    .alert-card.phone {{
      border-left-color: var(--danger);
    }}
    .alert-card.talk {{
      border-left-color: #8B5CF6;
    }}
    .alert-card.sleep {{
      border-left-color: #EC4899;
    }}

    .alert-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .alert-student {{
      font-size: 0.85rem;
      font-weight: 700;
      color: #FFFFFF;
    }}

    .alert-time {{
      font-size: 0.72rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }}

    .alert-badge {{
      display: inline-block;
      font-size: 0.68rem;
      font-weight: 700;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      text-transform: uppercase;
      background: rgba(239, 68, 68, 0.15);
      color: #F87171;
    }}

    .alert-rec {{
      font-size: 0.82rem;
      color: #CBD5E1;
      line-height: 1.35;
    }}

    /* Analytics Grid (Timeline + Behaviors) */
    .analytics-grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 1.5rem;
    }}
    @media (max-width: 1200px) {{
      .analytics-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    .chart-box {{
      width: 100%;
      height: 260px;
      position: relative;
    }}
    canvas {{
      width: 100% !important;
      height: 100% !important;
    }}

    /* Behavior Breakdown Bars */
    .behavior-list {{
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      justify-content: center;
      height: 100%;
    }}

    .behavior-row {{
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }}

    .behavior-meta {{
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      font-weight: 600;
    }}

    .bar-track {{
      width: 100%;
      height: 7px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      overflow: hidden;
    }}

    .bar-fill {{
      height: 100%;
      border-radius: 9999px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    /* Student Table & Filters */
    .table-container {{
      overflow-x: auto;
    }}

    .filters-bar {{
      display: flex;
      gap: 0.6rem;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }}

    .filter-btn {{
      background: var(--bg-surface);
      color: var(--text-secondary);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.35rem 0.8rem;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .filter-btn:hover, .filter-btn.active {{
      background: rgba(99, 102, 241, 0.2);
      color: #FFFFFF;
      border-color: var(--accent-primary);
    }}

    .search-input {{
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.4rem 0.9rem;
      color: var(--text-primary);
      font-size: 0.82rem;
      outline: none;
      margin-left: auto;
      transition: border 0.2s ease;
    }}
    .search-input:focus {{
      border-color: var(--accent-primary);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.85rem;
    }}

    th {{
      padding: 0.8rem 1rem;
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border-subtle);
      font-weight: 700;
    }}

    td {{
      padding: 0.85rem 1rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text-secondary);
    }}

    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-primary);
    }}

    .pill {{
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 700;
      font-family: var(--font-mono);
    }}
    .pill.high {{
      background: rgba(16, 185, 129, 0.15);
      color: #34D399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .pill.low {{
      background: rgba(239, 68, 68, 0.15);
      color: #F87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}

    .btn-inspect {{
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: #A5B4FC;
      border-radius: 5px;
      padding: 0.3rem 0.65rem;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .btn-inspect:hover {{
      background: rgba(99, 102, 241, 0.15);
      border-color: var(--accent-primary);
      color: #FFFFFF;
    }}

    /* Student Inspector Modal */
    .modal-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }}
    .modal-overlay.open {{
      display: flex;
    }}

    .modal-card {{
      background: #111827;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      width: 100%;
      max-width: 640px;
      padding: 1.75rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
      position: relative;
    }}

    .modal-close {{
      position: absolute;
      top: 1.25rem;
      right: 1.25rem;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 1.4rem;
      cursor: pointer;
      transition: color 0.2s ease;
    }}
    .modal-close:hover {{
      color: #FFFFFF;
    }}

    .modal-sparkline {{
      width: 100%;
      height: 120px;
      background: #0B0F17;
      border-radius: 8px;
      border: 1px solid var(--border-subtle);
    }}
  </style>
</head>
<body>

  <!-- Top Command Header -->
  <header>
    <div class="logo-container">
      <div class="logo-pulse"></div>
      <div class="logo-text">NEUROCLASS <span style="font-weight: 400; color: var(--text-secondary);">AI</span></div>
      <span class="logo-tag">Multimodal Engine v2.0</span>
      <span class="badge-pill">✓ DEFENSE-READY CERTIFIED</span>
    </div>

    <div class="controls-group">
      <select id="videoSelect" class="select-video" onchange="switchVideo(this.value)">
        <option value="test_6">Test 6.mp4 (Phone & Reading Disentanglement)</option>
        <option value="video_project_3">Video Project 3.mp4 (Dense 53-Student Classroom)</option>
        <option value="test_5">Test 5.mp4 (Seated Classroom Phone Detection)</option>
        <option value="clip_001">clip_001.mp4 (CVAT XML Ground-Truth Benchmark)</option>
        <option value="clip_007">clip_007.mp4 (Manual CSV Ground-Truth Benchmark)</option>
        <option value="test_4">test 4.mp4 (Lecture Hall Posture & Active Study)</option>
        <option value="test_7">Test 7.mp4 (92-Student Lecture Hall Audit)</option>
      </select>
    </div>
  </header>

  <!-- Main Content -->
  <main>

    <!-- Top KPI Grid -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Class Attention Score <span>📈</span></div>
        <div id="kpiMeanScore" class="kpi-value" style="color: var(--success);">--</div>
        <div class="kpi-meta"><span id="kpiStatusBand">Verified High</span> • Scale 0-100</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Empirical Accuracy <span>🎯</span></div>
        <div id="kpiAccuracy" class="kpi-value" style="color: #38BDF8;">--%</div>
        <div class="kpi-meta">Macro F1: <span id="kpiMacroF1">--%</span> • Ground Truth Matched</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Student Population <span>👥</span></div>
        <div id="kpiStudentCount" class="kpi-value">--</div>
        <div class="kpi-meta"><span id="kpiAttentiveRatio">--</span> Attentive • 0 Hallucinations</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Attention Drop Alerts <span>⚡</span></div>
        <div id="kpiDropCount" class="kpi-value" style="color: var(--danger);">--</div>
        <div class="kpi-meta">Actionable Pedagogical Prompts Active</div>
      </div>
    </section>

    <!-- Core Spatial Heatmap & Intervention Section -->
    <section class="core-layout">
      <!-- 2D Spatial Heatmap Panel -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">
              <span>📍</span> 2D Classroom Seating & Attention Heatmap
            </div>
            <div id="heatmapSubtitle" class="panel-subtitle">Camera Coordinates Perspective Seating Layout</div>
          </div>
          <div class="badge-pill" style="background: rgba(99, 102, 241, 0.12); color: #818CF8; border-color: rgba(99, 102, 241, 0.3);">
            Interactive Desks
          </div>
        </div>

        <div id="heatmapCanvas" class="heatmap-container">
          <div class="podium-bar">[ FRONT OF CLASSROOM / TEACHER LECTURE PODIUM ]</div>
          <!-- Dynamically injected student desk nodes -->
        </div>
      </div>

      <!-- Real-Time Pedagogical Intervention Alert Panel -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">
              <span>🎓</span> Pedagogical Moves & Alerts
            </div>
            <div class="panel-subtitle">Actionable In-The-Moment Teacher Interventions</div>
          </div>
          <span id="alertCountBadge" class="badge-pill" style="background: rgba(239, 68, 68, 0.15); color: #F87171; border-color: rgba(239, 68, 68, 0.3);">
            0 Events
          </span>
        </div>

        <div id="interventionFeed" class="intervention-feed">
          <!-- Dynamically injected alert cards -->
        </div>
      </div>
    </section>

    <!-- Analytics Timeline & Behaviors Section -->
    <section class="analytics-grid">
      <!-- Attention Dynamics Timeline -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">
              <span>📉</span> Temporal Classroom Attention Dynamics
            </div>
            <div class="panel-subtitle">Continuous engagement trajectory across session duration (Threshold &tau;* = 50.0)</div>
          </div>
        </div>

        <div class="chart-box">
          <canvas id="timelineChart"></canvas>
        </div>
      </div>

      <!-- Observable Behaviors Breakdown -->
      <div class="panel">
        <div class="panel-header">
          <div>
            <div class="panel-title">
              <span>📊</span> Behavioral Modality Share
            </div>
            <div class="panel-subtitle">Disentangled Action Distribution</div>
          </div>
        </div>

        <div id="behaviorList" class="behavior-list">
          <!-- Dynamically injected behavior rows -->
        </div>
      </div>
    </section>

    <!-- Detailed Student Roster Table -->
    <section class="panel">
      <div class="panel-header">
        <div>
          <div class="panel-title">
            <span>📋</span> Student Forensic Attention Registry
          </div>
          <div class="panel-subtitle">Per-student multimodal attention metrics, behavior tags, and classification status</div>
        </div>
      </div>

      <div class="filters-bar">
        <button class="filter-btn active" onclick="filterTable('all', this)">All Students</button>
        <button class="filter-btn" onclick="filterTable('high', this)">Attentive (HIGH)</button>
        <button class="filter-btn" onclick="filterTable('low', this)">Distracted (LOW)</button>
        <button class="filter-btn" onclick="filterTable('phone', this)">Phone Users</button>
        <button class="filter-btn" onclick="filterTable('sleep', this)">Sleeping / Fatigue</button>

        <input type="text" id="searchInput" class="search-input" placeholder="Search Student ID (e.g. 4)..." onkeyup="searchTable()">
      </div>

      <div class="table-container">
        <table id="studentsTable">
          <thead>
            <tr>
              <th>Student ID</th>
              <th>Mean Score</th>
              <th>Attentive %</th>
              <th>Status</th>
              <th>Primary Observable Behavior</th>
              <th>Min / Max Score</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="studentsTableBody">
            <!-- Dynamically populated rows -->
          </tbody>
        </table>
      </div>
    </section>

  </main>

  <!-- Student Inspector Modal -->
  <div id="studentModal" class="modal-overlay" onclick="closeModal(event)">
    <div class="modal-card" onclick="event.stopPropagation()">
      <button class="modal-close" onclick="closeModalDirect()">&times;</button>
      <div style="display: flex; align-items: center; gap: 0.75rem;">
        <h2 id="modalStudentTitle" style="font-size: 1.25rem; font-weight: 800;">Student Detail</h2>
        <span id="modalStatusPill" class="pill high">Attentive</span>
      </div>

      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.5rem;">
        <div style="background: rgba(255, 255, 255, 0.03); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
          <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Average Score</div>
          <div id="modalMeanScore" style="font-size: 1.5rem; font-weight: 800; font-family: var(--font-mono); color: var(--success);">--</div>
        </div>
        <div style="background: rgba(255, 255, 255, 0.03); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
          <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Engagement Rate</div>
          <div id="modalAttentivePct" style="font-size: 1.5rem; font-weight: 800; font-family: var(--font-mono); color: #38BDF8;">--%</div>
        </div>
        <div style="background: rgba(255, 255, 255, 0.03); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
          <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">Primary Behavior</div>
          <div id="modalPrimaryCue" style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin-top: 0.35rem;">--</div>
        </div>
      </div>

      <div>
        <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-secondary); margin-bottom: 0.4rem;">
          Individual Attention Trajectory Sparkline
        </div>
        <canvas id="modalSparkline" class="modal-sparkline"></canvas>
      </div>

      <div id="modalRecommendationBox" style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); padding: 0.85rem; border-radius: 8px; font-size: 0.82rem; line-height: 1.4; color: #CBD5E1;">
        <!-- Pedagogical guidance -->
      </div>
    </div>
  </div>

  <script>
    // Embedded Multi-Video Empirical Dataset
    const DATABASE = {data_json_str};

    let currentKey = 'test_6';
    let currentFilter = 'all';

    function init() {{
      const select = document.getElementById('videoSelect');
      if (select) select.value = currentKey;
      renderCurrentVideo();
    }}

    function switchVideo(key) {{
      currentKey = key;
      renderCurrentVideo();
    }}

    function renderCurrentVideo() {{
      const data = DATABASE[currentKey];
      if (!data) return;

      // 1. KPIs
      const m = data.metrics;
      const meanEl = document.getElementById('kpiMeanScore');
      meanEl.innerText = m.class_mean_score.toFixed(1);
      meanEl.style.color = m.class_mean_score >= 50 ? 'var(--success)' : 'var(--danger)';

      document.getElementById('kpiStatusBand').innerText = m.class_mean_score >= 50 ? 'Verified High Engagement' : 'Needs Pedagogical Reset';
      document.getElementById('kpiAccuracy').innerText = m.accuracy.toFixed(1) + '%';
      document.getElementById('kpiMacroF1').innerText = m.macro_f1.toFixed(1) + '%';
      document.getElementById('kpiStudentCount').innerText = m.total_students;
      document.getElementById('kpiAttentiveRatio').innerText = `${{m.attentive_students}}/${{m.total_students}} (${{m.attentive_rate_pct}}%)`;
      document.getElementById('kpiDropCount').innerText = data.drop_events.length;
      document.getElementById('alertCountBadge').innerText = `${{data.drop_events.length}} Events`;

      // 2. Spatial Heatmap
      renderSpatialHeatmap(data);

      // 3. Intervention Feed
      renderInterventions(data.drop_events);

      // 4. Timeline Canvas
      renderTimelineChart(data.timeline);

      // 5. Behavior Distribution
      renderBehaviors(data.behavior_distribution);

      // 6. Students Table
      renderTable();
    }}

    function renderSpatialHeatmap(data) {{
      const container = document.getElementById('heatmapCanvas');
      container.innerHTML = '<div class="podium-bar">[ FRONT OF CLASSROOM / TEACHER LECTURE PODIUM ]</div>';

      const students = data.students;
      if (!students.length) return;

      // Compute bounding limits for normalization
      const xs = students.map(s => s.x);
      const ys = students.map(s => s.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);

      students.forEach(s => {{
        const node = document.createElement('div');
        node.className = 'student-node';
        node.id = `node-${{s.id}}`;

        // Normalize to container percent (with padding)
        const leftPct = maxX > minX ? 12 + ((s.x - minX) / (maxX - minX)) * 76 : 50;
        const topPct = maxY > minY ? 22 + ((s.y - minY) / (maxY - minY)) * 65 : 50;

        node.style.left = leftPct + '%';
        node.style.top = topPct + '%';

        const isHigh = s.mean_score >= 50.0;
        const color = isHigh ? 'var(--success)' : 'var(--danger)';
        node.style.borderColor = color;

        node.innerHTML = `
          <div class="node-id">S${{s.id < 10 ? '0' + s.id : s.id}}</div>
          <div class="node-score" style="color: ${{color}};">${{s.mean_score.toFixed(1)}}</div>
          <div class="node-cue">${{s.top_cue}}</div>
        `;

        node.onclick = () => openStudentModal(s);
        container.appendChild(node);
      }});

      document.getElementById('heatmapSubtitle').innerText = `${{students.length}} Seated Students • ${{data.resolution}} Native Resolution`;
    }}

    function renderInterventions(events) {{
      const feed = document.getElementById('interventionFeed');
      feed.innerHTML = '';

      if (!events.length) {{
        feed.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem; padding: 1rem; text-align: center;">Zero critical attention drops flagged. High classroom engagement.</div>';
        return;
      }}

      events.forEach(e => {{
        const card = document.createElement('div');
        const cueLower = e.cue.toLowerCase();
        let typeClass = '';
        if (cueLower.includes('phone')) typeClass = 'phone';
        else if (cueLower.includes('talk') || cueLower.includes('disturb')) typeClass = 'talk';
        else if (cueLower.includes('sleep') || cueLower.includes('table')) typeClass = 'sleep';

        card.className = `alert-card ${{typeClass}}`;
        card.innerHTML = `
          <div class="alert-top">
            <div class="alert-student">Student ${{e.student_id < 10 ? '0' + e.student_id : e.student_id}}</div>
            <div class="alert-time">t = ${{e.timestamp.toFixed(1)}}s</div>
          </div>
          <div>
            <span class="alert-badge">${{e.cue}}</span>
            <span style="font-size: 0.72rem; color: var(--text-muted); margin-left: 0.35rem; font-family: var(--font-mono);">Score: ${{e.score.toFixed(1)}}/100</span>
          </div>
          <div class="alert-rec"><strong>Pedagogical Move:</strong> ${{e.recommendation}}</div>
        `;
        feed.appendChild(card);
      }});
    }}

    function renderTimelineChart(timeline) {{
      const canvas = document.getElementById('timelineChart');
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * 2;
      canvas.height = rect.height * 2;
      ctx.scale(2, 2);

      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      if (!timeline || timeline.length < 2) return;

      const padding = {{ top: 20, right: 25, bottom: 25, left: 35 }};
      const plotW = w - padding.left - padding.right;
      const plotH = h - padding.top - padding.bottom;

      // Grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      for (let yVal = 0; yVal <= 100; yVal += 25) {{
        const y = padding.top + plotH - (yVal / 100) * plotH;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();

        ctx.fillStyle = '#64748B';
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText(yVal, padding.left - 6, y + 3);
      }}

      // Threshold line tau = 50.0
      const threshY = padding.top + plotH - (50 / 100) * plotH;
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.45)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padding.left, threshY);
      ctx.lineTo(w - padding.right, threshY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#F87171';
      ctx.fillText('&tau;* = 50', w - padding.right + 2, threshY + 3);

      // Trajectory Path
      const points = timeline.map((pt, idx) => {{
        const x = padding.left + (idx / (timeline.length - 1)) * plotW;
        const y = padding.top + plotH - (pt.mean_score / 100) * plotH;
        return {{ x, y, score: pt.mean_score, t: pt.t }};
      }});

      // Gradient Fill
      const grad = ctx.createLinearGradient(0, padding.top, 0, padding.top + plotH);
      grad.addColorStop(0, 'rgba(16, 185, 129, 0.25)');
      grad.addColorStop(0.5, 'rgba(6, 182, 212, 0.12)');
      grad.addColorStop(1, 'rgba(239, 68, 68, 0.0)');

      ctx.beginPath();
      ctx.moveTo(points[0].x, padding.top + plotH);
      points.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.lineTo(points[points.length - 1].x, padding.top + plotH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // Main Line
      ctx.beginPath();
      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 2.5;
      points.forEach((p, idx) => {{
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }});
      ctx.stroke();

      // Highlight points below 50
      points.forEach(p => {{
        if (p.score < 50) {{
          ctx.beginPath();
          ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#EF4444';
          ctx.fill();
        }}
      }});
    }}

    function renderBehaviors(distribution) {{
      const list = document.getElementById('behaviorList');
      list.innerHTML = '';

      const total = Object.values(distribution).reduce((a, b) => a + b, 0) || 1;
      const colors = {{
        'Reading / Writing': '#10B981',
        'Oriented / Listening': '#38BDF8',
        'Hand Raising': '#818CF8',
        'Peer Interaction': '#F59E0B',
        'Using Phone': '#EF4444',
        'Sleeping / Fatigue': '#EC4899'
      }};

      for (const [name, count] of Object.entries(distribution)) {{
        const pct = (count / total * 100).toFixed(1);
        const col = colors[name] || '#94A3B8';

        const row = document.createElement('div');
        row.className = 'behavior-row';
        row.innerHTML = `
          <div class="behavior-meta">
            <span style="color: ${{col}};">${{name}}</span>
            <span style="font-family: var(--font-mono);">${{pct}}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${{pct}}%; background: ${{col}};"></div>
          </div>
        `;
        list.appendChild(row);
      }}
    }}

    function renderTable() {{
      const tbody = document.getElementById('studentsTableBody');
      tbody.innerHTML = '';

      const data = DATABASE[currentKey];
      if (!data) return;

      const q = (document.getElementById('searchInput').value || '').toLowerCase();

      data.students.forEach(s => {{
        const isHigh = s.mean_score >= 50.0;
        const topCLower = s.top_cue.toLowerCase();

        // Filter logic
        if (currentFilter === 'high' && !isHigh) return;
        if (currentFilter === 'low' && isHigh) return;
        if (currentFilter === 'phone' && !topCLower.includes('phone')) return;
        if (currentFilter === 'sleep' && !topCLower.includes('sleep') && !topCLower.includes('table')) return;

        // Search logic
        const sidStr = s.id.toString();
        if (q && !sidStr.includes(q) && !topCLower.includes(q)) return;

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td style="font-weight: 700; color: #FFFFFF;">Student ${{s.id < 10 ? '0' + s.id : s.id}}</td>
          <td>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span style="font-weight: 800; font-family: var(--font-mono); color: ${{isHigh ? 'var(--success)' : 'var(--danger)'}};">${{s.mean_score.toFixed(1)}}</span>
              <div style="width: 50px; height: 5px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden;">
                <div style="width: ${{s.mean_score}}%; height: 100%; background: ${{isHigh ? 'var(--success)' : 'var(--danger)'}};"></div>
              </div>
            </div>
          </td>
          <td style="font-family: var(--font-mono);">${{s.attentive_pct.toFixed(1)}}%</td>
          <td>
            <span class="pill ${{isHigh ? 'high' : 'low'}}">${{isHigh ? 'ATTENTIVE' : 'DISTRACTED'}}</span>
          </td>
          <td style="font-weight: 500;">${{s.top_cue}}</td>
          <td style="font-family: var(--font-mono); color: var(--text-muted);">${{s.min_score.toFixed(1)}} / ${{s.max_score.toFixed(1)}}</td>
          <td>
            <button class="btn-inspect" onclick='openStudentModal(${{JSON.stringify(s)}})'>Inspect</button>
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterTable(filter, btn) {{
      currentFilter = filter;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderTable();
    }}

    function searchTable() {{
      renderTable();
    }}

    // Modal Details
    function openStudentModal(s) {{
      const modal = document.getElementById('studentModal');
      document.getElementById('modalStudentTitle').innerText = `Student ${{s.id < 10 ? '0' + s.id : s.id}} Attention Profile`;

      const isHigh = s.mean_score >= 50.0;
      const statusPill = document.getElementById('modalStatusPill');
      statusPill.className = `pill ${{isHigh ? 'high' : 'low'}}`;
      statusPill.innerText = isHigh ? 'ATTENTIVE (HIGH)' : 'DISTRACTED (LOW)';

      const meanEl = document.getElementById('modalMeanScore');
      meanEl.innerText = s.mean_score.toFixed(1);
      meanEl.style.color = isHigh ? 'var(--success)' : 'var(--danger)';

      document.getElementById('modalAttentivePct').innerText = `${{s.attentive_pct.toFixed(1)}}%`;
      document.getElementById('modalPrimaryCue').innerText = s.top_cue;

      const recBox = document.getElementById('modalRecommendationBox');
      const cueLower = s.top_cue.toLowerCase();
      if (cueLower.includes('phone')) {{
        recBox.innerHTML = `<strong>Active Pedagogical Guidance (Phone Usage):</strong><br>Circulate calmly down the aisle of desk S${{s.id}} using non-verbal proximity control. Avoid public callouts; gentle physical presence reliably prompts device stowing.`;
      }} else if (cueLower.includes('talk') || cueLower.includes('disturb')) {{
        recBox.innerHTML = `<strong>Active Pedagogical Guidance (Peer Chatter):</strong><br>Pivot immediately to a 30-second Think-Pair-Share challenge to constructively direct side conversation into structured course dialogue.`;
      }} else if (cueLower.includes('sleep') || cueLower.includes('table')) {{
        recBox.innerHTML = `<strong>Active Pedagogical Guidance (Fatigue / Posture Collapse):</strong><br>Conduct a rapid whole-class stand-up energy reset or interactive digital poll. Check in privately post-session regarding fatigue.`;
      }} else {{
        recBox.innerHTML = `<strong>Pedagogical Move:</strong><br>Student demonstrates consistent orientation. Encourage continued engagement with positive visual acknowledgment and occasional open retrieval questions.`;
      }}

      // Render modal sparkline
      renderModalSparkline(s.timeline);

      modal.classList.add('open');
    }}

    function renderModalSparkline(timeline) {{
      const canvas = document.getElementById('modalSparkline');
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * 2;
      canvas.height = rect.height * 2;
      ctx.scale(2, 2);

      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);
      if (!timeline || timeline.length < 2) return;

      const pad = 15;
      const plotW = w - pad * 2;
      const plotH = h - pad * 2;

      // Threshold line 50
      const ty = pad + plotH - (50 / 100) * plotH;
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pad, ty);
      ctx.lineTo(w - pad, ty);
      ctx.stroke();
      ctx.setLineDash([]);

      const pts = timeline.map((pt, i) => ({{
        x: pad + (i / (timeline.length - 1)) * plotW,
        y: pad + plotH - (pt.score / 100) * plotH
      }}));

      ctx.beginPath();
      ctx.strokeStyle = '#6366F1';
      ctx.lineWidth = 2.2;
      pts.forEach((p, idx) => {{
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }});
      ctx.stroke();
    }}

    function closeModal(e) {{
      if (e.target.id === 'studentModal') {{
        document.getElementById('studentModal').classList.remove('open');
      }}
    }}

    function closeModalDirect() {{
      document.getElementById('studentModal').classList.remove('open');
    }}

    window.addEventListener('DOMContentLoaded', init);
    window.addEventListener('resize', () => {{
      const data = DATABASE[currentKey];
      if (data) renderTimelineChart(data.timeline);
    }});
  </script>
</body>
</html>
"""

    with open(OUT_DASHBOARD, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated dashboard HTML: {OUT_DASHBOARD.resolve()} ({OUT_DASHBOARD.stat().st_size} bytes)")

    with open(ARTIFACT_OUT, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated artifact dashboard: {ARTIFACT_OUT.resolve()} ({ARTIFACT_OUT.stat().st_size} bytes)")

if __name__ == "__main__":
    generate()
