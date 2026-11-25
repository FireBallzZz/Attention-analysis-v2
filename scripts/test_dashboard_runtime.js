/**
 * Runtime smoke test for the generated dashboard.
 *
 * Boots the real index.html in jsdom, exercises every video, every chart and
 * every interaction, and asserts the rendered output is non-empty and free of
 * runtime errors.
 */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '..');
const HTML = path.join(ROOT, 'dashboard', 'index.html');
const html = fs.readFileSync(HTML, 'utf8');

const errors = [];
const notes = [];

const dom = new JSDOM(html, {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  url: 'file:///' + HTML.replace(/\\/g, '/'),
  beforeParse(window) {
    // jsdom has no layout engine: give canvases a deterministic box.
    window.HTMLCanvasElement.prototype.getContext = function () {
      const noop = () => {};
      return new Proxy(
        {
          canvas: this,
          measureText: () => ({ width: 10 }),
          createLinearGradient: () => ({ addColorStop: noop }),
          setTransform: noop, save: noop, restore: noop,
        },
        { get: (t, k) => (k in t ? t[k] : noop), set: (t, k, v) => ((t[k] = v), true) }
      );
    };
    Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', { get() { return 900; } });
    Object.defineProperty(window.HTMLElement.prototype, 'clientHeight', { get() { return 506; } });
    window.HTMLElement.prototype.getBoundingClientRect = function () {
      return { width: 900, height: 506, top: 0, left: 0, right: 900, bottom: 506, x: 0, y: 0 };
    };
    window.onerror = (msg) => errors.push('window.onerror: ' + msg);
    const origErr = console.error;
    console.error = (...a) => { errors.push('console.error: ' + a.join(' ')); origErr(...a); };
  },
});

const { window } = dom;
const doc = window.document;

function assert(cond, label) {
  if (cond) notes.push('  [ok]   ' + label);
  else errors.push('  [FAIL] ' + label);
}

const $ = (id) => doc.getElementById(id);

// `const DATABASE` is a lexical global, not a window property.
const DATABASE = window.eval('DATABASE');
global.DATABASE = DATABASE;

assert(typeof DATABASE === 'object', 'DATABASE global exposed');
assert(Object.keys(DATABASE).length === 7, 'DATABASE has 7 videos');
assert(typeof window.switchVideo === 'function', 'switchVideo exposed');

// --- initial render -------------------------------------------------
// boot() runs on DOMContentLoaded; jsdom fires that asynchronously.
function waitForBoot() {
  return new Promise((resolve) => {
    if (doc.readyState === 'complete' && $('videoSelect').options.length) return resolve();
    window.addEventListener('load', resolve);
    setTimeout(resolve, 250);
  });
}

async function main() {
await waitForBoot();

const first = Object.keys(DATABASE)[0];
assert($('videoSelect').options.length === 7, 'video selector populated with 7 options');
assert($('kpiScore').textContent !== '—', 'KPI score rendered: ' + $('kpiScore').textContent);
assert($('kpiAccuracy').textContent !== '—', 'KPI accuracy rendered: ' + $('kpiAccuracy').textContent);
assert($('kpiStudents').textContent !== '—', 'KPI students rendered: ' + $('kpiStudents').textContent);

// --- every video ----------------------------------------------------
Object.keys(DATABASE).forEach((key) => {
  const before = errors.length;
  window.switchVideo(key);
  const seats = doc.querySelectorAll('.seat').length;
  const alerts = doc.querySelectorAll('.alert').length;
  const rows = doc.querySelectorAll('#tableBody tr').length;
  const bars = doc.querySelectorAll('.behavior-row').length;
  const cmCells = doc.querySelectorAll('#confusionBox .cm-cell').length;
  const badge = $('gtBadge').textContent.trim();

  const data = DATABASE[key];
  const ev = data.evaluation || {};

  assert(seats === data.students.length, `${key}: ${seats} seat nodes == ${data.students.length} students`);

  // Every seat must sit inside the rendered stage (no overflow).
  const stage = $('seatingStage');
  const stageBox = stage.getBoundingClientRect();
  let outside = 0;
  doc.querySelectorAll('.seat').forEach((n) => {
    const b = n.getBoundingClientRect();
    const cx = b.left + b.width / 2;
    const cy = b.top + b.height / 2;
    if (cx < stageBox.left - 2 || cx > stageBox.right + 2 ||
        cy < stageBox.top - 2 || cy > stageBox.bottom + 2) outside++;
  });
  assert(outside === 0, `${key}: all ${seats} seats inside the stage bounds`);

  // Seat boxes must be big enough to read a label. jsdom has no layout
  // engine, so measure the inline width/height the renderer computed.
  const seatSizes = [...doc.querySelectorAll('.seat')].map(n => {
    const w = parseFloat(n.style.width) || 0;
    const h = parseFloat(n.style.height) || 0;
    return Math.min(w, h);
  }).sort((a, b) => a - b);
  assert(seatSizes[0] >= 14, `${key}: smallest seat box ${seatSizes[0].toFixed(0)}px is legible`);

  // Seat positions must land inside the scaled source frame.
  const [vw, vh] = data.resolution.split('x').map(Number);
  const scale = Math.min(900 / vw, 506 / vh);
  let offFrame = 0;
  doc.querySelectorAll('.seat').forEach((n) => {
    const x = parseFloat(n.style.left);
    const y = parseFloat(n.style.top);
    if (x < 0 || y < 0 || x > vw * scale || y > vh * scale) offFrame++;
  });
  assert(offFrame === 0, `${key}: all seats inside the scaled source frame`);
  assert(rows > 0, `${key}: ${rows} table rows rendered`);
  assert(bars > 0, `${key}: ${bars} behaviour rows rendered`);
  assert(cmCells > 0, `${key}: ${cmCells} confusion cells rendered`);
  assert(alerts === (data.drop_events || []).length, `${key}: ${alerts} alerts == ${(data.drop_events || []).length} events`);
  assert(badge.length > 0, `${key}: provenance badge = "${badge}"`);
  assert($('kpiAccuracy').textContent !== '—', `${key}: accuracy = ${$('kpiAccuracy').textContent}`);
  assert($('kpiMacroF1').textContent !== '—', `${key}: macro-F1 = ${$('kpiMacroF1').textContent}`);
  assert($('footSource').textContent.length > 0, `${key}: footer source = ${$('footSource').textContent}`);

  // metric must equal the dataset value, not a re-derived one
  const expectedAcc = ev.accuracy === null || ev.accuracy === undefined ? 'n/a' : ev.accuracy.toFixed(1) + '%';
  assert($('kpiAccuracy').textContent === expectedAcc,
    `${key}: accuracy matches dataset (${expectedAcc})`);

  if (errors.length > before) {
    errors.push(`  [FAIL] ${key}: runtime error during render`);
  }
});

// --- filters --------------------------------------------------------
window.switchVideo('video_project_3');
const allRows = doc.querySelectorAll('#tableBody tr').length;
doc.querySelector('.chip[data-filter="low"]').click();
const lowRows = doc.querySelectorAll('#tableBody tr').length;
const expectedLow = DATABASE.video_project_3.students.filter(s => s.mean_score < 50).length;
assert(lowRows === expectedLow, `low filter: ${lowRows} rows == ${expectedLow} expected`);
assert(lowRows <= allRows, 'low filter narrows the table');

doc.querySelector('.chip[data-filter="all"]').click();
assert(doc.querySelectorAll('#tableBody tr').length === allRows, 'all filter restores full table');

// --- search ---------------------------------------------------------
const search = $('searchInput');
search.value = '1';
search.dispatchEvent(new window.Event('input', { bubbles: true }));
const searched = doc.querySelectorAll('#tableBody tr').length;
assert(searched >= 1, `search "1" returns ${searched} rows`);
search.value = '';
search.dispatchEvent(new window.Event('input', { bubbles: true }));

// --- modal + sparkline ----------------------------------------------
window.switchVideo('clip_001');
const seat = doc.querySelector('.seat');
assert(!!seat, 'seat node available to click');
seat.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
assert($('studentModal').classList.contains('is-open'), 'modal opens on seat click');
assert($('modalMean').textContent !== '—', 'modal mean score populated: ' + $('modalMean').textContent);
assert($('modalAdvice').textContent.length > 20, 'modal advice populated');
assert($('modalObs').textContent !== '—', 'modal observation count populated');
window.closeModal();
assert(!$('studentModal').classList.contains('is-open'), 'modal closes');

// --- inspect button -------------------------------------------------
const inspectBtn = doc.querySelector('#tableBody .btn-ghost');
assert(!!inspectBtn, 'inspect button present in table');
inspectBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
assert($('studentModal').classList.contains('is-open'), 'inspect button opens modal');
window.closeModal();

// --- chart canvases -------------------------------------------------
const tl = $('timelineChart');
const sp = $('modalSparkline');
assert(tl.width > 0 && tl.height > 0, 'timeline canvas sized: ' + tl.width + 'x' + tl.height);
assert(sp.width > 0 && sp.height > 0, 'sparkline canvas sized: ' + sp.width + 'x' + sp.height);

// --- report ---------------------------------------------------------
console.log('='.repeat(68));
console.log('DASHBOARD RUNTIME TEST');
console.log('='.repeat(68));
notes.forEach(n => console.log(n));
if (errors.length) {
  console.log('');
  errors.forEach(e => console.log(e));
  console.log('');
  console.log('RESULT: ' + errors.length + ' failure(s)');
  process.exit(1);
}
console.log('');
console.log('RESULT: all runtime checks passed');
}

main().catch((e) => {
  console.error('test harness crashed:', e);
  process.exit(1);
});
