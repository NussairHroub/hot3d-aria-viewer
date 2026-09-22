/* HOT3D Aria viewer — shared runtime.
 *
 * Every page loads this file. It renders the header and theme control, decodes the payload
 * format the extraction scripts emit, and holds the small maths the viewers share so the same
 * pose convention is used everywhere.
 *
 * Payload conventions (see scripts/hot3d_extract.py):
 *   - numeric arrays are base64 little-endian float32, masks are base64 uint8
 *   - a frame with no annotation is NaN (uint8 masks use 255 for "not annotated")
 *   - rotations are quaternions in w,x,y,z order, mapping object/device -> world
 *   - frame i of a sequence payload is frame i of that sequence's rgb mp4
 */

const HOT3D = (() => {
  const PAGES = [
    ['index.html', 'Overview'],
    ['recordings.html', 'Recordings'],
    ['sequence.html', 'Sequence viewer'],
    ['scene3d.html', '3D scene'],
    ['hands.html', 'Hands'],
    ['objects.html', 'Objects'],
    ['cameras.html', 'Cameras'],
    ['gaze.html', 'Eye gaze'],
    ['quality.html', 'Annotation quality'],
    ['data.html', 'About the data'],
  ];

  /* ---------- chrome ---------- */

  function mountHeader() {
    const here = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
    const head = document.createElement('header');
    head.className = 'site-head';
    const nav = PAGES.map(([href, label]) => {
      const cur = href.toLowerCase() === here ? ' aria-current="page"' : '';
      return `<a href="${href}"${cur}>${label}</a>`;
    }).join('');
    head.innerHTML =
      `<div class="inner">
         <a class="brand" href="index.html">HOT3D Aria <span class="dot">viewer</span></a>
         <nav class="nav" aria-label="Sections">${nav}</nav>
         <button class="theme-btn" type="button" id="themeBtn" aria-label="Switch colour theme">◐</button>
       </div>`;
    document.body.prepend(head);
    const btn = head.querySelector('#themeBtn');
    btn.addEventListener('click', () => {
      const root = document.documentElement;
      const now = root.getAttribute('data-theme');
      const dark = now ? now === 'dark'
        : matchMedia('(prefers-color-scheme: dark)').matches;
      const next = dark ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('hot3d-theme', next); } catch (e) { /* private mode */ }
    });
  }

  function restoreTheme() {
    try {
      const saved = localStorage.getItem('hot3d-theme');
      if (saved === 'dark' || saved === 'light') document.documentElement.setAttribute('data-theme', saved);
    } catch (e) { /* storage can throw; the media query still themes the page */ }
  }

  function mountFooter(extra) {
    const f = document.createElement('div');
    f.className = 'site-foot wrap';
    f.innerHTML =
      `${extra || ''} Built from the HOT3D release (Meta Platforms Technologies) —
       <a href="https://facebookresearch.github.io/hot3d/">project page</a>,
       <a href="https://arxiv.org/abs/2411.19167">paper</a>. Figures on these pages are computed
       from the released annotation files; nothing is re-estimated.
       Video, stills, object models and annotations shown here are modified HOT3D material under
       <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>, and
       <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> for the
       hand annotations — see
       <a href="https://github.com/NussairHroub/hot3d-aria-viewer/blob/main/DATA_LICENSE.md">the
       licence notice</a> for what was changed.`;
    document.body.append(f);
  }

  /* ---------- payload decoding ---------- */

  function bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  /** base64 float32 -> Float32Array (optionally reshaped as an array of rows) */
  function f32(b64, cols) {
    const raw = bytes(b64);
    const flat = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    if (!cols || cols === 1) return flat;
    const rows = flat.length / cols, out = new Array(rows);
    for (let i = 0; i < rows; i++) out[i] = flat.subarray(i * cols, i * cols + cols);
    return out;
  }
  /** base64 uint8 -> Uint8Array (1 true, 0 false, 255 not annotated) */
  const u8 = (b64) => bytes(b64);

  /* A payload is several megabytes, so a slow link can look like a hung page. Every fetch that
     goes through json() reports its bytes to one shared bar under the header: the reader sees
     movement, and a stall is visibly a stall rather than a mystery. */
  const inflight = new Map();
  let bar = null;

  function progressBar() {
    if (bar) return bar;
    bar = document.createElement('div');
    bar.id = 'hot3d-progress';
    bar.setAttribute('role', 'status');
    bar.innerHTML = '<span class="pg-track"><span class="pg-fill"></span></span><span class="pg-text"></span>';
    Object.assign(bar.style, {
      position: 'fixed', left: '0', right: '0', top: '0', zIndex: '60',
      display: 'none', alignItems: 'center', gap: '10px', padding: '5px 14px',
      font: '12px/1.4 var(--mono, monospace)', color: 'var(--ink-2)',
      background: 'var(--surface)', borderBottom: '1px solid var(--line)',
    });
    const track = bar.querySelector('.pg-track');
    Object.assign(track.style, { flex: '1', height: '3px', background: 'var(--surface-2)', borderRadius: '2px', overflow: 'hidden' });
    Object.assign(bar.querySelector('.pg-fill').style, { display: 'block', height: '100%', width: '0%', background: 'var(--accent)' });
    document.body.append(bar);
    return bar;
  }

  function renderProgress() {
    const b = progressBar();
    if (!inflight.size) { b.style.display = 'none'; return; }
    let loaded = 0, total = 0, unknown = false;
    for (const s of inflight.values()) {
      loaded += s.loaded;
      if (s.total) total += s.total; else unknown = true;
    }
    const mb = (n) => (n / 1e6).toFixed(1);
    b.style.display = 'flex';
    b.querySelector('.pg-fill').style.width =
      total && !unknown ? `${Math.min(100, 100 * loaded / total).toFixed(1)}%` : '100%';
    b.querySelector('.pg-text').textContent = total && !unknown
      ? `loading ${mb(loaded)} / ${mb(total)} MB`
      : `loading ${mb(loaded)} MB`;
  }

  async function fetchJsonWithProgress(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    const total = Number(res.headers.get('content-length')) || 0;
    if (!res.body || !res.body.getReader) return res.json();   // older browsers: no stream, no bar
    const state = { loaded: 0, total };
    inflight.set(path, state);
    renderProgress();
    try {
      const reader = res.body.getReader();
      const chunks = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        state.loaded += value.length;
        renderProgress();
      }
      let size = 0;
      for (const c of chunks) size += c.length;
      const buf = new Uint8Array(size);
      let at = 0;
      for (const c of chunks) { buf.set(c, at); at += c.length; }
      return JSON.parse(new TextDecoder().decode(buf));
    } finally {
      inflight.delete(path);
      renderProgress();
    }
  }

  const cache = new Map();
  async function json(path) {
    if (!cache.has(path)) {
      // a failed fetch must not be cached as a permanent failure: drop it so a retry can work
      const p = fetchJsonWithProgress(path).catch(err => { cache.delete(path); throw err; });
      cache.set(path, p);
    }
    return cache.get(path);
  }
  const index = () => json('data/index.json');
  const sequence = (seq) => json(`data/seq/${seq}.json`);
  /** Hand annotations live in their own file: the release licenses them CC BY-NC-SA, while the
   *  rest of a recording is CC BY-SA, so the two are never merged into one payload.
   *  Shape: {seq, F, hands: {left, right}, masks: {hand_visible, hand_pose_available}, summary}.
   *  MANO parameters are not republished here (their use is gated by the SMPL-X/MANO licence). */
  const hands = (seq) => json(`data/hands/${seq}.hands.json`);

  /* ---------- pose maths (quaternions are w,x,y,z) ---------- */

  function quatToMat3(q) {
    const [w, x, y, z] = q;
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ];
  }
  /** world point of a local point under (translation t, rotation q) */
  function applyPose(t, q, p) {
    const m = quatToMat3(q);
    return [
      t[0] + m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
      t[1] + m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
      t[2] + m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
    ];
  }
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  /** running length of a (F,3) track, skipping frames that are not annotated */
  function pathLength(T) {
    let d = 0;
    for (let i = 1; i < T.length; i++) {
      const a = T[i - 1], b = T[i];
      if (!isFinite(a[0]) || !isFinite(b[0])) continue;
      d += dist(a, b);
    }
    return d;
  }

  /* ---------- formatting ---------- */

  const fmt = {
    m: (v, d = 2) => (isFinite(v) ? `${v.toFixed(d)} m` : '—'),
    cm: (v, d = 1) => (isFinite(v) ? `${(v * 100).toFixed(d)} cm` : '—'),
    pct: (v, d = 1) => (isFinite(v) ? `${(v * 100).toFixed(d)}%` : '—'),
    s: (v, d = 2) => (isFinite(v) ? `${v.toFixed(d)} s` : '—'),
    deg: (v, d = 1) => (isFinite(v) ? `${(v * 180 / Math.PI).toFixed(d)}°` : '—'),
    n: (v) => (isFinite(v) ? v.toLocaleString('en-US') : '—'),
    clock: (v) => {
      if (!isFinite(v)) return '—';
      const m = Math.floor(v / 60), s = v - 60 * m;
      return `${m}:${s.toFixed(2).padStart(5, '0')}`;
    },
    /** object and participant ids read better with the underscore kept but not wrapped */
    seq: (s) => s.replace('_', '‑'),
  };

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  /** the colour used for a given annotation channel, on every page */
  const channel = {
    object: () => css('--c-object'),
    hand: () => css('--c-hand'),
    gaze: () => css('--c-gaze'),
    headset: () => css('--c-headset'),
    points: () => css('--c-points'),
  };

  /* ---------- tiny plotting: one shared line chart ---------- */

  /**
   * Draw one or more series against the frame timeline into a canvas.
   * series: [{v: Float32Array, color, label, dash}] — NaN gaps are left open, never bridged.
   */
  function linePlot(canvas, t, series, opts = {}) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 600, h = canvas.clientHeight || 160;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const padL = opts.padL ?? 46, padR = 8, padT = 8, padB = 20;
    let lo = opts.min, hi = opts.max;
    if (lo === undefined || hi === undefined) {
      lo = Infinity; hi = -Infinity;
      for (const s of series) for (const v of s.v) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      const pad = (hi - lo) * .08 || .5; lo -= pad; hi += pad;
    }
    const x = (i) => padL + (w - padL - padR) * (t[i] - t[0]) / Math.max(t[t.length - 1] - t[0], 1e-6);
    const y = (v) => h - padB - (h - padT - padB) * (v - lo) / Math.max(hi - lo, 1e-9);

    ctx.strokeStyle = css('--line'); ctx.fillStyle = css('--muted');
    ctx.font = `11px ${css('--mono') || 'monospace'}`;
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {                      // horizontal grid + value labels
      const v = lo + (hi - lo) * g / 4, yy = Math.round(y(v)) + .5;
      ctx.globalAlpha = .6; ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
      ctx.globalAlpha = 1; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText((opts.fmt ? opts.fmt(v) : v.toFixed(2)), padL - 6, yy);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let g = 0; g <= 4; g++) {                      // time axis
      const i = Math.round((t.length - 1) * g / 4);
      ctx.fillText(`${t[i].toFixed(0)}s`, x(i), h - padB + 4);
    }
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < s.v.length; i++) {
        const v = s.v[i];
        if (!isFinite(v)) { pen = false; continue; }
        if (!pen) { ctx.moveTo(x(i), y(v)); pen = true; } else ctx.lineTo(x(i), y(v));
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    return { x, y, padL, padR, padT, padB, w, h, lo, hi };
  }

  /** vertical playhead for a chart drawn by linePlot */
  function playhead(canvas, geom, t, i) {
    const ctx = canvas.getContext('2d');
    ctx.save();
    ctx.strokeStyle = css('--ink'); ctx.globalAlpha = .45; ctx.lineWidth = 1;
    const xx = Math.round(geom.x(i)) + .5;
    ctx.beginPath(); ctx.moveTo(xx, geom.padT); ctx.lineTo(xx, geom.h - geom.padB); ctx.stroke();
    ctx.restore();
  }

  function init(footerExtra) {
    restoreTheme();
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => { mountHeader(); mountFooter(footerExtra); });
    } else { mountHeader(); mountFooter(footerExtra); }
  }

  return { init, mountHeader, mountFooter, restoreTheme, f32, u8, json, index, sequence, hands,
           quatToMat3, applyPose, dist, pathLength, fmt, css, channel, linePlot, playhead, PAGES };
})();

HOT3D.restoreTheme();
