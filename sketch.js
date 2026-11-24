/* ───────────────────────────────────────────────────────────────────────────
   EYE CONTACT
   A drawing made through the time you spend in front of your computer.

   1. Tracker          MediaPipe Face Landmarker → eye / head / blink numbers
   2. SessionRecorder  timestamped samples, gaps, blink events (no video, ever)
   3. Timeline         recorded samples → a clean "presence clock" of gaze states
   4. Generator        particle system: a fixed attractor (the computer's gaze)
                       against a variable force (the human's gaze)
   5. Renderer         strokes → canvas (p5), PNG and SVG
   6. App              screens, session flow, export, development panel
   ─────────────────────────────────────────────────────────────────────────── */
'use strict';

const CFG = {
  // drawing sheet (logical units; 4:5 portrait) and attractor position
  W: 2400, H: 3000, CX: 1200, CY: 1380, UNIT: 1000,
  PAPER: '#f4f1ea', INK: '#14120e',

  // tracking
  SAMPLE_HZ: 12,            // face analysis rate while a face is present
  ABSENT_DIVIDER: 4,        // analyse 1 in N ticks while nobody is there
  GAP_MS: 2500,             // a silence longer than this is "tracking unavailable"
  BLINK_ON: 0.55, BLINK_OFF: 0.35,
  CALIB_SECONDS: 4,

  // gaze model: iris offset (ratio of half eye width) + head angle → radians off-camera
  EYE_GX: 1.05, EYE_GY: 2.0, HEAD_G: 1.0,
  // sign conventions (verify with the live readout in dev mode: +x = toward screen right, +y = up)
  YAW_SIGN: 1, PITCH_SIGN: 1, ROLL_SIGN: 1,
  CAM_IN: 0.10, CAM_OUT: 0.42,   // radians: fully "with the camera" → fully "elsewhere"

  // timeline
  BIN: 0.2,                 // seconds per bin of the presence clock
  EXPORT_SCALE: 2,          // PNG = 4800 × 6000
};

/* ═══════════════════════════ utilities ═══════════════════════════ */
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const $ = (id) => document.getElementById(id);

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussFrom(rng) {
  return () => {
    let u = 0; while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * rng());
  };
}
function median(a) {
  if (!a.length) return 0;
  const s = a.slice().sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function fmtDuration(ms) {
  const s = Math.round(ms / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  if (h) return `${h} h ${String(m).padStart(2, '0')} min`;
  if (m) return `${m} min ${String(r).padStart(2, '0')} s`;
  return `${r} s`;
}
const pad2 = (n) => String(n).padStart(2, '0');
function fmtClock(ms) {
  const s = Math.floor(ms / 1000);
  return `${pad2(Math.floor(s / 3600))}:${pad2(Math.floor((s % 3600) / 60))}:${pad2(s % 60)}`;
}

/* ═══════════════════════════ 2 · recorder ═══════════════════════════
   Samples, struct-of-arrays. Fields per sample:
   ix, iy   iris offset (screen-oriented: +x right, +y up), ratio of half eye width
   yaw, pitch, roll   head orientation, radians
   blink    eyelid closure 0..1
   state    0 = nobody in frame, 1 = face present, 2 = tracking unavailable      */
class Recorder {
  constructor() {
    this.cap = 1 << 15; this.n = 0;
    this.t = new Float64Array(this.cap);
    this.s = new Uint8Array(this.cap);
    this.f = new Float32Array(this.cap * 6);
    this.blinks = [];   // [t_ms, duration_ms] flattened
  }
  _grow() {
    this.cap *= 2;
    const t = new Float64Array(this.cap); t.set(this.t); this.t = t;
    const s = new Uint8Array(this.cap); s.set(this.s); this.s = s;
    const f = new Float32Array(this.cap * 6); f.set(this.f); this.f = f;
  }
  push(t, s, ix = 0, iy = 0, yaw = 0, pitch = 0, roll = 0, blink = 0) {
    if (this.n === this.cap) this._grow();
    const i = this.n++, o = i * 6;
    this.t[i] = t; this.s[i] = s;
    this.f[o] = ix; this.f[o + 1] = iy; this.f[o + 2] = yaw; this.f[o + 3] = pitch; this.f[o + 4] = roll; this.f[o + 5] = blink;
  }
  toJSON() {
    const r = (v) => Math.round(v * 1e4) / 1e4;
    return {
      n: this.n,
      t: Array.from(this.t.subarray(0, this.n), (v) => Math.round(v)),
      s: Array.from(this.s.subarray(0, this.n)),
      f: Array.from(this.f.subarray(0, this.n * 6), r),
      blinks: this.blinks.map(Math.round),
    };
  }
  static fromJSON(j) {
    const rec = new Recorder();
    while (rec.cap < j.n) rec._grow();
    rec.n = j.n; rec.t.set(j.t); rec.s.set(j.s); rec.f.set(j.f); rec.blinks = j.blinks.slice();
    return rec;
  }
}

/* Turns tracker output into a session: real timestamps, honest gaps, blink events. */
class SessionRecorder {
  constructor(start) {
    this.start = start;
    this.rec = new Recorder();
    this.events = [];          // { t, type, ... }  visibility, gaps, camera status
    this.lastWall = null; this.lastT = 0; this.lastState = -1;
    this.inBlink = false; this.blinkT = 0; this.blinkPeak = 0;
    this.hiddenTouched = false;
    this.latest = null;
  }
  note(type, extra = {}) {
    const t = Date.now() - this.start;
    this.events.push({ t, type, ...extra });
    if (type === 'hidden') this.hiddenTouched = true;
  }
  onSample(wall, state, v) {
    const t = wall - this.start, rec = this.rec;
    // A long silence is a hole in the record. It is never filled in.
    if (this.lastWall !== null && wall - this.lastWall > CFG.GAP_MS && this.lastState !== 2) {
      rec.push(this.lastT + 1, 2);
      this.events.push({ t: this.lastT, type: 'gap', ms: wall - this.lastWall, hidden: this.hiddenTouched });
      this.lastState = 2; this.inBlink = false;
    }
    if (state === 1) {
      rec.push(t, 1, v.ix, v.iy, v.yaw, v.pitch, v.roll, v.blink);
      // blink events (with hysteresis) are kept as their own timestamped list
      if (!this.inBlink && v.blink > CFG.BLINK_ON) { this.inBlink = true; this.blinkT = t; this.blinkPeak = v.blink; }
      else if (this.inBlink) {
        this.blinkPeak = Math.max(this.blinkPeak, v.blink);
        if (v.blink < CFG.BLINK_OFF) {
          const dur = Math.max(80, t - this.blinkT);
          if (dur <= 900) rec.blinks.push(this.blinkT, dur);
          this.inBlink = false;
        }
      }
    } else {
      this.inBlink = false;
      if (this.lastState !== state) rec.push(t, state);
    }
    if (this.lastWall === null || state !== 1 || wall - this.lastWall <= CFG.GAP_MS) this.hiddenTouched = document.hidden;
    this.lastState = state; this.lastWall = wall; this.lastT = t;
    this.latest = { t, state, ...(v || {}) };
  }
  finish(end) {
    const duration = end - this.start;
    return { version: 1, start: this.start, end, duration, rec: this.rec, events: this.events, calib: this.calib || zeroCalib() };
  }
}
const zeroCalib = () => ({ ix: 0, iy: 0, yaw: 0, pitch: 0, roll: 0, done: false });

/* ═══════════════════════════ 1 · tracker ═══════════════════════════ */
const LM = { // MediaPipe face mesh landmark indices
  rA: 33, rB: 133, rIris: 468,   // subject's right eye: corners (image-left, image-right), iris centre
  lA: 362, lB: 263, lIris: 473,  // subject's left eye
};

class Tracker {
  constructor(video) {
    this.video = video; this.landmarker = null; this.stream = null; this.track = null;
    this.ic = null; this.worker = null; this.sink = null; this.busy = false;
    this.tickN = 0; this.lastState = 0; this.lastTs = 0;
    this.ema = null; this.lastEmaT = 0;
    this.bsIdx = null; this.muted = false; this.ended = false;
    this.onStatus = () => {};
  }

  async load(progress) {
    progress && progress('Loading the face model…');
    const vision = await import('./vendor/vision_bundle.mjs');
    const fileset = await vision.FilesetResolver.forVisionTasks('./vendor/wasm');
    const make = (delegate) => vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: './vendor/face_landmarker.task', delegate },
      runningMode: 'VIDEO', numFaces: 1,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
      minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    });
    try { this.landmarker = await make('GPU'); }
    catch (e) { console.warn('GPU delegate unavailable, using CPU', e); this.landmarker = await make('CPU'); }
  }

  async openCamera() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 30 }, facingMode: 'user' },
      audio: false,
    });
    this.track = this.stream.getVideoTracks()[0];
    this.track.onmute = () => { this.muted = true; this.onStatus('muted'); };
    this.track.onunmute = () => { this.muted = false; this.onStatus('live'); };
    this.track.onended = () => { this.ended = true; this.onStatus('ended'); };
    this.video.srcObject = this.stream;
    await this.video.play();
    // ImageCapture lets us pull frames even if the <video> stops repainting in a background tab.
    if ('ImageCapture' in window) { try { this.ic = new ImageCapture(this.track); } catch { this.ic = null; } }
  }

  /* A Worker drives the clock: page timers are throttled hard in hidden tabs, workers far less. */
  run(sink) {
    this.sink = sink;
    const src = 'let id=null;onmessage=e=>{clearInterval(id);if(e.data>0)id=setInterval(()=>postMessage(0),e.data)}';
    this.worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    this.worker.onmessage = () => this.tick();
    this.worker.postMessage(Math.round(1000 / CFG.SAMPLE_HZ));
  }

  async tick() {
    if (this.busy || !this.sink) return;
    // when nobody is there, look less often
    if (this.lastState === 0 && (this.tickN++ % CFG.ABSENT_DIVIDER)) return;
    this.busy = true;
    const wall = Date.now();
    try {
      if (this.ended || this.muted || this.track.readyState !== 'live') { this.lastState = 2; this.sink(wall, 2); return; }
      let img = null;
      if (this.ic) { try { img = await this.ic.grabFrame(); } catch { this.ic = null; } }
      const src = img || this.video;
      const w = img ? img.width : this.video.videoWidth, h = img ? img.height : this.video.videoHeight;
      if (!w || !h) { this.lastState = 2; this.sink(wall, 2); return; }
      const ts = Math.max(this.lastTs + 1, performance.now()); this.lastTs = ts;
      const res = this.landmarker.detectForVideo(src, ts);
      if (img && img.close) img.close();
      const v = res.faceLandmarks && res.faceLandmarks.length ? this.measure(res, w, h, wall) : null;
      if (v) { this.lastState = 1; this.sink(wall, 1, v); }
      else { this.lastState = 0; this.ema = null; this.sink(wall, 0); }
    } catch (e) {
      console.warn('tracking tick failed', e);
      this.lastState = 2; this.sink(wall, 2);
    } finally { this.busy = false; }
  }

  measure(res, w, h, wall) {
    const lm = res.faceLandmarks[0];
    // iris position inside each eye, in units of half the eye's width, along the eye's own axis
    const eye = (a, b, iris) => {
      const ax = lm[a].x * w, ay = lm[a].y * h, bx = lm[b].x * w, by = lm[b].y * h;
      const hx = bx - ax, hy = by - ay, hl = Math.hypot(hx, hy) || 1;
      const ux = hx / hl, uy = hy / hl;
      const dx = lm[iris].x * w - (ax + bx) / 2, dy = lm[iris].y * h - (ay + by) / 2;
      return [(dx * ux + dy * uy) / (hl / 2), (-dx * uy + dy * ux) / (hl / 2)]; // [along, toward-down]
    };
    const [rx, ry] = eye(LM.rA, LM.rB, LM.rIris), [lx, ly] = eye(LM.lA, LM.lB, LM.lIris);
    // camera images are not mirrored: image-right is the person's left, i.e. screen-left
    let ix = -(rx + lx) / 2, iy = -(ry + ly) / 2;

    // eyelid closure from blendshapes
    let blink = 0;
    const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
    if (cats) {
      if (!this.bsIdx) this.bsIdx = [cats.findIndex((c) => c.categoryName === 'eyeBlinkLeft'), cats.findIndex((c) => c.categoryName === 'eyeBlinkRight')];
      blink = (cats[this.bsIdx[0]].score + cats[this.bsIdx[1]].score) / 2;
    }

    // head orientation from the facial transformation matrix (column-major 4×4)
    let yaw = 0, pitch = 0, roll = 0;
    const M = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
    if (M) {
      const d = M.data;
      const fx = d[8], fy = d[9], fz = d[10];          // where the face points
      yaw = -Math.atan2(fx, fz) * CFG.YAW_SIGN;       // + = turned toward screen right
      pitch = Math.atan2(fy, Math.hypot(fx, fz)) * CFG.PITCH_SIGN; // + = up
      roll = Math.atan2(d[4], d[5]) * CFG.ROLL_SIGN;
    }

    // light time-aware smoothing; the eyes are held while the lids are moving so blinks don't read as glances
    const frozen = blink > 0.3;
    if (!this.ema) { this.ema = { ix, iy, yaw, pitch, roll }; this.lastEmaT = wall; }
    else {
      const dt = wall - this.lastEmaT; this.lastEmaT = wall;
      const a = 1 - Math.exp(-dt / 140), e = this.ema;
      if (!frozen) { e.ix += (ix - e.ix) * a; e.iy += (iy - e.iy) * a; }
      e.yaw += (yaw - e.yaw) * a; e.pitch += (pitch - e.pitch) * a; e.roll += (roll - e.roll) * a;
    }
    return { ix: this.ema.ix, iy: this.ema.iy, yaw: this.ema.yaw, pitch: this.ema.pitch, roll: this.ema.roll, blink };
  }

  /* camera light off, nothing retained */
  stop() {
    if (this.worker) { this.worker.postMessage(0); this.worker.terminate(); this.worker = null; }
    this.sink = null;
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null; this.track = null; this.ic = null;
    this.video.pause(); this.video.srcObject = null;
    if (this.landmarker) { try { this.landmarker.close(); } catch { /* ignore */ } this.landmarker = null; }
  }
}

/* ═══════════════════════════ dev · simulated sessions ═══════════════════════════ */
const SCENARIOS = {
  focused:   { pCam: 0.62, dwell: [2, 7],   leaveEvery: 45 * 60, leaveFor: [40, 240], head: 0.6, spread: 0.30 },
  wandering: { pCam: 0.30, dwell: [3, 13],  leaveEvery: 28 * 60, leaveFor: [30, 200], head: 0.9, spread: 0.45 },
  restless:  { pCam: 0.24, dwell: [1.2, 6], leaveEvery: 7 * 60,  leaveFor: [20, 300], head: 1.8, spread: 0.65 },
  distant:   { pCam: 0.07, dwell: [4, 22],  leaveEvery: 35 * 60, leaveFor: [30, 200], head: 0.7, spread: 0.55 },
};

function simulateSession(minutes, scenario, seed) {
  const P = SCENARIOS[scenario] || SCENARIOS.wandering;
  const rng = mulberry32((seed * 7919 + 13) >>> 0), gauss = gaussFrom(rng);
  const rec = new Recorder(), dt = 1 / CFG.SAMPLE_HZ, total = minutes * 60;
  const bias = rng() * 6.283, biasStrength = 0.4 + rng() * 0.5;
  let t = 0, gx = 0, gy = 0, tx = 0, ty = 0, dwell = 1;
  let hx = 0, hy = 0, hr = 0, ox = 0, oy = 0, or_ = 0, headT = [0, 0];
  let absentUntil = -1, blinkLeft = 0, blinkDur = 0.16, nextLeave = -Math.log(1 - rng()) * P.leaveEvery;
  while (t < total) {
    t += dt;
    if (t < absentUntil) continue;
    if (absentUntil > 0 && t >= absentUntil) { absentUntil = -1; nextLeave = t - Math.log(1 - rng()) * P.leaveEvery; }
    if (t > nextLeave) {
      rec.push(t * 1000, 0);
      absentUntil = t + lerp(P.leaveFor[0], P.leaveFor[1], rng() ** 1.3);
      continue;
    }
    dwell -= dt;
    if (dwell <= 0) {
      if (rng() < P.pCam) { tx = gauss() * 0.03; ty = gauss() * 0.03; }
      else {
        const mag = 0.14 + (rng() ** 0.8) * P.spread;
        const ang = rng() < biasStrength ? bias + gauss() * 0.9 : rng() * 6.283;
        tx = Math.cos(ang) * mag; ty = Math.sin(ang) * mag * 0.75;
      }
      const [lo, hi] = P.dwell;
      dwell = lerp(lo, hi, rng() ** 1.6) * (rng() < 0.04 ? 5 : 1);
      headT = [tx * 0.4, ty * 0.3];
    }
    const k = 1 - Math.exp(-dt / 0.09);
    gx += (tx - gx) * k + gauss() * 0.004; gy += (ty - gy) * k + gauss() * 0.004;
    const kh = 1 - Math.exp(-dt / 1.9);
    hx += (headT[0] - hx) * kh; hy += (headT[1] - hy) * kh;
    ox += (-ox * 0.02) + gauss() * 0.004 * P.head; oy += (-oy * 0.02) + gauss() * 0.003 * P.head; or_ += (-or_ * 0.015) + gauss() * 0.003 * P.head;
    const yaw = hx + ox, pitch = hy + oy, roll = or_;
    const ix = (gx - yaw * CFG.HEAD_G) / CFG.EYE_GX, iy = (gy - pitch * CFG.HEAD_G) / CFG.EYE_GY;
    let blink = 0.04 + Math.abs(gauss()) * 0.02;
    if (blinkLeft > 0) { blink = 0.95 * Math.sin(Math.PI * (1 - blinkLeft / blinkDur)); blinkLeft -= dt; }
    else if (rng() < 0.27 * dt) { blinkLeft = blinkDur = 0.12 + rng() * 0.1; }
    rec.push(t * 1000, 1, ix, iy, yaw, pitch, roll, Math.max(0, blink));
  }
  // blink events from the same hysteresis rule used live
  let inB = false, bt = 0;
  for (let i = 0; i < rec.n; i++) {
    if (rec.s[i] !== 1) { inB = false; continue; }
    const b = rec.f[i * 6 + 5], tt = rec.t[i];
    if (!inB && b > CFG.BLINK_ON) { inB = true; bt = tt; }
    else if (inB && b < CFG.BLINK_OFF) { rec.blinks.push(bt, Math.max(80, tt - bt)); inB = false; }
  }
  const end = Date.now(), start = end - total * 1000;
  return { version: 1, start, end, duration: total * 1000, rec, events: [], calib: zeroCalib(), simulated: scenario };
}

/* ═══════════════════════════ 3 · timeline ═══════════════════════════
   Recorded samples → bins on a "presence clock". Time spent absent or untracked simply
   does not exist on this clock: nothing is invented for it, and unrelated moments are
   never joined by an interpolated line.                                                */
const FIELDS = ['pull', 'c', 'gx', 'gy', 'gvx', 'gvy', 'yaw', 'pitch', 'roll', 'blink'];
const [F_PULL, F_C, F_GX, F_GY, F_GVX, F_GVY, F_YAW, F_PITCH, F_ROLL, F_BLINK] = FIELDS.map((_, i) => i);

function sessionStats(session) {
  const { rec, duration } = session;
  let present = 0, absent = 0, interrupted = 0;
  for (let i = 0; i < rec.n; i++) {
    const next = i + 1 < rec.n ? rec.t[i + 1] : duration;
    const span = Math.max(0, next - rec.t[i]);
    if (rec.s[i] === 1) { const p = Math.min(span, 600); present += p; interrupted += span - p; }
    else if (rec.s[i] === 0) absent += span;
    else interrupted += span;
  }
  const gaps = session.events.filter((e) => e.type === 'gap');
  return {
    present, absent, interrupted,
    hiddenGaps: gaps.filter((g) => g.hidden).length,
    sleepGaps: gaps.filter((g) => !g.hidden && g.ms > 60000).length,
    blinks: rec.blinks.length / 2,
  };
}

function buildTimeline(session) {
  const { rec, calib } = session, BW = CFG.BIN, n = rec.n;
  const cal = calib || zeroCalib();
  const A = { gx: [], gy: [], yaw: [], pitch: [], roll: [], blink: [], start: [] };
  const bl = rec.blinks; let bp = 0;

  let i = 0;
  while (i < n) {
    if (rec.s[i] !== 1) { i++; continue; }
    let j = i;
    while (j + 1 < n && rec.s[j + 1] === 1 && rec.t[j + 1] - rec.t[j] <= 600) j++;
    const t0 = rec.t[i], t1 = rec.t[j];
    const nb = Math.max(1, Math.floor((t1 - t0) / 1000 / BW) + 1), base = A.gx.length;
    let k = i;
    for (let b = 0; b < nb; b++) {
      const tb = t0 + b * BW * 1000;
      while (k < j - 1 && rec.t[k + 1] <= tb) k++;
      const k2 = Math.min(k + 1, j);
      const span = rec.t[k2] - rec.t[k];
      const a = span > 0 ? clamp((tb - rec.t[k]) / span, 0, 1) : 0;
      const g = (o) => lerp(rec.f[k * 6 + o], rec.f[k2 * 6 + o], a);
      const yaw = g(2), pitch = g(3);
      A.gx.push((g(0) - cal.ix) * CFG.EYE_GX + (yaw - cal.yaw) * CFG.HEAD_G);
      A.gy.push((g(1) - cal.iy) * CFG.EYE_GY + (pitch - cal.pitch) * CFG.HEAD_G);
      A.yaw.push(yaw - cal.yaw); A.pitch.push(pitch - cal.pitch); A.roll.push(g(4) - cal.roll);
      A.blink.push(0); A.start.push(b === 0 ? 1 : 0);
    }
    while (bp < bl.length && bl[bp] < t0) bp += 2;
    while (bp < bl.length && bl[bp] <= t1) { A.blink[base + Math.min(nb - 1, Math.floor((bl[bp] - t0) / 1000 / BW))]++; bp += 2; }
    i = j + 1;
  }

  const N = A.gx.length;
  const out = { n: N, bw: BW, pre: FIELDS.map(() => new Float64Array(N + 1)), stats: null };
  if (!N) return out;

  const aUp = 1 - Math.exp(-BW / 1.4), aDown = 1 - Math.exp(-BW / 0.4);   // meeting is gradual, looking away is quick
  const aV = 1 - Math.exp(-BW / 0.3), aH = 1 - Math.exp(-BW / 0.4);
  let pull = 0, gvx = 0, gvy = 0, yaw = A.yaw[0], pitch = A.pitch[0], roll = A.roll[0];
  let sumC = 0, sumGX = 0, sumGY = 0, sumHead = 0, sumYaw = 0;
  for (let b = 0; b < N; b++) {
    const th = Math.hypot(A.gx[b], A.gy[b]);
    const c = 1 - smooth(CFG.CAM_IN, CFG.CAM_OUT, th);
    pull += (c - pull) * (c > pull ? aUp : aDown);
    if (A.start[b] || b === 0) { gvx = 0; gvy = 0; }
    else {
      gvx += ((A.gx[b] - A.gx[b - 1]) / BW / 1.5 - gvx) * aV;
      gvy += ((A.gy[b] - A.gy[b - 1]) / BW / 1.5 - gvy) * aV;
    }
    const py = yaw, pp = pitch;
    yaw += (A.yaw[b] - yaw) * aH; pitch += (A.pitch[b] - pitch) * aH; roll += (A.roll[b] - roll) * aH;
    sumHead += (Math.abs(yaw - py) + Math.abs(pitch - pp)) / BW;
    const m = Math.hypot(gvx, gvy), s = m > 1 ? 1 / m : 1;
    const row = [pull, c, A.gx[b], A.gy[b], gvx * s, gvy * s, yaw, pitch, roll, A.blink[b]];
    for (let f = 0; f < FIELDS.length; f++) out.pre[f][b + 1] = out.pre[f][b] + row[f];
    sumC += c; sumGX += A.gx[b]; sumGY += A.gy[b]; sumYaw += yaw;
  }
  out.stats = {
    present: N * BW, meanC: sumC / N, awayFrac: 1 - sumC / N,
    meanGX: sumGX / N, meanGY: sumGY / N, headEnergy: sumHead / N, meanYaw: sumYaw / N,
    blinkBins: A.blink.reduce((q, v, b) => (v > 0 ? (q.push(b), q) : q), []),
    blinkTotal: A.blink.reduce((q, v) => q + v, 0),
  };
  return out;
}

function hashSession(session) {
  const { rec } = session;
  let h = 2166136261 >>> 0;
  const mix = (v) => { h ^= v | 0; h = Math.imul(h, 16777619) >>> 0; };
  mix(rec.n); mix(Math.round(session.duration / 1000));
  for (let i = 0; i < rec.n; i++) {
    mix(Math.round(rec.t[i] / 50)); mix(rec.s[i]);
    if (rec.s[i] === 1) for (let o = 0; o < 6; o++) mix(Math.round(rec.f[i * 6 + o] * 400));
  }
  for (let i = 0; i < rec.blinks.length; i++) mix(Math.round(rec.blinks[i]));
  return h >>> 0;
}

/* ═══════════════════════════ 4 · generator ═══════════════════════════
   The computer's gaze is a fixed attractor at the centre. The human's gaze is a variable
   force. Each particle lives through a stretch of the recorded presence clock and is
   pushed by whatever the person's eyes and head were doing during that stretch.

     meeting the gaze   inward pull + orbiting swirl, dense rings around the centre
     looking away       outward release along the direction of the gaze; gaze velocity sweeps
     returning          the pull builds gradually; earlier trajectories stay on the sheet
     blinking           a small kick to the field and a few fine filaments from the core
     head movement      rotates and domain-warps the flow field
     absence            the clock does not run: nothing is emitted, nothing moves
     duration           more particles, finer branching, more permitted layering (but bounded) */
const GEN = {
  kA: 0.00012, kT: 0.00016, kR: 0.0012, kO: 0.00040, kS: 0.00020, kF: 0.00012, kB: 0.0011,
  vmax: 0.0115, cell: 24, K0: 160, K1: 250,
};

class Generator {
  constructor(session, opts = {}) {
    this.session = session;
    this.tl = buildTimeline(session);
    this.seed = (opts.seed != null ? opts.seed : hashSession(session)) >>> 0;
    this.rng = mulberry32(this.seed ^ 0x9e3779b9);
    this.gauss = gaussFrom(this.rng);
    this.strokes = []; this.jobs = []; this.next = 0; this.done = false;
    const { n, stats } = this.tl;
    this.F = new Float64Array(FIELDS.length);
    this.pre = this.tl.pre;
    this.buf = new Float32Array(3 * 900);

    if (!n) { this.done = true; return; }
    const Dp = stats.present, Dmin = Dp / 60;

    // number of lines: grows with logarithm of time, never unbounded
    let S = clamp(650 + 850 * Math.log2(1 + Dmin), 650, 7600);
    S *= clamp(0.3 + Dp / 30, 0.3, 1);
    this.S = Math.round(S);
    this.capScale = clamp(0.9 + 0.11 * Math.log2(1 + Dmin), 0.9, 1.9);
    this.branchBudget = Math.round(this.S * 0.45);
    this.branchScale = 0.8 + clamp(Math.log2(1 + Dmin) / 6, 0, 1.4);

    // how many bins one step reads: each particle sees a stretch of the session
    const K = (GEN.K0 + GEN.K1) / 2;
    const stepSec = clamp(Dp / (3 * K), CFG.BIN, 3.0);
    this.sb = clamp(Math.round(stepSec / CFG.BIN), 1, n);

    // session-level character taken from the data
    this.spinP = 0.5 + 0.36 * Math.tanh(stats.meanGX * 5);
    this.freq = 0.7 + 1.2 * clamp(stats.headEnergy / 0.22, 0, 1);
    this.kO = GEN.kO * (0.65 + 1.0 * stats.awayFrac);
    this.zoom = 1;

    this.gridW = Math.ceil(CFG.W / GEN.cell); this.gridH = Math.ceil(CFG.H / GEN.cell);
    this.grid = new Float32Array(this.gridW * this.gridH);

    noiseSeed(this.seed);
    noiseDetail(2 + (Dmin > 20 ? 1 : 0) + (Dmin > 120 ? 1 : 0), 0.5);

    // jobs, in session order
    const rng = this.rng, jobs = this.jobs;
    for (let i = 0; i < this.S; i++) jobs.push({ prog: (i + 0.1 + 0.8 * rng()) / this.S, kind: 0 });
    const bb = stats.blinkBins, nb = Math.min(bb.length, Math.floor(this.S * 0.16));
    for (let i = 0; i < nb; i++) {
      const bin = bb[Math.floor((i + 0.5) * bb.length / nb)];
      jobs.push({ prog: (bin + 0.5) / n, kind: 1, bin });
    }
    jobs.sort((a, b) => a.prog - b.prog);
    this.total = jobs.length;
  }

  get progress() { return this.total ? this.next / this.total : 1; }

  run(ms) {
    const t0 = performance.now();
    while (this.next < this.jobs.length && performance.now() - t0 < ms) this.process(this.jobs[this.next++]);
    if (this.next >= this.jobs.length) this.done = true;
    return this.done;
  }

  sampleAt(i0, len) {
    const n = this.tl.n, pre = this.pre, F = this.F, nf = pre.length;
    let i = i0 % n; if (i < 0) i += n;
    const inv = 1 / len;
    if (i + len <= n) for (let k = 0; k < nf; k++) F[k] = (pre[k][i + len] - pre[k][i]) * inv;
    else { const l2 = len - (n - i); for (let k = 0; k < nf; k++) F[k] = ((pre[k][n] - pre[k][i]) + (pre[k][l2] - pre[k][0])) * inv; }
    F[F_BLINK] *= len; // a count, not an average
  }

  process(job) {
    const rng = this.rng, n = this.tl.n, F = this.F, sb = this.sb;
    const bin = job.kind === 1 ? job.bin : Math.min(n - 1, Math.floor(job.prog * n));
    this.sampleAt(bin, sb);
    const away = 1 - F[F_PULL];
    const gm = Math.hypot(F[F_GX], F[F_GY]);
    const gang = Math.atan2(-F[F_GY], F[F_GX]);                 // canvas orientation (y down)
    const dirBias = clamp(gm / 0.45, 0, 1) * away;

    const P = {
      depth: 0, bin0: bin, k0: 0,
      spin: rng() < this.spinP ? 1 : -1,
      drag: 0.968 + rng() * 0.022, mass: 0.75 + rng() * 0.7,
      fq: this.freq * (0.8 + rng() * 0.45), ox: rng() * 40, oy: rng() * 40,
      prog: job.prog,
    };
    let ang, r0;
    if (job.kind === 1) {                                        // blink filament: from the core, outward
      ang = rng() * 6.283; r0 = 0.09 + rng() * 0.07;
      P.K = 70 + Math.floor(rng() * 90); P.w = 0.45 + rng() * 0.4; P.a = 0.25 + rng() * 0.3; P.kind = 1;
    } else {
      ang = rng() < 0.1 + 0.5 * dirBias ? gang + this.gauss() * (1.1 - 0.5 * dirBias) : rng() * 6.283;
      r0 = away > 0.5 ? 0.1 + 0.28 * Math.pow(rng(), 1.5) : 0.14 + 0.85 * Math.pow(rng(), 1.3);
      P.K = Math.floor(GEN.K0 + rng() * (GEN.K1 - GEN.K0));
      P.w = 0.5 + Math.pow(rng(), 2.2) * 1.0; P.a = 0.14 + Math.pow(rng(), 1.4) * 0.46; P.kind = 0;
    }
    P.a *= Math.pow(1500 / Math.max(this.S, 700), 0.1);
    P.x = Math.cos(ang) * r0; P.y = Math.sin(ang) * r0;
    const v0 = 0.003 + 0.006 * away;
    P.vx = Math.cos(ang) * v0 * (job.kind === 1 ? 1.4 : away) - Math.sin(ang) * 0.004 * P.spin;
    P.vy = Math.sin(ang) * v0 * (job.kind === 1 ? 1.4 : away) + Math.cos(ang) * 0.004 * P.spin;

    const stack = [P];
    while (stack.length) this.trace(stack.pop(), stack);
  }

  trace(P, stack) {
    const rng = this.rng, tl = this.tl, n = tl.n, F = this.F, sb = this.sb, buf = this.buf, grid = this.grid;
    const { kA, kT, kR, kS, kF, kB, vmax } = GEN, kO = this.kO;
    const U = CFG.UNIT, cell = GEN.cell, gw = this.gridW, gh = this.gridH;
    let { x, y, vx, vy } = P;
    const K = Math.min(P.K, 880), prog = P.prog;
    const pb = 0.0042 * this.branchScale * (P.kind === 1 ? 0.3 : 1);
    let count = 0, stress = 0;

    for (let k = 0; k < K; k++) {
      const idx = P.bin0 + (P.k0 + k) * sb;
      this.sampleAt(idx, sb);
      const pull = F[F_PULL], away = 1 - pull;
      const r = Math.hypot(x, y) + 1e-5, rx = x / r, ry = y / r, tx = -ry, ty = rx;

      // the computer's gaze: fixed, patient, always attracting in proportion to being met
      const inward = kA * pull * r / (r * r + 0.06);
      let ax = -rx * inward, ay = -ry * inward;
      const sw = kT * pull * P.spin / (r + 0.2);
      ax += tx * sw; ay += ty * sw;
      const rep = kR * Math.exp(-(r * r) / 0.01);          // keeps a small clear core
      ax += rx * rep; ay += ry * rep;

      // the human's gaze: released outward along its own direction, swept by its own movement
      const gm = Math.hypot(F[F_GX], F[F_GY]);
      const gmag = clamp(gm / 0.5, 0, 1);
      let dx = rx, dy = ry;
      if (gm > 1e-4) { const m = 0.2 + 0.4 * gmag; dx = rx * (1 - m) + (F[F_GX] / gm) * m; dy = ry * (1 - m) - (F[F_GY] / gm) * m; }
      const dl = Math.hypot(dx, dy) || 1;
      const out = kO * away * (0.45 + 0.55 * gmag);
      ax += (dx / dl) * out; ay += (dy / dl) * out;
      ax += kS * F[F_GVX] * (0.4 + away); ay -= kS * F[F_GVY] * (0.4 + away);

      // procedural flow field; head orientation rotates and warps it
      const yaw = F[F_YAW], pitch = F[F_PITCH], roll = F[F_ROLL];
      const nv = noise(x * P.fq + P.ox + yaw * 1.1, y * P.fq + P.oy - pitch * 1.1, (idx / n) * 1.6 + roll * 0.7);
      const fa = (nv - 0.5) * 6.283 * 1.5 + roll * 1.4 + yaw * 0.8;
      const fw = kF * (0.3 + 0.7 * smooth(0.06, 0.7, r)) * (0.5 + 0.5 * away);
      ax += Math.cos(fa) * fw; ay += Math.sin(fa) * fw;

      // blink: a small, brief disturbance
      const bc = F[F_BLINK];
      if (bc > 0.01) { const a2 = rng() * 6.283, m = kB * Math.min(bc, 2); vx += Math.cos(a2) * m; vy += Math.sin(a2) * m; }

      vx = vx * P.drag + ax / P.mass; vy = vy * P.drag + ay / P.mass;
      const om = P.spin * (0.010 + 0.032 * pull) / (r + 0.3);   // field-like curl: velocity turns, speed is kept
      const co = Math.cos(om), si = Math.sin(om), nvx = vx * co - vy * si;
      vy = vx * si + vy * co; vx = nvx;
      const sp = Math.hypot(vx, vy);
      if (sp > vmax) { vx *= vmax / sp; vy *= vmax / sp; }
      x += vx; y += vy;

      const X = CFG.CX + x * U, Y = CFG.CY + y * U;
      if (X < -60 || X > CFG.W + 60 || Y < -60 || Y > CFG.H + 60) break;

      const o = count * 3;
      buf[o] = X; buf[o + 1] = Y; buf[o + 2] = clamp(0.5 + 0.9 * (1 - sp / vmax), 0.5, 1.4);
      count++;

      // density control: dense heart, open periphery, never a black mass
      const gi = Math.floor(clamp(Y, 0, CFG.H - 1) / cell) * gw + Math.floor(clamp(X, 0, CFG.W - 1) / cell);
      const cap = (1.3 + 5.2 * Math.exp(-Math.pow(r / 0.42, 1.6)) * (0.4 + 0.6 * prog)) * this.capScale;
      if (grid[gi] > cap && k > 24 && x * vx + y * vy < r * sp * 0.35) { if (++stress > 3) break; } else if (stress > 0) stress -= 0.25;
      grid[gi] += sp * U / cell;

      // branching
      if (P.depth < 2 && this.branchBudget > 0 && k > 12 && k < K - 40 && rng() < pb * (0.6 + away)) {
        this.branchBudget--;
        const sgn = rng() < 0.5 ? -1 : 1, da = sgn * (0.4 + rng() * 0.9), c = Math.cos(da), s = Math.sin(da);
        stack.push({
          depth: P.depth + 1, kind: P.kind, bin0: P.bin0, k0: P.k0 + k, K: Math.min(K - k, 60 + Math.floor(rng() * 110)),
          spin: P.spin, drag: P.drag, mass: P.mass, fq: P.fq, ox: P.ox, oy: P.oy, prog,
          x, y, vx: vx * c - vy * s, vy: vx * s + vy * c, w: P.w * 0.72, a: P.a * 0.85,
        });
      }
    }
    if (count >= 14) this.strokes.push({ pts: buf.slice(0, count * 3), n: count, w: P.w, a: P.a });
  }

  result() { return { strokes: this.strokes, seed: this.seed, S: this.S, stats: this.tl.stats }; }
}

/* ═══════════════════════════ 5 · renderer ═══════════════════════════ */
const tmpL = new Float32Array(2000), tmpR = new Float32Array(2000);

/* strokes are tapered ribbons: they thin out at both ends, and thicken where the pen slowed */
function paintArt(ctx, art, pxPerUnit, paper = true) {
  if (paper) { ctx.fillStyle = CFG.PAPER; ctx.fillRect(0, 0, CFG.W, CFG.H); }
  ctx.fillStyle = CFG.INK;
  const minW = 0.6 / pxPerUnit;           // never thinner than a ~0.6 device-pixel hairline
  for (const s of art.strokes) {
    const p = s.pts, n = s.n;
    for (let i = 0; i < n; i++) {
      const i0 = i > 0 ? i - 1 : 0, i1 = i < n - 1 ? i + 1 : n - 1;
      let tx = p[i1 * 3] - p[i0 * 3], ty = p[i1 * 3 + 1] - p[i0 * 3 + 1];
      const l = Math.hypot(tx, ty) || 1; tx /= l; ty /= l;
      const u = n > 1 ? i / (n - 1) : 0;
      const env = Math.pow(Math.min(1, u / 0.12), 0.8) * Math.pow(Math.min(1, (1 - u) / 0.35), 1.2);
      const hw = Math.max(s.w * p[i * 3 + 2], minW) * env * 0.5;
      const x = p[i * 3], y = p[i * 3 + 1];
      tmpL[i * 2] = x - ty * hw; tmpL[i * 2 + 1] = y + tx * hw;
      tmpR[i * 2] = x + ty * hw; tmpR[i * 2 + 1] = y - tx * hw;
    }
    ctx.globalAlpha = s.a;
    ctx.beginPath();
    ctx.moveTo(tmpL[0], tmpL[1]);
    for (let i = 1; i < n; i++) ctx.lineTo(tmpL[i * 2], tmpL[i * 2 + 1]);
    for (let i = n - 1; i >= 0; i--) ctx.lineTo(tmpR[i * 2], tmpR[i * 2 + 1]);
    ctx.closePath(); ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function renderCanvas(art, pxW) {
  const scale = pxW / CFG.W;
  const c = document.createElement('canvas');
  c.width = Math.round(CFG.W * scale); c.height = Math.round(CFG.H * scale);
  const ctx = c.getContext('2d');
  ctx.scale(scale, scale);
  paintArt(ctx, art, scale);
  return c;
}

function simplify(pts, n, tol) {         // Ramer–Douglas–Peucker on x,y of a packed stroke
  const keep = new Uint8Array(n); keep[0] = keep[n - 1] = 1;
  const st = [[0, n - 1]];
  while (st.length) {
    const [a, b] = st.pop();
    let dmax = 0, idx = -1;
    const ax = pts[a * 3], ay = pts[a * 3 + 1], bx = pts[b * 3], by = pts[b * 3 + 1];
    const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((pts[i * 3] - ax) * dy - (pts[i * 3 + 1] - ay) * dx) / L;
      if (d > dmax) { dmax = d; idx = i; }
    }
    if (dmax > tol) { keep[idx] = 1; st.push([a, idx], [idx, b]); }
  }
  const out = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(pts[i * 3].toFixed(1) + ',' + pts[i * 3 + 1].toFixed(1));
  return out;
}

function buildSVG(art, withPaper = true) {
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${CFG.W} ${CFG.H}" width="${CFG.W}" height="${CFG.H}">`];
  if (withPaper) parts.push(`<rect width="${CFG.W}" height="${CFG.H}" fill="${CFG.PAPER}"/>`);
  parts.push(`<g fill="none" stroke="${CFG.INK}" stroke-linecap="round" stroke-linejoin="round">`);
  for (const s of art.strokes) {
    const pts = simplify(s.pts, s.n, 0.35);
    if (pts.length < 2) continue;
    parts.push(`<path d="M${pts.join('L')}" stroke-width="${(s.w * 0.7).toFixed(2)}" stroke-opacity="${s.a.toFixed(2)}"/>`);
  }
  parts.push('</g></svg>');
  return parts.join('\n');
}

/* ═══════════════════════════ 6 · app ═══════════════════════════ */
const App = {
  screen: 'intro',
  tracker: null, srec: null, session: null, art: null, artCanvas: null, artW: 0,
  timers: {}, wakeLock: null, devVisible: false,

  show(name) { this.screen = name; document.body.dataset.screen = name; },

  /* ── begin: camera, model, optional calibration ── */
  async begin() {
    $('intro-error').hidden = true;
    this.show('loading'); $('loading-text').textContent = 'Waiting for the camera…';
    const tracker = new Tracker($('cam'));
    this.tracker = tracker;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('This browser cannot access a camera here. Use https or localhost.');
      await tracker.openCamera();
      await tracker.load((m) => { $('loading-text').textContent = m; });
    } catch (e) {
      tracker.stop(); this.tracker = null;
      this.show('intro');
      const denied = e && (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError');
      $('intro-error').textContent = denied ? 'Camera access was declined. Eye Contact needs it to look at you.'
        : e && e.name === 'NotFoundError' ? 'No camera was found.' : 'Could not start: ' + (e && e.message ? e.message : e);
      $('intro-error').hidden = false;
      return;
    }
    this.calibrate();
  },

  calibrate() {
    this.show('calib');
    const tracker = this.tracker, samples = [], t0 = Date.now();
    let finished = false;
    const finish = (skipped) => {
      if (finished) return; finished = true;
      clearInterval(this.timers.calib);
      let cal = zeroCalib();
      if (!skipped && samples.length >= 12) {
        cal = { ix: median(samples.map((s) => s.ix)), iy: median(samples.map((s) => s.iy)), yaw: median(samples.map((s) => s.yaw)),
          pitch: median(samples.map((s) => s.pitch)), roll: median(samples.map((s) => s.roll)), done: true };
      }
      this.startRecording(cal);
    };
    tracker.run((wall, state, v) => {
      if (state === 1 && v.blink < 0.3 && wall - t0 > 800) samples.push(v);
    });
    $('btn-skip-calib').onclick = () => finish(true);
    const tick = () => {
      const left = CFG.CALIB_SECONDS - (Date.now() - t0) / 1000;
      $('calib-count').textContent = Math.max(1, Math.ceil(left));
      if (left <= 0) finish(false);
    };
    tick(); this.timers.calib = setInterval(tick, 200);
  },

  /* ── recording ── */
  async startRecording(cal) {
    const tracker = this.tracker;
    const start = Date.now();
    const sr = new SessionRecorder(start); sr.calib = cal;
    this.srec = sr;
    tracker.sink = null; tracker.lastState = 0; tracker.ema = null;
    tracker.sink = (wall, state, v) => sr.onSample(wall, state, v);
    tracker.onStatus = (s) => {
      sr.note('camera-' + s);
      $('status-text').textContent = s === 'live' ? 'Camera active' : 'Camera unavailable — tracking paused';
      $('dot').classList.toggle('off', s !== 'live');
    };
    this.show('record');
    $('status-text').textContent = 'Camera active'; $('dot').classList.remove('off');
    $('record-note').textContent = 'Keep this tab open. If it is hidden for long, or the computer sleeps, the gaps are recorded as gaps.';

    const upd = () => {
      const el = Date.now() - start;
      $('elapsed').textContent = fmtClock(el);
      document.title = `● ${fmtClock(el)} — Eye Contact`;
      if (this.devVisible && sr.latest) this.devReadout(sr);
    };
    upd(); this.timers.elapsed = setInterval(upd, 1000);

    this.onVis = () => {
      sr.note(document.hidden ? 'hidden' : 'visible');
      if (!document.hidden) this.acquireWake();
    };
    document.addEventListener('visibilitychange', this.onVis);
    this.onUnload = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', this.onUnload);
    this.acquireWake();
  },

  async acquireWake() {   // keeps the display (and camera) awake while recording; re-requested after tab switches
    try { if ('wakeLock' in navigator && !document.hidden) this.wakeLock = await navigator.wakeLock.request('screen'); } catch { /* optional */ }
  },

  endSession() {
    clearInterval(this.timers.elapsed);
    document.removeEventListener('visibilitychange', this.onVis);
    window.removeEventListener('beforeunload', this.onUnload);
    if (this.wakeLock) { try { this.wakeLock.release(); } catch { /* ignore */ } this.wakeLock = null; }
    const end = Date.now();
    if (this.tracker) { this.tracker.stop(); this.tracker = null; }     // camera off
    document.title = 'Eye Contact';
    const session = this.srec.finish(end);
    this.srec = null;
    this.makeDrawing(session);
  },

  /* ── drawing ── */
  makeDrawing(session, opts = {}) {
    this.session = session;
    this.show('making'); $('making-bar').style.width = '0%';
    const gen = new Generator(session, opts);
    const pump = () => {
      gen.run(24);
      $('making-bar').style.width = Math.round(gen.progress * 100) + '%';
      if (!gen.done) setTimeout(pump, 0);
      else this.reveal(gen.result());
    };
    setTimeout(pump, 30);
  },

  reveal(art) {
    this.art = art; this.artCanvas = null;
    const s = this.session, st = sessionStats(s);
    const d0 = new Date(s.start), d1 = new Date(s.end);
    const tf = (d) => d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    $('m-date').textContent = d0.toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    $('m-times').textContent = `${tf(d0)} – ${tf(d1)}${d1.toDateString() !== d0.toDateString() ? ' (next day)' : ''}`;
    $('m-dur').textContent = fmtDuration(s.duration);
    const notes = [];
    if (s.simulated) notes.push(`Simulated session (${s.simulated}).`);
    if (st.present < 1500) notes.push('You were barely in view, so there is little here.');
    else notes.push(`In view for ${fmtDuration(st.present)}.`);
    if (st.interrupted > 5000) {
      const why = [st.hiddenGaps && 'tab hidden', st.sleepGaps && 'computer asleep', 'camera unavailable'].filter(Boolean);
      notes.push(`Tracking was unavailable for ${fmtDuration(st.interrupted)} (${why.slice(0, 2).join(' or ')}); those moments are left blank.`);
    }
    $('final-note').textContent = notes.join(' ');
    this.show('final');
    this.fit();
    this.devSummary(art, st);
  },

  fit() {
    if (this.screen !== 'final' || !window._p5ready) return;
    const stage = $('stage'), aw = stage.clientWidth - 8, ah = stage.clientHeight - 8;
    let w = aw, h = w * CFG.H / CFG.W;
    if (h > ah) { h = ah; w = h * CFG.W / CFG.H; }
    w = Math.max(200, Math.floor(w)); h = Math.floor(w * CFG.H / CFG.W);
    resizeCanvas(w, h);
    const px = Math.round(w * pixelDensity());
    if (this.art && (!this.artCanvas || Math.abs(this.artW - px) > 2)) {
      this.artCanvas = renderCanvas(this.art, px); this.artW = px;
    }
    redraw();
  },

  /* ── export ── */
  stamp() {
    const d = new Date(this.session.start);
    return `eye-contact-${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}-${pad2(d.getHours())}${pad2(d.getMinutes())}`;
  },
  download(blob, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  },
  async savePNG() {
    if (!this.art) return;
    const btn = $('btn-save'); btn.textContent = 'Preparing…';
    await new Promise((r) => setTimeout(r, 30));
    for (const sc of [CFG.EXPORT_SCALE, 1.5, 1]) {
      try {
        const c = renderCanvas(this.art, Math.round(CFG.W * sc));
        const blob = await new Promise((res) => c.toBlob(res, 'image/png'));
        if (blob) { this.download(blob, this.stamp() + '.png'); break; }
      } catch (e) { console.warn('export at', sc, 'failed', e); }
    }
    btn.textContent = 'Save drawing';
  },
  saveSVG() {
    if (!this.art) return;
    this.download(new Blob([buildSVG(this.art)], { type: 'image/svg+xml' }), this.stamp() + '.svg');
  },
  again() {
    this.art = null; this.artCanvas = null; this.session = null;
    this.show('intro');
  },

  /* ── development ── */
  devReadout(sr) {
    const l = sr.latest, f = (v) => (v == null ? '–' : v.toFixed(3));
    $('dev-out').textContent = `recording · state ${l.state}\nirisX ${f(l.ix)} irisY ${f(l.iy)}\nyaw ${f(l.yaw)} pitch ${f(l.pitch)} roll ${f(l.roll)}\nblink ${f(l.blink)} · blinks ${sr.rec.blinks.length / 2}\nsamples ${sr.rec.n} · gaps ${sr.events.filter((e) => e.type === 'gap').length}\n(+x = screen right, +y = up; flip YAW_SIGN / PITCH_SIGN in CFG if reversed)`;
  },
  devSummary(art, st) {
    if (!this.devVisible) return;
    const s = art.stats || {};
    $('dev-out').textContent = `seed ${art.seed}\nstrokes ${art.strokes.length} (target ${art.S})\npresent ${(st.present / 1000).toFixed(0)}s · absent ${(st.absent / 1000).toFixed(0)}s · lost ${(st.interrupted / 1000).toFixed(0)}s\nmeanC ${s.meanC ? s.meanC.toFixed(2) : '–'} · away ${s.awayFrac ? s.awayFrac.toFixed(2) : '–'} · head ${s.headEnergy ? s.headEnergy.toFixed(3) : '–'}\nblinks ${st.blinks}`;
  },
  initDev() {
    const toggle = (on) => { this.devVisible = on; $('dev').hidden = !on; };
    if (/[?&]dev\b/.test(location.search)) toggle(true);
    window.addEventListener('keydown', (e) => { if ((e.key === 'd' || e.key === 'D') && !/input|select/i.test(e.target.tagName) && this.screen !== 'record') toggle(!this.devVisible); });
    $('dev-close').onclick = () => toggle(false);
    const seedOpt = () => { const v = $('dev-seed').value.trim(); return v === '' ? {} : { seed: Number(v) >>> 0 }; };
    $('dev-gen').onclick = () => {
      if (this.screen === 'record') return;
      const sess = simulateSession(Number($('dev-min').value) || 5, $('dev-scn').value, Number($('dev-dseed').value) || 1);
      this.makeDrawing(sess, seedOpt());
    };
    $('dev-redraw').onclick = () => { if (this.session) this.makeDrawing(this.session, seedOpt()); };
    $('dev-dl').onclick = () => {
      if (!this.session) return;
      const s = this.session;
      this.download(new Blob([JSON.stringify({ ...s, rec: s.rec.toJSON() })], { type: 'application/json' }), this.stamp() + '-data.json');
    };
    $('dev-load').onchange = async (e) => {
      const file = e.target.files[0]; if (!file) return;
      const j = JSON.parse(await file.text());
      j.rec = Recorder.fromJSON(j.rec);
      this.makeDrawing(j, seedOpt());
    };
    // URL shortcut:  ?dev&sim=wandering&min=30&dseed=3&seed=123
    const q = new URLSearchParams(location.search);
    if (q.has('sim')) {
      this._autoSim = () => {
        const sess = simulateSession(Number(q.get('min')) || 10, q.get('sim'), Number(q.get('dseed')) || 1);
        this.makeDrawing(sess, q.has('seed') ? { seed: Number(q.get('seed')) >>> 0 } : {});
      };
    }
  },

  init() {
    $('btn-begin').onclick = () => this.begin();
    $('btn-end').onclick = () => this.endSession();
    $('btn-save').onclick = () => this.savePNG();
    $('btn-svg').onclick = () => this.saveSVG();
    $('btn-again').onclick = () => this.again();
    this.initDev();
  },
};

/* ═══════════════════════════ p5 hooks ═══════════════════════════ */
function setup() {
  const c = createCanvas(300, 375);
  c.parent('stage');
  noLoop();
  window._p5ready = true;
  App.init();
  if (App._autoSim) App._autoSim();
}
function draw() {
  background(CFG.PAPER);
  if (App.artCanvas) {
    drawingContext.imageSmoothingQuality = 'high';
    drawingContext.drawImage(App.artCanvas, 0, 0, width, height);
  }
}
let _rz = null;
function windowResized() { clearTimeout(_rz); _rz = setTimeout(() => App.fit(), 150); }

window.EyeContact = { Tracker, SessionRecorder, App, simulateSession, Generator, buildTimeline, sessionStats, CFG, GEN };
