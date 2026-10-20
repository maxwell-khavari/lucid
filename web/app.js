// Lucid demo. Channel A: the network, with frames it doubts handed to the solver.
// Channel B: the reference solver on every frame, with the same input.
import { step, splat, N } from "../sim/solver.js";

const M = N * N;
const ERR_W = [1, 1, 0.5]; // matches ERR_WEIGHT in lucid/model.py
const TAU_LO = -2.5, TAU_HI = -0.5; // log10 range of the threshold (predicted error / flow size)
const HIST = 300; // frames kept in the traces

const $ = (id) => document.getElementById(id);
const ui = {
  hybrid: $("hybrid"), truth: $("truth"), who: $("who"), status: $("status"), backend: $("backend"),
  netOn: $("net-on"), netOff: $("net-off"), doubt: $("doubt"), reset: $("reset"), kick: $("kick"),
  trust: $("trust"), trustV: $("trust-v"),
  sNet: $("s-net"), sSolver: $("s-solver"), sErr: $("s-err"), sDoubt: $("s-doubt"),
  rFrame: $("r-frame"), rFps: $("r-fps"), rInf: $("r-inf"), rSol: $("r-sol"), rHand: $("r-hand"),
  traceDoubt: $("trace-doubt"), traceErr: $("trace-err"),
  fieldBtns: [...document.querySelectorAll("[data-field]")],
  cbars: [...document.querySelectorAll(".cbar")],
};

const field = () => ({ u: new Float32Array(M), v: new Float32Array(M), d: new Float32Array(M) });
let hyb = field(), ref = field();
const force = { fx: new Float32Array(M), fy: new Float32Array(M), s: new Float32Array(M) };
const doubtMap = new Float32Array(M);
let safety = true, showDoubt = false, frame = 0, kickNext = false, shown = "density";
const recent = []; // 1 when the solver took the frame
const hist = { doubt: [], tau: [], hand: [], err: [] };
const perf = { inf: 0, sol: 0, stamps: [], handoffs: 0 };
const debug = (window.lucidDebug = { frames: 0, solverFrames: 0, maxDoubt: 0 }); // read by tests
let doubtScale = 1e-3; // running mean of per-cell predicted error, for the overlay

function tau() {
  const v = Number(ui.trust.value) / 100;
  return Math.pow(10, TAU_LO + (TAU_HI - TAU_LO) * v);
}

// Pointer stirring. Same splat shape as training; velocity in cells per frame.
const pointer = { down: false, x: 0, y: 0, px: 0, py: 0, last: 0 };
function toCells(e) {
  const r = ui.hybrid.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * N, ((e.clientY - r.top) / r.height) * N];
}
ui.hybrid.addEventListener("pointerdown", (e) => {
  ui.hybrid.setPointerCapture(e.pointerId);
  [pointer.x, pointer.y] = toCells(e);
  pointer.px = pointer.x; pointer.py = pointer.y; pointer.down = true; pointer.last = performance.now();
});
ui.hybrid.addEventListener("pointermove", (e) => {
  if (!pointer.down) return;
  [pointer.x, pointer.y] = toCells(e);
  pointer.last = performance.now();
});
const up = () => (pointer.down = false);
ui.hybrid.addEventListener("pointerup", up);
ui.hybrid.addEventListener("pointercancel", up);

function buildForce() {
  force.fx.fill(0); force.fy.fill(0); force.s.fill(0);
  const clamp = (a) => Math.max(-3.5, Math.min(3.5, a));
  if (pointer.down) {
    const vx = clamp(pointer.x - pointer.px), vy = clamp(pointer.y - pointer.py);
    splat(force, pointer.x, pointer.y, vx, vy, 3.5, 0.07, 0.05);
    pointer.px = pointer.x; pointer.py = pointer.y;
  }
  if (kickNext) {
    // A stir far stronger than anything in training: the case the self-check exists for.
    const a = Math.random() * 2 * Math.PI;
    splat(force, Math.random() * N, Math.random() * N, 3.5 * Math.cos(a), 3.5 * Math.sin(a), 2.5, 0.3, 0.3);
    kickNext = false;
  }
  if (performance.now() - pointer.last > 2500) {
    // Idle: an auto-stirrer on a slow Lissajous path, like the training forcing.
    const t = frame;
    for (const [ax, ay, wx, wy, ph] of [[18, 14, 0.05, 0.07, 1.3], [12, 20, 0.09, 0.04, 4.0]]) {
      const cx = 32 + ax * Math.sin(wx * t + ph), cy = 32 + ay * Math.sin(wy * t + 1.3 * ph);
      const vx = ax * wx * Math.cos(wx * t + ph), vy = ay * wy * Math.cos(wy * t + 1.3 * ph);
      splat(force, cx, cy, vx, vy, 3.5, 0.07, 0.05);
    }
  }
}

function relError(a, b) {
  let num = 0, den = 0;
  for (let c = 0; c < M; c++) {
    num += ERR_W[0] * Math.abs(a.u[c] - b.u[c]) + ERR_W[1] * Math.abs(a.v[c] - b.v[c]) + ERR_W[2] * Math.abs(a.d[c] - b.d[c]);
    den += ERR_W[0] * Math.abs(b.u[c]) + ERR_W[1] * Math.abs(b.v[c]) + ERR_W[2] * Math.abs(b.d[c]);
  }
  return num / (den + 1e-6);
}

// ---------- colour ----------
const hex = (s) => { s = s.trim().replace("#", ""); return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16)); };
const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"].map(hex);
let theme;
function readTheme() {
  const css = getComputedStyle(document.documentElement);
  const v = (k) => hex(css.getPropertyValue(k));
  const dark = css.getPropertyValue("color-scheme").trim() === "dark";
  theme = {
    well: v("--well"), ink: v("--ink"), signal: v("--signal"), trace: css.getPropertyValue("--trace").trim(),
    muted: css.getPropertyValue("--muted").trim(), faint: css.getPropertyValue("--faint").trim(),
    grid: css.getPropertyValue("--grid").trim(), inkCss: css.getPropertyValue("--ink").trim(),
    signalCss: css.getPropertyValue("--signal").trim(),
    // Diverging map for vorticity: two poles and a neutral midpoint.
    neg: dark ? hex("#5b9cff") : hex("#2166ac"), mid: dark ? hex("#2a2e35") : hex("#f4f4f4"), pos: dark ? hex("#ff6b6b") : hex("#b2182b"),
  };
}
readTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readTheme(); drawColorbars(); });

const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  return lerp(stops[i], stops[i + 1], t - i);
}

// Each field: how to read a cell, how to colour a value, and the colorbar labels.
let scale = { speed: 1, vorticity: 0.2 };
const FIELDS = {
  density: {
    value: (s, c) => s.d[c],
    color: (x) => lerp(theme.well, theme.ink, 1 - Math.exp(-2.0 * Math.max(0, x))),
    bar: (t) => lerp(theme.well, theme.ink, 1 - Math.exp(-2.0 * 2 * t)),
    labels: () => ["0", "2", "smoke density d (arbitrary units)"],
  },
  speed: {
    value: (s, c) => Math.hypot(s.u[c], s.v[c]),
    color: (x) => ramp(VIRIDIS, x / scale.speed),
    bar: (t) => ramp(VIRIDIS, t),
    labels: () => ["0", scale.speed.toFixed(2), "speed |u|, cells per substep"],
  },
  vorticity: {
    value: (s, c) => vort[s === hyb ? 0 : 1][c],
    color: (x) => { const t = Math.max(-1, Math.min(1, x / scale.vorticity)); return t < 0 ? lerp(theme.mid, theme.neg, -t) : lerp(theme.mid, theme.pos, t); },
    bar: (t) => (t < 0.5 ? lerp(theme.neg, theme.mid, t * 2) : lerp(theme.mid, theme.pos, (t - 0.5) * 2)),
    labels: () => [`−${scale.vorticity.toFixed(2)}`, `+${scale.vorticity.toFixed(2)}`, "vorticity ω = ∂v/∂x − ∂u/∂y, per substep"],
  },
};
const vort = [new Float32Array(M), new Float32Array(M)];
function computeVorticity(s, out) {
  for (let i = 0; i < N; i++) {
    const im = (i + N - 1) % N, ip = (i + 1) % N;
    for (let j = 0; j < N; j++) {
      const jm = (j + N - 1) % N, jp = (j + 1) % N;
      out[i * N + j] = 0.5 * (s.v[i * N + jp] - s.v[i * N + jm]) - 0.5 * (s.u[ip * N + j] - s.u[im * N + j]);
    }
  }
}
function updateScales() {
  // Both channels share one scale, taken from the reference, smoothed so colours do not flicker.
  let sp = 0, w = 0;
  for (let c = 0; c < M; c++) { sp = Math.max(sp, Math.hypot(ref.u[c], ref.v[c])); w = Math.max(w, Math.abs(vort[1][c])); }
  scale.speed = Math.max(0.05, 0.95 * scale.speed + 0.05 * sp);
  scale.vorticity = Math.max(0.01, 0.95 * scale.vorticity + 0.05 * w);
}

function makeLayer() {
  const c = document.createElement("canvas"); c.width = c.height = N;
  return { c, ctx: c.getContext("2d"), img: new ImageData(N, N) };
}
const layers = { hybrid: makeLayer(), truth: makeLayer() };
function draw(canvas, layer, s, doubt) {
  const f = FIELDS[shown], p = layer.img.data;
  const sig = theme.signal;
  for (let c = 0; c < M; c++) {
    let [r, g, b] = f.color(f.value(s, c));
    if (doubt) {
      const k = Math.min(1, doubt[c] / (doubtScale * 3)) * 0.6;
      r += (sig[0] - r) * k; g += (sig[1] - g) * k; b += (sig[2] - b) * k;
    }
    p[4 * c] = r; p[4 * c + 1] = g; p[4 * c + 2] = b; p[4 * c + 3] = 255;
  }
  layer.ctx.putImageData(layer.img, 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(layer.c, 0, 0, canvas.width, canvas.height);
}

function drawColorbars() {
  const f = FIELDS[shown], [lo, hi, unit] = f.labels();
  for (const bar of ui.cbars) {
    const cv = bar.querySelector("canvas"), ctx = cv.getContext("2d");
    const img = ctx.createImageData(cv.width, 1);
    for (let x = 0; x < cv.width; x++) {
      const [r, g, b] = f.bar(x / (cv.width - 1));
      img.data.set([r, g, b, 255], 4 * x);
    }
    for (let y = 0; y < cv.height; y++) ctx.putImageData(img, 0, y);
    bar.querySelector('[data-cbar="min"]').textContent = lo;
    bar.querySelector('[data-cbar="max"]').textContent = hi;
    bar.querySelector('[data-cbar="unit"]').textContent = unit;
  }
}

// ---------- traces ----------
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
const L = 40, R = 6, T = 6, B = 18; // plot margins
function axes(ctx, w, h, ticks) {
  ctx.font = "10.5px 'IBM Plex Mono', ui-monospace, monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (const [y, label] of ticks) {
    ctx.strokeStyle = theme.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, Math.round(y) + 0.5); ctx.lineTo(w - R, Math.round(y) + 0.5); ctx.stroke();
    ctx.fillStyle = theme.faint; ctx.fillText(label, L - 6, y);
  }
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  ctx.fillText(`t−${HIST}`, L, h - 3);
  ctx.textAlign = "right"; ctx.fillText("now", w - R, h - 3);
}
function line(ctx, xs, ys, color, width, dash) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
  ctx.beginPath();
  xs.forEach((x, i) => (i ? ctx.lineTo(x, ys[i]) : ctx.moveTo(x, ys[i])));
  ctx.stroke(); ctx.setLineDash([]);
}
const SUP = { "-3": "10⁻³", "-2": "10⁻²", "-1": "10⁻¹", "0": "10⁰" };
function drawTraces() {
  const n = hist.doubt.length;
  // Predicted error, log scale from 1e-3 to 1.
  {
    const { ctx, w, h } = fitCanvas(ui.traceDoubt);
    const pw = w - L - R, ph = h - T - B;
    const yOf = (v) => T + ph * (1 - (Math.log10(Math.max(1e-3, Math.min(1, v))) + 3) / 3);
    axes(ctx, w, h, [-3, -2, -1, 0].map((e) => [yOf(10 ** e), SUP[e]]));
    const xOf = (i) => L + pw * (1 - (n - 1 - i) / (HIST - 1));
    const xs = hist.doubt.map((_, i) => xOf(i));
    ctx.fillStyle = theme.signalCss;
    hist.hand.forEach((hnd, i) => { if (hnd) ctx.fillRect(Math.round(xOf(i)) - 1, T + ph - 9, 2, 9); });
    line(ctx, xs, hist.tau.map(yOf), theme.muted, 1.25, [4, 3]);
    line(ctx, xs, hist.doubt.map(yOf), theme.trace, 1.75);
  }
  // Error of A vs B, linear, in percent.
  {
    const { ctx, w, h } = fitCanvas(ui.traceErr);
    const pw = w - L - R, ph = h - T - B;
    const peak = Math.max(...hist.err, 0);
    const top = Math.max(10, Math.ceil(peak / 5) * 5);
    const yOf = (v) => T + ph * (1 - Math.min(v, top) / top);
    axes(ctx, w, h, [0, top / 2, top].map((v) => [yOf(v), `${v}%`]));
    const xOf = (i) => L + pw * (1 - (n - 1 - i) / (HIST - 1));
    line(ctx, hist.err.map((_, i) => xOf(i)), hist.err.map(yOf), theme.inkCss, 1.75);
  }
}

// ---------- controls ----------
function setSafety(on) {
  safety = on;
  ui.netOn.setAttribute("aria-pressed", String(on));
  ui.netOff.setAttribute("aria-pressed", String(!on));
}
ui.netOn.onclick = () => setSafety(true);
ui.netOff.onclick = () => setSafety(false);
ui.doubt.onchange = () => (showDoubt = ui.doubt.checked);
ui.kick.onclick = () => (kickNext = true);
for (const b of ui.fieldBtns) {
  b.onclick = () => {
    shown = b.dataset.field;
    ui.fieldBtns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    drawColorbars();
  };
}
function warmUp() {
  // Start from a stirred flow, as the eval does, instead of an empty tank.
  hyb = field(); ref = field(); recent.length = 0; frame = 0; perf.handoffs = 0;
  for (const k in hist) hist[k].length = 0;
  pointer.last = -1e9;
  for (let k = 0; k < 40; k++) { buildForce(); step(ref, force); frame++; }
  hyb.u.set(ref.u); hyb.v.set(ref.v); hyb.d.set(ref.d);
}
ui.reset.onclick = warmUp;
warmUp();
const showTau = () => (ui.trustV.textContent = `τ = ${tau().toExponential(1)}`);
ui.trust.oninput = showTau;
showTau();
drawColorbars();

// ---------- model ----------
function loadScript(src) {
  return new Promise((ok, fail) => {
    const s = document.createElement("script");
    s.src = src; s.onload = ok; s.onerror = () => fail(new Error(`could not load ${src}`));
    document.head.appendChild(s);
  });
}

async function loadOrt() {
  for (const base of window.ORT_CDNS) {
    try {
      await loadScript(base + "ort.webgpu.min.js");
      ort.env.wasm.wasmPaths = base;
      return;
    } catch (e) { console.warn(e.message); }
  }
  throw new Error("the inference library could not be downloaded. Check your connection and reload.");
}

async function createSession() {
  const model = await fetch("lucid.onnx");
  if (!model.ok) throw new Error(`the model file could not be loaded (HTTP ${model.status}).`);
  const bytes = new Uint8Array(await model.arrayBuffer());
  if ("gpu" in navigator) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
      ui.backend.textContent = "ONNX Runtime, WebGPU";
      return s;
    } catch (e) { console.warn("WebGPU unavailable, using WebAssembly:", e.message); }
  }
  ui.backend.textContent = "ONNX Runtime, WebAssembly";
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

async function main() {
  if (typeof WebAssembly !== "object") throw new Error("this browser does not support WebAssembly.");
  await loadOrt();
  const session = await createSession();
  ui.status.textContent = "";
  let slowFrames = 0;
  const x = new Float32Array(6 * M);
  const ema = (a, b) => (a ? 0.9 * a + 0.1 * b : b);

  async function tick() {
    buildForce();
    x.set(hyb.u, 0); x.set(hyb.v, M); x.set(hyb.d, 2 * M);
    x.set(force.fx, 3 * M); x.set(force.fy, 4 * M); x.set(force.s, 5 * M);
    const t0 = performance.now();
    const out = await session.run({ x: new ort.Tensor("float32", x, [1, 6, N, N]) });
    const tInf = performance.now() - t0;
    perf.inf = ema(perf.inf, tInf);
    slowFrames = tInf > 120 ? slowFrames + 1 : 0;
    if (slowFrames === 30) ui.status.textContent = "This device is running the network slowly, so the fluid will look slow. The results are the same.";
    const y = out.y.data;
    let doubt = 0, size = 0;
    for (let c = 0; c < M; c++) {
      doubtMap[c] = y[3 * M + c]; doubt += doubtMap[c];
      size += ERR_W[0] * Math.abs(hyb.u[c]) + ERR_W[1] * Math.abs(hyb.v[c]) + ERR_W[2] * Math.abs(hyb.d[c]);
    }
    // Relative to the flow's size, as in eval/evaluate.py. The floor (below the 1st percentile of
    // flow sizes in the eval) keeps a nearly empty tank from dividing by zero.
    doubt = doubt / Math.max(size, 0.03 * M);
    const useSolver = safety && doubt > tau();
    if (useSolver) {
      step(hyb, force);
    } else {
      hyb.u.set(y.subarray(0, M)); hyb.v.set(y.subarray(M, 2 * M)); hyb.d.set(y.subarray(2 * M, 3 * M));
    }
    const t1 = performance.now();
    step(ref, force);
    perf.sol = ema(perf.sol, performance.now() - t1);
    frame++;
    debug.frames++; debug.solverFrames += useSolver ? 1 : 0; debug.maxDoubt = Math.max(debug.maxDoubt, doubt);
    if (useSolver) perf.handoffs++;
    recent.push(useSolver ? 1 : 0);
    if (recent.length > 240) recent.shift();
    const err = 100 * relError(hyb, ref);
    hist.doubt.push(doubt); hist.tau.push(tau()); hist.hand.push(useSolver ? 1 : 0); hist.err.push(err);
    if (hist.doubt.length > HIST) for (const k in hist) hist[k].shift();

    if (shown === "vorticity") { computeVorticity(hyb, vort[0]); computeVorticity(ref, vort[1]); }
    else computeVorticity(ref, vort[1]);
    updateScales();
    if (shown !== "density" && frame % 15 === 0) drawColorbars();
    doubtScale = 0.9 * doubtScale + 0.1 * (doubt * size / M + 1e-6);
    draw(ui.hybrid, layers.hybrid, hyb, showDoubt ? doubtMap : null);
    draw(ui.truth, layers.truth, ref, null);
    drawTraces();

    const now = performance.now();
    perf.stamps.push(now);
    while (perf.stamps[0] < now - 1000) perf.stamps.shift();
    const solverFrac = recent.reduce((a, b) => a + b, 0) / recent.length;
    ui.who.textContent = useSolver ? "solver took this frame" : "network";
    ui.who.className = useSolver ? "pill solver" : "pill";
    ui.sNet.textContent = `${Math.round(100 * (1 - solverFrac))}%`;
    ui.sSolver.textContent = `${Math.round(100 * solverFrac)}%`;
    ui.sErr.textContent = `${err.toFixed(1)}%`;
    ui.sDoubt.textContent = doubt.toExponential(1);
    ui.rFrame.textContent = frame;
    ui.rFps.textContent = perf.stamps.length;
    ui.rInf.textContent = perf.inf.toFixed(1);
    ui.rSol.textContent = perf.sol.toFixed(1);
    ui.rHand.textContent = perf.handoffs;
    requestAnimationFrame(safeTick);
  }
  async function safeTick() {
    try { await tick(); }
    catch (e) { fail(e); }
  }
  requestAnimationFrame(safeTick);
}

function fail(e) {
  console.error(e);
  ui.status.innerHTML = "";
  ui.status.append(`Something went wrong: ${e.message} `);
  const b = document.createElement("button");
  b.className = "btn"; b.textContent = "Reload";
  b.onclick = () => location.reload();
  ui.status.append(b);
}

main().catch(fail);
