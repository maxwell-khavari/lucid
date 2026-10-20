// Lucid demo: the network runs the left pane, the real solver runs the right pane on the same input.
import { step, splat, N } from "../sim/solver.js";

const M = N * N;
const ERR_W = [1, 1, 0.5]; // matches ERR_WEIGHT in lucid/model.py
const TAU_LO = -2.5, TAU_HI = -0.5; // log10 range of the doubt threshold (predicted error / flow size)

const $ = (id) => document.getElementById(id);
const ui = {
  hybrid: $("hybrid"), truth: $("truth"), who: $("who"), status: $("status"),
  netOn: $("net-on"), netOff: $("net-off"), doubt: $("doubt"), reset: $("reset"), kick: $("kick"),
  trust: $("trust"), trustV: $("trust-v"),
  sNet: $("s-net"), sSolver: $("s-solver"), sErr: $("s-err"), sDoubt: $("s-doubt"),
};

const field = () => ({ u: new Float32Array(M), v: new Float32Array(M), d: new Float32Array(M) });
let hyb = field(), ref = field();
const force = { fx: new Float32Array(M), fy: new Float32Array(M), s: new Float32Array(M) };
const doubtMap = new Float32Array(M);
let safety = true, showDoubt = false, frame = 0, kickNext = false;
const recent = []; // 1 when the solver took the frame
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

// Rendering: smoke as ink on paper, doubt as a thin accent tint.
const css = getComputedStyle(document.documentElement);
const hex = (s) => { s = s.trim().replace("#", ""); return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16)); };
let paper = hex(css.getPropertyValue("--panel")), ink = hex(css.getPropertyValue("--ink")), acc = hex(css.getPropertyValue("--accent"));
function makeLayer() {
  const c = document.createElement("canvas"); c.width = c.height = N;
  return { c, ctx: c.getContext("2d"), img: new ImageData(N, N) };
}
const layers = { hybrid: makeLayer(), truth: makeLayer() };
function draw(canvas, layer, s, doubt) {
  const p = layer.img.data;
  for (let c = 0; c < M; c++) {
    const a = 1 - Math.exp(-2.0 * Math.max(0, s.d[c]));
    let r = paper[0] + (ink[0] - paper[0]) * a, g = paper[1] + (ink[1] - paper[1]) * a, b = paper[2] + (ink[2] - paper[2]) * a;
    if (doubt) {
      const k = Math.min(1, doubt[c] / (doubtScale * 3)) * 0.6;
      r += (acc[0] - r) * k; g += (acc[1] - g) * k; b += (acc[2] - b) * k;
    }
    p[4 * c] = r; p[4 * c + 1] = g; p[4 * c + 2] = b; p[4 * c + 3] = 255;
  }
  layer.ctx.putImageData(layer.img, 0, 0);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(layer.c, 0, 0, canvas.width, canvas.height);
}

function setSafety(on) {
  safety = on;
  ui.netOn.setAttribute("aria-pressed", String(on));
  ui.netOff.setAttribute("aria-pressed", String(!on));
}
ui.netOn.onclick = () => setSafety(true);
ui.netOff.onclick = () => setSafety(false);
ui.doubt.onchange = () => (showDoubt = ui.doubt.checked);
ui.kick.onclick = () => (kickNext = true);
function warmUp() {
  // Start from a stirred flow, as the eval does, instead of an empty tank.
  hyb = field(); ref = field(); recent.length = 0; frame = 0;
  pointer.last = -1e9;
  for (let k = 0; k < 40; k++) { buildForce(); step(ref, force); frame++; }
  hyb.u.set(ref.u); hyb.v.set(ref.v); hyb.d.set(ref.d);
}
ui.reset.onclick = warmUp;
warmUp();
const showTau = () => (ui.trustV.textContent = tau().toExponential(1));
ui.trust.oninput = showTau;
showTau();

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
    try { return await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] }); }
    catch (e) { console.warn("WebGPU unavailable, using WebAssembly:", e.message); }
  }
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}

async function main() {
  if (typeof WebAssembly !== "object") throw new Error("this browser does not support WebAssembly.");
  await loadOrt();
  const session = await createSession();
  ui.status.textContent = "";
  let slowFrames = 0;
  const x = new Float32Array(6 * M);

  async function tick() {
    buildForce();
    x.set(hyb.u, 0); x.set(hyb.v, M); x.set(hyb.d, 2 * M);
    x.set(force.fx, 3 * M); x.set(force.fy, 4 * M); x.set(force.s, 5 * M);
    const t0 = performance.now();
    const out = await session.run({ x: new ort.Tensor("float32", x, [1, 6, N, N]) });
    slowFrames = performance.now() - t0 > 120 ? slowFrames + 1 : 0;
    if (slowFrames === 30) ui.status.textContent = "This device is running the network slowly, so the fluid will look slow. The results are the same.";
    const y = out.y.data;
    let doubt = 0;
    let size = 0;
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
    step(ref, force);
    frame++;
    debug.frames++; debug.solverFrames += useSolver ? 1 : 0; debug.maxDoubt = Math.max(debug.maxDoubt, doubt);
    recent.push(useSolver ? 1 : 0);
    if (recent.length > 240) recent.shift();

    doubtScale = 0.9 * doubtScale + 0.1 * (doubt * size / M + 1e-6);
    draw(ui.hybrid, layers.hybrid, hyb, showDoubt ? doubtMap : null);
    draw(ui.truth, layers.truth, ref, null);
    const solverFrac = recent.reduce((a, b) => a + b, 0) / recent.length;
    ui.who.textContent = useSolver ? "solver took this frame" : "network";
    ui.who.className = useSolver ? "solver" : "";
    ui.sNet.textContent = `${Math.round(100 * (1 - solverFrac))}%`;
    ui.sSolver.textContent = `${Math.round(100 * solverFrac)}%`;
    ui.sErr.textContent = `${(100 * relError(hyb, ref)).toFixed(1)}%`;
    ui.sDoubt.textContent = doubt.toExponential(1);
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
  b.className = "plain"; b.textContent = "Reload";
  b.onclick = () => location.reload();
  ui.status.append(b);
}

main().catch(fail);
