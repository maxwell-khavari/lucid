// Reference fluid solver, the same algorithm as sim/solver.py, step for step.
// Runs in the browser (as the fallback) and in node (for the parity test).
// Fields are Float32Array(N*N), row-major [y][x], periodic edges.

export const N = 64;
export const SUBSTEPS = 4;
export const JACOBI_ITERS = 80;
const DT = 1.0;
const VEL_DECAY = 0.99;
const DENS_DECAY = 0.995;
const VEL_CLAMP = 4.0;

const wrap = (a, n) => ((a % n) + n) % n;

function sample(f, x, y, n) {
  const x0f = Math.floor(x), y0f = Math.floor(y);
  const tx = Math.fround(x - x0f), ty = Math.fround(y - y0f);
  const x0 = wrap(x0f, n), y0 = wrap(y0f, n);
  const x1 = (x0 + 1) % n, y1 = (y0 + 1) % n;
  const top = Math.fround((1 - tx) * f[y0 * n + x0] + tx * f[y0 * n + x1]);
  const bot = Math.fround((1 - tx) * f[y1 * n + x0] + tx * f[y1 * n + x1]);
  return (1 - ty) * top + ty * bot;
}

function project(u, v, n, iters, scratch) {
  const { div, p, q } = scratch;
  for (let i = 0; i < n; i++) {
    const im = wrap(i - 1, n), ip = (i + 1) % n;
    for (let j = 0; j < n; j++) {
      const jm = wrap(j - 1, n), jp = (j + 1) % n;
      div[i * n + j] = 0.5 * ((u[i * n + jp] - u[i * n + jm]) + (v[ip * n + j] - v[im * n + j]));
    }
  }
  p.fill(0);
  let a = p, b = q;
  for (let k = 0; k < iters; k++) {
    for (let i = 0; i < n; i++) {
      const im = wrap(i - 1, n), ip = (i + 1) % n;
      for (let j = 0; j < n; j++) {
        const jm = wrap(j - 1, n), jp = (j + 1) % n;
        b[i * n + j] = (a[i * n + jm] + a[i * n + jp] + a[im * n + j] + a[ip * n + j] - div[i * n + j]) / 4.0;
      }
    }
    const t = a; a = b; b = t;
  }
  for (let i = 0; i < n; i++) {
    const im = wrap(i - 1, n), ip = (i + 1) % n;
    for (let j = 0; j < n; j++) {
      const jm = wrap(j - 1, n), jp = (j + 1) % n;
      b[i * n + j] = u[i * n + j] - 0.5 * (a[i * n + jp] - a[i * n + jm]);
      div[i * n + j] = v[i * n + j] - 0.5 * (a[ip * n + j] - a[im * n + j]);
    }
  }
  u.set(b);
  v.set(div);
}

function makeScratch(n) {
  const m = n * n;
  return { div: new Float32Array(m), p: new Float32Array(m), q: new Float32Array(m),
    u2: new Float32Array(m), v2: new Float32Array(m), d2: new Float32Array(m) };
}

let cached = null;

// Advance one frame in place. state = {u, v, d}, force = {fx, fy, s}.
export function step(state, force, substeps = SUBSTEPS, iters = JACOBI_ITERS) {
  const n = N, m = n * n;
  if (!cached) cached = makeScratch(n);
  const sc = cached;
  const { u, v, d } = state;
  const { fx, fy, s } = force;
  for (let k = 0; k < substeps; k++) {
    for (let c = 0; c < m; c++) {
      u[c] = Math.min(Math.max(u[c] + DT * fx[c], -VEL_CLAMP), VEL_CLAMP);
      v[c] = Math.min(Math.max(v[c] + DT * fy[c], -VEL_CLAMP), VEL_CLAMP);
      d[c] = d[c] + DT * s[c];
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const c = i * n + j;
        const bx = Math.fround(j - DT * u[c]), by = Math.fround(i - DT * v[c]);
        sc.u2[c] = sample(u, bx, by, n);
        sc.v2[c] = sample(v, bx, by, n);
        sc.d2[c] = sample(d, bx, by, n);
      }
    }
    project(sc.u2, sc.v2, n, iters, sc);
    for (let c = 0; c < m; c++) {
      u[c] = sc.u2[c] * VEL_DECAY;
      v[c] = sc.v2[c] * VEL_DECAY;
      d[c] = sc.d2[c] * DENS_DECAY;
    }
  }
  return state;
}

// Gaussian splat with periodic distance; the same formula sim/forcing.py uses for training.
export function splat(force, cx, cy, vx, vy, radius, amount, strength) {
  const n = N;
  for (let i = 0; i < n; i++) {
    let dy = Math.abs(i - cy) % n; dy = Math.min(dy, n - dy);
    for (let j = 0; j < n; j++) {
      let dx = Math.abs(j - cx) % n; dx = Math.min(dx, n - dx);
      const w = Math.exp(-(dx * dx + dy * dy) / (2 * radius * radius));
      const c = i * n + j;
      force.fx[c] += w * vx * strength;
      force.fy[c] += w * vy * strength;
      force.s[c] += w * amount;
    }
  }
}
