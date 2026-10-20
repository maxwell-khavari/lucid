// Reads {state: [3*N*N], forces: [[3*N*N], ...]} on stdin, runs the JS solver, prints the states.
import { step, N } from "../sim/solver.js";

let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => {
  const { state, forces } = JSON.parse(raw);
  const m = N * N;
  const s = {
    u: Float32Array.from(state.slice(0, m)),
    v: Float32Array.from(state.slice(m, 2 * m)),
    d: Float32Array.from(state.slice(2 * m)),
  };
  const out = [];
  for (const f of forces) {
    step(s, { fx: Float32Array.from(f.slice(0, m)), fy: Float32Array.from(f.slice(m, 2 * m)), s: Float32Array.from(f.slice(2 * m)) });
    out.push([...s.u, ...s.v, ...s.d]);
  }
  process.stdout.write(JSON.stringify(out));
});
