"""The browser solver (sim/solver.js) must match the training solver (sim/solver.py).
Run: python3 -m unittest discover tests"""

import json
import pathlib
import subprocess
import unittest

import torch

from sim.forcing import Stirrers
from sim.solver import N, step

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestParity(unittest.TestCase):
    def test_js_matches_python(self):
        gen = torch.Generator().manual_seed(7)
        st = Stirrers(1, "cpu", gen)
        x = torch.zeros(1, 3, N, N)
        for t in range(40):  # warm up so the start state has real structure
            x = step(x, st.force(t))
        start = x.clone()
        forces = [st.force(40 + t) for t in range(10)]
        ref = []
        for f in forces:
            x = step(x, f)
            ref.append(x[0].flatten())
        payload = json.dumps({"state": start[0].flatten().tolist(),
                              "forces": [f[0].flatten().tolist() for f in forces]})
        out = subprocess.run(["node", str(ROOT / "tests" / "parity.mjs")], input=payload,
                             capture_output=True, text=True, check=True).stdout
        js = torch.tensor(json.loads(out))
        for t, r in enumerate(ref):
            scale = r.abs().max().item()
            err = (js[t] - r).abs().max().item()
            self.assertLess(err / scale, 1e-4, f"frame {t}: max abs diff {err:.2e} vs scale {scale:.2e}")


if __name__ == "__main__":
    unittest.main()
