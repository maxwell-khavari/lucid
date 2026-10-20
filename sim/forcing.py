"""Random stirring for training data: a few stirrers move along smooth Lissajous paths,
pushing the fluid and injecting smoke. Splats use the same formula as splat() in sim/solver.js,
so mouse input in the browser looks like the forcing the model was trained on."""

import math

import torch

from sim.solver import N

MAX_STIRRERS = 4


def splat(cx, cy, vx, vy, radius, amount, strength, n=N):
    """Batched Gaussian splats. Every argument is a [B, K] tensor; returns a [B, 3, n, n] force."""
    dev = cx.device
    i = torch.arange(n, device=dev, dtype=torch.float32)
    dx = (i.view(1, 1, 1, n) - cx[..., None, None]).abs() % n
    dx = torch.minimum(dx, n - dx)
    dy = (i.view(1, 1, n, 1) - cy[..., None, None]).abs() % n
    dy = torch.minimum(dy, n - dy)
    w = torch.exp(-(dx * dx + dy * dy) / (2 * radius[..., None, None] ** 2))
    fx = (w * (vx * strength)[..., None, None]).sum(1)
    fy = (w * (vy * strength)[..., None, None]).sum(1)
    s = (w * amount[..., None, None]).sum(1)
    return torch.stack([fx, fy, s], 1)


class Stirrers:
    """B independent episodes of random stirring. Call force(t) for each frame t."""

    def __init__(self, b, device, gen=None):
        self.b, self.dev, self.gen = b, device, gen
        self.resample(torch.ones(b, dtype=torch.bool, device=device))

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.dev, generator=self.gen)

    def resample(self, mask):
        b, k = self.b, MAX_STIRRERS
        new = {
            "c": self._rand(b, k, 2) * N,
            "amp": 4 + self._rand(b, k, 2) * 20,
            "w": 0.02 + self._rand(b, k, 2) * 0.13,
            "ph": self._rand(b, k, 2) * 2 * math.pi,
            "r": 2 + self._rand(b, k) * 3,
            "amount": self._rand(b, k) * 0.08,
            "strength": 0.01 + self._rand(b, k) * 0.05,
            "on": self._rand(b, k) < 0.6,
        }
        new["on"][:, 0] = True
        if not hasattr(self, "p"):
            self.p = new
            return
        for key, val in new.items():
            m = mask.view(b, *([1] * (val.dim() - 1)))
            self.p[key] = torch.where(m, val, self.p[key])

    def force(self, t):
        p = self.p
        flip = self._rand(self.b, MAX_STIRRERS) < 0.01
        p["on"] = p["on"] ^ flip
        t = t.float()[:, None, None] if torch.is_tensor(t) else float(t)
        pos = p["c"] + p["amp"] * torch.sin(p["w"] * t + p["ph"])
        vel = p["amp"] * p["w"] * torch.cos(p["w"] * t + p["ph"])
        on = p["on"].float()
        return splat(pos[..., 0] % N, pos[..., 1] % N, vel[..., 0], vel[..., 1],
                     p["r"], p["amount"] * on, p["strength"] * on)
