"""Lucid: a small U-Net that replaces one solver frame and predicts its own error.

Input  [B, 6, N, N]: state (u, v, d) and this frame's forcing (fx, fy, s), in raw units.
Output [B, 4, N, N]: the next state (u, v, d) in raw units, and a per-cell predicted error.
Normalization lives inside the model so the exported ONNX file takes raw fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.solver import VEL_CLAMP

IN_SCALE = (1.0, 1.0, 0.5, 5.0, 5.0, 10.0)
OUT_SCALE = (0.025, 0.025, 0.08)    # typical per-frame change of (u, v, d); the head predicts change / OUT_SCALE
ERR_WEIGHT = (1.0, 1.0, 0.5)         # per-channel weights of the error the head predicts


def state_error(pred, true):
    """Per-cell error [B, N, N] in normalized units; the quantity the error head learns."""
    w = torch.tensor(ERR_WEIGHT, device=pred.device).view(1, 3, 1, 1)
    return ((pred - true).abs() * w).sum(1)


def cpad(x):
    # Circular padding written as concatenation so it exports cleanly to ONNX.
    x = torch.cat([x[..., -1:], x, x[..., :1]], -1)
    return torch.cat([x[..., -1:, :], x, x[..., :1, :]], -2)


class Conv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=0)

    def forward(self, x):
        return F.silu(self.conv(cpad(x)))


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a, self.b = Conv(cin, cout), Conv(cout, cout)

    def forward(self, x):
        return self.b(self.a(x))


class Lucid(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192)):
        super().__init__()
        w = widths
        self.register_buffer("in_scale", torch.tensor(IN_SCALE).view(1, 6, 1, 1))
        self.register_buffer("out_scale", torch.tensor(OUT_SCALE).view(1, 3, 1, 1))
        self.enc = nn.ModuleList([Block(6, w[0])] + [Block(w[i], w[i + 1]) for i in range(len(w) - 1)])
        self.down = nn.ModuleList([Conv(w[i], w[i], stride=2) for i in range(len(w) - 1)])
        self.dec = nn.ModuleList([Block(w[i + 1] + w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.head = nn.Conv2d(w[0], 3, 1)
        self.err = nn.Conv2d(w[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.err.weight)
        nn.init.constant_(self.err.bias, -4.0)  # start near typical per-cell error, not at random

    def forward(self, x):
        state = x[:, :3]
        h = x * self.in_scale
        skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        for blk in self.dec:
            h = F.interpolate(h, scale_factor=2.0, mode="nearest")
            h = blk(torch.cat([h, skips.pop()], 1))
        nxt = state + self.head(h) * self.out_scale
        uv = torch.clamp(nxt[:, :2], -VEL_CLAMP, VEL_CLAMP)
        d = F.relu(nxt[:, 2:3])
        err = torch.exp(torch.clamp(self.err(h), -12.0, 4.0))
        return torch.cat([uv, d, err], 1)


if __name__ == "__main__":
    m = Lucid()
    print(sum(p.numel() for p in m.parameters()), "params")
    print(m(torch.zeros(2, 6, 64, 64)).shape)
