"""Reference fluid solver: 2D incompressible stable fluids (Stam 1999) on a periodic grid.

This is the "real, slow method" Lucid falls back to. It is batched and runs on CPU or MPS.
sim/solver.js implements the same steps in the same order; tests/test_parity.py checks they agree.

State is a [B, 3, N, N] float32 tensor with channels (u, v, d):
x-velocity, y-velocity (both in cells per substep), and smoke density.
Forcing is a [B, 3, N, N] tensor with channels (fx, fy, s) and is held constant over one frame.
"""

import torch

N = 64
SUBSTEPS = 4          # solver substeps per frame; the network predicts one whole frame
JACOBI_ITERS = 80     # pressure iterations per substep
DT = 1.0
VEL_DECAY = 0.99      # per substep
DENS_DECAY = 0.995    # per substep
VEL_CLAMP = 4.0       # cells per substep


def _l(f):  # f[i, j-1]
    return torch.roll(f, 1, -1)


def _r(f):  # f[i, j+1]
    return torch.roll(f, -1, -1)


def _u(f):  # f[i-1, j]
    return torch.roll(f, 1, -2)


def _d(f):  # f[i+1, j]
    return torch.roll(f, -1, -2)


def _sample(f, x, y):
    """Bilinear sample of f [B, n, n] at positions x, y (cells), wrapping at the edges."""
    b, n = f.shape[0], f.shape[-1]
    x0f = torch.floor(x)
    y0f = torch.floor(y)
    tx = x - x0f
    ty = y - y0f
    x0 = x0f.long() % n
    y0 = y0f.long() % n
    x1 = (x0 + 1) % n
    y1 = (y0 + 1) % n
    flat = f.reshape(b, -1)

    def g(yy, xx):
        return torch.gather(flat, 1, (yy * n + xx).reshape(b, -1)).reshape(b, n, n)

    top = (1 - tx) * g(y0, x0) + tx * g(y0, x1)
    bot = (1 - tx) * g(y1, x0) + tx * g(y1, x1)
    return (1 - ty) * top + ty * bot


def _grid(n, device):
    i = torch.arange(n, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(i, i, indexing="ij")
    return xx, yy


def project(u, v, iters=JACOBI_ITERS):
    div = 0.5 * ((_r(u) - _l(u)) + (_d(v) - _u(v)))
    p = torch.zeros_like(u)
    for _ in range(iters):
        p = (_l(p) + _r(p) + _u(p) + _d(p) - div) / 4.0
    u = u - 0.5 * (_r(p) - _l(p))
    v = v - 0.5 * (_d(p) - _u(p))
    return u, v


def substep(u, v, d, fx, fy, s, iters=JACOBI_ITERS):
    u = u + DT * fx
    v = v + DT * fy
    d = d + DT * s
    u = torch.clamp(u, -VEL_CLAMP, VEL_CLAMP)
    v = torch.clamp(v, -VEL_CLAMP, VEL_CLAMP)
    xx, yy = _grid(u.shape[-1], u.device)
    bx = xx - DT * u
    by = yy - DT * v
    u2 = _sample(u, bx, by)
    v2 = _sample(v, bx, by)
    d2 = _sample(d, bx, by)
    u2, v2 = project(u2, v2, iters)
    return u2 * VEL_DECAY, v2 * VEL_DECAY, d2 * DENS_DECAY


def step(state, force, substeps=SUBSTEPS, iters=JACOBI_ITERS):
    """Advance one frame. state, force: [B, 3, N, N]. Returns the next state."""
    u, v, d = state[:, 0], state[:, 1], state[:, 2]
    fx, fy, s = force[:, 0], force[:, 1], force[:, 2]
    for _ in range(substeps):
        u, v, d = substep(u, v, d, fx, fy, s, iters)
    return torch.stack([u, v, d], 1)
