"""Measure whether Lucid knows when it is wrong, and what that buys.

Held-out episodes (their own seed) start from a solver warm-up. The reference runs the solver on
every frame. A hybrid runs Lucid each frame and hands the frame to the solver when a gate fires.
Two scenarios:
  normal  forcing drawn from the training distribution
  kicks   the same, plus sudden strong stirs on 10% of frames, outside anything seen in training
Gates, compared at matched solver-call rates (scores are relative to the current state's size):
  learned   Lucid's own predicted error
  residual  divergence of Lucid's predicted velocity (a physics check, no learning)
  input     how far this frame's forcing exceeds the training range (a hand-written rule)
  oracle    the true one-step error (needs the solver every frame, so not usable in practice)
  random    the solver on a random fraction of frames
  schedule  the solver every k-th frame

python3 -m eval.evaluate --ckpt ckpt/lucid.pt --out results/eval.json
"""

import argparse
import json

import torch

from lucid.model import Lucid, state_error
from sim.forcing import Stirrers, splat
from sim.solver import N, step

WARMUP = 40
TRAIN_FORCE_MAX = 0.06 * 3.75  # strength x stirrer speed at the edge of the training range


def episodes(n, frames, seed, kicks=False):
    gen = torch.Generator().manual_seed(seed)
    st = Stirrers(n, "cpu", gen)
    x = torch.zeros(n, 3, N, N)
    for t in range(WARMUP):
        x = step(x, st.force(t))
    forces = [st.force(WARMUP + t) for t in range(frames)]
    if kicks:
        kg = torch.Generator().manual_seed(seed + 1)
        for t in range(frames):
            on = (torch.rand(n, generator=kg) < 0.1).float().view(n, 1, 1, 1)
            cx, cy = torch.rand(n, 1, generator=kg) * N, torch.rand(n, 1, generator=kg) * N
            ang = torch.rand(n, 1, generator=kg) * 2 * torch.pi
            f = splat(cx, cy, 3.5 * torch.cos(ang), 3.5 * torch.sin(ang), torch.full((n, 1), 2.5),
                      torch.full((n, 1), 0.3), torch.full((n, 1), 0.3))
            forces[t] = forces[t] + f * on
    ref = [x]
    for f in forces:
        ref.append(step(ref[-1], f))
    return forces, ref


def divergence(y):
    u, v = y[:, 0], y[:, 1]
    du = torch.roll(u, -1, -1) - torch.roll(u, 1, -1)
    dv = torch.roll(v, -1, -2) - torch.roll(v, 1, -2)
    return (0.5 * (du + dv)).abs().mean((1, 2))


def size(h):
    return state_error(h, torch.zeros_like(h)).mean((1, 2)) + 1e-6


def rel_error(h, r):
    """Mean per-cell state error relative to the reference's own size, per episode."""
    return state_error(h, r).mean((1, 2)) / size(r)


SCORES = {
    "learned": lambda h, f, y: y[:, 3].mean((1, 2)) / size(h),
    "residual": lambda h, f, y: divergence(y[:, :3]) / size(h),
    "input": lambda h, f, y: (f[:, :2].abs().amax((1, 2, 3)) / TRAIN_FORCE_MAX),
    "oracle": lambda h, f, y: state_error(y[:, :3], step(h, f)).mean((1, 2)) / size(h),
}


@torch.no_grad()
def run(model, forces, ref, score=None, thr=None, rate=None, every=None, rng=None):
    """Roll a hybrid; returns per-frame mean relative error and the fraction of solver calls."""
    h = ref[0].clone()
    errs, calls = [], 0.0
    for t, f in enumerate(forces):
        y = model(torch.cat([h, f], 1))
        pred = y[:, :3].clone()
        if score is not None:
            use = SCORES[score](h, f, y) > thr
        elif rate is not None:
            use = torch.rand(h.shape[0], generator=rng) < rate
        elif every is not None:
            use = torch.full((h.shape[0],), t % every == 0)
        else:
            use = torch.zeros(h.shape[0], dtype=torch.bool)
        if use.any():
            pred[use] = step(h[use], f[use])
        h = pred
        calls += use.float().mean().item()
        errs.append(rel_error(h, ref[t + 1]).mean().item())
    return errs, calls / len(forces)


@torch.no_grad()
def pure_scores(model, forces, ref):
    """Gate scores and true one-step errors along a pure Lucid rollout."""
    h = ref[0].clone()
    sc = {k: [] for k in SCORES}
    true = []
    for f in forces:
        y = model(torch.cat([h, f], 1))
        for k, fn in SCORES.items():
            sc[k].append(fn(h, f, y))
        true.append(state_error(y[:, :3], step(h, f)).mean((1, 2)) / size(h))
        h = y[:, :3]
    return {k: torch.cat(v) for k, v in sc.items()}, torch.cat(true)


def auroc(score, label):
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=score.dtype)
    pos = label.sum().item()
    neg = len(label) - pos
    return (ranks[label].sum().item() - pos * (pos + 1) / 2) / (pos * neg)


def point(errs, calls):
    return {"calls": round(calls, 4), "error": round(sum(errs) / len(errs), 5)}


def scenario(model, args, kicks):
    forces, ref = episodes(args.episodes, args.frames, args.seed, kicks)
    sc, true = pure_scores(model, forces, ref)
    label = true > torch.quantile(true, 0.8)
    errs, _ = run(model, forces, ref)
    out = {"auroc": {k: round(auroc(v, label), 4) for k, v in sc.items() if k != "oracle"},
           "pure_lucid": {"error": round(sum(errs) / len(errs), 5), "error_by_frame": errs},
           "curves": {}}
    for k, v in sc.items():
        out["curves"][k] = [point(*run(model, forces, ref, score=k, thr=torch.quantile(v, q).item()))
                            for q in (0.95, 0.9, 0.8, 0.7, 0.5)]
    rng = torch.Generator().manual_seed(1)
    out["curves"]["random"] = [point(*run(model, forces, ref, rate=p, rng=rng)) for p in (0.05, 0.1, 0.2, 0.3, 0.5)]
    out["curves"]["schedule"] = [point(*run(model, forces, ref, every=k)) for k in (20, 10, 5, 3, 2)]
    return out


def main(args):
    torch.manual_seed(0)
    model = Lucid()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    out = {"episodes": args.episodes, "frames": args.frames, "seed": args.seed}
    for name, kicks in (("normal", False), ("kicks", True)):
        out[name] = scenario(model, args, kicks)
        print(name, "auroc", out[name]["auroc"], "pure", out[name]["pure_lucid"]["error"])
        for g, c in out[name]["curves"].items():
            print("  ", g, [(p["calls"], p["error"]) for p in c])
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/lucid.pt")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/eval.json")
    main(ap.parse_args())
