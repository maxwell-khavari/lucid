"""Train Lucid. All data is generated on the fly by the reference solver; nothing is read from disk.

Each training input is either a true solver state or a state the model itself drifted into by
rolling forward r frames. The target is always one solver frame from that input, so the network
learns the solver's step and its error head learns how wrong it is, including on its own drift.

python3 train.py --steps 20000 --out ckpt/lucid.pt
"""

import argparse
import collections
import json
import math
import pathlib
import random
import threading
import time

import torch
import torch.nn.functional as F

from lucid.model import OUT_SCALE, Lucid, state_error
from sim.forcing import Stirrers
from sim.solver import N, step

K = 4  # frames per stored window, so the model can roll up to K-1 frames before the target


class Pool:
    """P episodes advanced in lockstep on the CPU by a background thread."""

    def __init__(self, p, replay, seed):
        self.p = p
        self.gen = torch.Generator().manual_seed(seed)
        self.st = Stirrers(p, "cpu", self.gen)
        self.x = torch.zeros(p, 3, N, N)
        self.t = torch.zeros(p, dtype=torch.long)
        self.buf = collections.deque(maxlen=replay)
        self.lock = threading.Lock()
        self.stop = False
        # Warm the pool up with staggered ages so it is not full of empty tanks.
        for i in range(60):
            self._advance()
            reset = torch.rand(p, generator=self.gen) < 0.02
            self._reset(reset)

    def _reset(self, mask):
        if mask.any():
            self.x[mask] = 0
            self.t[mask] = 0
            self.st.resample(mask)

    def _advance(self):
        f = self.st.force(self.t)
        self.x = step(self.x, f)
        self.t += 1
        return f

    def tick(self):
        xs, fs = [self.x.clone()], []
        for _ in range(K):
            fs.append(self._advance())
            xs.append(self.x.clone())
        self._reset(torch.rand(self.p, generator=self.gen) < 0.03)
        win = (torch.stack(xs, 1).half(), torch.stack(fs, 1).half())
        with self.lock:
            self.buf.append(win)

    def run(self):
        while not self.stop:
            self.tick()

    def sample(self, b):
        with self.lock:
            xs, fs = random.choice(self.buf)
        idx = torch.randperm(self.p)[:b]
        return xs[idx].float(), fs[idx].float()


def norm(x):
    return x / torch.tensor(OUT_SCALE, device=x.device).view(1, 3, 1, 1)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(8)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    pool = Pool(args.pool, args.replay, args.seed)
    for _ in range(4):
        pool.tick()
    th = threading.Thread(target=pool.run, daemon=True)
    th.start()

    model = Lucid().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a")
    t0 = time.time()
    for it in range(1, args.steps + 1):
        lr = args.lr * min(1.0, it / 300) * 0.5 * (1 + math.cos(math.pi * it / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        xs, fs = pool.sample(args.batch)
        # Curriculum: one-step only at first, then mix in the model's own drift.
        max_r = 0 if it < args.warm else K - 1
        r = 0 if random.random() < 0.75 else random.randint(0, max_r)
        if r == 0:
            inp, f, target = xs[:, 0], fs[:, 0], xs[:, 1]
        else:
            with torch.no_grad():
                h = xs[:, 0].to(dev)
                model.eval()
                for k in range(r):
                    h = model(torch.cat([h, fs[:, k].to(dev)], 1))[:, :3]
                model.train()
            inp, f = h.cpu(), fs[:, r]
            target = step(inp, f)
        inp, f, target = inp.to(dev), f.to(dev), target.to(dev)
        y = model(torch.cat([inp, f], 1))
        pred, err_hat = y[:, :3], y[:, 3]
        loss_pred = F.mse_loss(norm(pred - inp), norm(target - inp))
        e = state_error(pred.detach(), target)
        loss_err = F.smooth_l1_loss(torch.log(err_hat + 1e-5), torch.log(e + 1e-5))
        loss = loss_pred + args.err_weight * loss_err
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            rec = {"step": it, "r": r, "loss_pred": loss_pred.item(), "loss_err": loss_err.item(),
                   "mean_err": e.mean().item(), "lr": lr, "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if it % 500 == 0:
                print(rec, flush=True)
        if it % 2000 == 0 or it == args.steps:
            torch.save({"model": model.state_dict(), "step": it, "args": vars(args)}, out)
    pool.stop = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pool", type=int, default=192)
    ap.add_argument("--replay", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--err-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/lucid.pt")
    ap.add_argument("--log", default="results/train_log.jsonl")
    train(ap.parse_args())
