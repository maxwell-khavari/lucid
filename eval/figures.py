"""Extra figures: where the network thinks it is wrong, and a timeline of one kicked episode.

python3 -m eval.figures
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from eval.evaluate import SCORES, rel_error  # noqa: E402
from eval.gif import KICKS, THRESHOLD  # noqa: E402
from lucid.model import Lucid, state_error  # noqa: E402
from sim.forcing import Stirrers, splat  # noqa: E402
from sim.solver import N, step  # noqa: E402

BG, INK, MUTED, GRID, SIGNAL, TRACE = "#ffffff", "#0e1116", "#5b6472", "#e3e6ea", "#c2189b", "#2f6fde"
SMOKE = LinearSegmentedColormap.from_list("smoke", ["#e9ecf0", "#0e1116"])
HEAT = LinearSegmentedColormap.from_list("heat", ["#ffffff", "#f3c6e6", "#c2189b", "#4a0a3c"])


def load():
    m = Lucid()
    m.load_state_dict(torch.load("weights/lucid.pt", map_location="cpu")["model"])
    return m.eval()


def kick_force(t, f):
    if t not in KICKS:
        return f
    cx, cy, ang = KICKS[t]
    one = lambda v: torch.tensor([[float(v)]])
    return f + splat(one(cx), one(cy), one(3.5 * np.cos(ang)), one(3.5 * np.sin(ang)), one(2.5), one(0.3), one(0.3))


@torch.no_grad()
def episode(model, frames=150):
    st = Stirrers(1, "cpu", torch.Generator().manual_seed(3))
    x = torch.zeros(1, 3, N, N)
    for t in range(40):
        x = step(x, st.force(t))
    off, on, ref = x.clone(), x.clone(), x.clone()
    rec = {"doubt": [], "hand": [], "err_off": [], "err_on": [], "snap": None}
    for t in range(frames):
        f = kick_force(t, st.force(40 + t))
        y_off = model(torch.cat([off, f], 1))
        if t == min(KICKS):
            truth = step(off, f)
            rec["snap"] = {"pred": y_off[0, 2].numpy(), "ref": truth[0, 2].numpy(),
                           "pred_err": y_off[0, 3].numpy(), "true_err": state_error(y_off[:, :3], truth)[0].numpy()}
        off = y_off[:, :3]
        y = model(torch.cat([on, f], 1))
        d = SCORES["learned"](on, f, y).item()
        used = d > THRESHOLD
        on = step(on, f) if used else y[:, :3]
        ref = step(ref, f)
        rec["doubt"].append(d)
        rec["hand"].append(used)
        rec["err_off"].append(100 * rel_error(off, ref).item())
        rec["err_on"].append(100 * rel_error(on, ref).item())
    return rec


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)


def where_wrong(snap, out="results/where_wrong.png"):
    pe, te = snap["pred_err"], snap["true_err"]
    r = np.corrcoef(pe.ravel(), te.ravel())[0, 1]
    vmax = np.quantile(te, 0.995)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), facecolor=BG)
    panels = [(snap["pred"], SMOKE, (0, 2), "Network's next frame"),
              (pe, HEAT, (0, vmax), "Where it thinks it's wrong (ε̂)"),
              (te, HEAT, (0, vmax), "Where it is actually wrong"),
              (snap["ref"], SMOKE, (0, 2), "Real solver")]
    for ax, (img, cmap, (lo, hi), title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="bicubic")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.suptitle(f"The frame a strong stir hits. Predicted and actual error maps share one colour scale; "
                 f"pixel correlation r = {r:.2f}.", x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG)
    print("wrote", out, "r =", round(r, 3))


def timeline(rec, out="results/timeline.png"):
    t = np.arange(len(rec["doubt"]))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, facecolor=BG,
                                 gridspec_kw={"height_ratios": [1, 1.1]})
    for ax in (a1, a2):
        style(ax)
        for k in KICKS:
            ax.axvline(k, color=SIGNAL, lw=1, alpha=0.25)
    a1.semilogy(t, rec["doubt"], color=TRACE, lw=1.8, label="Predicted error ε̂")
    a1.axhline(THRESHOLD, color=MUTED, lw=1.2, ls="--", label="Threshold τ")
    hand = np.array(rec["hand"])
    a1.plot(t[hand], np.full(hand.sum(), 1.2e-3), "|", color=SIGNAL, ms=12, mew=2, label="Frame handed to solver")
    a1.set_ylim(1e-3, 8)
    a1.set_ylabel("ε̂ (relative)", color=MUTED)
    a1.legend(frameon=False, fontsize=9, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    a1.set_title("Three strong stirs (pink lines). The network's doubt spikes, crosses τ, and the solver takes those frames.",
                 loc="left", fontsize=10.5, color=INK)
    a2.plot(t, rec["err_off"], color="#8a93a1", lw=1.8, label="Safety net off")
    a2.plot(t, rec["err_on"], color=INK, lw=1.8, label="Safety net on")
    a2.set_ylim(bottom=0)
    a2.set_ylabel("Error vs real solver (%)", color=MUTED)
    a2.set_xlabel("Frame", color=MUTED)
    a2.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG)
    print("wrote", out, "handoffs", int(hand.sum()), "final err off/on",
          round(rec["err_off"][-1], 1), round(rec["err_on"][-1], 1))


if __name__ == "__main__":
    rec = episode(load())
    where_wrong(rec["snap"])
    timeline(rec)
