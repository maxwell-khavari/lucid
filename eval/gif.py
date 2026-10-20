"""Render results/demo.gif: safety net off, safety net on, and the real solver on identical input.

python3 -m eval.gif
"""

import torch
from PIL import Image, ImageDraw, ImageFont

from eval.evaluate import SCORES, rel_error
from lucid.model import Lucid
from sim.forcing import Stirrers, splat
from sim.solver import N, step

PAPER, INK, ACCENT, MUTED = (233, 236, 240), (14, 17, 22), (194, 24, 155), (91, 100, 114)
SCALE, PAD, HEAD = 3, 14, 36
THRESHOLD = 0.012
KICKS = {40: (20, 40, 0.0), 75: (44, 18, 2.2), 110: (30, 30, 4.1)}


def render(d, scale=SCALE):
    a = (1 - torch.exp(-2.0 * d.clamp(min=0))).unsqueeze(-1)
    paper, ink = torch.tensor(PAPER).float(), torch.tensor(INK).float()
    img = (paper + (ink - paper) * a).byte().numpy()
    return Image.fromarray(img).resize((N * scale, N * scale), Image.BICUBIC)


@torch.no_grad()
def frames(n=150, scale=SCALE, k=1.0):
    """Yield the demo frames as RGB images. k scales padding and text with the panes."""
    model = Lucid()
    model.load_state_dict(torch.load("weights/lucid.pt", map_location="cpu")["model"])
    model.eval()
    st = Stirrers(1, "cpu", torch.Generator().manual_seed(3))
    x = torch.zeros(1, 3, N, N)
    for t in range(40):
        x = step(x, st.force(t))
    off, on, ref = x.clone(), x.clone(), x.clone()
    font = ImageFont.load_default(size=round(15 * k))
    small = ImageFont.load_default(size=round(13 * k))
    pad, head = round(PAD * k), round(HEAD * k)
    w = N * scale
    for t in range(n):
        f = st.force(40 + t)
        if t in KICKS:
            cx, cy, ang = KICKS[t]
            one = lambda v: torch.tensor([[float(v)]])
            f = f + splat(one(cx), one(cy), one(3.5 * torch.cos(torch.tensor(ang))), one(3.5 * torch.sin(torch.tensor(ang))),
                          one(2.5), one(0.3), one(0.3))
        off = model(torch.cat([off, f], 1))[:, :3]
        y = model(torch.cat([on, f], 1))
        used = (SCORES["learned"](on, f, y) > THRESHOLD).item()
        on = step(on, f) if used else y[:, :3]
        ref = step(ref, f)
        canvas = Image.new("RGB", (3 * w + 4 * pad, w + head + 2 * pad + round(40 * k)), (243, 244, 246))
        dr = ImageDraw.Draw(canvas)
        panes = [(off, "Safety net off"), (on, "Safety net on"), (ref, "Real solver")]
        for i, (s, label) in enumerate(panes):
            px = pad + i * (w + pad)
            canvas.paste(render(s[0, 2], scale), (px, head + pad))
            dr.text((px, pad + round(8 * k)), label, fill=INK, font=font)
            if i < 2:
                err = rel_error(s, ref).item()
                dr.text((px, head + pad + w + round(6 * k)), f"error vs solver {100 * err:4.1f}%", fill=MUTED, font=small)
        if used:
            dr.text((pad + w + pad, head + pad + w + round(22 * k)), "doubt: solver took this frame", fill=ACCENT, font=small)
        if any(0 <= t - kk < 6 for kk in KICKS):
            dr.text((pad + 2 * (w + pad), head + pad + w + round(6 * k)), "kick", fill=ACCENT, font=small)
        yield canvas


def main(out="results/demo.gif", n=150):
    images = [c.quantize(colors=96, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE) for c in frames(n)]
    images[0].save(out, save_all=True, append_images=images[1:], duration=55, loop=0, optimize=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
