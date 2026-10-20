"""Render results/demo.gif: safety net off, safety net on, and the real solver on identical input.

python3 -m eval.gif
"""

import torch
from PIL import Image, ImageDraw, ImageFont

from eval.evaluate import SCORES, rel_error
from lucid.model import Lucid
from sim.forcing import Stirrers, splat
from sim.solver import N, step

PAPER, INK, ACCENT, MUTED = (235, 229, 215), (29, 27, 23), (180, 68, 28), (107, 101, 90)
SCALE, PAD, HEAD = 3, 14, 36
THRESHOLD = 0.012
KICKS = {40: (20, 40, 0.0), 75: (44, 18, 2.2), 110: (30, 30, 4.1)}


def render(d):
    a = (1 - torch.exp(-2.0 * d.clamp(min=0))).unsqueeze(-1)
    paper, ink = torch.tensor(PAPER).float(), torch.tensor(INK).float()
    img = (paper + (ink - paper) * a).byte().numpy()
    return Image.fromarray(img).resize((N * SCALE, N * SCALE), Image.BICUBIC)


@torch.no_grad()
def main(out="results/demo.gif", frames=150):
    model = Lucid()
    model.load_state_dict(torch.load("weights/lucid.pt", map_location="cpu")["model"])
    model.eval()
    st = Stirrers(1, "cpu", torch.Generator().manual_seed(3))
    x = torch.zeros(1, 3, N, N)
    for t in range(40):
        x = step(x, st.force(t))
    off, on, ref = x.clone(), x.clone(), x.clone()
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
    w = N * SCALE
    images = []
    for t in range(frames):
        f = st.force(40 + t)
        kicked = t in KICKS
        if kicked:
            cx, cy, ang = KICKS[t]
            one = lambda v: torch.tensor([[float(v)]])
            f = f + splat(one(cx), one(cy), one(3.5 * torch.cos(torch.tensor(ang))), one(3.5 * torch.sin(torch.tensor(ang))),
                          one(2.5), one(0.3), one(0.3))
        off = model(torch.cat([off, f], 1))[:, :3]
        y = model(torch.cat([on, f], 1))
        used = (SCORES["learned"](on, f, y) > THRESHOLD).item()
        on = step(on, f) if used else y[:, :3]
        ref = step(ref, f)
        canvas = Image.new("RGB", (3 * w + 4 * PAD, w + HEAD + 2 * PAD + 40), (244, 240, 230))
        dr = ImageDraw.Draw(canvas)
        panes = [(off, "Safety net off", MUTED), (on, "Safety net on", ACCENT if used else MUTED), (ref, "Real solver", MUTED)]
        for i, (s, label, col) in enumerate(panes):
            px = PAD + i * (w + PAD)
            canvas.paste(render(s[0, 2]), (px, HEAD + PAD))
            dr.text((px, PAD + 8), label, fill=INK, font=font)
            if i < 2:
                err = rel_error(s, ref).item()
                dr.text((px, HEAD + PAD + w + 6), f"error vs solver {100 * err:4.1f}%", fill=MUTED, font=small)
        if used:
            dr.text((PAD + w + PAD, HEAD + PAD + w + 22), "doubt: solver took this frame", fill=ACCENT, font=small)
        if any(0 <= t - k < 6 for k in KICKS):
            dr.text((PAD + 2 * (w + PAD), HEAD + PAD + w + 6), "kick", fill=ACCENT, font=small)
        images.append(canvas.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    images[0].save(out, save_all=True, append_images=images[1:], duration=55, loop=0, optimize=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
