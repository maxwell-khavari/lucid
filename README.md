# Lucid

**A tiny neural fluid simulator that knows when it's wrong.**

**[Try it in your browser](https://maxwell-khavari.github.io/lucid/)**: drag to stir, press Kick, and flip the safety net.

![Lucid with the safety net on and off](results/demo.gif)

AI surrogates can stand in for slow physics solvers, but they fail silently: nothing tells you when the output has stopped being right. Lucid is a 1.2M-parameter network (4.9 MB) that runs a 2D fluid one frame at a time and also predicts its own error. When it doubts itself, it hands that frame to the real solver.

## Result

![Error against solver calls](results/headline.png)

Results are on held-out episodes (24 episodes of 150 frames, in `results/eval.json`).

**When the fluid is pushed outside the training range** (sudden strong stirs on 10% of frames):
- The network alone drifts to **24.6%** error against the real solver.
- Lucid's self-check sends **14% of frames** to the solver and holds error at **6.1%**.
- Spending a similar budget on random or scheduled frames only gets to **18 to 21%**.
- The self-check flags its own bad frames with an **AUROC of 0.99**.

**Baselines, stated plainly:**
- At low budgets (under 10% solver frames), a physics check on the network's output does as well or slightly better. It is the divergence of the predicted velocity, which a true fluid keeps at zero.
- A hand-written rule that knows exactly what an out-of-range stir looks like is ahead at a 10% budget.
- From about 14% upward, the self-check gives the lowest error of any gate, and it needed no rule written for the case.

**Under normal stirring** the network is accurate on its own: 6.2% average error over 150 frames, and 0.6% per frame. Its self-check does no better than calling the solver on a fixed schedule, because nothing unusual is happening.

The self-check earns its keep when inputs leave what the model learned. That is where silent failures happen.

## What this is and isn't
- **Is:** a small, complete, reproducible demonstration of a surrogate that estimates its own error and defers to a trusted method. You can play with it.
- **Isn't:** a speedup. At 64x64 the classical solver is about as fast as the network. Surrogates save compute where the real method is expensive: 3D, high resolution, weather, engineering CFD. The self-check idea carries over to those settings, but this repo does not show that.
- **Isn't:** a new research concept. Hybrid surrogate-plus-solver methods exist; see [Prior work](#prior-work). The new parts are a playable, open-weights, in-browser version and a direct comparison of gates.

## How it works
1. **Solver.** `sim/solver.py` and `sim/solver.js` implement stable fluids (Stam 1999) on a 64x64 periodic grid: 4 substeps per frame, 80 Jacobi pressure iterations each. A test checks the Python and JS versions agree to within 1e-4.
2. **Data.** Random stirrers (`sim/forcing.py`) drive the solver while training runs. No dataset is stored or downloaded.
3. **Model.** `lucid/model.py` is a U-Net with circular padding. One head predicts the next frame's change. The other predicts the per-cell error of that prediction.
4. **Training.** `train.py` takes about 2.3 hours on an Apple M4. Inputs are true solver states, plus states the network drifted into by rolling itself forward, so the error head learns what its own mistakes look like.
5. **Gate.** The score is the predicted error divided by the flow's size. Above the threshold, the solver takes the frame.

## Reproduce
```bash
pip install torch numpy matplotlib onnx onnxruntime
python3 -m unittest discover tests              # JS and Python solvers agree
python3 train.py --steps 24000                  # ~2.3 h on an M4, writes ckpt/lucid.pt
python3 -m eval.evaluate                        # ~35 min, writes results/eval.json
python3 -m eval.plots                           # writes results/headline.png
python3 -m export.onnx                          # writes web/lucid.onnx, checks it against PyTorch
python3 -m http.server 8000                     # then open http://localhost:8000/web/
```
The trained weights are included: `web/lucid.onnx` for the browser and `weights/lucid.pt` for PyTorch.

## Use the model
```python
import torch
from lucid.model import Lucid

model = Lucid()
model.load_state_dict(torch.load("weights/lucid.pt", map_location="cpu")["model"])
model.eval()
x = torch.zeros(1, 6, 64, 64)            # (u, v, d, fx, fy, s)
y = model(x)                             # (next u, next v, next d, predicted error per cell)
```

## Prior work
- J. Stam, *Stable Fluids*, SIGGRAPH 1999. The reference solver.
- XRePIT (arXiv:2510.21804). A deep-learning and CFD hybrid in OpenFOAM that falls back to the solver when a residual crosses a threshold.
- Hybrid Neural-MPM (arXiv:2505.18926). A real-time neural fluid that falls back to a classical MPM solver.
- Hybrid Neural World Models (arXiv:2605.28317). Uses a surrogate's error map to gate a reference-solver fallback.
- A. Graves, *Adaptive Computation Time*, 2016, and related work on learning when to spend compute.

Lucid is an independent implementation. It is not affiliated with or endorsed by the authors above.

## License and disclaimer
Copyright 2026 Maxwell Khavari. Code and weights are licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.
- **Free:** personal use, research, education and nonprofit use.
- **Commercial use:** needs written permission.

This is a research demo of a toy 2D fluid. It is provided **as is, without warranty of any kind**. As far as the law allows, the author is not liable for any damages arising from its use. It is not validated for, and must not be used for, engineering, safety, medical, financial or any other real-world decision. See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limits.
