# Model card: Lucid

## What it is
Lucid is a 1.2M-parameter convolutional network (U-Net, about 4.9 MB at float32). It does two jobs:
- It replaces one frame of a 2D incompressible fluid solver (stable fluids, 64x64 periodic grid, 4 substeps of 80 Jacobi pressure iterations).
- It predicts its own per-cell error for that frame.

A gate uses that prediction to hand frames it doubts to the real solver.

- **Inputs:** the velocity (u, v) and smoke density (d), plus this frame's forcing (fx, fy, s).
- **Outputs:** the next (u, v, d), plus a per-cell predicted error.
- **Files:** `weights/lucid.pt` (PyTorch) and `web/lucid.onnx` (ONNX, opset 17). The two agree to within 2e-6.

## Training data
All data was generated during training by the solver in `sim/solver.py`. The forcing is random stirring, from `sim/forcing.py`. No external dataset, no third-party model, and no personal data were used.

The network was trained for 24,000 steps (about 2.3 hours on one Apple M4). Training covers two kinds of input:
- true solver states
- states the network drifted into by rolling itself forward up to 3 frames

As a result, the error head has seen the network's own mistakes.

## Evaluation
`eval/evaluate.py` runs held-out episodes with their own seed, in two scenarios:
- **normal**, using the training forcing distribution
- **kicks**, adding sudden stirs on 10% of frames that are stronger than anything in training

It reports:
- error against the reference solver
- how well each gate ranks the true one-step error (AUROC)
- error against solver-call rate, for five gates

Results are in `results/eval.json` and the README.

## Intended use
Research and education. It demonstrates one idea: a learned surrogate that estimates its own error and defers to a trusted method.

## Out of scope / do not use for
This is a toy 2D fluid on a 64x64 grid.

**It is not validated for, and must not be used for:**
- engineering, design, or safety decisions of any kind
- weather, climate, or medical work
- financial decisions
- any setting where an error could harm a person or property

Its self-check is not a guarantee. A low predicted error does not mean the output is correct.

## Limitations
- At this resolution the real solver is about as fast as the network. The demo shows error control, not speedup. A surrogate saves compute only where the reference method is expensive.
- When the forcing stays inside the training distribution, the network's errors are spread evenly over time. Its self-check then does no better than calling the solver on a fixed schedule. The self-check pays off when inputs leave the training distribution.
- Under the solver's own floating-point behaviour, results from the browser (JS, float32) and from PyTorch match to within 1e-4 relative per frame. They are not bit-identical.

## License
PolyForm Noncommercial License 1.0.0 (see LICENSE). This license covers the code and the weights.
- **Free:** personal use, research, education and nonprofit use.
- **Not allowed without written permission:** commercial use.
- **No warranty:** the software comes as is, and the licensor is not liable for any damages, as far as the law allows.
