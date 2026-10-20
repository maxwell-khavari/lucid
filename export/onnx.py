"""Export a checkpoint to ONNX for the browser and check it against PyTorch.

python3 -m export.onnx --ckpt ckpt/lucid.pt --out web/lucid.onnx
"""

import argparse

import numpy as np
import torch

from eval.evaluate import episodes
from lucid.model import Lucid


def main(args):
    model = Lucid()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    torch.onnx.export(model, (torch.zeros(1, 6, 64, 64),), args.out, input_names=["x"],
                      output_names=["y"], opset_version=17, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    forces, ref = episodes(8, 20, 4242)
    worst = 0.0
    for t in range(20):
        x = torch.cat([ref[t], forces[t]], 1)
        with torch.no_grad():
            a = model(x).numpy()
        for i in range(x.shape[0]):
            b = sess.run(None, {"x": x[i:i + 1].numpy()})[0]
            worst = max(worst, float(np.abs(a[i:i + 1] - b).max()))
    print(f"max |torch - onnx| over 160 frames: {worst:.2e}")
    assert worst < 1e-4, "ONNX output does not match PyTorch"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/lucid.pt")
    ap.add_argument("--out", default="web/lucid.onnx")
    main(ap.parse_args())
