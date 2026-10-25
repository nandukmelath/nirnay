"""ONNX Runtime backend for the frozen encoder (the expensive part of every call).

  python -m nirnay.onnx_backend export <model_dir>     -> <model_dir>/encoder.onnx
The engine uses encoder.onnx automatically when it exists and onnxruntime is installed
(set NIRNAY_BACKEND=torch to force the PyTorch path).
"""
import os
import sys
from typing import Optional

import numpy as np
import torch


class _Wrap(torch.nn.Module):
    def __init__(self, enc, adapter=None, split=0, bottom_only=False):
        super().__init__()
        self.enc, self.adapter, self.split, self.bottom_only = enc, adapter, split, bottom_only

    def forward(self, ids, mask):
        m = mask.bool()
        if self.bottom_only:   # deep fusion: frozen layers [0, split) only; the head runs the rest
            h = self.enc(ids, m, stop_layer=self.split)
            return h, h[:, 0] * 0.0
        if self.adapter is not None:
            h = self.adapter(self.enc(ids, m, stop_layer=self.split), m)
        else:
            h = self.enc(ids, m)
        return h, self.enc.pool(h, m)


def _export(module, m, path, opset):
    ids = torch.full((2, 16), m.pad_id, dtype=torch.long)
    ids[:, 0] = m.cls_id
    mask = torch.ones(2, 16, dtype=torch.long)
    kw = dict(input_names=["ids", "mask"], output_names=["hidden", "pooled"],
              dynamic_axes={"ids": {0: "b", 1: "l"}, "mask": {0: "b", 1: "l"}, "hidden": {0: "b", 1: "l"}, "pooled": {0: "b"}},
              opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(module, (ids, mask), path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(module, (ids, mask), path, **kw)
    return path


def export(model_dir: str, opset: int = 17) -> str:
    """encoder.onnx (frozen encoder: question/option texts, and states on a non-split model);
    state.onnx as well on a split model (frozen bottom layers + trained adapter)."""
    from .engine import Nirnay
    os.environ["NIRNAY_BACKEND"] = "torch"
    m = Nirnay(model_dir, threads=8)
    if m.fusion == "deep":
        return _export(_Wrap(m.enc, None, m.split, bottom_only=True), m, os.path.join(model_dir, "state.onnx"), opset)
    paths = [_export(_Wrap(m.enc), m, os.path.join(model_dir, "encoder.onnx"), opset)]
    if m.adapter is not None:
        paths.append(_export(_Wrap(m.enc, m.adapter, m.split), m, os.path.join(model_dir, "state.onnx"), opset))
    return ", ".join(paths)


class OrtEncoder:
    def __init__(self, path: str, threads: int):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.set_denormal_as_zero", "1")
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def __call__(self, ids: torch.Tensor, mask: torch.Tensor):
        h, v = self.sess.run(None, {"ids": ids.numpy().astype(np.int64), "mask": mask.numpy().astype(np.int64)})
        return torch.from_numpy(h), torch.from_numpy(v)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) == 2 and argv[0] == "export":
        print("wrote", export(argv[1]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
