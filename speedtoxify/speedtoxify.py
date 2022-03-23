from typing import Optional, Any
from pathlib import Path

import numpy as np
from onnxruntime import InferenceSession
from detoxify import Detoxify

from .onnx import save_onnx


def sigmoid_np(x: np.ndarray):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class Speedtoxify(Detoxify):
    """Speedtoxify
    Same as Detoxify but optimized for inference only using onnxruntime. 
    """

    def __init__(self,
                 model_type: str = "original",
                 checkpoint: Optional[Any] = None,
                 device: str = "cpu",
                 force_export: bool = False,
                 cache_dir: Optional[Path] = None):
        # sets self.model, self.tokenizer, self.class_names
        super().__init__(model_type, checkpoint, device)

        if cache_dir is None:
            cache_dir = Path.home() / ".cache/detoxify_onnx"
        onnx_path = cache_dir / f"{model_type}.onnx"
        if not onnx_path.exists() or force_export:
            save_onnx(self.model, self.tokenizer, onnx_path)
        del self.model

        if device == "cuda":
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.session = InferenceSession(str(onnx_path), providers=providers)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="np",
                                truncation=True, padding=True)
        outputs = self.session.run(None, input_feed=dict(inputs))[0]
        scores = sigmoid_np(outputs)
        results = {}
        for i, cla in enumerate(self.class_names):
            if isinstance(text, str):
                results[cla] = scores[0][i]
            else:
                results[cla] = scores.T[i].tolist()
        return results
