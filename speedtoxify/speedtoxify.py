from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from detoxify import Detoxify
from onnxruntime import InferenceSession, SessionOptions

from .onnx import save_onnx


def sigmoid_np(x: np.ndarray):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class Speedtoxify(Detoxify):
    """Speedtoxify
    Same as Detoxify but optimized for inference only using ONNX runtime.
    """

    def __init__(
        self,
        model_type: str = "original",
        checkpoint: Optional[Any] = None,
        device: Literal["cpu", "cuda"] = "cpu",
        force_export: bool = False,
        cache_dir: Optional[Path] = None,
        session_options: Optional[SessionOptions] = None,
    ):
        """
        Creates a Detoxify model but exports the model to ONNX format and runs inference
        using ONNX runtime.

        Args:
            model_type (str, optional): The available model types in Detoxify.
                Defaults to "original".
            checkpoint (Any | None, optional): Checkpoint of the model to load.
                Defaults to None.
            device (Literal["cpu", "cuda"], optional): The device to run on.
                Currently supports "cpu" or "cuda". "onnxruntime-gpu" needs to be installed
                for inference on cuda. Defaults to "cpu".
            force_export (bool, optional): Whether or not to force re-export the ONNX model.
                Defaults to False.
            cache_dir (Path | None, optional): Directory to save the ONNX models.
                Defaults to Path.home() / ".cache/detoxify_onnx".
            session_options (SessionOptions | None, optional): ONNX runtime SessionOptions for
                configuring the ONNX session. Please refer to ONNX python API docs.
        """
        super().__init__(model_type, checkpoint, device)

        if cache_dir is None:
            cache_dir = Path.home() / ".cache/detoxify_onnx"

        onnx_path = cache_dir / f"{model_type}.onnx"

        # export onnx model
        if not onnx_path.exists() or force_export:
            save_onnx(self.model, self.tokenizer, onnx_path)
        # delete huggingface model to save memory
        del self.model

        if device == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]

        # start onnx runtime session
        self.session = InferenceSession(
            str(onnx_path), providers=providers, sess_options=session_options
        )

    def predict(self, text: Union[str, List[str]]) -> Dict[str, List[float]]:
        inputs = self.tokenizer(
            text, return_tensors="np", truncation=True, padding=True
        )
        outputs = self.session.run(None, input_feed=dict(inputs))[0]
        scores = sigmoid_np(outputs)
        results = {}
        for i, cla in enumerate(self.class_names):
            if isinstance(text, str):
                results[cla] = scores[0][i]
            else:
                results[cla] = scores.T[i].tolist()
        return results
