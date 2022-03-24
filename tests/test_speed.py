from typing import Literal
import time
import logging

import pytest

from speedtoxify import Speedtoxify
from detoxify import Detoxify

text = ["You are a good person"]


@pytest.mark.parametrize("model_name,iters,batch_size,device", [
    ("original", 125, 8, "cpu"),
    ("original", 1000, 1, "cpu"),
    ("original", 125, 8, "cuda"),
    ("original", 1000, 1, "cuda"),
])
def test_onnx_models(model_name: str, iters: int, batch_size: int, device: Literal["cpu", "cuda"]):
    model = Detoxify(model_name, device=device)
    onnx_model = Speedtoxify(model_name, device=device)

    batch = text * batch_size
    res = model.predict(batch)
    start = time.time()
    for _ in range(iters):
        model.predict(batch)
    pt_time = (time.time() - start) / (iters * batch_size)
    logging.info(f"Detoxify: {pt_time * 1000:.2f}ms per sample")

    onnx_res = onnx_model.predict(batch)
    start = time.time()
    for _ in range(iters):
        onnx_model.predict(batch)
    onnx_time = (time.time() - start) / (iters * batch_size)
    logging.info(f"Speedtoxify: {onnx_time * 1000:.2f}ms per sample")

    speed_up = pt_time / onnx_time
    logging.info(f"Speedup: {speed_up:.2f}x")

    assert all(k in onnx_res for k in res)
    for k in res:
        assert res[k] == pytest.approx(onnx_res[k], abs=1e-4)

    assert speed_up >= 1
