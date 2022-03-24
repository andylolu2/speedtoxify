<div align="center">

# Speedtoxify :rocket:

## Fast :speak_no_evil: Detoxify inference with ONNX runtime

</div>

Speedtoxify is a wrapper around [`detoxify`](https://github.com/unitaryai/detoxify) 
that speeds up inference by 2-4x by using [ONNX runtime](https://github.com/microsoft/onnxruntime). 

## :zap: Lightning fast

| Model          | Batch size | Device | Detoxify (ms/sample) | Speedtoxify (ms/sample) | Speedup |
| -------------- | ---------- | ------ | -------------------- | ----------------------- | ------- |
| original-small | 8          | cpu    | 13.34                | 5.43                    | 2.46x   |
| original-small | 1          | cpu    | 31.07                | 13.03                   | 2.38x   |
| original-small | 8          | cuda   | 1.55                 | 0.79                    | 1.98x   |
| original-small | 1          | cuda   | 11.17                | 3.24                    | 3.44x   |
| original       | 8          | cpu    | 22.99                | 5.39                    | 4.26x   |
| original       | 1          | cpu    | 31.48                | 13.11                   | 2.40x   |
| original       | 8          | cuda   | 1.60                 | 0.75                    | 2.12x   |
| original       | 1          | cuda   | 12.13                | 3.37                    | 3.60x   |

Evaluation script can be found in [test_speed.py](tests/test_speed.py).

## :star2: Quick start

```python
from speedtoxify import Speedtoxify

model = Speedtoxify("original-small")
# Exporting to onnx format to /home/andylo/.cache/detoxify_onnx/original-small.onnx...
# Using framework PyTorch: 1.11.0+cu102
# Removing shared weights from /home/andylo/.cache/detoxify_onnx/original-small.onnx...
# Validating ONNX model...
# 	-[✓] ONNX model output names match reference model ({'logits'})
# 	- Validating ONNX Model output "logits":
# 		-[✓] (2, 6) matches (2, 6)
# 		-[✓] all values close (atol: 1e-05)

res = model.predict("I hate you!")
print(res)
# {'toxicity': 0.9393415, 'severe_toxicity': 0.015587699, 'obscene': 0.039672945, 'threat': 0.0733101, 'insult': 0.15676126, 'identity_attack': 0.019178415}
```

Please refer to [detoxify](https://github.com/unitaryai/detoxify) for 
available model types. 

The first time `Speedtoxify("original-small")` is called, an onnx model is 
exported and stored at `~/.cache/detoxify_onnx`. 
This directory can be customized in the `cache_dir` argument to 
`Speedtoxify()`.

## Documentation

Please refer to [docs](docs).

## GPU inference

Please install `onnxruntime-gpu` for inference on gpus. Requires the 
machine have CUDA installed.

```terminal
pip install onnxruntime-gpu
```
