<div align="center">

# Speedtoxify :rocket:

## Fast :speak_no_evil: Detoxify inference with ONNX runtime

#### [:zap: Benchmarks](#zap-lightning-fast) | [:gear: install](#gear-installation) | [:star2: Quick Start](#star2-quick-start) | [:page_with_curl: Docs](#pagewithcurl-documentation)

</div>

Speedtoxify is a wrapper around [`detoxify`](https://github.com/unitaryai/detoxify) 
that speeds up inference by 2-4x by using [ONNX runtime](https://github.com/microsoft/onnxruntime). 

Detoxify is a NLP library for detecting toxic / inappropriate / profane texts. 
Speedtoxify makes use of their pretrained models and runs them in 
ONNX runtime for much faster inference speeds, which makes it the better option 
for being used in production. 

Speedtoxify provides the same Python API as Detoxify, so it can be used as a drop-in replacement. 

However, if your focus is on fine-tuning / re-training the models with your own 
data, please refer to Detoxify.

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

> Evaluation is done on my laptop with AMD 4900HS and Nvidia 2060 Max-Q.

## :gear: Installation

### Pip

```terminal
pip install speedtoxify
```

### GPU Inference

Please additionally install `onnxruntime-gpu` for inference on gpus. 
Requires the machine to have CUDA installed.

```terminal
pip install onnxruntime-gpu
```

## :star2: Quick start

Speedtoxify provides the identical Python API as Detoxify. 

```python
from speedtoxify import Speedtoxify

model = Speedtoxify("original-small")
# Exporting to onnx format to ~/.cache/detoxify_onnx/original-small.onnx...
# Using framework PyTorch: 1.11.0+cu102
# Removing shared weights from ~/.cache/detoxify_onnx/original-small.onnx...
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

## :page_with_curl: Documentation

Please refer to [docs](docs).
