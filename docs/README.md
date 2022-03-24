# speedtoxify package


### _class_ speedtoxify.Speedtoxify(model_type='original', checkpoint=None, device='cpu', force_export=False, cache_dir=None)
Bases: `detoxify.Detoxify`

Same as Detoxify but optimized for inference only using ONNX runtime.


#### \__init__(model_type='original', checkpoint=None, device='cpu', force_export=False, cache_dir=None)
Creates a Detoxify model but exports the model to ONNX format and runs inference
using ONNX runtime.


* **Parameters**

    
    * **model_type** (*str*, *optional*) – The available model types in Detoxify.
    Defaults to "original".


    * **checkpoint** (*Any* | **None**, *optional*) – Checkpoint of the model to load.
    Defaults to None.


    * **device** (*Literal*[**"cpu"**, **"cuda"**], *optional*) – The device to run on.
    Currently supports "cpu" or "cuda". "onnxruntime-gpu" needs to be installed
    for inference on cuda. Defaults to "cpu".


    * **force_export** (*bool*, *optional*) – Whether or not to force re-export the ONNX model.
    Defaults to False.


    * **cache_dir** (*Path* | **None**, *optional*) – Directory to save the ONNX models.
    Defaults to `Path.home() / ".cache/detoxify_onnx"`.
