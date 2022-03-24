from typing import Union
from pathlib import Path
import logging
import shutil

import onnx
from onnxruntime.transformers.onnx_model import OnnxModel
from transformers.onnx import FeaturesManager, export, validate_model_outputs


def deduplicate_shared_layers(path: Union[Path, str]):
    '''
    Removes the duplicated shared layers when exporting the model to ONNX format.

    Args:
        path (Path): Path to the .onnx file
    '''
    model = onnx.load(str(path))
    onnx_model = OnnxModel(model)

    initializer = model.graph.initializer
    duplicated_layers = set()
    for i in range(len(initializer)):
        if i in duplicated_layers:
            continue
        for j in range(i + 1, len(initializer)):
            if initializer[j].raw_data == initializer[i].raw_data:
                onnx_model.replace_input_of_all_nodes(
                    initializer[j].name, initializer[i].name)
                duplicated_layers.add(j)

    onnx_model.update_graph()
    onnx_model.save_model_to_file(str(path))


def save_onnx(model, tokenizer, output_path: Union[Path, str]):
    logger = logging.getLogger("transformers.onnx")
    level = logger.level
    logger.setLevel(logging.INFO)

    tmp_dir = Path("tmp")
    model.save_pretrained(tmp_dir)

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = FeaturesManager.get_model_from_feature(
            "sequence-classification", str(tmp_dir))
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model, feature="sequence-classification")
        onnx_config = model_onnx_config(model.config)

        opset = onnx_config.default_onnx_opset

        # export model to onnx format
        logger.info(f"Exporting to onnx format to {output_path}...")
        _, onnx_outputs = export(
            tokenizer,
            model,
            onnx_config,
            opset,
            output_path,
        )

        # remove duplicated shared layers in Albert
        logger.info(f"Removing shared weights from {output_path}...")
        deduplicate_shared_layers(output_path)

        # validate the models are the same
        atol = onnx_config.atol_for_validation
        validate_model_outputs(onnx_config, tokenizer,
                               model, output_path, onnx_outputs, atol)
    finally:
        shutil.rmtree(str(tmp_dir))
        logger.setLevel(level)
