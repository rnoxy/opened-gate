import torch

from opened_gate.train import ClassificationModule

path = "data/06_models/model.ckpt"
model = ClassificationModule.load_from_checkpoint(path).eval()

# Convert to ONNX
input_sample = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    input_sample,
    "app/model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
