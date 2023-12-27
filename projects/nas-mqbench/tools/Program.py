from typing import Any

import torch
import torchvision.models.resnet

import ppq.lib as PFL
from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
from ppq.api import ENABLE_CUDA_KERNEL, dump_torch_to_onnx, load_onnx_graph
from ppq.quantization.optim import (ParameterQuantizePass,
                                    RuntimeCalibrationPass)

from Quantizers import MyFP8Quantizer, MyInt8Quantizer

# 1. build pytorch module.
model = torchvision.models.resnet.resnet18()  #.cuda()


# 2. convert pytorch module to onnx with a sample input.
ONNX_FILE_NAME = 'Tmp.onnx'
sample_input = torch.rand(size=[1, 3, 224, 224])  #.cuda()
dump_torch_to_onnx(
    model=model, onnx_export_file=ONNX_FILE_NAME, 
    inputs=sample_input, input_shape=None)


# 3. create quantizer.
PerChannel  = True
Symmetrical = True
PowerOf2    = False
FP8         = False

graph = load_onnx_graph(onnx_import_file=ONNX_FILE_NAME)
if not FP8:
    quantizer = MyInt8Quantizer(
        graph=graph, 
        per_channel=PerChannel, 
        sym=Symmetrical, 
        power_of_2=PowerOf2)
else:
    quantizer = MyFP8Quantizer()


# 4. quantize operations, only conv and gemm will be quantized by this script.
for op in graph.operations.values():
    if op.type in {'Conv', 'Gemm'}:
        platform = TargetPlatform.INT8 if not FP8 else TargetPlatform.FP8
    else: platform = TargetPlatform.FP32

    quantizer.quantize_operation(
        op_name = op.name, platform=platform)


# 5. build quantization pipeline
pipeline = PFL.Pipeline([
    ParameterQuantizePass(),
    RuntimeCalibrationPass(),

    # LearnedStepSizePass is a network retraining procedure.
    # LearnedStepSizePass(steps=500)
])


# 6. run quantization, measure quantization loss.
# rewrite the definition of dataloader with real data first!
dataloader = [torch.rand(size=[1, 3, 224, 224]) for _ in range(32)]

def collate_fn(batch: Any):
    return batch  #.cuda()

# with ENABLE_CUDA_KERNEL():
executor = TorchExecutor(graph=graph, device='cpu')
executor.tracing_operation_meta(inputs=sample_input)

pipeline.optimize(
    graph=graph, dataloader=dataloader, verbose=True, 
    calib_steps=32, collate_fn=collate_fn, executor=executor)

error_dict = graphwise_error_analyse(
    graph=graph, running_device='cpu', 
    dataloader=dataloader, collate_fn=collate_fn)

    # quantization error are stored in error_dict
    # error_dict = {op name(str): error(float)}