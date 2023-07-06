# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmengine.utils import import_modules_from_strings
from mmrazor.registry import MODELS
from mmrazor.models import TensorRTQuantizer
from mmrazor.models.utils import str2class
from mmrazor.structures.quantization import BackendConfigs
custom_imports = 'projects.nas-mqbench.structures.quantization.backend_config.mutable_tensorrt'
mutabletensorrt = import_modules_from_strings(custom_imports)
BackendConfigs['mutabletensorrt'] = mutabletensorrt.get_mutabletensorrt_backend_config()


@MODELS.register_module()
class MutableTensorRTQuantizer(TensorRTQuantizer):
    """Quantizer for quantizing and deploying to TensorRT backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    TensorRT's some important features about quantization is as follows:
    * support_w_mode = ('per_tensor', 'per_channel')
    * support_a_mode = ('per_tensor')
    """

    def __init__(self, *args, tracer: Dict = dict(type='CustomTracer'), **kwargs):
        if 'skipped_module_classes' in tracer:
            tracer['skipped_module_classes'] = str2class(tracer['skipped_module_classes'])
        super().__init__(*args, tracer=tracer, **kwargs)

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'mutabletensorrt'
