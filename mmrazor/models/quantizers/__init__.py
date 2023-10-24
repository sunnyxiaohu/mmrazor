# Copyright (c) OpenMMLab. All rights reserved.
from .academic_quantizer import AcademicQuantizer
from .base import BaseQuantizer
from .native_quantizer import TorchNativeQuantizer
from .openvino_quantizer import OpenVINOQuantizer
from .tensorrt_quantizer import TensorRTQuantizer
from .superacme_quantizer import SuperAcmeQuantizer

__all__ = [
    'BaseQuantizer', 'AcademicQuantizer', 'TorchNativeQuantizer',
    'TensorRTQuantizer', 'OpenVINOQuantizer', 'SuperAcmeQuantizer'
]
