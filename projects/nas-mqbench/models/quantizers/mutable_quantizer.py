# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmengine.utils import import_modules_from_strings
from mmrazor.registry import MODELS
from mmrazor.models import TensorRTQuantizer
from mmrazor.models.utils import str2class
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.models.architectures.dynamic_ops import (DynamicConvMixin,
                                                      DynamicLinearMixin)
try:
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.fx.graph_module import ObservedGraphModule
    from torch.ao.quantization.quantize_fx import _fuse_fx
    from torch.fx.graph_module import GraphModule
    from torch.nn.intrinsic.qat import modules as qat_fused_modules
    from torch.nn.qat import modules as qat_modules
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    GraphModule = get_placeholder('torch>=1.13')
    ObservedGraphModule = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')
    qat_fused_modules = get_package_placeholder('torch>=1.13')
    qat_modules = get_package_placeholder('torch>=1.13')
from mmrazor.models.task_modules.tracer import build_graphmodule    
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

    def __init__(self, *args, tracer: Dict = dict(type='CustomTracer'),
                 quant_bits=None, quant_bits_skipped_module_names=None, **kwargs):
        if 'skipped_module_classes' in tracer:
            tracer['skipped_module_classes'] = str2class(tracer['skipped_module_classes'])
        super().__init__(*args, tracer=tracer, **kwargs)
        self.quant_bits = quant_bits
        if quant_bits_skipped_module_names is None:
            quant_bits_skipped_module_names = []
        self.quant_bits_skipped_module_names = quant_bits_skipped_module_names

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'mutabletensorrt'

    def prepare(self, model, concrete_args=None):
        """prepare graph to ObservedGraphModule.

        Returns:
            ObservedGraphModule: GraphModules after fuse and observer.

        Notes:
            'graph_module' after '_fuse_fx()' function will fuse conv, BN, ReLU
            into modules in SUPPORT_QAT_MODULES.
            'graph_module' after 'prepare()' function will become observed.

        Notes:
            Keep `is_qat` is True is because in Pytorch when `is_qat` is false,
            the `_fuse_fx()` function only fuse module into `nn.Squential`.
            In mmrazor, we aim to add more ptq algorithm into our pipeline such
            as Adaround, these kind of ptq method have some additional
            fake_quant  operations that we need it to be fused into our
            `SUPPORT_QAT_MODULES` type, which is a tricky way to deal with it.
        """
        # propogate qbconfig
        for name, mod in model.named_modules():
            if isinstance(mod, (DynamicConvMixin,
                    DynamicLinearMixin)) and not hasattr(mod, 'qbconfig'):
                mod.qbconfig = {
                    'quant_bits': OneShotMutableValue(
                        alias=name + 'quant_bits', value_list=self.quant_bits)
                } if self.quant_bits and name not in self.quant_bits_skipped_module_names else {}
        # import pdb; pdb.set_trace()
        self.swap_ff_with_fxff(model)
        traced_graph = self.tracer.trace(model, concrete_args=concrete_args)
        graph_module = build_graphmodule(model, traced_graph)

        # set the training modes of all modules to True to `_fuse_fx` correctly
        # todo: check freezebn
        self.sync_module_training_mode(graph_module, mode=True)

        graph_module = _fuse_fx(
            graph_module=graph_module,
            is_qat=True,
            backend_config=self.backend_config)

        prepared = prepare(
            model=graph_module,
            qconfig_mapping=self.qconfig_mapping,
            is_qat=True,
            node_name_to_scope=self.tracer.node_name_to_scope,
            example_inputs=self.example_inputs,
            backend_config=self.backend_config)
        prepared = self.del_redundant_fakequant(prepared)

        return prepared
