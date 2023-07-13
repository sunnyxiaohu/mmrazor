# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Any, List, Tuple

import copy

from mmengine.utils import import_modules_from_strings
from mmrazor.registry import MODELS
from mmrazor.models import OpenVINOQuantizer, LearnableFakeQuantize
from mmrazor.models.utils import str2class
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.models.architectures.dynamic_ops import (DynamicConvMixin,
                                                      DynamicLinearMixin)
from mmrazor.models.task_modules.tracer import build_graphmodule
from mmrazor.models.task_modules.tracer.fx.graph_utils import _get_attrs
from mmrazor.structures.quantization import BackendConfigs

try:
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.quantize_fx import _fuse_fx
except ImportError:
    from mmrazor.utils import get_placeholder
    prepare = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')

custom_imports = 'projects.nas-mqbench.structures.quantization.backend_config.mutable_openvino'
mutableopenvino = import_modules_from_strings(custom_imports)
BackendConfigs['mutableopenvino'] = mutableopenvino.get_mutableopenvino_backend_config()
custom_imports = 'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_qlinear'
dynamic_qlinear = import_modules_from_strings(custom_imports)


@MODELS.register_module()
class MutableOpenVINOQuantizer(OpenVINOQuantizer):
    """Quantizer for quantizing and deploying to openvino backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    openvino's some important features about quantization is as follows:
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
        return 'mutableopenvino'

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
                        alias=name + '.quant_bits', value_list=self.quant_bits)
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

        prepared = modify_fakequant_to_int8(
            prepared, tuple(self.quant_bits_skipped_module_names), True)
        return prepared


def modify_fakequant_to_int8(prepared_model,
                             module_patterns: Tuple,
                             inplace: bool = True):
    """Delete useless fakequant before modules whose type are in
    `module_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before and inner the modules
            whose name in `module_patterns` will be modified.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after modified.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and node.target in module_patterns:
            maybe_weight = _get_attrs(prepared_model, node.target)
            if not (hasattr(maybe_weight, 'weight_fake_quant') and isinstance(
                    maybe_weight.weight_fake_quant, LearnableFakeQuantize)):
                continue
            maybe_weight = maybe_weight.weight_fake_quant
            maybe_act = node.args[0]
            if not (maybe_act.op == 'call_module' and isinstance(
                    _get_attrs(prepared_model, maybe_act.target),
                    LearnableFakeQuantize)):
                continue
            maybe_act = _get_attrs(prepared_model, maybe_act.target)
            dynamic_qlinear.update_qdype_qmin_qmax(maybe_weight, 8)
            dynamic_qlinear.update_qdype_qmin_qmax(maybe_act, 8)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model
