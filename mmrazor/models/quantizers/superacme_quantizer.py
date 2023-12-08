# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union, Dict
import copy

import torch
import torch.nn as nn
try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer)
    from torch.ao.quantization.fake_quantize import FakeQuantizeBase                                       
    from torch.ao.quantization.fx import prepare
    from torch.ao.quantization.fx.graph_module import ObservedGraphModule
    from torch.ao.quantization.qconfig_mapping import (
        _FIXED_QPARAMS_OP_TO_OBSERVER, FixedQParamsFakeQuantize, QConfig,
        QConfigMapping, default_weight_fake_quant)
    from torch.ao.quantization.quantize_fx import _fuse_fx
    from torch.fx.graph_module import GraphModule
    from torch.nn.intrinsic.qat import modules as qat_fused_modules
    from torch.nn.qat import modules as qat_modules
    from torch.onnx import register_custom_op_symbolic
except ImportError:
    from mmrazor.utils import get_package_placeholder, get_placeholder
    FakeQuantizeBase = get_placeholder('torch>=1.13')
    GraphModule = get_placeholder('torch>=1.13')
    ObservedGraphModule = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    disable_observer = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    prepare = get_placeholder('torch>=1.13')
    QConfigMapping = get_placeholder('torch>=1.13')
    _fuse_fx = get_placeholder('torch>=1.13')
    qat_fused_modules = get_package_placeholder('torch>=1.13')
    qat_modules = get_package_placeholder('torch>=1.13')
    _FIXED_QPARAMS_OP_TO_OBSERVER = get_package_placeholder('torch>=1.13')
    FixedQParamsFakeQuantize = get_package_placeholder('torch>=1.13')
    QConfig = get_package_placeholder('torch>=1.13')
    default_weight_fake_quant = get_package_placeholder('torch>=1.13')

from mmrazor import digit_version
from mmrazor.registry import MODELS
from mmrazor.models import LearnableFakeQuantize
from mmrazor.models.utils import str2class
from mmrazor.models.task_modules.tracer.fx import build_graphmodule
from mmrazor.models.task_modules.tracer.fx.graph_utils import (_get_attrs,
    modify_fakequant_bits, register_mutables_for_dynamic_fakequant)

from mmrazor.structures.quantization import BackendConfigs
from mmrazor.structures.quantization.backend_config.superacme import get_superacme_backend_config
from ..utils.quantization_util import post_process_nodename
from .native_quantizer import TorchNativeQuantizer, SUPPORT_QAT_MODULES, MERGE_BN_MAPPINGS

BackendConfigs['superacme'] = get_superacme_backend_config()

@MODELS.register_module()
class SuperAcmeQuantizer(TorchNativeQuantizer):
    """Quantizer for quantizing and deploying to SuperAcme backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    SuperAcme's some important features about quantization is as follows:
    * support_w_mode = ('per_tensor', 'per_channel')
    * support_a_mode = ('per_tensor')
    * weight range should be symmetric, such as int 8 is [-127, 127] rather
    than [-128, 127]
    """
    def __init__(self, *args, tracer: Dict = dict(type='CustomTracer'),
                 quant_bits=None, quant_bits_skipped_module_names=None,
                 default_skipped_bit=8, nested_quant_bits_in_layer=False,
                 **kwargs):
        if 'skipped_module_classes' in tracer:
            tracer['skipped_module_classes'] = str2class(tracer['skipped_module_classes'])
        super().__init__(*args, tracer=tracer, **kwargs)
        self.quant_bits = quant_bits
        if quant_bits_skipped_module_names is None:
            quant_bits_skipped_module_names = []
        self.quant_bits_skipped_module_names = quant_bits_skipped_module_names
        self.default_skipped_bit = default_skipped_bit
        self.nested_quant_bits_in_layer = nested_quant_bits_in_layer

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

        prepared = del_fakequant_after_placeholder(prepared)
        # prepared = modify_fakequant_according_clip(prepared)
        prepared = modify_fakequant_bits(
            prepared, tuple(self.quant_bits_skipped_module_names), self.default_skipped_bit, True)
        # import pdb; pdb.set_trace()
        if self.quant_bits:
            prepared = register_mutables_for_dynamic_fakequant(
                prepared, tuple(self.quant_bits_skipped_module_names), self.quant_bits,
                self.default_skipped_bit, True, nested_quant_bits_in_layer=self.nested_quant_bits_in_layer)
        return prepared

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'superacme'

    @property
    def support_w_modes(self):
        """Supported quantization modes for weight about per_tensor or
        per_channel."""
        return ('per_tensor', 'per_channel')

    @property
    def support_a_modes(self):
        """Supported quantization modes for activation about per_tensor or
        per_channel."""
        return ('per_tensor')

    def export_onnx(self,
                    model: Union[torch.nn.Module, torch.jit.ScriptModule,
                                 torch.jit.ScriptFunction],
                    args: Union[Tuple[Any, ...], torch.Tensor],
                    output_path: str,
                    opset_version: Optional[int] = 11,
                    **kwargs):
        """Export the onnx model that can be deployed to SuperAcme backend."""

        symbolic_output_path = output_path.replace('.onnx', '_superacme_symbolic.onnx')
        torch.onnx.export(
            model,
            args,
            symbolic_output_path,
            opset_version=opset_version,
            **kwargs)
        post_process_nodename(symbolic_output_path)

        from .exporters.superacme_quantize_exporter import SuperAcmeQuantizeExportor
        exporter = SuperAcmeQuantizeExportor(symbolic_output_path, output_path)
        exporter.export()

    def post_process_for_deploy(self,
                                observed_module: ObservedGraphModule,
                                device: str = 'cpu',
                                update_weight_with_fakequant: bool = False,
                                keep_w_fake_quant: bool = False):
        """weight fake-quant for supported QAT modules.

        Args:
            observed_module (ObservedGraphModule): Modules after fused and
                observed.
            keep_w_fake_quant (bool, optional): Bool to determine whether to
                keep weight fake-quant op, depending on the backend. Defaults
                to False.

        Note:
            `post_process_weight_fakequant()` function is necessary that the
                `SUPPORT_QAT_MODULES` will be convert to normal modules, and
                BN will be really integrated into conv layers.
        """

        def traverse(module):
            for name, child in module.named_children():
                # Trace `SUPPORT_QAT_MODULES` recursively.
                if isinstance(child, SUPPORT_QAT_MODULES):
                    # We add w_fakequant once in case some ptq methods have
                    # specific operations such as Adaround. So we do Quantize
                    # to perform these operations and do dequantize to
                    # introduce quantization loss in advance.
                    weight_fakequant = child.weight_fake_quant

                    # `to_float()` function fuse BN into conv or conv_relu, and
                    # also convert a qat module to a normal module.
                    # source url: https://github.com/pytorch/pytorch/blob/master/torch/nn/intrinsic/qat/modules/conv_fused.py # noqa: E501
                    float_child = child.to_float()

                    if update_weight_with_fakequant:
                        from torch.ao.nn.intrinsic import _FusedModule
                        if issubclass(type(float_child), _FusedModule):
                            float_child[0].weight.data = weight_fakequant(
                                float_child[0].weight.data)
                        else:
                            float_child.weight.data = weight_fakequant(
                                float_child.weight.data)
                    # This is decided by backend type, some backend need
                    # explicitly keep the fake quant structure, others don't.
                    # TODO add deploy doc link
                    if keep_w_fake_quant:
                        # make weight fakequant fixed as the consistent
                        # fakequant, it will help to deploy our model to
                        # various backends.
                        self.qconfig.fixed_w_fakequant(ref_weight_fakequant=weight_fakequant)
                        for m in float_child.modules():
                            setattr(m, 'qconfig', self.qconfig.convert())
                        if type(child) in MERGE_BN_MAPPINGS:
                            cls = MERGE_BN_MAPPINGS[type(child)]
                            new_child = cls.from_float(float_child).to(device)
                        else:
                            new_child = type(child).from_float(float_child).to(
                                device)

                        # because weight fakequants and observers are replaced
                        # with base fakequants and base observers, some
                        # initialized args need to be update by running
                        # weight_fake_quant.
                        enable_observer(new_child)
                        new_child.weight_fake_quant(new_child.weight)
                        disable_observer(new_child)
                        new_child.weight_fake_quant.scale = weight_fakequant.scale
                    else:
                        new_child = float_child.to(device)
                    setattr(module, name, new_child)
                else:
                    traverse(child)

        observed_module.apply(enable_fake_quant)
        observed_module.apply(disable_observer)
        traverse(observed_module)

    @property
    def module_prev_wo_fakequant(self):
        """Configurate the modules that their previous nodes are redundant
        fakequants."""
        return (torch.nn.ReLU6, torch.nn.Identity)

    @property
    def module_next_wo_fakequant(self):
        """Configurate the modules that their next nodes are redundant
        fakequants."""
        return (torch.nn.MaxPool2d, torch.nn.modules.pooling.AdaptiveAvgPool2d,)

    @property
    def method_next_wo_fakequant(self):
        """Configurate the methods that their next nodes are redundant
        fakequants."""
        return ()

    @property
    def op_prev_wo_fakequant(self):
        """Configurate the OPs that their previous nodes are redundant
        fakequants."""
        return ()
    @property
    def function_prev_wo_fakequant(self):
        """Configurate the functions that their previous nodes are redundant
        fakequants."""
        return (torch.cat,)


def del_fakequant_after_placeholder(prepared_model,
                                    inplace: bool = True):
    """Delete useless fakequant after modules whose op is `placeholder`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    target_nodes = []
    for node in new_graph.nodes:
        if node.op == 'placeholder':
            target_nodes.append(node)

    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node not in target_nodes:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def modify_fakequant_according_clip(prepared_model,
                                    inplace: bool = True):
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    # import pdb; pdb.set_trace()
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
            _get_attrs(prepared_model, node.target), LearnableFakeQuantize):
            relu6 = node.args[0]
            if relu6 == 'call_module' and not isinstance(
                    _get_attrs(prepared_model, relu6.target), nn.ReLU6):
                continue
            fake_quant = _get_attrs(prepared_model, node.target)
            setattr(fake_quant, 'relu6_clip', True)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model
