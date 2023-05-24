# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union

import torch

from mmrazor.registry import MODELS
from mmrazor.models.quantizers.native_quantizer import *
from mmrazor.models.task_modules.tracer.fx.graph_utils import del_fakequant_after_placeholder



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
                        self.qconfig.fixed_w_fakequant()
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
                        new_child.weight_fake_quant.scale =weight_fakequant.scale
                    else:
                        new_child = float_child.to(device)
                    setattr(module, name, new_child)
                else:
                    traverse(child)

        observed_module.apply(enable_fake_quant)
        observed_module.apply(disable_observer)
        traverse(observed_module)

    def del_redundant_fakequant(self, prepared: GraphModule):
        """delete redundant fakequant op in prepared model.

        Returns:
            prepared (GraphModule): prepared model after delete redundant
                fakequant op.

        Notes:
             We can configure different ways to delete redundant nodes:
                @property
                def module_prev_wo_fakequant(self):
                    return (torch.nn.ReLU6, torch.nn.Identity)
        """
        extra_module_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_module_prev_wo_fakequant', tuple())
        prepared = del_fakequant_after_placeholder(prepared,inplace=True)
        prepared = del_fakequant_before_module(
            prepared,
            self.module_prev_wo_fakequant + extra_module_prev_wo_fakequant,
            inplace=True)

        extra_module_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_module_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_module(
            prepared,
            self.module_next_wo_fakequant + extra_module_next_wo_fakequant,
            inplace=True)

        extra_function_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_function_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_function(
            prepared,
            self.function_prev_wo_fakequant + extra_function_prev_wo_fakequant,
            inplace=True)

        extra_function_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_function_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_function(
            prepared,
            self.function_next_wo_fakequant + extra_function_next_wo_fakequant,
            inplace=True)

        extra_method_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_method_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_method(
            prepared,
            self.method_prev_wo_fakequant + extra_method_prev_wo_fakequant,
            inplace=True)

        extra_method_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_method_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_method(
            prepared,
            self.method_next_wo_fakequant + extra_method_next_wo_fakequant,
            inplace=True)

        extra_op_prev_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_op_prev_wo_fakequant', tuple())
        prepared = del_fakequant_before_op(
            prepared,
            self.op_prev_wo_fakequant + extra_op_prev_wo_fakequant,
            inplace=True)

        extra_op_next_wo_fakequant = self.extra_redundant_fakequants.get(
            'extra_op_next_wo_fakequant', tuple())
        prepared = del_fakequant_after_op(
            prepared,
            self.op_next_wo_fakequant + extra_op_next_wo_fakequant,
            inplace=True)
        return prepared

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
