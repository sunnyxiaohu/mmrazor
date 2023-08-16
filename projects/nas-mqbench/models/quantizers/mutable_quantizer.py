# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Any, List, Tuple
from types import MethodType
import operator
from inspect import signature
import copy

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.utils import import_modules_from_strings
from mmrazor.registry import MODELS
from mmrazor.models import OpenVINOQuantizer, LearnableFakeQuantize
from mmrazor.models.utils import str2class
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.models.architectures.dynamic_ops import (DynamicConvMixin,
                                                      DynamicMixin,
                                                      DynamicLinearMixin,
                                                      DynamicSequential)
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
custom_imports = 'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_lsq'
dynamic_lsq = import_modules_from_strings(custom_imports)


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
                 quant_bits=None, quant_bits_skipped_module_names=None,
                 default_skipped_bit=8, **kwargs):
        if 'skipped_module_classes' in tracer:
            tracer['skipped_module_classes'] = str2class(tracer['skipped_module_classes'])
        super().__init__(*args, tracer=tracer, **kwargs)
        self.quant_bits = quant_bits
        if quant_bits_skipped_module_names is None:
            quant_bits_skipped_module_names = []
        self.quant_bits_skipped_module_names = quant_bits_skipped_module_names
        self.default_skipped_bit = default_skipped_bit

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'mutableopenvino'

    @property
    def support_a_modes(self):
        """Supported quantization modes for activation about per_tensor or
        per_channel."""
        return ('per_tensor', 'per_channel')

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
        # import pdb; pdb.set_trace()
        # get all the dynamic seq names.
        dynamic_seq_names = {}
        for name, module in model.named_modules():
            if isinstance(module, DynamicSequential) and 'depth' in module.mutable_attrs:
                idx_nodes = {idx+1: [] for idx in range(module.pure_module_nums)}
                dynamic_seq_names[(name, module.mutable_depth)] = idx_nodes

        self.swap_ff_with_fxff(model)
        traced_graph = self.tracer.trace(model, concrete_args=concrete_args)
        graph_module = build_graphmodule(model, traced_graph)
        # import pdb; pdb.set_trace()
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

        if self.quant_bits:
            prepared = register_mutables_for_dynamic_fakequant(
                prepared, tuple(self.quant_bits_skipped_module_names), self.quant_bits,
                self.default_skipped_bit, True)
        prepared = modify_fakequant_to_int8(
            prepared, tuple(self.quant_bits_skipped_module_names), True)

        prepared = wrap_depth_scope(
            prepared, dynamic_seq_names, self.tracer.node_name_to_scope, True)
        return prepared

def wrap_depth_scope(prepared_model, dynamic_seq_names, node_name_to_scope, inplace: bool = True):
    # Since fx.trace can not handle nested nodes. e.g., we have `DynamicSequential`
    # `BigNASConv2d`, where the `BigNASConv2d` is nested in `DynamicSequential`, we
    # can not both treat then as `Node`(leaf module). for simplicy, we record the 
    # submodule's `depth_scope` and propogate `mutable_depth`.

    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)

    for name, depth_mutable in dynamic_seq_names:
        for node in new_graph.nodes:
            if node.name not in node_name_to_scope:
                continue
            scope_name = node_name_to_scope[node.name][0]
            if scope_name.startswith(name):
                depth_scope = int(scope_name.replace(name, '').split('.')[1]) + 1
                # print(node.name, scope_name, depth_scope)
                dynamic_seq_names[(name, depth_mutable)][depth_scope].append(node)
                if node.op == 'call_module' or node.op == 'call_function':
                    for maybe_act in reversed(node.args):
                        if maybe_act.op == 'call_module' and isinstance(
                                _get_attrs(prepared_model, maybe_act.target), LearnableFakeQuantize):
                            # print(maybe_act.name, depth_scope)
                            pos = len(dynamic_seq_names[(name, depth_mutable)][depth_scope]) - 1
                            if maybe_act not in dynamic_seq_names[(name, depth_mutable)][depth_scope]:
                                dynamic_seq_names[(name, depth_mutable)][depth_scope].insert(pos, maybe_act)
                elif node.op == 'call_method':
                    raise ValueError(f'Not supported call_method yet, get `{node.name}`')

    def module_call_wrapper(self, depth_scope, depth_mutable):
        _orig_module_call: Callable = self.forward
        self._DEPTH_SCOPE = depth_scope
        self._DEPTH_MUTABLE = depth_mutable
        self._already_patched = True

        def module_call(self, *args, **kwargs):
            assert hasattr(self, '_DEPTH_SCOPE')
            assert hasattr(self, '_DEPTH_MUTABLE')
            if not isinstance(args[0], torch.Tensor):
                print(f'Except torch.Tensor, {self} get invalid: {type(args[0])}')
            if self._DEPTH_MUTABLE.current_choice < self._DEPTH_SCOPE:
                # import pdb; pdb.set_trace()
                return args[0]
            return _orig_module_call(*args, **kwargs)
        return module_call

    class function_call_wrapper(object):
        SUPPORTED_FUNCS = (torch.add, operator.add)

        def __init__(self, name, func, depth_scope, depth_mutable):
            assert func in self.SUPPORTED_FUNCS, f'Only support {self.SUPPORTED_FUNCS}, get {func}'
            self.func = func
            self._DEPTH_SCOPE = depth_scope
            self._DEPTH_MUTABLE = depth_mutable
            self.__name__ = name

        def __call__(self, *args, **kwarg):
            if self.func in (torch.add, operator.add):
                return self.__add__(*args, **kwarg)

        def __add__(self, input, other, *args, alpha=1, out=None):
            # import pdb; pdb.set_trace()
            if self._DEPTH_MUTABLE.current_choice < self._DEPTH_SCOPE:
                # import pdb; pdb.set_trace()
                assert input is other, 'Get invalid input and other'
                for arg in args:
                    assert input is arg, 'Get invalid input and arg'
                return input
            else:
                if self.func == torch.add:
                    return self.func(input, other, *args, alpha=alpha, out=out)
                elif self.func == operator.add:
                    return self.func(input, other)

    memo = {}
    for (name, depth_mutable), idx_nodes in dynamic_seq_names.items():
        for idx, nodes in idx_nodes.items():
            for node in nodes:
                if node not in memo:
                    memo[node] = [(name, idx)]
                else:
                    memo[node].append((name, idx))
                    raise ValueError(f'Duplicated node: {node} in {memo[node]})')
                if node.op == 'call_module':
                    module = _get_attrs(prepared_model, node.target)
                    if not getattr(module, '_already_patched', False):
                        module.forward = MethodType(
                            module_call_wrapper(module, idx, depth_mutable), module)
                elif node.op == 'call_function':
                    # import pdb; pdb.set_trace()
                    new_func = function_call_wrapper(node.name, node.target, idx, depth_mutable)
                    node.target = new_func
                    node.op == 'call_module'

    # import pdb; pdb.set_trace()
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def register_mutables_for_dynamic_fakequant(prepared_model,
                                            module_patterns: Tuple,
                                            quant_bits: List = None,
                                            default_skipped_bit = 32,
                                            inplace: bool = True):
    """Register mutables for dynamic fakequant. It will follow the rules bellow
    to register mutables:
    1. if quant_bits contains `FLOAT_BITS`, all the dynamic fakequant will use
        register mutable by using `quant_bits` as candidate values.
    2. if not, all the dynamic fakequant except that matchs `module_patterns` will
        register mutable by using `quant_bits` as candidate values.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before and inner the modules
            whose name in `module_patterns` will be modified.
        quant_bits (List): Quant bits for building mutables.
        default_skipped_bit (int): If matched, the default skipped bit will be registered.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after modified.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    # get skipped nodes, weights and activation.
    # skipped_nodes = set()
    # since there will be multiple nodes share the same fake_quant
    skipped_fake_quant = set()
    for node in new_graph.nodes:
        if node.op == 'call_module' and node.target in module_patterns:
            maybe_weight = _get_attrs(prepared_model, node.target)
            if not (hasattr(maybe_weight, 'weight_fake_quant') and isinstance(
                    maybe_weight.weight_fake_quant, dynamic_lsq.DynamicLearnableFakeQuantize)):
                continue
            # skipped_nodes.add(node)
            skipped_fake_quant.add(maybe_weight.weight_fake_quant)
            maybe_act = node.args[0]
            if not (maybe_act.op == 'call_module' and isinstance(
                    _get_attrs(prepared_model, maybe_act.target), dynamic_lsq.DynamicLearnableFakeQuantize)):
                continue
            # skipped_nodes.add(maybe_act)
            skipped_fake_quant.add(_get_attrs(prepared_model, maybe_act.target))
    for node in new_graph.nodes:
        if node.op == 'call_module':
            maybe_dynamic = _get_attrs(prepared_model, node.target)
            # fp32 for BN when not fuse_bn
            if isinstance(maybe_dynamic, _BatchNorm):
                maybe_act = node.args[0]
                if not (maybe_act.op == 'call_module' and isinstance(
                        _get_attrs(prepared_model, maybe_act.target),
                        LearnableFakeQuantize)):
                    continue
                qbits = OneShotMutableValue(alias=maybe_act.target + '.quant_bits', value_list=[32])
                maybe_dynamic = _get_attrs(prepared_model, maybe_act.target)
                maybe_dynamic.register_mutable_attr('quant_bits', qbits)
            # for activations
            elif isinstance(maybe_dynamic, dynamic_lsq.DynamicLearnableFakeQuantize):
                if 'quant_bits' in maybe_dynamic.mutable_attrs:
                    continue
                this_bits = quant_bits
                if maybe_dynamic in skipped_fake_quant:
                    this_bits = [default_skipped_bit]
                qbits = OneShotMutableValue(alias=node.target + '.quant_bits', value_list=this_bits)
                maybe_dynamic.register_mutable_attr('quant_bits', qbits)
            # for weights
            elif hasattr(maybe_dynamic, 'weight_fake_quant') and isinstance(
                    maybe_dynamic.weight_fake_quant, dynamic_lsq.DynamicLearnableFakeQuantize):
                maybe_dynamic = maybe_dynamic.weight_fake_quant
                this_bits = quant_bits
                if maybe_dynamic in skipped_fake_quant:
                    this_bits = [default_skipped_bit]
                qbits = OneShotMutableValue(alias=node.target + '.weight_fake_quant.quant_bits', value_list=this_bits)
                maybe_dynamic.register_mutable_attr('quant_bits', qbits)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


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
            dynamic_lsq.update_qdype_qmin_qmax(maybe_weight, 8)
            dynamic_lsq.update_qdype_qmin_qmax(maybe_act, 8)

    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model
