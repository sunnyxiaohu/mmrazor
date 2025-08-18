# Copyright (c) OpenMMLab. All rights reserved.

from typing import List
import numpy as np
import difflib
import yaml
import logging

try:
    import onnx
    from onnx import helper, numpy_helper
except ImportError:
    from mmrazor.utils import get_package_placeholder
    onnx = get_package_placeholder('No module named onnx')
    numpy_helper = get_package_placeholder('No module named onnx.numpy_helper')
    helper = get_package_placeholder('No module named onnx.helper')

from mmrazor.models.quantizers.exporters.base_quantize_exporter import BaseQuantizeExportor ,PERCHANNEL_FAKEQUANTIZER,PERTENSOR_FAKEQUANTIZER
from mmengine import print_log


class XiongmaiQuantizeExportor(BaseQuantizeExportor):

    def __init__(self, onnx_model, export_path, export_ref_qfile=None) -> None:
        super().__init__(onnx_model, export_path)
        # self.optimizer.replace_resize_op_with_upsample(self.onnx_model, self.output2node)
        self._remap_input_and_node()
        self._remap_output_and_node()
        self.export_ref_qfile = export_ref_qfile

    def _insert_initializers_to_onnx(self, initializers: List):
        """Insert onnx initializers to the onnx graph."""
        inserted_init_names = set()
        for init in initializers:
            if init.name in inserted_init_names:
                continue

            self.onnx_model.graph.initializer.append(init)
            inserted_init_names.add(init.name)
            
    def clip_weight(self, node, name2data, named_initializer):
        tensor_name, maxinum, fl, qmin, qmax = self.parse_qparams(node)
        data = name2data[tensor_name]
        clip_range_min = -2 ** int(maxinum -1)
        clip_range_max = 2 ** int(maxinum -1) - 2**int(-fl)
        if len(maxinum.shape) > 0 and maxinum.shape[0] > 1:
            new_data = []
            transposed = False
            if data.shape[0] != maxinum.shape[0]:
                transposed = True
                data = data.transpose(1, 0, 2, 3)
            for c in range(data.shape[0]):
                new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
            new_data = np.array(new_data)
            if not np.allclose(data, new_data):
                print_log(f'Clip weights <{tensor_name}> to per-channel ranges.', logger='current', level=logging.WARNING)
            if transposed:
                new_data = new_data.transpose(1, 0, 2, 3)
        else:
            new_data = np.clip(data, clip_range_min, clip_range_max)
            if not np.allclose(data, new_data):
                print_log(f'Clip weights <{tensor_name}> from range [{np.min(data)}, {np.max(data)}] to range [{clip_range_min}, {clip_range_max}].', logger='current', level=logging.WARNING)
        new_data = numpy_helper.from_array(new_data)
        named_initializer[tensor_name].raw_data = new_data.raw_data
        
    def weight_preprocess(self, target_tensor, out2node, inp2node, named_initializer):
        def find_weight(tensor):
            if tensor not in named_initializer:
                _node = out2node[tensor]
                for inp in _node.input:
                    return find_weight(inp)
            return tensor
        weight = find_weight(target_tensor)

        # TODO need more general method, like onnxruntime infer
        data = numpy_helper.to_array(named_initializer[weight])
        data = np.tanh(data)
        data = data / (np.max(np.abs(data)) + 1e-5)
        data = numpy_helper.from_array(data)
        named_initializer[weight].raw_data = data.raw_data

        redundant_nodes = []

        def find_redundant_nodes(tensor):
            if tensor == target_tensor:
                return
            nodes = inp2node[tensor]
            for node, idx in nodes:
                if node not in redundant_nodes:
                    redundant_nodes.append(node)
                    redundant_nodes.extend(self._get_constant_inputs(node))
                find_redundant_nodes(node.output[0])
        find_redundant_nodes(weight)
        return weight, redundant_nodes
    
    def prepare_initializer(self,onnx_model):
        named_initializer = {}
        for init in onnx_model.graph.initializer:
            named_initializer[init.name] = init
        return named_initializer
    
    def deal_with_weight_fakequant(self, node, out2node, inp2node, named_initializer):
        next_nodes = inp2node[node.output[0]]
        assert len(next_nodes) == 1
        next_node, idx = next_nodes[0]
        assert next_node.op_type in ['Conv', 'Gemm', 'ConvTranspose']
        redundant_nodes = []
        if node.input[0] not in named_initializer:
            node.input[0], redundant_nodes = \
                self.weight_preprocess(node.input[0], out2node, inp2node, named_initializer)
        next_node.input[idx] = node.input[0]
        return redundant_nodes

    def deal_with_activation_fakequant(self, node, inp2node):
        next_nodes = inp2node[node.output[0]]
        for next_node, idx in next_nodes:
            next_node.input[idx] = node.input[0]
        return
            
    def clip_and_collect_params(self, symbolic_nodes: List):
        """gen clip range yamlfile."""
        named_initializer = self.prepare_initializer(onnx_model=self.onnx_model)
        nodes_to_be_removed = []
        clip_ranges = {}
        for node in symbolic_nodes:
            if 'activation_post_process_' in node.name:

                if node.output[0] not in self.input2node:
                    assert node.output[0] in [x.name for x in self.graph.output]
                    self.input2node[node.output[0]] = []

                next_nodes = self.input2node[node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node, self.output2node, self.input2node, named_initializer)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, self.name2data, named_initializer)
                else:
                    # fake quantize for activations
                    self.deal_with_activation_fakequant(node, self.input2node)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node)
                    for out in self.graph.output:
                        if out.name == node.output[0]:
                            out.name = tensor_name
            else:
                redundant_nodes = self.deal_with_weight_fakequant(node, self.output2node, self.input2node, named_initializer)
                nodes_to_be_removed.extend(redundant_nodes)
                self.clip_weight(node, self.name2data, named_initializer)
                tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(
                    node)
            clip_ranges[tensor_name] = {'dtype': 'dynamic_fixed_point',
                                        'method': 'layer',
                                        'max_value': None,
                                        'min_value': None,
                                        'fl': [int(zero_point)],
                                        'qtype': 'i8'
                                        }
        return clip_ranges, nodes_to_be_removed

    def post_process_clip_ranges(self, clip_ranges, graph, inp2node, outp2node, name2data):
        def find_the_closest_clip_range(node):
            if node.input[0] in clip_ranges:
                return node.input[0]
            # look forward
            ret = None
            if node.op_type in ['Flatten', 'Resize', 'Relu', 'Clip','Concat', 'MaxPool'] and node.output[0] in inp2node:
                ret = find_the_closest_clip_range(inp2node[node.output[0]][0][0])
            # # Temporal plan, may not correct.
            # if ret is not None:
            #     return ret
            # # look backward
            # if node.op_type in ['Flatten', 'Resize', 'Relu', 'Clip','Concat', 'MaxPool'] and node.input[0] in outp2node:
            #     ret = find_the_closest_clip_range(outp2node[node.input[0]])
            return ret

        for node in graph.node:
            if node.op_type in ['Flatten', 'Resize', 'Relu','Clip', 'Concat', 'MaxPool']:
                tensor_name = find_the_closest_clip_range(node)
                if tensor_name:
                    for i in range(len(node.input)):
                        clip_ranges[node.input[i]] = clip_ranges[tensor_name]
                        inputname = node.input[i]
                        print_log(f'Pass <{tensor_name}> clip range to <{node.name}> input <{inputname}>.', logger='current', level=logging.DEBUG)

        # 1. handle weight name and bias
        for node in graph.node:
            if node.op_type in ['Gemm', 'Conv']:
                for i in range(len(node.input)):
                    inputname = node.input[i]
                    if '.weight' in inputname and inputname in clip_ranges:
                        qrange = clip_ranges.pop(inputname)
                        clip_ranges[f'{node.name}:weight'] = qrange
                    elif '.bias' in inputname:
                        if node.input[0] in clip_ranges:
                            input_fl = clip_ranges[node.input[0]]['fl'][0]
                        else:  # The first layer
                            # import pdb; pdb.set_trace()
                            input_fl = -1
                        weight_fl = clip_ranges[f'{node.name}:weight']['fl'][0]
                        fl = input_fl + weight_fl
                        max_val_pos = np.abs(self.name2data[inputname]).max()
                        maxinum = np.ceil(np.log2(max_val_pos) + 1)
                        max_fl = (16 - maxinum)
                        if fl > max_fl:
                            print_log(f'Setting <{node.name}> bias may uncorrect. excepted fl: {fl}, max_fl: {max_fl}', logger='current', level=logging.WARNING)
                        clip_ranges[f'{node.name}:bias'] = {'dtype': 'dynamic_fixed_point',
                                                            'method': 'layer',
                                                            'max_value': None,
                                                            'min_value': None,
                                                            'fl': [int(fl)],
                                                            'qtype': 'i16'
                                                            }
        if self.export_ref_qfile is None:                                                            
            return clip_ranges                                  
        # 2. tensor name remapping
        # new_clip_ranges = {}
        # for k, v in clip_ranges.items():
        #     new_k = '@Conv_' + k if '/Conv' in k else k
        #     new_k = '@Relu_' + new_k if 'Relu_' in new_k else new_k
        #     new_k = '@Reshape_' + new_k if 'Flatten_' in new_k else new_k
        #     new_clip_ranges[new_k] = v
        # clip_ranges = new_clip_ranges
        ref_clip_ranges = yaml.load(open(self.export_ref_qfile, 'r', encoding='utf-8'), yaml.FullLoader)
        names = list(clip_ranges.keys())
        ref_names = list(ref_clip_ranges.keys())
        new_clip_ranges = {}
        k_refk_mapping = {}
        refk_k_mapping = {}
        for k in names:
            refk = difflib.get_close_matches(k, ref_names, n=1, cutoff=0.0)[0]
            k_refk_mapping[k] = refk

        for refk in ref_names:
            k = difflib.get_close_matches(refk, names, n=1, cutoff=0.0)[0]
            refk_k_mapping[refk] = k
            if k_refk_mapping[k] == refk: # double-direct match
                # k_refk_mapping.pop(k)
                v = clip_ranges.pop(k)
                refv = ref_clip_ranges.pop(refk)
                new_clip_ranges[refk] = v
                print_log(f'Match <{k}> to <{refk}>', logger='current', level=logging.INFO)
        # rematch the last names
        names = list(clip_ranges.keys())
        ref_names = list(ref_clip_ranges.keys())        
        for k, v in clip_ranges.items():
            refk = difflib.get_close_matches(k, ref_names, n=1, cutoff=0.0)[0]
            new_clip_ranges[refk] = v
            ref_clip_ranges.pop(refk)
            print_log(f'Match <{k}> to <{refk}>', logger='current', level=logging.INFO)
        # add the last ref_names
        new_clip_ranges.update(ref_clip_ranges)

        return new_clip_ranges

    def _collect_symbolic_constant_inputs(self, symbolic_nodes: List):
        """Collect these constant nodes which is the input of all the symbolic
        node."""

        collected_constant_names = set()
        constant_inputs_out = list()
        for node in symbolic_nodes:
            constant_inputs = self._get_constant_inputs(node)
            for constant in constant_inputs:
                if constant.name in collected_constant_names:
                    continue
                constant_inputs_out.append(constant)
                collected_constant_names.add(constant.name)
        return constant_inputs_out

    def _remove_symbolic_related(self):
        """removeing symbolic related nodes and initializers in the original
        onnx model ."""
        # import pdb; pdb.set_trace()
        symbolic_nodes = self.collect_symbolic_nodes(self.onnx_model)
        self.clip_ranges, nodes_to_be_removed = self.clip_and_collect_params(symbolic_nodes)

        symbolic_nodes.extend(nodes_to_be_removed)

        collect_func = self._collect_symbolic_constant_inputs
        # Usually different activation fakequants share the same constant
        # input, and different weight fakequants share the same constant input.
        symbolic_constant_inputs = collect_func(symbolic_nodes)

        self._remove_symbolic_related_from_onnx(symbolic_nodes,
                                                symbolic_constant_inputs)

        self.optimizer.optimize(self.onnx_model)

        self.clip_ranges = self.post_process_clip_ranges(
            self.clip_ranges, self.graph, self.input2node, self.output2node, self.name2data)

    def export(self):
        """Export end to end onnx model."""
        self._remove_symbolic_related()
        onnx.save(self.onnx_model, self.export_path)

        context_filename = self.export_path.replace('.onnx','_xiongmai_quantization.cfg')

        with open(context_filename, 'w', encoding="utf-8") as f:
            yaml.dump(self.clip_ranges, f, default_flow_style=False, allow_unicode=True, Dumper=CustomDumper)


class CustomDumper(yaml.Dumper):
    """ 禁止 YAML 输出时使用引用（& 和 *） """
    def ignore_aliases(self, data):
        return True  # 总是返回 True，确保不使用 YAML 变量引用

    # """ 禁止 YAML 输出 None 值（即不写入 null） """
    # @classmethod
    # def represent_mapping(cls, tag, mapping, flow_style=None):
    #     # 过滤掉所有值为 None 的键
    #     new_mapping = {k: v for k, v in mapping.items() if v is not None}
    #     return dumper.represent_mapping(tag, new_mapping, flow_style)

