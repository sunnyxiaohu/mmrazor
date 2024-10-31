# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import import_modules_from_strings

try:
    import onnx
except ImportError:
    from mmrazor.utils import get_package_placeholder
    onnx = get_package_placeholder('No module named onnx')


def _check_valid_source(source):
    """Check if the source's format is valid."""
    if not isinstance(source, str):
        raise TypeError(f'source should be a str '
                        f'instance, but got {type(source)}')

    assert len(source.split('.')) > 1, \
        'source must have at least one `.`'


def str2class(str_inputs):
    clss = []
    if not isinstance(str_inputs, tuple) and not isinstance(str_inputs, list):
        str_inputs_list = [str_inputs]
    else:
        str_inputs_list = str_inputs
    for s_class in str_inputs_list:
        _check_valid_source(s_class)
        mod_str = '.'.join(s_class.split('.')[:-1])
        cls_str = s_class.split('.')[-1]
        try:
            mod = import_modules_from_strings(mod_str)
        except ImportError:
            raise ImportError(f'{mod_str} is not imported correctly.')
        imported_cls: type = getattr(mod, cls_str)
        if not isinstance(imported_cls, type):
            raise TypeError(f'{cls_str} should be a type '
                            f'instance, but got {type(imported_cls)}')
        clss.append(imported_cls)
    if isinstance(str_inputs, list):
        return clss
    elif isinstance(str_inputs, tuple):
        return tuple(clss)
    else:
        return clss[0]


def post_process_nodename(onnx_file, onnx_node_tensor_translate_mapping=None,
                          debug_mode=False):
    if debug_mode:
        return

    onnx_model = onnx.load(onnx_file)
    # Note that for superacme (.hir) files, only the activation tensor name will influence it's storage size.
    onnx_node_tensor_name_mapping = {}
    if onnx_node_tensor_translate_mapping is None:
        onnx_node_tensor_translate_mapping = {}
    else:
        # check provided node/tensor name valididation
        node_names = [node.name for node in onnx_model.graph.node]
        for node in onnx_model.graph.input:
            node_names.append(node.name)
        for node in onnx_model.graph.output:
            node_names.append(node.name)
        tensor_names = []
        for node in onnx_model.graph.node:
            tensor_names.extend([name for name in node.input] + [name for name in node.output])
        for name in onnx_node_tensor_translate_mapping:
            if name not in node_names + tensor_names:
                raise ValueError(f'Provided node/tensor name: {name} not in the onnx_model: {node_names+tensor_names}')
        onnx_node_tensor_name_mapping.update(onnx_node_tensor_translate_mapping)

    def find_new_node_name(old_name, node_idx):
        if old_name in onnx_node_tensor_name_mapping:
            new_name = onnx_node_tensor_name_mapping[old_name]
        else:
            new_name = f'{node_idx}'
            while (new_name in onnx_node_tensor_name_mapping):
                node_idx += 1
                new_name = f'{node_idx}'
        next_node_idx = node_idx+1 if new_name == f'{node_idx}' else node_idx
        return new_name, next_node_idx

    node_idx = 0
    for node in onnx_model.graph.node:
        new_name, node_idx = find_new_node_name(node.name, node_idx)
        onnx_node_tensor_name_mapping[node.name] = new_name
        for input_name in node.input:
            new_name, node_idx = find_new_node_name(input_name, node_idx)
            onnx_node_tensor_name_mapping[input_name] = new_name
        for output_name in node.output:
            new_name, node_idx = find_new_node_name(output_name, node_idx)
            onnx_node_tensor_name_mapping[output_name] = new_name

    for node in onnx_model.graph.input:
        node.name = onnx_node_tensor_name_mapping.get(node.name, node.name)
    for node in onnx_model.graph.output:
        node.name = onnx_node_tensor_name_mapping.get(node.name, node.name)
    for node in onnx_model.graph.node:
        node.name = onnx_node_tensor_name_mapping.get(node.name, node.name)
        for idx, input_name in enumerate(node.input):
            node.input[idx] = onnx_node_tensor_name_mapping.get(input_name, input_name)
        for idx, output_name in enumerate(node.output):
            node.output[idx] = onnx_node_tensor_name_mapping.get(output_name, output_name)
    for initializer in onnx_model.graph.initializer:
        initializer.name = onnx_node_tensor_name_mapping.get(initializer.name, initializer.name)
    # TDL: graph.value_info
    # import pdb; pdb.set_trace()
    onnx.save(onnx_model, onnx_file)
