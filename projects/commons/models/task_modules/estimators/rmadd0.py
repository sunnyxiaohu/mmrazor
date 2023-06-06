import onnx
from onnx import numpy_helper

def update_inp2node_out2node(graph):
    out2node = {}
    inp2node = {}
    for node in graph.node:
        for out in node.output:
            # suppose each node only has one output
            out2node[out] = node
        for idx, inp in enumerate(node.input):
            # one node may have multiple inputs
            if inp not in inp2node:
                inp2node[inp] = []
            inp2node[inp].append([node, idx])
    return out2node, inp2node

def prepare_data(graph):
    params = {}
    for init in graph.initializer:
        params[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    params[node.output[0]] = numpy_helper.to_array(attr.t)
    return params

def get_constant_inputs(node, out2node):
    node_list = []
    for inp in node.input:
        if inp in out2node and out2node[inp].op_type == 'Constant':
            node_list.append(out2node[inp])
    return node_list

class OnnxPreprocess(object):
    def remove_fake_pad_op(self, graph, name2data, inp2node, out2node):
        nodes_to_be_removed = []
        for idx, node in enumerate(graph.node):
            node = graph.node[idx]
            if node.op_type == 'Add' and node.input[1] in name2data.keys():
                addval = name2data[node.input[1]]
                if addval.size==1 and addval==0:
                    print(f"Remove Add op: <{node.name}>.")
                    next_nodes = inp2node[node.output[0]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = node.input[0]
                    nodes_to_be_removed.append(node)
                    nodes_to_be_removed.extend(get_constant_inputs(node, out2node))
        for node in nodes_to_be_removed:
            graph.node.remove(node)
        return

def prepare_initializer(graph):
    named_initializer = {}
    for init in graph.initializer:
        named_initializer[init.name] = init
    return named_initializer

# onnx_path = "./temp_model/temp.onnx"
# model = onnx.load(onnx_path)
# graph = model.graph
# out2node, inp2node = update_inp2node_out2node(graph)
# name2data = prepare_data(graph)
# named_initializer = prepare_initializer(graph)

# preprocess = OnnxPreprocess()
# preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
# out2node, inp2node = update_inp2node_out2node(graph)


#onnx.save(model, "noadd0.onnx")
