import math
import warnings
from typing import Dict

import torch
from torch.ao.quantization import (FakeQuantizeBase, disable_fake_quant, disable_observer,
                                   enable_fake_quant, enable_observer)

from .recorder import MeasureRecorder


class MeasurePrinter():
    """Helper class for print top-k record."""
    def __init__(self, data: Dict[str, float], measure: str, label: str = 'Layer',
                 k: int = None, order: str = 'large_to_small', 
                 percentage: bool = False) -> None:

        if order not in {'large_to_small', 'small_to_large', None}:
            raise ValueError('Parameter "order" can only be "large_to_small" or "small_to_large"')
        self.collection = [(name, value) for name, value in data.items()]
        if order is not None:
            self.collection = sorted(self.collection, key=lambda x: x[1])
            if order == 'large_to_small': self.collection = self.collection[::-1]
        if k is not None: self.collection = self.collection[:k]

        if order is None:
            sorted_collection = sorted(self.collection, key=lambda x: x[1])
            largest_element, smallest_element = sorted_collection[-1][1], sorted_collection[0][1]
        elif order == 'large_to_small':
            largest_element, smallest_element = self.collection[0][1], self.collection[-1][1]
        else: largest_element, smallest_element = self.collection[-1][1], self.collection[0][1]
        self.normalized_by = largest_element - smallest_element
        self.min = smallest_element

        max_name_length = len(label)
        for name, _ in self.collection:
            max_name_length = max(len(name), max_name_length)
        self.max_name_length = max_name_length
        self.measure_str = measure
        self.label = label
        self.percentage = percentage

    def print(self, max_blocks: int = 20):
        print(f'{self.label}{" " * (self.max_name_length - len(self.label))}  | {self.measure_str} ')
        for name, value in self.collection:
            normalized_value = (value - self.min) / (self.normalized_by + 1e-7)
            if math.isnan(value):
                warnings.warn('MeasurePrinter found an NaN value in your data.')
                normalized_value = 0
            num_of_blocks = round(normalized_value * max_blocks)
            
            if not self.percentage:
                print(f'{name}:{" " * (self.max_name_length - len(name))} | '
                    f'{"█" * num_of_blocks}{" " * (max_blocks - num_of_blocks)} | '
                    f'{value:.4f}')
            else:
                print(f'{name}:{" " * (self.max_name_length - len(name))} | '
                    f'{"█" * num_of_blocks}{" " * (max_blocks - num_of_blocks)} | '
                    f'{value * 100:.3f}%')


def get_activation_hook(name, results):
    def wrapper(mod, input, output):
        key = f'{name}_quantize' if mod.fake_quant_enabled[0] else f'{name}_float'
        if key not in results:
            results[key] = []
        if not isinstance(output, torch.Tensor):
            assert isinstance(output, tuple)
            assert isinstance(output[0], torch.Tensor)
            warnings.warn(f'{name} get {len(output)} outputs, handle the first')
            output = output[0]
        results[key].append(output)
    return wrapper


def graphwise_analysis(model, dataloader, method, step=-1, verbose=False, topk=1):
    # Note: make sure that dataloader is randomness.
    # step1: add hooks for recording activation map.
    results = {}
    error_results = {}
    names = []
    recorders = {}
    step = step if 0 <= step <= len(dataloader) else len(dataloader)
    for name, module in model.qmodels['predict'].named_modules():
        if name.startswith('activation_post_process') and isinstance(module, FakeQuantizeBase):
            module.register_forward_hook(get_activation_hook(name, results))
            names.append(name)
            recorders[name] = MeasureRecorder(measurement=method)

    model.eval()
    # step2: quantize graph and get it's activation map.
    model.apply(enable_fake_quant)
    model.apply(enable_observer)
    for idx, data_batch in enumerate(dataloader):
        if idx >= step:
            break        
        model.test_step(data_batch)

    # step3: dequantize graph and get it's activation map.
    model.apply(disable_fake_quant)
    model.apply(disable_observer)
    for idx, data_batch in enumerate(dataloader):
        if idx >= step:
            break        
        model.test_step(data_batch)

    assert 2 * len(names) == len(results)

    for name in names:
        recorder = recorders[name]
        fkey = f'{name}_float'
        qkey = f'{name}_quantize'
        fresult = results[fkey]
        qresult = results[qkey]
        for idx in range(step):
            recorder.update(y_real=fresult[idx], y_pred=qresult[idx])
        error_results[name] = recorder.measure

    if verbose:
        method_str = 'MEASUREMENT'
        if method == 'snr': method_str = 'NOISE:SIGNAL POWER RATIO'
        if method == 'cosine': method_str = 'COSINE SIMILARITY'
        if method == 'mse': method_str = 'MSE LOSS(UNSCALED)'
        printer = MeasurePrinter(error_results, order='large_to_small', measure=method_str, percentage=method in {'snr', 'cosine'})
        printer.print()

        context_topk = 3
        node_name_to_scope = model.quantizer.tracer.node_name_to_scope
        for name, value in printer.collection[:topk]:
            for node in model.qmodels['predict'].graph.nodes:
                if node.name == name:
                    break

            # 1. get the information of the current node
            print(f'\nNode {name}: {node.__dict__}, \n and Used by:')
            for us in list(node.users):
                try:
                    print(f'\t\t user({us.name}): {node_name_to_scope[us.name]}')
                except KeyError:
                    print('\t\t ---------')
            # 2. find the context by node._next and node._prev
            print(f'Top {context_topk} Context of Node: {name}')
            k = 1
            cnode = node._prev
            context_str = []
            while(k <= context_topk and cnode):
                cstr = f'** -{k} **: {cnode.name}, target: {cnode.target}'  # {cnode.__dict__}')
                context_str.insert(0, cstr)
                for us in list(cnode.users):
                    try:
                        cstr = f'\t\t user({us.name}): {node_name_to_scope[us.name]}'
                    except KeyError:
                        cstr = f'\t\t user({us.name}): ---------'
                    context_str.append(cstr)
                cnode = cnode._prev
                k += 1
            k = 1
            cnode = node._next
            while(k <= context_topk and cnode):
                cstr = f'** +{k} **: {cnode.name}, target: {cnode.target}'  # {cnode.__dict__}')
                context_str.append(cstr)
                for us in list(cnode.users):
                    try:
                        cstr = f'\t\t user({us.name}): {node_name_to_scope[us.name]}'
                    except KeyError:
                        cstr = f'\t\t user({us.name}): ---------'
                    context_str.append(cstr)
                cnode = cnode._next
                k += 1
            print('\n'.join(context_str))

    return error_results
