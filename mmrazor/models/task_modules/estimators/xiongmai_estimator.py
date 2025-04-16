# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy
import os.path as osp
import sys
import zlib
import time
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import MNN
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.utils.data import DataLoader

from mmengine import print_log
from mmengine.dist import get_rank, get_world_size
from mmengine.fileio import load, dump
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist

from mmrazor.registry import METRICS, TASK_UTILS

from ...utils.quantization_util import post_process_nodename
from .resource_estimator import ResourceEstimator

logger = MMLogger.get_current_instance()


@TASK_UTILS.register_module()
class XiongmaiResourceEstimator(ResourceEstimator):
    """Estimator for calculating the resources consume.

    Args:
        dpumodel_cfg (dict): Cfg for estimating dpumodel.
        input_shape (tuple): Input data's default shape, for calculating
            resources consume. Defaults to (1, 3, 224, 224).
        units (dict): Dict that contains converted FLOPs/params/latency units.
            Default to dict(flops='M', params='M', latency='ms').
        as_strings (bool): Output FLOPs/params/latency counts in a string
            form. Default to False.
        flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
            Default to None.
        latency_cfg (dict): Cfg for estimating latency. Default to None.

    """

    def __init__(
        self,
        dpumodel_cfg: dict,
        input_shape: Tuple = (1, 3, 112, 112),
        units: Dict = dict(flops='M', params='M', latency='ms'),
        as_strings: bool = False,
        flops_params_cfg: Optional[dict] = None,
        latency_cfg: Optional[dict] = None,
        dataloader: Optional[DataLoader] = None,
        only_backend: bool = False,
    ):
        super().__init__(
            input_shape,
            units,
            as_strings,
            flops_params_cfg=flops_params_cfg,
            latency_cfg=latency_cfg,
            dataloader=dataloader)
        self.dpumodel_cfg = dpumodel_cfg
        self.dpumodel = TASK_UTILS.build(self.dpumodel_cfg, default_args=dict(dataloader=self.dataloader))
        self.only_backend = only_backend

    @torch.no_grad()
    def estimate(self,
                 model: torch.nn.Module,
                 flops_params_cfg: dict = None,
                 latency_cfg: dict = None) -> Dict[str, Union[float, str]]:
        """Estimate the resources(flops/params/latency) of the given model.

        This method will first parse the merged :attr:`self.flops_params_cfg`
        and the :attr:`self.latency_cfg` to check whether the keys are valid.

        Args:
            model: The measured model.
            flops_params_cfg (dict): Cfg for estimating FLOPs and parameters.
                Default to None.
            latency_cfg (dict): Cfg for estimating latency. Default to None.

            NOTE: If the `flops_params_cfg` and `latency_cfg` are both None,
            this method will only estimate FLOPs/params with default settings.

        Returns:
            Dict[str, Union[float, str]]): A dict that contains the resource
                results(FLOPs, params and latency).
        """
        # resource_metrics = super().estimate(
        #     model, flops_params_cfg=flops_params_cfg, latency_cfg=latency_cfg)

        resource_metrics = dict()
        self.dpumodel.import_torch(model)
        self.dpumodel.dpu_convert()
        self.dpumodel.dpu_profiler()
        if self.dpumodel.infer_metric is not None:
            if not self.only_backend:
                fakequant_metrics = self.dpumodel.torch_fixed_inference()
                print_log(f'torch fakequant metrics: {fakequant_metrics}', logger='current')
            metrics = self.dpumodel.fixed_inference()
            resource_metrics.update(metrics)
        heron_metircs = self.dpumodel.res_extract()
        self.dpumodel.reset_model()

        resource_metrics.update(heron_metircs)
        return resource_metrics


@TASK_UTILS.register_module()
class XiongmaiModelWrapper:
    """Xiongmai Wrapper class.
    """

    def __init__(self,
                 work_dir,
                 dataloader,
                 num_infer=None,
                 is_quantized=False,
                 outputs_mapping=None,
                 onnx_node_tensor_translate_mapping=None,
                 onnx_node_debug_mode=False,
                 use_flip=True,
                 profiler_args='',
                 infer_metric=None):
        name = f'{self.__class__.__name__}'
        work_dir = os.path.join(work_dir, f'rank_{get_rank()}')
        mkdir_or_exist(work_dir)
        self.dataloader = dataloader
        self.onnx_file = osp.join(work_dir, f'{name}.onnx')
        self.dpu_file = osp.join(work_dir, f'{name}.dpu')
        self.profiler_net_res = osp.join(work_dir, f'{name}_net_profiler.txt')
        self.profiler_layer_res = osp.join(work_dir, f'{name}_layer_profiler.csv')
        # sann config path load
        # heron tool load
        if isinstance(next(iter(self.dataloader))['inputs'], torch.Tensor):
            self.inputs_is_tensor = True
            self.shape = (1, ) + next(iter(self.dataloader))['inputs'].shape[1:]
        else:
            self.inputs_is_tensor = False
            self.shape = (1, ) + next(iter(self.dataloader))['inputs'][0].shape
        self.is_quantized = is_quantized
        self.infer_metric = infer_metric
        if infer_metric is not None:
            self.infer_metric = METRICS.build(infer_metric)
            if hasattr(self.dataloader.dataset, 'metainfo'):
                self.infer_metric.dataset_meta = self.dataloader.dataset.metainfo
            else:
                print_log(
                    f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                    'metainfo. ``dataset_meta`` in metric will be None.',
                    logger='current',
                    level=logging.WARNING)
        self.num_infer = num_infer if num_infer is not None and 0 <= num_infer < len(self.dataloader) else len(self.dataloader)
        self.model = None
        self.outputs_mapping = outputs_mapping
        self.use_flip = use_flip
        self.profiler_args = profiler_args
        self.onnx_node_tensor_translate_mapping = onnx_node_tensor_translate_mapping
        self.onnx_node_debug_mode = onnx_node_debug_mode

    def import_torch(self, model):
        self.model = model
        device = next(model.parameters()).device
        if self.inputs_is_tensor:
            dummy_data = next(iter(self.dataloader))['inputs'][0, None].float().to(device)
        else:
            dummy_data = next(iter(self.dataloader))['inputs'][0].unsqueeze(0).float().to(device)
        if self.is_quantized:
            observed_model = model.get_deploy_model()
            model.quantizer.export_onnx(observed_model, dummy_data, self.onnx_file,
                                        onnx_node_tensor_translate_mapping=self.onnx_node_tensor_translate_mapping,
                                        debug_mode=self.onnx_node_debug_mode)
            self.observed_model = observed_model
        else:
            model = fuse_conv_bn(model)
            torch.onnx.export(
                model,
                dummy_data,
                self.onnx_file,
                keep_initializers_as_inputs=False,
                verbose=False,
                opset_version=11)
            post_process_nodename(self.onnx_file, onnx_node_tensor_translate_mapping=self.onnx_node_tensor_translate_mapping,
                                  debug_mode=self.onnx_node_debug_mode)

    def dpu_convert(self):
        # convert and compiler
        pass

    def dpu_profiler(self):
        pass

    def res_extract(self):
        results = {}
        return results

    def reset_model(self):
        torch.cuda.empty_cache()

    def torch_fixed_inference(self):
        # import pdb; pdb.set_trace()
        for i, data in enumerate(self.dataloader):
            if i >= self.num_infer:
                break
            inputs, data_samples = data['inputs'], data['data_samples']
            outputs = self.model.val_step(data)
            self.infer_metric.process(inputs, [out.to_dict() for out in outputs])
        num_samples = min(get_world_size() * self.num_infer, len(self.dataloader.dataset))
        metrics = self.infer_metric.evaluate(num_samples) if self.num_infer > 0 else {}
        return metrics

    def fixed_inference(self):
        """Fixed point inference."""
        metrics = {}
        return metrics


def _fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    if hasattr(conv, 'transposed') and conv.transposed:
        shape = [1, -1] + [1] * (len(conv.weight.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv.weight.shape) - 2)
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w * factor.reshape(shape))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, (_ConvNd, nn.Linear)):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module
