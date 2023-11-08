# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import zlib
from typing import Dict, Optional, Tuple, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
from mmengine.dist import broadcast_object_list, get_rank
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist
from torch.utils.data import DataLoader

from mmrazor.models import ResourceEstimator
from mmrazor.registry import METRICS, TASK_UTILS

logger = MMLogger.get_current_instance()


@TASK_UTILS.register_module()
class HERONResourceEstimator(ResourceEstimator):
    """Estimator for calculating the resources consume.

    Args:
        heronmodel_cfg (dict): Cfg for estimating heronmodel.
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
        heronmodel_cfg: dict,
        input_shape: Tuple = (1, 3, 112, 112),
        units: Dict = dict(flops='M', params='M', latency='ms'),
        as_strings: bool = False,
        flops_params_cfg: Optional[dict] = None,
        latency_cfg: Optional[dict] = None,
        dataloader: Optional[DataLoader] = None,
    ):
        super().__init__(
            input_shape,
            units,
            as_strings,
            flops_params_cfg=flops_params_cfg,
            latency_cfg=latency_cfg,
            dataloader=dataloader)
        self.heronmodel_cfg = heronmodel_cfg
        #self._check_update_heronmodel_cfg(self.heronmodel_cfg)
        self.heronmodel = HERONModelWrapper(val_data=self.dataloader.dataset, **self.heronmodel_cfg)

    # def _check_update_heronmodel_cfg(self, heronmodel_cfg: dict) -> None:
    #     qdef_file_dir = heronmodel_cfg.pop('qdef_file_dir', '')
    #     for key, value in heronmodel_cfg.items():
    #         if key not in HERONModelWrapper.__init__.__code__.co_varnames[1:]:
    #             raise KeyError(f'Got invalid key `{key}` in heronmodel_cfg.')
    #         qdef_names = ['qfnodes', 'qfops']
    #         if key in qdef_names:
    #             heronmodel_cfg[key] = osp.join(qdef_file_dir, value)

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
        resource_metrics = super().estimate(
            model, flops_params_cfg=flops_params_cfg, latency_cfg=latency_cfg)

        self.heronmodel.import_torch(model)
        self.heronmodel.sann_convert()
        self.heronmodel.hir_convert()
        # self.heronmodel.profiler()
        # if self.heronmodel.infer_metric is not None:
        #     logger.info(
        #         f'Floating inference: {self.heronmodel.floating_inference()}')
        #     logger.info(
        #         f'Fixed inference: {self.heronmodel.fixed_inference()}')
        heron_metircs = self.heronmodel.res_extract()
        self.heronmodel.reset_model()

        resource_metrics.update(heron_metircs)
        return resource_metrics


class HERONModelWrapper:
    """HERON Wrapper class.
    """

    def __init__(self,
                 work_dir,
                 val_data,
                 ptq_json = None,
                 HeronCompiler = None,
                 HeronProfiler = None
                 ):
        name = f'{self.__class__.__name__}'
        work_dir = os.path.join(work_dir, f'rank_{get_rank()}')
        mkdir_or_exist(work_dir)
        self.val_data = val_data
        self.onnx_file = osp.join(work_dir, f'{name}.onnx')
        self.mnn_quant_fix_file = osp.join(work_dir, f'{name}_minmax-a8w8_sann-v03.mnn')
        self.mnn_quant_file = osp.join(work_dir, f'{name}.mnn')
        self.hir_file = osp.join(work_dir, f'{name}.hir')
        self.profiler_net_res = osp.join(work_dir, f'{name}_net_profiler.txt')
        self.profiler_layer_res = osp.join(work_dir, f'{name}_layer_profiler.txt')
        # sann config path load
        self.mnn_ptq_w8a8_json = ptq_json
        # heron tool load
        self.HeronCompiler = HeronCompiler
        self.HeronProfiler = HeronProfiler
        self.reset_data()

    def reset_data(self):
        """Reset dataset iterator."""
        self.it_val_data = iter(self.val_data)

    def pre_func(self, index):
        """Pre-processsing function; for loading input images."""
        print('\r{0}'.format(index), end='')
        ## Get the images and labels
        data_batch = next(self.it_val_data, (None))
        ## MMDataset return data after transforms pipeline.
        self.data = np.asarray(data_batch['inputs'])
        self.data = np.expand_dims(self.data, axis=0)
        self.label = data_batch['data_samples']
        ## Note: NPUC inference engine takes in numpy.ndarray object.
        x = self.data
        return x

    def import_torch(self, model):
        device = next(model.parameters()).device
        dummy_data = self.val_data[0]['inputs'].float().unsqueeze(0).to(device)
        torch.onnx.export(
            model,
            dummy_data,
            self.onnx_file,
            keep_initializers_as_inputs=False,
            verbose=False,
            opset_version=11)

    def sann_convert(self):
        # prepare sann deb and environment 
        command_line = 'sann build --input '+self.onnx_file+' --output '+self.mnn_quant_file+' --config '+self.mnn_ptq_w8a8_json+' > /dev/null'
        os.system(command_line)

    def hir_convert(self):
        # prepare heronrt and environment
        # command_line = self.HeronCompiler+' -i '+self.mnn_quant_fix_file+' -o '+self.hir_file +' --dataFormat RGBA 2>&1 | tee ' + self.profiler_net_res
        command_line = self.HeronCompiler+' -i '+self.mnn_quant_fix_file+' -o '+self.hir_file +' --dataFormat RGBA > ' + self.profiler_net_res
        os.system(command_line)
        command_line = self.HeronProfiler+ ' -m '+self.hir_file+' > ' + self.profiler_layer_res
        os.system(command_line)

    def res_extract(self):
        result_net = open(self.profiler_net_res).readlines()
        ddr =float(result_net[2].split()[-1][:-3])
        sram = float(result_net[3].split()[-1][:-3])
        params = float(result_net[4].split()[-1][:-3])
        bandwidth = ddr+sram
        result_layers = open(self.profiler_layer_res).readlines()
        latency = float(result_layers[-2].split('(')[0].split()[-1])
        results = {
            'heron_bandwidth': bandwidth,
            'heron_latency': latency,
            'heron_params': params,
        }
        return results

    def reset_model(self):        
        self.data = None        
        self.label = None        
        torch.cuda.empty_cache()
