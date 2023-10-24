# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy
import os.path as osp
import sys
import zlib
from typing import Dict, Optional, Tuple, Union

import numpy as np
import MNN
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mmengine import print_log
from mmengine.dist import get_rank
from mmengine.fileio import load, dump
from mmengine.logging import MMLogger
from mmengine.utils import mkdir_or_exist

from mmrazor.registry import METRICS, TASK_UTILS

from .resource_estimator import ResourceEstimator

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
        self.heronmodel = TASK_UTILS.build(self.heronmodel_cfg, default_args=dict(val_data=self.dataloader.dataset))

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
        self.heronmodel.import_torch(model)
        self.heronmodel.hir_convert()
        self.heronmodel.hir_profiler()
        if self.heronmodel.infer_metric is not None:
            self.heronmodel.reset_data()
            fakequant_metrics = self.heronmodel.torch_fixed_inference()
            print_log(f'torch fakequant metrics: {fakequant_metrics}', logger='current')
            self.heronmodel.reset_data()
            metrics = self.heronmodel.fixed_inference()
            resource_metrics.update(metrics)
        heron_metircs = self.heronmodel.res_extract()
        self.heronmodel.reset_model()

        resource_metrics.update(heron_metircs)
        return resource_metrics


@TASK_UTILS.register_module()
class HERONModelWrapper:
    """HERON Wrapper class.
    """

    def __init__(self,
                 work_dir,
                 val_data,
                 num_infer=None,
                 mnn_quant_json=None,
                 is_quantized=False,
                 infer_metric=None,
                 to_rgb=True,
                 outputs_mapping=None):
        name = f'{self.__class__.__name__}'
        work_dir = os.path.join(work_dir, f'rank_{get_rank()}')
        mkdir_or_exist(work_dir)
        self.val_data = val_data
        self.onnx_file = osp.join(work_dir, f'{name}.onnx')
        self.fixed_sann_file = self.onnx_file.replace('.onnx', '_fixed.sann')
        self.float_sann_file = self.onnx_file.replace('.onnx', '_float.sann')
        self.hir_file = osp.join(work_dir, f'{name}.hir')
        self.profiler_net_res = osp.join(work_dir, f'{name}_net_profiler.txt')
        self.profiler_layer_res = osp.join(work_dir, f'{name}_layer_profiler.csv')
        # sann config path load
        self.mnn_quant_json = mnn_quant_json
        # heron tool load
        self.reset_data()
        self.shape = (1, ) + next(self.it_val_data)['inputs'].shape
        self.is_quantized = is_quantized
        if self.is_quantized:
            quant_json = load(self.mnn_quant_json)
            quant_json['qatfile'] = self.onnx_file.replace('.onnx', '_superacme_clip_ranges.json')
            dump(quant_json, self.mnn_quant_json, indent=4)
        self.infer_metric = infer_metric
        if infer_metric is not None:
            self.infer_metric = METRICS.build(infer_metric)
        self.num_infer = num_infer if num_infer is not None and 0 < num_infer < len(self.val_data) else len(self.val_data)
        self.outputs_mapping = outputs_mapping
        self.model = None
        self.to_rgb = to_rgb

    def reset_data(self):
        """Reset dataset iterator."""
        self.it_val_data = iter(self.val_data)

    def import_torch(self, model):
        self.model = model
        device = next(model.parameters()).device
        dummy_data = next(self.it_val_data)['inputs'].float().unsqueeze(0).to(device)
        if self.is_quantized:
            observed_model = model.get_deploy_model()
            model.quantizer.export_onnx(observed_model, dummy_data, self.onnx_file)
        else:
            torch.onnx.export(
                model,
                dummy_data,
                self.onnx_file,
                keep_initializers_as_inputs=False,
                verbose=False,
                opset_version=11)

    def hir_convert(self):
        # convert and compiler
        command_line = 'sann build --input '+self.onnx_file+' --output '+self.hir_file+' --config '+self.mnn_quant_json+' > '+self.profiler_net_res
        os.system(command_line)

    def hir_profiler(self):
        command_line = 'sann profiler -m '+self.hir_file+' -o '+self.profiler_layer_res+' > /dev/null'
        os.system(command_line)

    def res_extract(self):
        ddr_io, sram_io, params, ddr_occp, sram_occp = None, None, None, None, None
        for line in open(self.profiler_net_res).readlines():
            if 'total DDR import and export data size' in line:
                ddr_io = float(line.split(':')[-1].split('MiB')[0])
            if 'total SRAM import and export data size' in line:
                sram_io = float(line.split(':')[-1].split('MiB')[0])
            if 'total static data size' in line:
                params = float(line.split(':')[-1].split('MiB')[0])
            if 'total DDR memory occupancy' in line:
                ddr_occp = float(line.split(':')[-1].split('MiB')[0])
            if 'total SRAM memory occupancy' in line:
                sram_occp = float(line.split(':')[-1].split('MiB')[0])
        assert (ddr_io is not None and sram_io is not None and params is not None) and (
            ddr_occp is not None and sram_occp is not None)
        fps = None
        for line in open(self.profiler_layer_res).readlines():
            if 'Fps' in line:
                fps = float(line.split(',')[-1].split(':')[-1])
        assert fps is not None
        results = {
            'ddr_io': ddr_io,
            'sram_io': sram_io,
            'params': params,
            'ddr_occp': ddr_occp,
            'sram_occp': sram_occp,
            'fps': fps
        }
        return results

    def reset_model(self):        
        self.data = None
        self.label = None
        torch.cuda.empty_cache()

    def torch_fixed_inference(self):
        for i, data in enumerate(self.it_val_data):
            if i >= self.num_infer:
                break
            inputs, data_samples = data['inputs'], data['data_samples']
            # if not self.to_rgb:
            #     inputs = inputs.flip(0)
            data = {'inputs': inputs.reshape(self.shape), 'data_samples': [data_samples]}
            # 1. mode = 'predict'
            outputs = self.model.val_step(data)
            self.infer_metric.process(inputs, [outputs[0].to_dict()])
            # # # 2. mode = 'tensor'
            # data = self.model.data_preprocessor(data, False)
            # cls_score = self.model._run_forward(data, mode='tensor')  # type: ignore
            # scores = F.softmax(cls_score, dim=1)
            # labels = scores.argmax(dim=1, keepdim=True).detach()
            # data_samples.set_pred_score(scores.squeeze()).set_pred_label(labels.squeeze())
            # self.infer_metric.process(inputs, [data_samples.to_dict()])
        metrics = self.infer_metric.evaluate(self.num_infer) if self.num_infer > 0 else {}
        return metrics

    def fixed_inference(self):
        """Fixed point inference."""
        interpreter = MNN.Interpreter(self.fixed_sann_file)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        interpreter.resizeTensor(input_tensor, self.shape)
        interpreter.resizeSession(session)
        # import pdb; pdb.set_trace()
        for i, data in enumerate(self.it_val_data):
            if i >= self.num_infer:
                break
            inputs, data_samples = data['inputs'], data['data_samples']
            if self.to_rgb:
                inputs = inputs.flip(0)
            input_data = MNN.Tensor(self.shape, MNN.Halide_Type_Uint8,
                inputs.reshape(self.shape).numpy().astype(np.uint8), MNN.Tensor_DimensionType_Caffe)
            # input_data = MNN.Tensor(self.shape, MNN.Halide_Type_Float,
            #     inputs[0].numpy().astype(np.float32), MNN.Tensor_DimensionType_Caffe)
            input_tensor.copyFrom(input_data)
            interpreter.runSession(session)
            outputs = interpreter.getSessionOutputAll(session)
            # # Note that here different tasks may need different post-process.
            # # Check and modify from its corresponding Head._get_predictions.
            assert len(self.outputs_mapping) == 1, f'Unsuported outputs_mapping len.'
            assert len(self.outputs_mapping) == len(outputs), f'Unmatched outputs_mapping and outputs.'
            for k, v in outputs.items():
                v_np = v.getNumpyData()
                output_data = MNN.Tensor(v_np.shape, MNN.Halide_Type_Float,
                    v_np.astype(np.float32), MNN.Tensor_DimensionType_Caffe)
                v.copyToHostTensor(output_data)
                cls_score = torch.from_numpy(output_data.getNumpyData())
                scores = F.softmax(cls_score, dim=1)
                labels = scores.argmax(dim=1, keepdim=True).detach()
                data_samples.set_pred_score(scores.squeeze()).set_pred_label(labels.squeeze())
            self.infer_metric.process(inputs, [data_samples.to_dict()])
            ########################DEBUG EXAMPLE#############################
            # from mmrazor.models import LearnableFakeQuantize
            # before_results = {}
            # after_results = {}
            # def before_after_hook(name):
            #     def wrapper(module, input, output):
            #         before_results[name] = copy.deepcopy(input[0].cpu().numpy())
            #         after_results[name] = copy.deepcopy(output.cpu().numpy())
            #     return wrapper
            # for name, module in self.model.named_modules():
            #     if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU6, LearnableFakeQuantize)):
            #         module.register_forward_hook(before_after_hook(name))
            # # 1. mode = 'predict'
            # # outputs = self.model.val_step(data)
            # # self.infer_metric.process(inputs, [outputs[0].to_dict()])
            # # 2. mode = 'tensor'
            # data = {'inputs': inputs.reshape(self.shape), 'data_samples': [data_samples]}
            # data = self.model.data_preprocessor(data, False)
            # outputs = self.model._run_forward(data, mode='tensor')  # type: ignore

            # begin_results = {}
            # def begin_callback(tensors, op):
            #     # import pdb; pdb.set_trace()
            #     assert tensors[0].getDataType() in [MNN.Halide_Type_Uint8, MNN.Halide_Type_Int8, MNN.Halide_Type_Float]
            #     if tensors[0].getDataType() == MNN.Halide_Type_Uint8:
            #         dtype = 'uint8'
            #     elif tensors[0].getDataType() == MNN.Halide_Type_Int8:
            #         dtype = 'int8'
            #     elif tensors[0].getDataType() == MNN.Halide_Type_Float:
            #         dtype = 'float32'
            #     input_data2 = MNN.Tensor(tensors[0].getShape(), tensors[0].getDataType(),
            #         np.ones(tensors[0].getShape(), dtype=dtype), MNN.Tensor_DimensionType_Caffe)                    
            #     tensors[0].copyToHostTensor(input_data2)
            #     begin_results[op.getName()] = copy.deepcopy(input_data2.getNumpyData())
            #     return True
            # end_results = {}
            # def end_callback(tensors, op):
            #     # import pdb; pdb.set_trace()
            #     assert tensors[0].getDataType() in [MNN.Halide_Type_Uint8, MNN.Halide_Type_Int8, MNN.Halide_Type_Float]
            #     if tensors[0].getDataType() == MNN.Halide_Type_Uint8:
            #         dtype = 'uint8'
            #     elif tensors[0].getDataType() == MNN.Halide_Type_Int8:
            #         dtype = 'int8'
            #     elif tensors[0].getDataType() == MNN.Halide_Type_Float:
            #         dtype = 'float32'
            #     input_data2 = MNN.Tensor(tensors[0].getShape(), tensors[0].getDataType(),
            #         np.ones(tensors[0].getShape(), dtype=dtype), MNN.Tensor_DimensionType_Caffe)
            #     tensors[0].copyToHostTensor(input_data2)
            #     end_results[op.getName()] = copy.deepcopy(input_data2.getNumpyData())
            #     return True
            # # interpreter.runSessionWithCallBack(session, begin_callback, end_callback)
            # interpreter.runSessionWithCallBackInfo(session, begin_callback, end_callback)
            ########################DEBUG EXAMPLE#############################

        metrics = self.infer_metric.evaluate(self.num_infer) if self.num_infer > 0 else {}
        return metrics
