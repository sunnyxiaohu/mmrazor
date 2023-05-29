from typing import Dict, List, Optional, Union,Tuple,Sequence
import logging
from mmengine.evaluator import Evaluator
from mmengine.runner import TestLoop
from torch.utils.data import DataLoader
from mmengine.logging import print_log
from mmrazor.registry import LOOPS
import torch
from mmengine.analysis import get_model_complexity_info

@LOOPS.register_module()
class Subnet2onnxLoop(TestLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 task_type : str = 'det',
                 input_shape : Tuple = (1, 3, 384, 640),
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        
        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        self.input_shape = input_shape
        self.task_type = task_type

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        # convert to onnx
        dummy_input = torch.rand(self.input_shape).cuda() 
        if self.task_type=='det':
            torch.onnx.export(
                self.runner.model, 
                dummy_input, 
                'subnet.onnx',
                input_names=['input'],
                export_params=True,
                keep_initializers_as_inputs=True,
                do_constant_folding=True,
                verbose=False,
                opset_version=11) 
        elif self.task_type == 'cls':
            torch.onnx.export(
                self.runner.model, 
                dummy_input, 
                'subnet.onnx', 
                keep_initializers_as_inputs=False, 
                verbose=False, 
                opset_version=11)
        print ('convert to onnx successfully')
        self.runner.model =self.runner.model.cpu()
        analysis_results = get_model_complexity_info(
            self.runner.model,
            (self.input_shape[1],self.input_shape[2],self.input_shape[3])
        )
        print ('flops: ', analysis_results['flops_str'])
        print ('params: ', analysis_results['params_str'])
        return metrics
    