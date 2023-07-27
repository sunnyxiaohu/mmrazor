# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer, disable_fake_quant)
    from torch.nn.intrinsic.qat import freeze_bn_stats
except ImportError:
    from mmrazor.utils import get_placeholder

    disable_observer = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    freeze_bn_stats = get_placeholder('torch>=1.13')

from mmengine.dist import all_reduce_params, is_distributed
from torch.utils.data import DataLoader

from mmrazor.models import register_torch_fake_quants, register_torch_observers
from mmrazor.models.fake_quants import (enable_param_learning,
                                        enable_static_estimate, enable_val)
from mmrazor.models.utils import add_prefix
from mmrazor.structures import export_fix_subnet
from mmrazor.registry import LOOPS, TASK_UTILS                                      
from mmrazor.engine.runner import QATEpochBasedLoop
from mmrazor.engine.runner.utils import CalibrateBNMixin

TORCH_observers = register_torch_observers()
TORCH_fake_quants = register_torch_fake_quants()


@LOOPS.register_module()
class QNASEpochBasedLoop(QATEpochBasedLoop):
    """`EpochBasedLoop` for `LEARNED STEP SIZE QUANTIZATION`

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats. Defaults to -1, which means no need to freeze bn.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            qat_begin: int = 1,
            freeze_bn_begin: int = -1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            freeze_bn_begin=freeze_bn_begin,
            dynamic_intervals=dynamic_intervals)

        self._is_first_batch = True
        self.distributed = is_distributed()
        self.qat_begin = qat_begin

    def prepare_for_run_epoch(self):
        """Toggle the state of the observers and fake quantizers before qat
        training."""
        model = self.runner.model
        model = getattr(model, 'module', model)        
        if (self.freeze_bn_begin > 0
                and self._epoch + 1 >= self.freeze_bn_begin):
            self.runner.model.apply(freeze_bn_stats)
        if (self.qat_begin > 0
                and self._epoch + 1 >= self.qat_begin):
            self.runner.model.apply(enable_param_learning)
            model.current_stage = 'qat'
        else:
            self.runner.model.apply(enable_val)
            self.runner.model.apply(disable_fake_quant)
            model.current_stage = 'float'     
        print_log(f'Switch current_stage: "{model.current_stage}"', logger='current')

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_val)

    @property
    def is_first_batch(self):
        return (self.qat_begin > 0
                    and self._epoch + 1 == self.qat_begin
                        and self._is_first_batch)

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        # import pdb; pdb.set_trace()
        for idx, data_batch in enumerate(self.dataloader):
            if self.is_first_batch:
                # lsq observer init
                self.runner.model.apply(enable_static_estimate)

            self.run_iter(idx, data_batch)

            if self.is_first_batch:
                # In the first batch, scale in LearnableFakeQuantize is
                # calculated through lsq observer. As the values of `scale` of
                # different observers in different rank are usually different,
                # we have to sync the `scale` here.
                if self.distributed:
                    all_reduce_params(
                        self.runner.model.parameters(), op='mean')

                # Change back to param learning mode
                self._is_first_batch = False
                self.runner.model.apply(enable_param_learning)

        self.runner.model.sync_qparams(src_mode='loss')
        # Make sure the registered buffer such as `observer_enabled` is
        # correct in the saved checkpoint.
        self.prepare_for_val()
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class QNASValLoop(ValLoop, CalibrateBNMixin):
    """`ValLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 evaluate_fixed_subnet: bool = False,
                 calibrate_sample_num: int = 4096,
                 estimator_cfg: Optional[Dict] = dict(type='mmrazor.ResourceEstimator'),                 
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        if self.runner.distributed:
            assert hasattr(self.runner.model.module, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.module.data_preprocessor
            model = self.runner.model.module
        else:
            assert hasattr(self.runner.model, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.data_preprocessor
            model = self.runner.model
        self.model = model
        self.architecture = model.architecture.architecture
        self.architecture.data_preprocessor = data_preprocessor
        self.evaluate_fixed_subnet = evaluate_fixed_subnet
        self.calibrate_sample_num = calibrate_sample_num
        self.estimator = TASK_UTILS.build(estimator_cfg)        

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        # import pdb; pdb.set_trace()
        all_metrics = dict()
        if self.model.current_stage == 'float':
            evaluate_once_func = self._evaluate_once
        else:
            evaluate_once_func = self._qat_evaluate_once

        if self.evaluate_fixed_subnet:
            metrics = evaluate_once_func
            all_metrics.update(add_prefix(metrics, 'fix_subnet'))
        elif hasattr(self.model, 'sample_kinds'):
            sample_kinds = ['max', 'min'] if self.model.current_stage == 'float' else ['max', 'qmax', 'min']
            for kind in sample_kinds:  # self.model.sample_kinds:
                if kind == 'max':
                    self.model.mutator.set_max_choices()
                    metrics = evaluate_once_func(kind=kind)
                    all_metrics.update(add_prefix(metrics, 'max_subnet'))
                elif kind == 'min':
                    self.model.mutator.set_min_choices()
                    metrics = evaluate_once_func(kind=kind)
                    all_metrics.update(add_prefix(metrics, 'min_subnet'))
                elif 'random' in kind:
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices())
                    metrics = evaluate_once_func(kind=kind)
                    all_metrics.update(add_prefix(metrics, f'{kind}_subnet'))
                elif 'qmax' in kind:
                    def qmax(mutables):
                        choice = mutables[0].max_choice
                        if mutables[0].alias and 'quant_bits' in mutables[0].alias and choice == 32 and len(mutables[0].choices) >= 2:
                            choice = mutables[0].choices[-2]
                        return choice
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices(kind=qmax))
                    metrics = evaluate_once_func(kind=kind)
                    all_metrics.update(add_prefix(metrics, 'qmax_subnet'))

        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')

    def _evaluate_once(self, kind='') -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        if self.calibrate_sample_num > 0:
            self.calibrate_bn_statistics(self.runner.train_dataloader,
                                         model = self.architecture,
                                         calibrate_sample_num = self.calibrate_sample_num)
        self.architecture.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.architecture)

        _, sliced_model = export_fix_subnet(
            self.architecture, slice_weight=True)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        resource_metrics = self.estimator.estimate(sliced_model)
        # for mode in self.params_modes:
        #     filename = f'{self.runner.log_dir}/subnet-{kind}_{mode}_params_boxplot_ep{self.runner.epoch}.png'
        #     if mode == 'fuse_conv_bn':
        #         from mqbench.cle_superacme.batch_norm_fold import fold_all_batch_norms
        #         folded_pairs = fold_all_batch_norms(sliced_model, self.input_shapes)
        #     elif mode == 'cle':
        #         from mqbench.cle_superacme.cle import apply_cross_layer_equalization
        #         apply_cross_layer_equalization(model=sliced_model, input_shape=self.input_shapes)
        #     draw_params_boxplot(sliced_model, filename, topk_params=self.topk_params)

        metrics.update(resource_metrics)
        return metrics

    def _qat_evaluate_once(self, kind='') -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        qat_metrics = dict()

        if self.calibrate_sample_num > 0:
            self.calibrate_bn_statistics(self.runner.train_dataloader,
                                         model = self.model,
                                         calibrate_sample_num = self.calibrate_sample_num)
        self.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.model)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        for key, value in metrics.items():
            qat_key = 'qat.' + key
            # ori_key = 'original.' + key
            qat_metrics[qat_key] = value
            # self.runner.message_hub.log_scalars.pop(f'val/{ori_key}', None)

        if self.calibrate_sample_num > 0:
            self.calibrate_bn_statistics(self.runner.train_dataloader,
                                         model = self.architecture,
                                         calibrate_sample_num = self.calibrate_sample_num)
        self.architecture.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.architecture)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        for key, value in metrics.items():
            # qat_key = 'qat.' + key
            ori_key = 'original.' + key
            qat_metrics[ori_key] = value
            # self.runner.message_hub.log_scalars.pop(f'val/{qat_key}', None)

        return qat_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], model):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement

        outputs = model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
