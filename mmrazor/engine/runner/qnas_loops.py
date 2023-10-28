# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import logging
import os
import os.path as osp
import math
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict, OrderedDict
from copy import deepcopy
from pulp import LpProblem, lpSum, LpVariable, LpInteger, LpMinimize, value, LpMaximize

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from mmengine import fileio
from mmengine.dist import broadcast_object_list, all_reduce_params, is_distributed
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop, autocast
from mmengine.utils import is_list_of, import_modules_from_strings

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer, disable_fake_quant,
                                       FakeQuantizeBase)
except ImportError:
    from mmrazor.utils import get_placeholder

    disable_observer = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    FakeQuantizeBase = get_placeholder('torch>=1.13')

from mmrazor.models import (BaseAlgorithm, register_torch_fake_quants,
                            register_torch_observers, LearnableFakeQuantize,
                            BatchLSQObserver)
from mmrazor.models.architectures.dynamic_ops import (DynamicConvMixin,
                                                      DynamicLinearMixin)
from mmrazor.models.architectures.dynamic_qops import (freeze_bn_stats,
    fix_calib_stats, unfix_calib_stats, DynamicLearnableFakeQuantize)                                                      
from mmrazor.models.fake_quants import (enable_param_learning,
                                        enable_static_estimate, enable_val)
from mmrazor.models.task_modules.tracer.fx.graph_utils import _get_attrs
from mmrazor.models.utils import add_prefix, get_module_device
from mmrazor.structures import Candidates, export_fix_subnet, convert_fix_subnet
from mmrazor.structures.subnet.fix_subnet import _load_fix_subnet_by_mutable

from mmrazor.registry import LOOPS, TASK_UTILS
from mmrazor.utils import SupportRandomSubnet
from mmrazor.engine.runner import QATEpochBasedLoop, EvolutionSearchLoop
from mmrazor.engine.runner.utils import check_subnet_resources, crossover
from mmrazor.engine.runner.utils.calibrate_bn_mixin import AverageMeter

TORCH_observers = register_torch_observers()
TORCH_fake_quants = register_torch_fake_quants()


@contextlib.contextmanager
def adabn_context(model):
    running_means = []
    running_vars = []
    num_batches_trackeds = []
    for mod in model.modules():
        if isinstance(mod, _BatchNorm) and mod.track_running_stats:
            running_means.append(mod.running_mean.data.clone())
            running_vars.append(mod.running_var.data.clone())
            num_batches_trackeds.append(mod.num_batches_tracked.clone())
            # mod.reset_running_stats()
    yield
    i = 0
    for mod in model.modules():
        if isinstance(mod, _BatchNorm) and mod.track_running_stats:
            mod.running_mean.data.copy_(running_means[i])
            mod.running_var.data.copy_(running_vars[i])
            mod.num_batches_tracked.data.copy_(num_batches_trackeds[i])
            i += 1

class CalibrateMixin:
    fp16: bool = False

    @torch.no_grad()
    def calibrate_bn_observer_statistics(self,
                                dataloader: DataLoader,
                                calibrate_sample_num: int = 2000,
                                model = None) -> None:
        if model is None:
            model = self.runner.model
        model.apply(unfix_calib_stats)

        def record_bn_statistics_hook(bn_module: _BatchNorm, input: Tensor,
                                      output: Tensor) -> None:
            mean_average_meter: AverageMeter = bn_module.__mean_average_meter__
            var_average_meter: AverageMeter = bn_module.__var_average_meter__

            real_input = input[0]
            mean = real_input.mean((0, 2, 3))
            var = real_input.var((0, 2, 3), unbiased=True)

            mean_average_meter.update(mean, real_input.size(0))
            var_average_meter.update(var, real_input.size(0))

        def record_observer_statistics_hook(observer_module, input: Tensor,
                                            output: Tensor) -> None:
            max_average_meter: AverageMeter = observer_module.__max_average_meter__
            min_average_meter: AverageMeter = observer_module.__min_average_meter__

            real_input = input[0]
            max_val = observer_module.max_val
            min_val = observer_module.min_val
            max_average_meter.update(max_val, real_input.size(0))
            min_average_meter.update(min_val, real_input.size(0))

        hook_handles = []

        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                print_log('register `record_bn_statistics_hook` to module: '
                          f'{name}', logger='current', level=logging.DEBUG)
                module.__mean_average_meter__ = AverageMeter()
                module.__var_average_meter__ = AverageMeter()
                handle = module.register_forward_hook(
                    record_bn_statistics_hook)
                hook_handles.append(handle)
            if isinstance(module, BatchLSQObserver):
                print_log('register `record_observer_statistics_hook` to module: '
                          f'{name}', logger='current', level=logging.DEBUG)
                module.__max_average_meter__ = AverageMeter()
                module.__min_average_meter__ = AverageMeter()
                module.reset_min_max_vals()
                handle = module.register_forward_hook(
                    record_observer_statistics_hook)
                hook_handles.append(handle)

        model.train()
        print_log('Start calibrating bn and observer statistics', logger='current')
        print_log(f'Total sample number for calibration: {calibrate_sample_num}',
                  logger='current')
        remaining = calibrate_sample_num
        for data_batch in dataloader:
            # TODO: Handle remaining can not be divided evenly by total data_batch
            # if len(data_batch) >= remaining:
            #     data_batch = data_batch[:remaining]
            if isinstance(data_batch, torch.Tensor):
                data_batch_nums = len(data_batch)
            else:
                data_batch_nums = len(data_batch['inputs'])
            if dist.is_initialized() and dist.is_available():
                data_batch_tensor = torch.tensor(
                    [data_batch_nums], device=get_module_device(model))
                dist.all_reduce(
                    data_batch_tensor, dist.ReduceOp.SUM, async_op=False)
                data_batch_nums = data_batch_tensor.item()
            remaining -= data_batch_nums

            print_log(f'Remaining samples for calibration: {remaining}',
                      logger='current', level=logging.DEBUG)
            with autocast(enabled=self.fp16):
                model.test_step(data_batch)

            if remaining <= 0:
                break

        for name, module in model.named_modules():
            if isinstance(module, _BatchNorm):
                mean_average_meter = module.__mean_average_meter__
                var_average_meter = module.__var_average_meter__
                if mean_average_meter.count == 0 or \
                        var_average_meter.count == 0:
                    assert mean_average_meter.count == 0 and \
                        var_average_meter.count == 0
                    print_log(f'layer {name} is not chosen, ignored',
                              logger='current', level=logging.DEBUG)
                    continue

                calibrated_bn_mean = mean_average_meter.avg
                calibrated_bn_var = var_average_meter.avg

                feature_dim = calibrated_bn_mean.size(0)

                print_log(
                    f'layer: {name}, '
                    f'current feature dimension: {feature_dim}, '
                    'number of samples for calibration: '
                    f'{mean_average_meter.count}, '
                    'l2 norm of calibrated running mean: '
                    f'{calibrated_bn_mean.norm()}, '
                    'l2 norm of calibrated running var: '
                    f'{calibrated_bn_var.norm()}, '
                    'l2 norm of original running mean: '
                    f'{module.running_mean[:feature_dim].norm()}, '
                    'l2 norm of original running var: '
                    f'{module.running_var[:feature_dim].norm()}, ',
                    logger='current', level=logging.DEBUG)

                module.running_mean[:feature_dim].copy_(calibrated_bn_mean)
                module.running_var[:feature_dim].copy_(calibrated_bn_var)

                del module.__mean_average_meter__
                del module.__var_average_meter__

            if isinstance(module, BatchLSQObserver):
                max_average_meter = module.__max_average_meter__
                min_average_meter = module.__min_average_meter__
                if max_average_meter.count == 0 or \
                        min_average_meter.count == 0:
                    assert max_average_meter.count == 0 and \
                        min_average_meter.count == 0
                    print_log(f'layer {name} is not chosen, ignored',
                              logger='current', level=logging.DEBUG)
                    continue

                calibrated_max_val = max_average_meter.avg
                calibrated_min_val = min_average_meter.avg

                module.max_val.copy_(calibrated_max_val)
                module.min_val.copy_(calibrated_min_val)

                del module.__max_average_meter__
                del module.__min_average_meter__

        print_log('Remove all hooks for calibration',
                  logger='current', level=logging.DEBUG)
        print_log('Calibrate bn and observer statistics done', logger='current')
        for handle in hook_handles:
            handle.remove()
        model.apply(fix_calib_stats)


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
            is_first_batch: bool = True,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            freeze_bn_begin=freeze_bn_begin,
            dynamic_intervals=dynamic_intervals)

        self._is_first_batch = is_first_batch
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
        self.runner.model.apply(unfix_calib_stats)
        print_log(f'Switch current_stage: "{model.current_stage}"', logger='current')

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_val)
        self.runner.model.apply(fix_calib_stats)

    @property
    def is_first_batch(self):
        return (self.qat_begin > 0
                    and self._epoch + 1 == self.qat_begin
                        and self._is_first_batch)

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        # if self._epoch < 5:
        #     self.runner.model.module.sample_kinds = ['min']
        # else:
        #     self.runner.model.module.sample_kinds = ['max', 'min', 'random0', 'random1']
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
class QNASValLoop(ValLoop, CalibrateMixin):
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
                 quant_bits = None,
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
        self.quant_bits = quant_bits

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')

        all_metrics = dict()
        if self.model.current_stage == 'float':
            evaluate_once_func = self._evaluate_once
        else:
            evaluate_once_func = self._qat_evaluate_once

        if self.evaluate_fixed_subnet:
            metrics = evaluate_once_func()
            all_metrics.update(add_prefix(metrics, 'fix_subnet'))
        elif hasattr(self.model, 'sample_kinds'):
            if self.model.current_stage == 'float':
                sample_kinds = ['max', 'min']
            else:
                search_groups = self.model.mutator.search_groups
                quant_bits = list(filter(lambda x: x[1][0].alias and 'quant_bits' in x[1][0].alias, search_groups.items()))
                if len(quant_bits) == 0:
                    sample_kinds = ['max', 'min']
                else:
                    sample_kinds = []
                    if self.quant_bits is None:
                        for qb in quant_bits[1:]:
                            assert qb[1][0].choices == quant_bits[0][1][0].choices
                        choices = quant_bits[0][1][0].choices
                    else:
                        choices = self.quant_bits
                    for bit in choices:
                        sample_kinds.extend([f'max_q{bit}', f'min_q{bit}'])

            def qmaxmin(bit=32, is_max=True):
                def sample(mutables):
                    choice = mutables[0].max_choice if is_max else mutables[0].min_choice
                    if mutables[0].alias and 'quant_bits' in mutables[0].alias:
                        if bit in mutables[0].choices:
                            choice = bit
                        elif self.quant_bits is not None:
                            pass
                        else:
                            assert bit in mutables[0].choices
                    return choice
                return sample

            for kind in sample_kinds:  # self.model.sample_kinds:
                if kind == 'max':
                    self.model.mutator.set_max_choices()
                elif kind == 'min':
                    self.model.mutator.set_min_choices()
                elif 'random' in kind:
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices())
                elif kind.startswith('max_q') or kind.startswith('min_q'):
                    is_max = kind.split('_')[0] == 'max'
                    bit = int(kind.split('_')[1].replace('q', ''))
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices(kind=qmaxmin(bit, is_max)))
                metrics = evaluate_once_func(kind=kind)
                all_metrics.update(add_prefix(metrics, f'{kind}_subnet'))
        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')

    def _evaluate_once(self, kind='') -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        with adabn_context(self.architecture):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.runner.train_dataloader,
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
        with adabn_context(self.model):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.runner.train_dataloader,
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

        with adabn_context(self.architecture):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.runner.train_dataloader,
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


def nonqmin(mutables):
    """Sample choice for mutables except `quant_bits`"""
    if mutables[0].alias and 'quant_bits' in mutables[0].alias:
        choice = mutables[0].current_choice
    else:
        choice = mutables[0].min_choice
    return choice

@LOOPS.register_module()
class QNASEvolutionSearchLoop(EvolutionSearchLoop, CalibrateMixin):
    """Loop for evolution searching."""

    def __init__(self, *args, export_fix_subnet=None, solve_mode='ilp', **kwargs):
        super().__init__(*args, **kwargs)
        self.export_float_model = False
        self.export_fix_subnet = export_fix_subnet
        self.solve_mode = solve_mode
        assert solve_mode in ['evo', 'prob', 'ilp']

    def get_qrange_probs(self):
        # 1. get all the spec_modules.
        spec_modules = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, (DynamicConvMixin, DynamicLinearMixin)):
                spec_modules.append(name)
        # 2. forward and get it's corresponding flops / params.
        self.model.mutator.set_min_choices()
        results = self.estimator.estimate_separation_modules(
            model=self.model, flops_params_cfg=dict(spec_modules=spec_modules, seperate_return=True))
        # filter results
        filterd_results = {}
        for key, val in results.items():
            if val['flops'] != 0 or val['params'] != 0:
                filterd_results[key] = val
        # get every FakeQaunt's qrange
        qrange_results = defaultdict(int)
        prefix = 'architecture.qmodels.tensor'
        prepared_model = _get_attrs(self.model, prefix)
        kinds = ['max', 'min']
        kinds += ['random'] * 3
        for kind in kinds:
            self.model.mutator.set_choices(self.model.mutator.sample_choices(kind=kind))
            with adabn_context(self.model):
                if self.calibrate_sample_num > 0:
                    self.calibrate_bn_observer_statistics(self.calibrate_dataloader,
                                                          model=self.model,
                                                          calibrate_sample_num=self.calibrate_sample_num)
                for node in prepared_model.graph.nodes:
                    if node.op == 'call_module':
                        maybe_lsq = _get_attrs(prepared_model, node.target)
                        if hasattr(maybe_lsq, 'weight_fake_quant'):
                            # weights
                            maybe_lsq = maybe_lsq.weight_fake_quant
                        if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                            # activation
                            quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                            bit = quant_bits.current_choice
                            scale, zero_point = maybe_lsq.calculate_qparams()
                            qrange_results[quant_bits.alias] += scale.item() * (2**bit - 1)                         
        # import pdb; pdb.set_trace()
        # normalize qrange_results
        belta, eps = 2.0, 10e-6
        filter_fns = [
            lambda x: 'activation_post_process_' in x[0],
            lambda x: 'activation_post_process_' not in x[0]
            # lambda x: 'activation_post_process_' in x[0],
        ]
        for fn in filter_fns:
            f_qrange_results = dict(filter(fn, qrange_results.items()))
            values = np.array(list(f_qrange_results.values()))
            values = belta * ((values - values.min()) / (values.max() - values.min() + eps) - 0.5)
            f_qrange_results = dict(zip(f_qrange_results.keys(), values))
            qrange_results.update(f_qrange_results)

        # Suppose fakequant A and fakequant B,
        # if qrange(A) > qrange(B), then the higher bit-width of A should get more selection prob
        # if qrange(A) < qrange(B), then the lower bit-width of A should get less selection prob.

        def adjust_function(fixed_point, ratio, inputs, ftype='linear'):
            assert ftype in ['linear', 'exp', 'log']
            def softmax(x, T=1.0):
                return np.exp(x / T) / sum(np.exp(x / T))

            outputs = []
            for input in inputs:
                if ftype == 'linear':
                    output = ratio * (input - fixed_point[0]) + fixed_point[1]
                elif ftype == 'exp':
                    output = np.exp(ratio * (input - fixed_point[0])) + (1 - fixed_point[1])
                outputs.append(output)
            outputs = softmax(np.array(outputs))
            return dict(zip(inputs, outputs))

        prob_results = {}
        for node in prepared_model.graph.nodes:
            if node.op == 'call_module':
                maybe_lsq = _get_attrs(prepared_model, node.target)
                if hasattr(maybe_lsq, 'weight_fake_quant'):
                    # weights
                    maybe_lsq = maybe_lsq.weight_fake_quant
                if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                    # activation
                    quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                    fixed_point = (np.mean(quant_bits.choices), 0.5)
                    ratio = qrange_results[quant_bits.alias]
                    prob_results[quant_bits.alias] = adjust_function(fixed_point, ratio, quant_bits.choices, ftype='linear')

        return filterd_results, prob_results

    def sample_qrange_prob_once(self, prob_results):
        prefix = 'architecture.qmodels.tensor'
        prepared_model = _get_attrs(self.model, prefix)        
        for node in prepared_model.graph.nodes:
            if node.op == 'call_module':
                maybe_lsq = _get_attrs(prepared_model, node.target)
                if hasattr(maybe_lsq, 'weight_fake_quant'):
                    # weights
                    maybe_lsq = maybe_lsq.weight_fake_quant
                if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                    # activation
                    quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                    quant_bits.current_choice = np.random.choice(
                        quant_bits.choices, p=list(prob_results[quant_bits.alias].values()))
        candidate = self.model.mutator.current_choices
        return candidate

    def _map_actfqnode2outputs(self, prepared_model):
        actfqnode2outputs = {}
        for node in prepared_model.graph.nodes:
            for maybe_act_fq in node.args:
                if not hasattr(maybe_act_fq, 'op') or maybe_act_fq.op != 'call_module':
                    continue
                maybe_act_fq = _get_attrs(prepared_model, maybe_act_fq.target)
                if not isinstance(maybe_act_fq, DynamicLearnableFakeQuantize):
                    continue
                quant_bits = maybe_act_fq.mutable_attrs['quant_bits']
                if quant_bits.alias not in actfqnode2outputs:
                    actfqnode2outputs[quant_bits.alias] = [node]
                else:
                    actfqnode2outputs[quant_bits.alias].append(node)

        return actfqnode2outputs

    def sample_ilp_once(self, filterd_results, prob_results, w_alpha=1.0, act_alpha=1.0, default_bit=4) -> None:
        others_prob_results = deepcopy(prob_results)
        filterd_mods = dict((name, mod) for name, mod in self.model.named_modules() if name in filterd_results)
        prefix = 'architecture.qmodels.tensor'
        prepared_model = _get_attrs(self.model, prefix)
        # find activation fakequant that have multiple targets, which is in filterd_results.
        actfqnode2outputs = self._map_actfqnode2outputs(prepared_model)
        filterd_actfqnode2outputs = defaultdict(list)
        for fq, outputs in actfqnode2outputs.items():
            count = 0
            for output in outputs:
                if output.op == 'call_module' and _get_attrs(prepared_model, output.target) in filterd_mods.values():
                    filterd_actfqnode2outputs[fq].append(_get_attrs(prepared_model, output.target))
                    count += 1
            if count == 1:
                del filterd_actfqnode2outputs[fq]
        # build interger linear programming and solve it.
        problem = LpProblem('Bit-width allocation', LpMinimize)
        variables = {}
        target, bitops, bitparams = 0, 0, 0
        for name, mod in filterd_mods.items():
            w_quant_bits = mod.weight_fake_quant.mutable_attrs['quant_bits']
            act_quant_bits = mod._ACT_QUANT_BITS
            for wb in w_quant_bits.choices:
                w_quant_bits.current_choice = wb
                for ab in act_quant_bits.choices:
                    act_quant_bits.current_choice = ab
                    key = f'{w_quant_bits.alias}.{wb}-{act_quant_bits.alias}.{ab}'
                    variables[key] = LpVariable(key, 0, 1, LpInteger)
                    # target
                    target += (w_alpha * prob_results[w_quant_bits.alias][wb] + act_alpha * prob_results[act_quant_bits.alias][ab]) * variables[key]
                    bitops += filterd_results[name]['flops'] / w_quant_bits.choices[0] / act_quant_bits.choices[0] * wb * ab * variables[key]
                    bitparams += filterd_results[name]['params'] / w_quant_bits.choices[0] * wb * variables[key]
            others_prob_results.pop(w_quant_bits.alias, None)
            others_prob_results.pop(act_quant_bits.alias, None)
            # constraint 1: only select one bit for weight and one bit for activation.
            problem += sum(variables[f'{w_quant_bits.alias}.{wb}-{act_quant_bits.alias}.{ab}']
                           for wb in w_quant_bits.choices for ab in act_quant_bits.choices) == 1
        for name, mod in filterd_mods.items():
            act_quant_bits = mod._ACT_QUANT_BITS
            # constraint 2: make sure multiple targets share the same quant_bits.
            outputs = filterd_actfqnode2outputs.pop(act_quant_bits.alias, None)
            if outputs is not None:
                base_w_quant_bits = outputs[0].weight_fake_quant.mutable_attrs['quant_bits']
                for ab in act_quant_bits.choices:
                    base_w_value = sum(variables[f'{base_w_quant_bits.alias}.{wb}-{act_quant_bits.alias}.{ab}']
                                       for wb in base_w_quant_bits.choices)
                    for output in outputs[1:]:
                        selected_w_quant_bits = output.weight_fake_quant.mutable_attrs['quant_bits']
                        selected_w_value = sum(variables[f'{selected_w_quant_bits.alias}.{wb}-{act_quant_bits.alias}.{ab}']
                                               for wb in base_w_quant_bits.choices)
                        problem += base_w_value == selected_w_value

        problem += target
        # constraint 3
        if 'flops' in self.constraints_range:
            problem += bitops >= self.constraints_range['flops'][0]
            problem += bitops <= self.constraints_range['flops'][1]
        # constraint 4
        if 'params' in self.constraints_range:
            problem += bitparams >= self.constraints_range['params'][0]
            problem += bitparams <= self.constraints_range['params'][1]            
        # TODO: constraint 5: avg_quant_bits for weight and act, seperately.
        ret = problem.solve()
        if ret != 1:
            return None
        # get results for weight-only nn.Modules
        for name, mod in filterd_mods.items():
            w_quant_bits = mod.weight_fake_quant.mutable_attrs['quant_bits']
            act_quant_bits = mod._ACT_QUANT_BITS
            chosen = False
            for wb in w_quant_bits.choices:
                for ab in act_quant_bits.choices:
                    key = f'{w_quant_bits.alias}.{wb}-{act_quant_bits.alias}.{ab}'
                    if value(variables[key]) == 1.0:
                        chosen = True
                        break
                if chosen is True:
                    break
            assert chosen
            act_quant_bits.current_choice = default_bit if act_alpha == 0.0 and default_bit in act_quant_bits.choices else ab
            w_quant_bits.current_choice = default_bit if w_alpha ==0.0 and default_bit in w_quant_bits.choices else wb
        # get results for Others
        others_select_mode = 2
        for node in prepared_model.graph.nodes:
            if node.op == 'call_module':
                maybe_lsq = _get_attrs(prepared_model, node.target)
                if hasattr(maybe_lsq, 'weight_fake_quant'):
                    # weights
                    maybe_lsq = maybe_lsq.weight_fake_quant
                if isinstance(maybe_lsq, DynamicLearnableFakeQuantize):
                    # activation
                    quant_bits = maybe_lsq.mutable_attrs['quant_bits']
                    if quant_bits.alias not in others_prob_results:
                        continue
                    if others_select_mode == 1:
                        # select the highest bit;
                        quant_bits.current_choice = list(others_prob_results[quant_bits.alias].keys())[-1]
                    elif others_select_mode == 2:
                        # select the bit with the smallest prob;
                        keys = list(others_prob_results[quant_bits.alias].keys())
                        values = list(others_prob_results[quant_bits.alias].values())
                        quant_bits.current_choice = keys[values.index(min(values))]
                    elif others_select_mode == 3:
                        # select the highest bit;
                        quant_bits.current_choice = default_bit
                    else:
                        # random                        
                        quant_bits.current_choice = random.choice(quant_bits.choices)

        candidate = self.model.mutator.current_choices
        return candidate

    def sample_candidates(self) -> None:
        """Update candidate pool contains specified number of candicates."""
        candidates_resources = []
        init_candidates = len(self.candidates)
        idx = 0
        if self.solve_mode == 'ilp':
            num_candidates = self.num_candidates - init_candidates
            act_alphas = [3*(i+1.0) / num_candidates for i in range(num_candidates)]
            w_act_alphas = [(1.0, 0.0), (0.0, 1.0)]
            w_act_alphas += [(1.0, i) for i in act_alphas]
            filterd_results, prob_results = self.get_qrange_probs()
        elif self.solve_mode == 'prob':
            _, prob_results = self.get_qrange_probs()
        while len(self.candidates) < self.num_candidates:
            idx += 1
            if self.solve_mode == 'ilp' and idx <= len(w_act_alphas):
                candidate = self.sample_ilp_once(
                    filterd_results, prob_results, *w_act_alphas[idx-1])
                if candidate is None:
                    continue
            elif self.solve_mode == 'prob':
                candidate = self.sample_qrange_prob_once(prob_results)
            elif idx == 1:
                candidate = self.model.mutator.sample_choices('max')
            elif idx == 2:
                candidate = self.model.mutator.sample_choices('min')
            else:
                candidate = self.model.mutator.sample_choices()
            self.model.mutator.set_choices(candidate)
            is_pass, result = self._check_constraints(
                random_subnet=candidate)
            if is_pass:
                self.candidates.append(candidate)
                candidates_resources.append(result)
        self.candidates = Candidates(self.candidates.data)

        if len(candidates_resources) > 0:
            self.candidates.update_resources(
                candidates_resources,
                start=len(self.candidates.data) - len(candidates_resources))
            assert init_candidates + len(
                candidates_resources) == self.num_candidates

        # broadcast candidates to val with multi-GPUs.
        broadcast_object_list(self.candidates.data)

    @torch.no_grad()
    def _val_candidate(self, use_predictor: bool = False) -> Dict:
        """Run validation.

        Args:
            use_predictor (bool): Whether to use predictor to get metrics.
                Defaults to False.
        """
        if use_predictor:
            assert self.predictor is not None
            metrics = self.predictor.predict(self.model)
        else:
            with adabn_context(self.model):
                if self.calibrate_sample_num > 0:
                    self.calibrate_bn_observer_statistics(self.calibrate_dataloader,
                                                          model = self.model,
                                                          calibrate_sample_num = self.calibrate_sample_num)
                self.runner.model.eval()
                for data_batch in self.dataloader:
                    outputs = self.runner.model.val_step(data_batch)
                    self.evaluator.process(
                        data_samples=outputs, data_batch=data_batch)
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        return metrics

    def _check_constraints(
            self, random_subnet: SupportRandomSubnet) -> Tuple[bool, Dict]:
        """Check whether is beyond constraints.

        Returns:
            bool, result: The result of checking.
        """
        is_pass, results = check_subnet_resources(
            model=self.model,
            subnet=random_subnet,
            estimator=self.estimator,
            constraints_range=self.constraints_range,
            export=False)

        return is_pass, results

    def _save_best_fix_subnet(self):
        """Save best subnet in searched top-k candidates."""
        best_random_subnet = self.top_k_candidates.subnets[0]
        self.model.mutator.set_choices(best_random_subnet)
        with adabn_context(self.model):
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.calibrate_dataloader,
                                                      model=self.model,
                                                      calibrate_sample_num=self.calibrate_sample_num)
                self.model.sync_qparams(src_mode='predict')
            best_fix_subnet, sliced_model = \
                export_fix_subnet(self.model.architecture, slice_weight=True)
        float_sliced_model = None
        with adabn_context(self.model.architecture.architecture):
            if self.export_float_model and self.calibrate_sample_num > 0:
                self.calibrate_bn_observer_statistics(self.calibrate_dataloader,
                                                      model = self.model.architecture.architecture,
                                                      calibrate_sample_num = self.calibrate_sample_num)
                _, float_sliced_model = export_fix_subnet(self.model.architecture, slice_weight=True)
        if self.runner.rank == 0:
            timestamp_subnet = time.strftime('%Y%m%d_%H%M', time.localtime())
            model_name = f'subnet_{timestamp_subnet}.pth'
            save_path = osp.join(self.runner.work_dir, model_name)
            torch.save({
                'state_dict': sliced_model.state_dict(),
                'meta': {}
            }, save_path)
            self.runner.logger.info(f'Subnet checkpoint {model_name} saved in '
                                    f'{self.runner.work_dir}')
            if self.export_float_model:
                model_name = 'float_' + model_name
                save_path = osp.join(self.runner.work_dir, model_name)
                torch.save({
                    'state_dict': float_sliced_model.state_dict(),
                    'meta': {}
                }, save_path)
                self.runner.logger.info(f'Subnet checkpoint {model_name} saved in '
                                        f'{self.runner.work_dir}')
            save_name = 'best_fix_subnet.yaml'
            best_fix_subnet = convert_fix_subnet(best_fix_subnet)
            fileio.dump(best_fix_subnet,
                        osp.join(self.runner.work_dir, save_name))
            self.runner.logger.info(
                f'Subnet config {save_name} saved in {self.runner.work_dir}.')
            save_name = 'best_fix_subnet_by_module_name.yaml'
            best_fix_subnet = self.convert_fix_subnet_by_module_name(sliced_model.qmodels)
            fileio.dump(best_fix_subnet,
                        osp.join(self.runner.work_dir, save_name))
            self.runner.logger.info(
                f'Subnet config {save_name} saved in {self.runner.work_dir}.')

            self.runner.logger.info('Search finished.')

    def convert_fix_subnet_by_module_name(self, model):
        fix_subnet = {}
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantizeBase):
                quant_info = {
                    'bit': module.bitwidth,
                    'quant_min': module.quant_min,
                    'quant_max': module.quant_max,
                }
                fix_subnet[name] = quant_info
        return fix_subnet

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_train')

        self.prepare_for_val()
        if self.export_fix_subnet:
            fix_subnet = fileio.load(self.export_fix_subnet)
            _load_fix_subnet_by_mutable(self.model, fix_subnet)
            fix_subnet = self.model.mutator.current_choices
            _, result = self._check_constraints(random_subnet=fix_subnet)
            self.candidates = Candidates([fix_subnet])
            self.candidates.update_resources([result])
            self.update_candidates_scores()
            self.top_k_candidates = self.candidates
            self._save_best_fix_subnet()
            return

        if self.predictor_cfg is not None:
            self._init_predictor()

        if self.resume_from:
            self._resume()

        while self._epoch < self._max_epochs:
            self.run_epoch()
            self._save_searcher_ckpt()

        self._save_best_fix_subnet()

        self.runner.call_hook('after_train')

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_val)
        self.runner.model.apply(fix_calib_stats)
