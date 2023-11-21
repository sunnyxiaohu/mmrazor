# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified from https://github.com/pytorch/pytorch
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math
import warnings
import weakref
from collections import Counter
from functools import wraps
from typing import Callable, List, Optional, Sequence, Union

import torch
from torch.optim import Optimizer

from mmengine import MessageHub
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper, _ParamScheduler

from mmrazor.registry import PARAM_SCHEDULERS

INF = int(1e9)

OptimizerType = Union[OptimWrapper, Optimizer]


@PARAM_SCHEDULERS.register_module()
class QRangeParamScheduler(_ParamScheduler):
    """Reduce the parameters of each parameter group when a metric has stopped
    improving. Models often benefit from reducing the parameters by a factor of
    2-10 once learning stagnates. This scheduler reads a metrics quantity and
    if no improvement is seen for a ``patience`` number of epochs, the
    parameters are reduced.

    The implementation is motivated by `PyTorch ReduceLROnPlateau`_.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        monitor (str): The name of the metric to measure whether
            the performance of the model is improved.
        rule (str): One of `less`, `greater`. In `less` rule, parameters will
            be reduced when the quantity monitored has stopped
            decreasing; in `greater` rule it will be reduced when the
            quantity monitored has stopped increasing. Defaults to 'less'.
            The ``rule`` is the renaming of ``mode`` in pytorch.
        factor (float): Factor by which the parameters will be
            reduced. new_param = param * factor. Defaults to 0.1.
        patience (int): Number of epochs with no improvement after
            which parameters will be reduced. For example, if
            ``patience = 2``, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the parameters after
            the 3rd epoch if the monitor value still hasn't improved then.
            Defaults to 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Defaults to 1e-4.
        threshold_rule (str): One of `rel`, `abs`. In `rel` rule,
            dynamic_threshold = best * ( 1 + threshold ) in 'greater'
            rule or best * ( 1 - threshold ) in `less` rule.
            In `abs` rule, dynamic_threshold = best + threshold in
            `greater` rule or best - threshold in `less` rule.
            Defaults to 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after parameters have been reduced. Defaults to 0.
        min_value (float or list[float]): A scalar or a sequence of scalars.
            A lower bound on the parameters of each parameter group
            respectively. Defaults to 0. .
        eps (float): Minimal decay applied to parameters. If the difference
            between new and old parameters are smaller than eps, the update is
            ignored. Defaults to 1e-8.
        begin (int): Step at which to start triggering the scheduler
            to monitor in val within the interval calculated
            according to epoch of training. Defaults to 0.
        end (int): Step at which to stop triggering the scheduler
            to monitor in val within the interval calculated
            according to epoch of training. Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.

    .. _PyTorch ReduceLROnPlateau:
       https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
    """

    need_val_args = True

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 monitor: str = 'loss',
                 min_value: Union[float, Sequence[float]] = 1e-5,
                 max_value: Union[float, Sequence[float]] = 1e-4,
                 eps: float = 1e-8,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, OptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end
        # import pdb; pdb.set_trace()
        assert by_epoch, \
            f'Now {type(self).__name__} only support by_epoch=True'
        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))

        self.last_step = last_step

        self._global_step = 0
        self.verbose = verbose

        if isinstance(min_value, (list, tuple)):
            if len(min_value) != len(optimizer.param_groups):
                raise ValueError('expected {} min_lrs, got {}'.format(
                    len(optimizer.param_groups), len(min_value)))
            self.min_values = list(min_value)
        else:
            self.min_values = [min_value] * len(  # type: ignore
                optimizer.param_groups)

        if isinstance(max_value, (list, tuple)):
            if len(max_value) != len(optimizer.param_groups):
                raise ValueError('expected {} min_lrs, got {}'.format(
                    len(optimizer.param_groups), len(max_value)))
            self.max_values = list(max_value)
        else:
            self.max_values = [max_value] * len(  # type: ignore
                optimizer.param_groups)

        self.eps = eps

        self.monitor = monitor

        # remove call self.step() and init self._global_step = 0
        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

    def step(self, metrics=None):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule.

        Args:
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
                Defaults to None.
        """
        if metrics is None:
            # only to count self._global_step
            self._global_step += 1
            return

        if not isinstance(metrics, dict):
            raise TypeError('metrics type should be dict,'
                            f' but got type {type(metrics)}')

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1

            # convert `metric` to float, in case it's a zero-dim Tensor
            model = metrics.pop(self.monitor, None)
            mmonitor = f'val/{self.monitor}'
            message_hub = MessageHub.get_current_instance()
            if mmonitor in message_hub.runtime_info:
                message_hub.runtime_info.pop(mmonitor, None)

            if model is not None:
                candidate_params = [(name, param) for name, param in model.named_parameters()]
                # only apply on conv
                # TODO(shiguang): 
                # 1. apply on other parameters?
                # 2. consider symetric or unsymetric?
                # 3. use quantile instead of minmax to make it more stable.
                # 4. group parameters by introducing cle.
                datas = []
                datas_var = []
                import pdb; pdb.set_trace()
                for name, param in candidate_params:
                    if param.dim() > 2:
                        lower_val, upper_val = torch.aminmax(param.flatten())
                        qrange = max(upper_val - lower_val, self.eps)
                        datas.append(qrange)
                        datas_var.append(param.var())
                datas = torch.stack(datas, dim=0)
                datas_var = torch.stack(datas_var, dim=0)
                datas = list((datas.mean() / datas).detach().cpu().numpy())

                for i, (name, param) in enumerate(candidate_params):
                    if param.dim() > 2:
                        continue
                    datas.insert(i, 1.0)

                values = self._get_value(datas)

                for i, data in enumerate(
                        zip(self.optimizer.param_groups, values)):
                    param_group, value = data
                    if abs(param_group[self.param_name] - value) > self.eps:
                        param_group[self.param_name] = value
                        self.print_value(self.verbose, candidate_params[i][0], value)
            else:
                raise KeyError(f'Excepted key in {list(metrics.keys())},'
                               f' but got key {self.monitor} is not in dict')

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

    def print_value(self, is_verbose: bool, group: Union[int, str], value: float) -> None:
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            step_name = 'epoch' if self.by_epoch else 'iter'
            print_log(
                f'Adjusting parameter value of group {group} to {value:.4e} '
                f'in {step_name} {self.last_step}.',
                logger='current')

    def _get_value(self, factors):
        """Compute value using chainable form of the scheduler."""
        values = []
        for group, factor, min_v, max_v in zip(self.optimizer.param_groups, factors, 
                                               self.min_values, self.max_values):
            if factor == 1.0:
                value = float(group[self.param_name])
            else:
                value = max(min(float(group[self.param_name]) * factor, max_v), min_v)
            values.append(value)
        return values

