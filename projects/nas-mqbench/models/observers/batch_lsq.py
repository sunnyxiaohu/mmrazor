# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.distributed as dist

from mmrazor.registry import MODELS

try:
    from torch.ao.quantization.observer import (MinMaxObserver,
                                                PerChannelMinMaxObserver)
except ImportError:
    from mmrazor.utils import get_placeholder
    MinMaxObserver = get_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_placeholder('torch>=1.13')


def update_estimator_mode_0(mod):
    """Enable validation, if applicable. Example usage::

    # model is any PyTorch model model.apply(update_estimator_mode_0)
    """
    if isinstance(mod, BatchLSQObserver):
        mod.estimator_mode[0] = 0

def update_estimator_mode_1(mod):
    """Enable validation, if applicable. Example usage::

    # model is any PyTorch model model.apply(update_estimator_mode_1)
    """
    if isinstance(mod, BatchLSQObserver):
        mod.estimator_mode[0] = 1

def update_estimator_mode_2(mod):
    """Enable validation, if applicable. Example usage::

    # model is any PyTorch model model.apply(update_estimator_mode_2)
    """
    if isinstance(mod, BatchLSQObserver):
        mod.estimator_mode[0] = 2
        mod.reset_min_max_vals()


@MODELS.register_module()
class BatchLSQObserver(MinMaxObserver):
    """LSQ observer.

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>
    """

    def __init__(self, *args, extreme_estimator=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.extreme_estimator = extreme_estimator
        # estimator_mode defines the way to estimate min_val and max_val
        # 0: no estimate min_val and max_val
        # 1: estimate min_val and max_val only by current batch
        # 2: estimate min_val and max_val by avraging multiple batches.
        self.register_buffer('estimator_mode',
                             torch.tensor([1], dtype=torch.uint8))


    def forward(self, x_orig):
        """Records the running minimum, maximum and tensor_norm of ``x``."""
        if x_orig.numel() == 0 or self.estimator_mode[0] == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)

        if self.extreme_estimator == 0:
            min_val_cur, max_val_cur = torch.aminmax(x)
        elif self.extreme_estimator == 1:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[1] = 0
            new_axis_list[0] = 1
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val_cur, max_val_cur = min_val_cur.mean(), max_val_cur.mean()
        elif self.extreme_estimator == 2:
            mean, var = x.mean(), x.var()
            min_val_cur, max_val_cur = mean - 3 * var, mean + 3 * var
        else:
            raise ValueError(f'Unsupported extreme estimator type: {self.extreme_estimator}')

        if self.estimator_mode[0] == 1:
            self.min_val.copy_(min_val_cur)
            self.max_val.copy_(max_val_cur)
        elif self.estimator_mode[0] == 2:
            if self.min_val == float('inf') or self.max_val == float('-inf'):
                min_val = min_val_cur
                max_val = max_val_cur
            else:
                min_val = torch.stack([min_val_cur, self.min_val]).mean()
                max_val = torch.stack([max_val_cur, self.max_val]).mean()
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        else:
            raise ValueError(f'Unsupported estimator_mode: {self.estimator_mode}')
        return x_orig

