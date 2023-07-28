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


def sync_tensor(tensor):
    """Synchronize the target tensor during distributed training."""
    if torch.distributed.is_initialized() and tensor.is_cuda:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


@MODELS.register_module()
class BatchLSQObserver(MinMaxObserver):
    """LSQ observer.

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>
    """

    def __init__(self, *args, extreme_estimator=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.extreme_estimator = extreme_estimator


    def forward(self, x_orig):
        """Records the running minimum, maximum and tensor_norm of ``x``."""
        if x_orig.numel() == 0:
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

        if self.training:
            self.min_val.copy_(min_val_cur)
            self.max_val.copy_(max_val_cur)
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
        return x_orig

