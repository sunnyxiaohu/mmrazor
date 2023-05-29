# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmengine.dist import get_dist_info
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from torch import nn

from mmdet.utils.dist_utils import all_reduce_dict


def get_norm_states(module):
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states['.'.join([name, k])] = v
    return async_norm_states


@HOOKS.register_module()
class Old_SyncNormHook(Hook):
    """Synchronize Norm states after training epoch, currently used in YOLOX.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to switch to synchronizing norm interval. Default: 15.
        interval (int): Synchronizing norm interval. Default: 1.
    """

    def __init__(self, num_last_epochs=15, interval=1):
        self.interval = interval
        self.num_last_epochs = num_last_epochs

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            # Synchronize norm every epoch.
            self.interval = 1

    def after_train_epoch(self, runner):
        """Synchronizing norm."""
        epoch = runner.epoch
        module = runner.model
        if (epoch + 1) % self.interval == 0:
            _, world_size = get_dist_info()
            if world_size == 1:
                return
            norm_states = get_norm_states(module)
            if len(norm_states) == 0:
                return
            norm_states = all_reduce_dict(norm_states, op='mean')
            module.load_state_dict(norm_states, strict=False)


@HOOKS.register_module()
class Old_SyncNormHookIters(Hook):
    """Synchronize Norm states after training epoch, currently used in YOLOX.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to switch to synchronizing norm interval. Default: 15.
        interval (int): Synchronizing norm interval. Default: 1.
    """

    def __init__(self, num_last_iters=15, interval=1):
        self.interval = interval
        self.num_last_iters = num_last_iters

    def before_train_iter(self, runner):
        iter = runner.iter
        if (iter + 1) == runner.max_iters - self.num_last_iters:
            runner.logger.info('Synchronize norm every 1000 iter!')
            # Synchronize norm every iters.
            self.interval = 1000

    def after_train_iter(self, runner):
        """Synchronizing norm."""
        iter = runner.iter
        module = runner.model
        if (iter + 1) % self.interval == 0:
            runner.logger.info('Synchronize norm !')
            _, world_size = get_dist_info()
            if world_size == 1:
                return
            norm_states = get_norm_states(module)
            if len(norm_states) == 0:
                return
            norm_states = all_reduce_dict(norm_states, op='mean')
            module.load_state_dict(norm_states, strict=False)