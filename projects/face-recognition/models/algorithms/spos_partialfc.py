from typing import Dict, List, Optional, Union
import os

import torch
from mmengine.logging import MessageHub
from mmengine.model import MMDistributedDataParallel
from mmengine.optim import OptimWrapper

from mmrazor.models import SPOS
from mmrazor.registry import MODEL_WRAPPERS, MODELS


@MODELS.register_module()
class SPOSPartialFC(SPOS):
    """Implementation of `SPOS <https://arxiv.org/abs/1904.00420>`_"""

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        optimizer_cfg = losses.pop('optimizer')
        message_hub = MessageHub.get_current_instance()
        cur_iter = message_hub.get_info('iter')
        # TODO(shiguang): More common realize.
        if cur_iter == 0:
            assert len(optim_wrapper.param_groups[-1]['params'][-1]) == 0

        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore

        optim_wrapper.optimizer.state.pop(optim_wrapper.param_groups[-1]['params'][-1], None)
        optim_wrapper.param_groups[-1]['params'][-1] = optimizer_cfg['params']
        optim_wrapper.optimizer.state[optimizer_cfg['params']] = optimizer_cfg['state']
        optim_wrapper.update_params(parsed_losses)
        return log_vars


@MODEL_WRAPPERS.register_module()
class SPOSPartialFCDDP(MMDistributedDataParallel):

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, True)
            losses = self.module._run_forward(data, mode='loss')  # type: ignore
        optimizer_cfg = losses.pop('optimizer')
        message_hub = MessageHub.get_current_instance()
        cur_iter = message_hub.get_info('iter')
        # TODO(shiguang): More common realize.
        if cur_iter == 0:
            assert len(optim_wrapper.param_groups[-1]['params'][-1]) == 0

        parsed_losses, log_vars = self.module.parse_losses(losses)  # type: ignore

        optim_wrapper.optimizer.state.pop(optim_wrapper.param_groups[-1]['params'][-1], None)
        optim_wrapper.param_groups[-1]['params'][-1] = optimizer_cfg['params']
        optim_wrapper.optimizer.state[optimizer_cfg['params']] = optimizer_cfg['state']
        optim_wrapper.update_params(parsed_losses)
        return log_vars
