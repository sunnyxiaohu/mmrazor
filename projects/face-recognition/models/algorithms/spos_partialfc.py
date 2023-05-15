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
        with optim_wrapper['architecture.backbone'].optim_context(self):
            data = self.data_preprocessor(data, True)
            feats = self.architecture.extract_feat(data['inputs'])
            # losses = self._run_forward(data, mode='loss')  # type: ignore
        with optim_wrapper['architecture.head'].optim_context(self):
            losses = self.architecture.head.loss(feats, data['data_samples'])
            optimizer_cfg = losses.pop('optimizer')
            # message_hub = MessageHub.get_current_instance()
            # cur_iter = message_hub.get_info('iter')
            # # TODO(shiguang): More common realize.
            # if cur_iter == 0:
            #     assert len(optim_wrapper.param_groups[-1]['params'][-1]) == 0
            optim_wrapper['architecture.head'].optimizer.state.pop(optim_wrapper['architecture.head'].param_groups[-1]['params'][-1], None)
            optim_wrapper['architecture.head'].param_groups[-1]['params'][-1] = optimizer_cfg['params']
            optim_wrapper['architecture.head'].optimizer.state[optimizer_cfg['params']] = optimizer_cfg['state']

        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        parsed_losses = optim_wrapper['architecture.head'].scale_loss(parsed_losses)
        optim_wrapper['architecture.head'].backward(parsed_losses)
        optim_wrapper['architecture.head'].step()
        optim_wrapper['architecture.backbone'].step()
        optim_wrapper['architecture.head'].zero_grad()
        optim_wrapper['architecture.backbone'].zero_grad()

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
        with optim_wrapper['architecture.backbone'].optim_context(self):
            data = self.module.data_preprocessor(data, True)
            feats = self.module.architecture.extract_feat(data['inputs'])
            # losses = self.module._run_forward(data, mode='loss')  # type: ignore
        with optim_wrapper['architecture.head'].optim_context(self):
            losses = self.module.architecture.head.loss(feats, data['data_samples'])
            optimizer_cfg = losses.pop('optimizer')
            # message_hub = MessageHub.get_current_instance()
            # cur_iter = message_hub.get_info('iter')
            # # TODO(shiguang): More common realize.
            # if cur_iter == 0:
            #     assert len(optim_wrapper.param_groups[-1]['params'][-1]) == 0
            optim_wrapper['architecture.head'].optimizer.state.pop(optim_wrapper['architecture.head'].param_groups[-1]['params'][-1], None)
            optim_wrapper['architecture.head'].param_groups[-1]['params'][-1] = optimizer_cfg['params']
            optim_wrapper['architecture.head'].optimizer.state[optimizer_cfg['params']] = optimizer_cfg['state']

        parsed_losses, log_vars = self.module.parse_losses(losses)  # type: ignore
        parsed_losses = optim_wrapper['architecture.head'].scale_loss(parsed_losses)
        optim_wrapper['architecture.head'].backward(parsed_losses)
        optim_wrapper['architecture.head'].step()
        optim_wrapper['architecture.backbone'].step()
        optim_wrapper['architecture.head'].zero_grad()
        optim_wrapper['architecture.backbone'].zero_grad()

        return log_vars
