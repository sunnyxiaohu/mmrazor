from typing import Dict, List, Optional, Union
import os

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from mmengine.device import get_device
from mmengine.dist import get_rank
from mmengine.optim import OptimWrapper
from mmengine.model import MMDistributedDataParallel

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

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            self.init_cfg['checkpoint'] += f'_gpu{get_rank()}'
        return super().init_weights()


@MODEL_WRAPPERS.register_module()
class SPOSPartialFCDDP(DistributedDataParallel):

    def __init__(self,
                 module: nn.Module,
                 exclude_module: str,
                 broadcast_buffers: bool = False,
                 find_unused_parameters: bool = False,
                 **kwargs) -> None:
        super(DistributedDataParallel, self).__init__()
        self.module = module
        device = get_device()
        # Wrap the submodule excluded `exclude_module` with parameters of `self.module` to
        # `MMDistributedDataParallel`

        def _wrap_model(module, exclude_module):
            assert exclude_module.split('.')[0] in module._modules, f'{exclude_module} have to be a submodule of module.'
            for name, sub_module in module._modules.items():
                # recursive to math to the corresponding level.
                if exclude_module.startswith(name) and len(exclude_module.split('.')) > 1:
                    exclude_module = '.'.join(exclude_module.split('.')[1:])
                    sub_module = _wrap_model(sub_module, exclude_module)
                elif name == exclude_module:
                    sub_module = sub_module.to(device)
                # module without parameters.
                elif next(sub_module.parameters(), None) is None:
                    sub_module = sub_module.to(device)
                elif all(not p.requires_grad for p in sub_module.parameters()):
                    sub_module = sub_module.to(device)
                else:
                    sub_module = MMDistributedDataParallel(
                        module=sub_module.to(device),
                        broadcast_buffers=broadcast_buffers,
                        find_unused_parameters=find_unused_parameters,
                        **kwargs)
                module._modules[name] = sub_module
            return module

        _wrap_model(module, exclude_module)

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


    def train(self, mode: bool = True) -> 'SPOSPartialFCDDP':
        """Sets the module in training mode. """
        self.training = mode
        self.module.train(mode)
        return self

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.val_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.test_step(data)
