# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import torch
from mmengine.device import get_device
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel

from mmrazor.models import BaseAlgorithm
from mmrazor.models.architectures.ops.mobilenet_series import MBBlock
from mmrazor.models.architectures.utils import set_dropout
from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators import NasMutator
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.registry import MODEL_WRAPPERS, MODELS

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]


@MODELS.register_module()
class BigNASPartialFC(BaseAlgorithm):
    """Implementation of `BigNas <https://arxiv.org/pdf/2003.11142>`_

    BigNAS is a NAS algorithm which searches the following items in MobileNetV3
    with the one-shot paradigm: kernel_sizes, out_channels, expand_ratios,
    block_depth and input sizes.

    BigNAS uses a `sandwich` strategy to sample subnets from the supernet,
    which includes the max subnet, min subnet and N random subnets. It doesn't
    require retraining, therefore we can directly get well-trained subnets
    after supernet training.

    The logic of the search part is implemented in
    :class:`mmrazor.engine.EvolutionSearchLoop`

    Args:
        architecture (dict|:obj:`BaseModel`): The config of :class:`BaseModel`
            or built model. Corresponding to supernet in NAS algorithm.
        mutator (VALID_MUTATOR_TYPE): The config of :class:`NasMutator` or
            built mutator.
        distiller (VALID_DISTILLER_TYPE): Cfg of :class:`ConfigurableDistiller`
            or built distiller.
        data_preprocessor (Optional[Union[dict, nn.Module]]): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        num_random_samples (int): number of random sample subnets.
            Defaults to 2.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.2.
        backbone_dropout_stages (List): Stages to be set dropout. Defaults to
            [6, 7].
        init_cfg (Optional[dict]): Init config for ``BaseModule``.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATOR_TYPE = None,
                 distiller: VALID_DISTILLER_TYPE = None,
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 num_random_samples: int = 2,
                 drop_path_rate: float = 0.2,
                 backbone_dropout_stages: List = [6, 7],
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(architecture, data_preprocessor, init_cfg)
        self.mutator = self._build_mutator(mutator)
        # NOTE: `mutator.prepare_from_supernet` must be called
        # before distiller initialized.
        self.mutator.prepare_from_supernet(self.architecture)

        self.distiller = self._build_distiller(distiller)
        self.distiller.prepare_from_teacher(self.architecture)
        self.distiller.prepare_from_student(self.architecture)

        self.sample_kinds = ['max', 'min']
        for i in range(num_random_samples):
            self.sample_kinds.append('random' + str(i))
        # self.sample_kinds = ['max']
        self.drop_path_rate = drop_path_rate
        self.backbone_dropout_stages = backbone_dropout_stages
        self._optim_wrapper_count_status_reinitialized = False

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE = None) -> NasMutator:
        """Build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, NasMutator):
            raise TypeError('mutator should be a `dict` or `NasMutator` '
                            f'instance, but got {type(mutator)}.')
        return mutator

    def _build_distiller(
            self,
            distiller: VALID_DISTILLER_TYPE = None) -> ConfigurableDistiller:
        """Build distiller."""
        if isinstance(distiller, dict):
            distiller = MODELS.build(distiller)
        if not isinstance(distiller, ConfigurableDistiller):
            raise TypeError('distiller should be a `dict` or '
                            '`ConfigurableDistiller` instance, but got '
                            f'{type(distiller)}')
        return distiller

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f'{self.__class__.__name__}.train_step not implemented')


@MODEL_WRAPPERS.register_module()
class BigNASPartialFCDDP(DistributedDataParallel):

    def __init__(self,
                 module: nn.Module,
                 exclude_module: str = None,
                 broadcast_buffers: bool = False,
                 find_unused_parameters: bool = False,
                 **kwargs) -> None:
        if exclude_module is None:
            super().__init__(
                module=module, broadcast_buffers=broadcast_buffers, find_unused_parameters=find_unused_parameters, **kwargs)
        else:
            super(DistributedDataParallel, self).__init__()
            self.module = module
            self.device = get_device()

            # Wrap the submodule excluded `exclude_module` with parameters of `self.module` to
            # `MMDistributedDataParallel`

            def _wrap_model(module, exclude_module):
                assert exclude_module.split(
                    '.'
                )[0] in module._modules, f'{exclude_module} have to be a submodule of module.'
                for name, sub_module in module._modules.items():
                    # recursive to math to the corresponding level.
                    if exclude_module.startswith(name) and len(
                            exclude_module.split('.')) > 1:
                        exclude_module = '.'.join(exclude_module.split('.')[1:])
                        sub_module = _wrap_model(sub_module, exclude_module)
                    elif name == exclude_module:
                        sub_module = sub_module.to(self.device)
                    # module without parameters.
                    elif next(sub_module.parameters(), None) is None:
                        sub_module = sub_module.to(self.device)
                    elif all(not p.requires_grad for p in sub_module.parameters()):
                        sub_module = sub_module.to(self.device)
                    else:
                        sub_module = MMDistributedDataParallel(
                            module=sub_module.to(self.device),
                            broadcast_buffers=broadcast_buffers,
                            find_unused_parameters=find_unused_parameters,
                            **kwargs)
                    module._modules[name] = sub_module
                return module

            _wrap_model(module, exclude_module)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            subnet_losses = dict()
            with self.module.distiller.student_recorders:
                with optim_wrapper['architecture.backbone'].optim_context(self):
                    feats = self.module.architecture.extract_feat(batch_inputs)
                with optim_wrapper['architecture.head'].optim_context(self):
                    student_subnet_losses = self.module.architecture.head.loss(feats, data_samples)
                    # clear state and update optim since partialfc
                    optimizer_cfg = student_subnet_losses.pop('optimizer')
                    optim_wrapper['architecture.head'].optimizer.state.pop(
                        optim_wrapper['architecture.head'].param_groups[-1]['params']
                        [-1], None)
                    optim_wrapper['architecture.head'].param_groups[-1]['params'][
                        -1] = optimizer_cfg['params']
                    optim_wrapper['architecture.head'].optimizer.state[
                        optimizer_cfg['params']] = optimizer_cfg['state']
                    subnet_losses.update(student_subnet_losses)
                    soft_loss = self.module.distiller.compute_distill_losses()
                    subnet_losses.update(soft_loss)                        
                    # parse loss, scale loss and backward
                    parsed_subnet_losses, _ = self.module.parse_losses(
                        subnet_losses)
                    parsed_subnet_losses = optim_wrapper['architecture.head'].scale_loss(
                        parsed_subnet_losses)
                    optim_wrapper['architecture.head'].backward(parsed_subnet_losses)

            return subnet_losses

        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper['architecture.head'],
                accumulative_counts=len(self.module.sample_kinds))
            self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.module.data_preprocessor(
            data, True).values()

        total_losses = dict()
        for kind in self.module.sample_kinds:
            # update the max subnet loss.
            if kind == 'max':
                self.module.mutator.set_max_choices()
                # get teacher recorders
                with self.module.distiller.teacher_recorders:
                    with optim_wrapper['architecture.backbone'].optim_context(self):
                        feats = self.module.architecture.extract_feat(batch_inputs)
                    with optim_wrapper['architecture.head'].optim_context(self):
                        max_subnet_losses = self.module.architecture.head.loss(feats, data_samples)
                        # clear state and update optim since partialfc
                        optimizer_cfg = max_subnet_losses.pop('optimizer')
                        optim_wrapper['architecture.head'].optimizer.state.pop(
                            optim_wrapper['architecture.head'].param_groups[-1]['params']
                            [-1], None)
                        optim_wrapper['architecture.head'].param_groups[-1]['params'][
                            -1] = optimizer_cfg['params']
                        optim_wrapper['architecture.head'].optimizer.state[
                            optimizer_cfg['params']] = optimizer_cfg['state']
                        # parse loss, scale loss and backward
                        parsed_max_subnet_losses, _ = self.module.parse_losses(
                            max_subnet_losses)
                        parsed_max_subnet_losses = optim_wrapper['architecture.head'].scale_loss(
                            parsed_max_subnet_losses)
                        optim_wrapper['architecture.head'].backward(parsed_max_subnet_losses)

                        total_losses.update(
                            add_prefix(max_subnet_losses, 'max_subnet'))
            # update the min subnet loss.
            elif kind == 'min':
                self.module.mutator.set_min_choices()
                # min_subnet_losses = distill_step(batch_inputs, data_samples)

                with optim_wrapper['architecture.backbone'].optim_context(self):
                    feats = self.module.architecture.extract_feat(batch_inputs)
                with optim_wrapper['architecture.head'].optim_context(self):
                    min_subnet_losses = self.module.architecture.head.loss(feats, data_samples)
                    # clear state and update optim since partialfc
                    optimizer_cfg = min_subnet_losses.pop('optimizer')
                    optim_wrapper['architecture.head'].optimizer.state.pop(
                        optim_wrapper['architecture.head'].param_groups[-1]['params']
                        [-1], None)
                    optim_wrapper['architecture.head'].param_groups[-1]['params'][
                        -1] = optimizer_cfg['params']
                    optim_wrapper['architecture.head'].optimizer.state[
                        optimizer_cfg['params']] = optimizer_cfg['state']
                    # parse loss, scale loss and backward
                    parsed_min_subnet_losses, _ = self.module.parse_losses(
                        min_subnet_losses)
                    parsed_min_subnet_losses = optim_wrapper['architecture.head'].scale_loss(
                        parsed_min_subnet_losses)
                    optim_wrapper['architecture.head'].backward(parsed_min_subnet_losses)

                total_losses.update(
                    add_prefix(min_subnet_losses, 'min_subnet'))
            # update the random subnets loss.
            elif 'random' in kind:
                self.module.mutator.set_choices(self.module.mutator.sample_choices())
                # random_subnet_losses = distill_step(batch_inputs, data_samples)

                with optim_wrapper['architecture.backbone'].optim_context(self):
                    feats = self.module.architecture.extract_feat(batch_inputs)
                with optim_wrapper['architecture.head'].optim_context(self):
                    random_subnet_losses = self.module.architecture.head.loss(feats, data_samples)
                    # clear state and update optim since partialfc
                    optimizer_cfg = random_subnet_losses.pop('optimizer')
                    optim_wrapper['architecture.head'].optimizer.state.pop(
                        optim_wrapper['architecture.head'].param_groups[-1]['params']
                        [-1], None)
                    optim_wrapper['architecture.head'].param_groups[-1]['params'][
                        -1] = optimizer_cfg['params']
                    optim_wrapper['architecture.head'].optimizer.state[
                        optimizer_cfg['params']] = optimizer_cfg['state']
                    # parse loss, scale loss and backward
                    parsed_random_subnet_losses, _ = self.module.parse_losses(
                        random_subnet_losses)
                    parsed_random_subnet_losses = optim_wrapper['architecture.head'].scale_loss(
                        parsed_random_subnet_losses)
                    optim_wrapper['architecture.head'].backward(parsed_random_subnet_losses)

                total_losses.update(
                    add_prefix(random_subnet_losses, f'{kind}_subnet'))

        # step and zero_grad
        optim_wrapper['architecture.head'].step()
        if hasattr(optim_wrapper['architecture.backbone'],
                   'loss_scaler') and hasattr(
                       optim_wrapper['architecture.head'], 'loss_scaler'):
            optim_wrapper[
                'architecture.backbone'].loss_scaler._scale = optim_wrapper[
                    'architecture.head'].loss_scaler._scale
            optim_wrapper[
                'architecture.backbone'].loss_scaler._growth_tracker = optim_wrapper[
                    'architecture.head'].loss_scaler._growth_tracker
        optim_wrapper['architecture.backbone'].step()
        optim_wrapper['architecture.head'].zero_grad()
        optim_wrapper['architecture.backbone'].zero_grad()

        # Clear data_buffer so that we could implement deepcopy.
        for key, recorder in self.module.distiller.teacher_recorders.recorders.items():
            recorder.reset_data_buffer()
        for key, recorder in self.module.distiller.student_recorders.recorders.items():
            recorder.reset_data_buffer()

        return total_losses

    @property
    def _optim_wrapper_count_status_reinitialized(self) -> bool:
        return self.module._optim_wrapper_count_status_reinitialized

    @_optim_wrapper_count_status_reinitialized.setter
    def _optim_wrapper_count_status_reinitialized(self, val: bool) -> None:
        assert isinstance(val, bool)

        self.module._optim_wrapper_count_status_reinitialized = val

    def train(self, mode: bool = True) -> 'SPOSPartialFCDDP':
        """Sets the module in training mode."""
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
