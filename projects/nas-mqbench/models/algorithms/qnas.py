# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.architectures.ops.mobilenet_series import MBBlock
from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators import NasMutator
from mmrazor.models.utils import (add_prefix,
                                  reinitialize_optim_wrapper_count_status)
from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODEL_WRAPPERS, MODELS

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]
LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class QNAS(BaseAlgorithm):
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
                 qat_distiller: VALID_DISTILLER_TYPE = None,
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
        self.distiller.prepare_from_teacher(self.architecture.architecture)
        self.distiller.prepare_from_student(self.architecture.architecture)

        self.qat_distiller = self._build_distiller(qat_distiller)
        self.qat_distiller.prepare_from_teacher(self.architecture.qmodels['loss'])
        self.qat_distiller.prepare_from_student(self.architecture.qmodels['loss'])

        self.sample_kinds = ['max', 'min']
        for i in range(num_random_samples):
            self.sample_kinds.append('random' + str(i))

        self.drop_path_rate = drop_path_rate
        self.backbone_dropout_stages = backbone_dropout_stages
        self._optim_wrapper_count_status_reinitialized = False
        # May be set by HOOK or Runner outside
        self.current_stage = 'float'

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
        if self.current_stage == 'float':
            distiller = self.distiller
            mode = 'float.loss'
        elif self.current_stage == 'qat':
            distiller = self.qat_distiller
            mode = 'loss'
        else:
            raise ValueError(f'Unsupported current_stage: {self.module.current_stage}')

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            subnet_losses = dict()
            with optim_wrapper.optim_context(
                    self), distiller.student_recorders:  # type: ignore
                _ = self(batch_inputs, data_samples, mode=mode)
                soft_loss = distiller.compute_distill_losses()

                subnet_losses.update(soft_loss)

                parsed_subnet_losses, _ = self.parse_losses(subnet_losses)
                optim_wrapper.update_params(parsed_subnet_losses)

            return subnet_losses

        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=len(self.sample_kinds))
            self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.data_preprocessor(data,
                                                            True).values()

        total_losses = dict()
        for kind in self.sample_kinds:
            # update the max subnet loss.
            if kind == 'max':
                self.mutator.set_max_choices()
                # qnas_backbone.set_dropout(
                #     dropout_stages=self.backbone_dropout_stages,
                #     drop_path_rate=self.drop_path_rate)
                with optim_wrapper.optim_context(
                        self
                ), distiller.teacher_recorders:  # type: ignore
                    max_subnet_losses = self(
                        batch_inputs, data_samples, mode=mode)
                    parsed_max_subnet_losses, _ = self.parse_losses(
                        max_subnet_losses)
                    optim_wrapper.update_params(parsed_max_subnet_losses)
                total_losses.update(
                    add_prefix(max_subnet_losses, 'max_subnet'))
            # update the min subnet loss.
            elif kind == 'min':
                self.mutator.set_min_choices()
                # qnas_backbone.set_dropout(
                #     dropout_stages=self.backbone_dropout_stages,
                #     drop_path_rate=0.)
                min_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(min_subnet_losses, 'min_subnet'))
            # update the random subnets loss.
            elif 'random' in kind:
                self.mutator.set_choices(self.mutator.sample_choices())
                # qnas_backbone.set_dropout(
                #     dropout_stages=self.backbone_dropout_stages,
                #     drop_path_rate=0.)
                random_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(random_subnet_losses, f'{kind}_subnet'))

        # Clear data_buffer so that we could implement deepcopy.
        for key, recorder in distiller.teacher_recorders.recorders.items():
            recorder.reset_data_buffer()
        for key, recorder in distiller.student_recorders.recorders.items():
            recorder.reset_data_buffer()

        return total_losses

    def sync_qparams(self, src_mode: str):
        """Same as in 'MMArchitectureQuant'. Sync all quantize parameters in
        different `forward_modes`. We could have several modes to generate
        graphs, but in training, only one graph will be update, so we need to
        sync qparams on the other graphs.

        Args:
            src_mode (str): The src modes of forward method.
        """
        self.architecture.sync_qparams(src_mode)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'float.loss':
            return self.architecture.architecture(inputs, data_samples, mode='loss')
        elif mode == 'float.tensor':
            return self.architecture.architecture(inputs, data_samples, mode='tensor')
        elif mode == 'float.predict':
            return self.architecture.architecture(inputs, data_samples, mode='predict')
        else:
            return super().forward(inputs, data_samples=data_samples, mode=mode)


@MODEL_WRAPPERS.register_module()
class QNASDDP(MMDistributedDataParallel):

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.module.architecture.qmodels = self.module.architecture._build_qmodels(
        #     self.module.architecture.architecture)
        self.module.architecture.sync_qparams('tensor')
        self.module.architecture.reset_observer_and_fakequant_statistics(self)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        if self.module.current_stage == 'float':
            distiller = self.module.distiller
            mode = 'float.loss'
        elif self.module.current_stage == 'qat':
            distiller = self.module.qat_distiller
            mode = 'loss'
        else:
            raise ValueError(f'Unsupported current_stage: {self.module.current_stage}')

        def distill_step(
                batch_inputs: torch.Tensor, data_samples: List[BaseDataElement]
        ) -> Dict[str, torch.Tensor]:
            subnet_losses = dict()
            with optim_wrapper.optim_context(
                    self
            ), distiller.student_recorders:  # type: ignore
                _ = self(batch_inputs, data_samples, mode=mode)
                soft_loss = distiller.compute_distill_losses()

                subnet_losses.update(soft_loss)

                parsed_subnet_losses, _ = self.module.parse_losses(
                    subnet_losses)
                optim_wrapper.update_params(parsed_subnet_losses)

            return subnet_losses

        if not self._optim_wrapper_count_status_reinitialized:
            reinitialize_optim_wrapper_count_status(
                model=self,
                optim_wrapper=optim_wrapper,
                accumulative_counts=len(self.module.sample_kinds))
            self._optim_wrapper_count_status_reinitialized = True

        batch_inputs, data_samples = self.module.data_preprocessor(
            data, True).values()

        total_losses = dict()
        for kind in self.module.sample_kinds:
            # update the max subnet loss.
            if kind == 'max':
                self.module.mutator.set_max_choices()
                # qnas_backbone.set_dropout(
                #     dropout_stages=self.module.backbone_dropout_stages,
                #     drop_path_rate=self.module.drop_path_rate)
                with optim_wrapper.optim_context(
                        self
                ), distiller.teacher_recorders:  # type: ignore
                    max_subnet_losses = self(
                        batch_inputs, data_samples, mode=mode)
                    parsed_max_subnet_losses, _ = self.module.parse_losses(
                        max_subnet_losses)
                    optim_wrapper.update_params(parsed_max_subnet_losses)
                total_losses.update(
                    add_prefix(max_subnet_losses, 'max_subnet'))
            # update the min subnet loss.
            elif kind == 'min':
                self.module.mutator.set_min_choices()
                # qnas_backbone.set_dropout(
                #     dropout_stages=self.module.backbone_dropout_stages,
                #     drop_path_rate=0.)
                min_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(min_subnet_losses, 'min_subnet'))
            # update the random subnets loss.
            elif 'random' in kind:
                self.module.mutator.set_choices(
                    self.module.mutator.sample_choices())
                # qnas_backbone.set_dropout(
                #     dropout_stages=self.module.backbone_dropout_stages,
                #     drop_path_rate=0.)
                random_subnet_losses = distill_step(batch_inputs, data_samples)
                total_losses.update(
                    add_prefix(random_subnet_losses, f'{kind}_subnet'))

        # Clear data_buffer so that we could implement deepcopy.
        for key, recorder in distiller.teacher_recorders.recorders.items():
            recorder.reset_data_buffer()
        for key, recorder in distiller.student_recorders.recorders.items():
            recorder.reset_data_buffer()

        return total_losses

    @property
    def _optim_wrapper_count_status_reinitialized(self) -> bool:
        return self.module._optim_wrapper_count_status_reinitialized

    @_optim_wrapper_count_status_reinitialized.setter
    def _optim_wrapper_count_status_reinitialized(self, val: bool) -> None:
        assert isinstance(val, bool)

        self.module._optim_wrapper_count_status_reinitialized = val

    def sync_qparams(self, src_mode: str):
        """Same as in 'MMArchitectureQuant'. Sync all quantize parameters in
        different `forward_modes`. We could have several modes to generate
        graphs, but in training, only one graph will be update, so we need to
        sync qparams on the other graphs.

        Args:
            src_mode (str): The src modes of forward method.
        """

        self.module.sync_qparams(src_mode)