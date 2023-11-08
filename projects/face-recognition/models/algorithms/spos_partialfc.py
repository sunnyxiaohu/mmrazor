from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn
from mmengine.runner import load_checkpoint
from mmengine.device import get_device
from mmengine.dist import get_rank
from mmengine.model import MMDistributedDataParallel, BaseModel
from mmengine.optim import OptimWrapper
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel.distributed import DistributedDataParallel

from mmrazor.models import SPOS
from mmrazor.models.mutators import NasMutator
from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODEL_WRAPPERS, MODELS

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]

@MODELS.register_module()
class SPOSPartialFC(SPOS):
    """Implementation of `SPOS <https://arxiv.org/abs/1904.00420>`_"""
    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATOR_TYPE = None,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None,
                 distiller: Optional[dict] = None,
                 teacher: Union[BaseModel, Dict] = None,
                 teacher_ckpt: Optional[str] = None,
                 teacher_norm_eval: bool = True):
        super().__init__(architecture, mutator=mutator, norm_training=norm_training,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.distiller = distiller
        if distiller is not None:
            self.distiller = MODELS.build(distiller)

            if isinstance(teacher, Dict):
                teacher = MODELS.build(teacher)
                teacher.init_weights()

            if not isinstance(teacher, BaseModel):
                raise TypeError('teacher should be a `dict` or '
                                f'`BaseModel` instance, but got '
                                f'{type(teacher)}')
            self.teacher = teacher
                                
            if teacher_ckpt:
                _ = load_checkpoint(self.teacher, teacher_ckpt)
                # avoid loaded parameters be overwritten
                self.teacher._is_init = True
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher_norm_eval = teacher_norm_eval

            # In ``ConfigurableDistller``, the recorder manager is just
            # constructed, but not really initialized yet.
            self.distiller.prepare_from_student(self.student)
            self.distiller.prepare_from_teacher(self.teacher)

            # may be modified by stop distillation hook
            self.distillation_stopped = False

    @property
    def student(self) -> nn.Module:
        """Alias for ``architecture``."""
        return self.architecture

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        losses = dict()
        with optim_wrapper['architecture.backbone'].optim_context(self):
            data = self.data_preprocessor(data, True)
        # 1. get teacher recorders.
        if self.distiller is not None:
            with self.distiller.teacher_recorders:
                with torch.no_grad():
                    _ = self.teacher(data['inputs'], data['data_samples'], mode='loss')

            with self.distiller.student_recorders:
                # 2.compute student loss and get it's recorders. 
                with optim_wrapper['architecture.backbone'].optim_context(self):
                    feats = self.architecture.extract_feat(data['inputs'])
                with optim_wrapper['architecture.head'].optim_context(self):
                    student_losses = self.architecture.head.loss(feats, data['data_samples'])
                    optimizer_cfg = student_losses.pop('optimizer')
                    losses.update(add_prefix(student_losses, 'student'))
            # 3. compute distill loss
            if not self.distillation_stopped:
                distill_losses = self.distiller.compute_distill_losses()
                losses.update(add_prefix(distill_losses, 'distill'))
        else:
            with optim_wrapper['architecture.backbone'].optim_context(self):
                feats = self.architecture.extract_feat(data['inputs'])
            with optim_wrapper['architecture.head'].optim_context(self):
                losses = self.architecture.head.loss(feats, data['data_samples'])
                optimizer_cfg = losses.pop('optimizer')
        # 4. update optimizer and step.
        optim_wrapper['architecture.head'].optimizer.state.pop(
            optim_wrapper['architecture.head'].param_groups[-1]['params']
            [-1], None)
        optim_wrapper['architecture.head'].param_groups[-1]['params'][
            -1] = optimizer_cfg['params']
        optim_wrapper['architecture.head'].optimizer.state[
            optimizer_cfg['params']] = optimizer_cfg['state']            
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        parsed_losses = optim_wrapper['architecture.head'].scale_loss(
            parsed_losses)
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

    def train(self, mode: bool = True) -> None:
        """Set distiller's forward mode."""
        super().train(mode)
        if mode and self.distiller is not None and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


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
        losses = dict()
        # import pdb; pdb.set_trace()
        with optim_wrapper['architecture.backbone'].optim_context(self):
            data = self.module.data_preprocessor(data, True)
        # 1. get teacher recorders.
        if self.module.distiller is not None:
            with self.module.distiller.teacher_recorders:
                with torch.no_grad():
                    _ = self.module.teacher(data['inputs'], data['data_samples'], mode='loss')

            with self.module.distiller.student_recorders:
                # 2.compute student loss and get it's recorders. 
                with optim_wrapper['architecture.backbone'].optim_context(self):
                    feats = self.module.architecture.extract_feat(data['inputs'])
                with optim_wrapper['architecture.head'].optim_context(self):
                    student_losses = self.module.architecture.head.loss(feats, data['data_samples'])
                    optimizer_cfg = student_losses.pop('optimizer')
                    losses.update(add_prefix(student_losses, 'student'))
            if not self.module.distillation_stopped:                    
                # 3. compute distill loss
                distill_losses = self.module.distiller.compute_distill_losses()
                losses.update(add_prefix(distill_losses, 'distill'))
        else:
            with optim_wrapper['architecture.backbone'].optim_context(self):
                feats = self.module.architecture.extract_feat(data['inputs'])
            with optim_wrapper['architecture.head'].optim_context(self):
                losses = self.module.architecture.head.loss(
                    feats, data['data_samples'])
                optimizer_cfg = losses.pop('optimizer')
        # 4. update optimizer and step.
        optim_wrapper['architecture.head'].optimizer.state.pop(
            optim_wrapper['architecture.head'].param_groups[-1]['params']
            [-1], None)
        optim_wrapper['architecture.head'].param_groups[-1]['params'][
            -1] = optimizer_cfg['params']
        optim_wrapper['architecture.head'].optimizer.state[
            optimizer_cfg['params']] = optimizer_cfg['state']
        parsed_losses, log_vars = self.module.parse_losses(
            losses)  # type: ignore
        parsed_losses = optim_wrapper['architecture.head'].scale_loss(
            parsed_losses)
        optim_wrapper['architecture.head'].backward(parsed_losses)
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

        return log_vars

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
