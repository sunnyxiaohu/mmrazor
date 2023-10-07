# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.optim import (OptimWrapper, DefaultOptimWrapperConstructor,
                            CosineAnnealingParamScheduler)
from mmrazor.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class IndependentOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    """Independent constructor for optimizers.
    """

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):               
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        assert 'custom_keys' not in self.paramwise_cfg, (
            '`paramwise_cfg` should not contain `custom_keys`')
        super().__init__(optim_wrapper_cfg, paramwise_cfg=paramwise_cfg)

    def __call__(self, model: nn.Module) -> OptimWrapper:
        param_names = dict(model.named_parameters())
        custom_keys = dict((k, dict(decay_mult=1.0)) for k in param_names)
        self.paramwise_cfg['custom_keys'] = custom_keys
        return super().__call__(model)
