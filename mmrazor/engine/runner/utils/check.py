# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Tuple

import torch

from mmrazor.models import BaseAlgorithm, ResourceEstimator
from mmrazor.structures import export_fix_subnet
from mmrazor.utils import SupportRandomSubnet

try:
    from mmdet.models.detectors import BaseDetector
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseDetector = get_placeholder('mmdet')


@torch.no_grad()
def check_subnet_resources(
    model,
    subnet: SupportRandomSubnet,
    estimator: ResourceEstimator,
    constraints_range: Dict[str, Any] = dict(flops=(0, 330))
) -> Tuple[bool, Dict]:
    """Check whether is beyond resources constraints.

    Returns:
        bool, result: The result of checking.
    """
    if constraints_range is None:
        return True, dict()

    assert hasattr(model, 'mutator') and isinstance(model, BaseAlgorithm)
    model.mutator.set_choices(subnet)
    # Support nested algorithm.
    model_to_check = model
    while(isinstance(model_to_check, BaseAlgorithm)):
        model_to_check = model_to_check.architecture
    _, sliced_model = export_fix_subnet(model_to_check, slice_weight=True)

    if isinstance(sliced_model, BaseDetector):
        results = estimator.estimate(model=sliced_model.backbone)
    else:
        results = estimator.estimate(model=sliced_model)

    for k, v in constraints_range.items():
        if not isinstance(v, (list, tuple)):
            v = (0, v)
        if results[k] < v[0] or results[k] > v[1]:
            return False, results

    return True, results
