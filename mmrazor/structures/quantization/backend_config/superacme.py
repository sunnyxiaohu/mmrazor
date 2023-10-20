# Copyright (c) OpenMMLab. All rights reserved.
import torch

try:
    from torch.ao.quantization.backend_config import (BackendConfig,
                                                      BackendPatternConfig,
                                                      DTypeConfig,
                                                      ObservationType)
except ImportError:
    from mmrazor.utils import get_placeholder
    BackendConfig = get_placeholder('torch>=1.13')
    BackendPatternConfig = get_placeholder('torch>=1.13')
    DTypeConfig = get_placeholder('torch>=1.13')
    ObservationType = get_placeholder('torch>=1.13')

from .common_operator_config_utils import (_get_binary_op_configs,
                                           _get_conv_configs,
                                           _get_linear_configs,
                                           _get_share_qparams_op_configs)


def get_superacme_backend_config() -> BackendConfig:
    """Return the `BackendConfig` for the superacme backend.

    Note:
        Learn more about BackendConfig, please refer to:
        https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/backend_config # noqa: E501
    """
    # dtype configs
    weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    non_weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
    )

    cat_config = BackendPatternConfig(torch.cat) \
        .set_observation_type(
            ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
        .add_dtype_config(non_weighted_op_qint8_dtype_config)
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    # (Shiguang): Avoid patterns like `Clip -> size -> view`,
    # the dtype(int) of view will propogate to size and Clip, then
    # no observers inserted.
    size_config = BackendPatternConfig('size') \
                .set_observation_type(
                    ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
                .set_dtype_configs(share_qparams_op_dtype_configs)
    # there might be things not supported in fx2trt, but it will error out
    # during fx2trt conversion and can support them after that
    return BackendConfig('superacme') \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_config(cat_config) \
        .set_backend_pattern_configs(
            _get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(
            _get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_config(size_config)


def get_superacme_backend_config_dict():
    """Return the `BackendConfig` for the superacme backend in dictionary
    form."""
    return get_superacme_backend_config().to_dict()


__all__ = [
    'get_superacme_backend_config',
    'get_superacme_backend_config_dict',
]
