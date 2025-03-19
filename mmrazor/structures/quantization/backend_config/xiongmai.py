# Copyright (c) OpenMMLab. All rights reserved.
import torch

try:
    from torch.ao.quantization.backend_config import BackendConfig, DTypeConfig
except ImportError:
    from mmrazor.utils import get_placeholder
    BackendConfig = get_placeholder('torch>=1.13')
    DTypeConfig = get_placeholder('torch>=1.13')

from .common_operator_config_utils import (  # noqa: F401,F403
    _get_binary_op_configs, _get_bn_configs, _get_cat_config,
    _get_conv_configs, _get_default_op_configs, _get_embedding_op_configs,
    _get_fixed_qparams_op_configs, _get_linear_configs, _get_ln_configs,
    _get_rnn_op_configs, _get_share_qparams_op_configs)

# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_xiongmai_backend_config() -> BackendConfig:
    """Return the `BackendConfig` for PyTorch Xiongmai backend (fbgemm/qnnpack).

    Note:
        Learn more about BackendConfig, please refer to:
        https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/backend_config # noqa: E501
    """
    # TODO: express this BackendConfig as a union of the FBGEMM and QNNPACK
    # BackendConfigs

    # ===================
    # |  DTYPE CONFIGS  |
    # ===================
    # weighted op int8 dtype config
    # this is config for ops that has quantized weights, like linear, conv
    weighted_op_int8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )

    default_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.qint8,
        output_dtype=torch.qint8,
    )

    conv_dtype_configs = [weighted_op_int8_dtype_config]
    linear_dtype_configs = [
        weighted_op_int8_dtype_config
    ]
    binary_op_dtype_configs = [weighted_op_int8_dtype_config]
    default_op_dtype_configs = [default_op_qint8_dtype_config]
    share_qparams_op_dtype_configs = [default_op_qint8_dtype_config]

    return BackendConfig('xiongmai') \
        .set_backend_pattern_configs(
            _get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(
            _get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(
            _get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_config(
            _get_cat_config(default_op_dtype_configs)) \
        .set_backend_pattern_configs(
            _get_default_op_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs))


def get_xiongmai_backend_config_dict():
    """Return the `BackendConfig` for PyTorch Xiongmai backend (fbgemm/qnnpack)
    in dictionary form."""
    return get_xiongmai_backend_config().to_dict()


__all__ = [
    'get_xiongmai_backend_config',
    'get_xiongmai_backend_config_dict',
]
