# Copyright (c) OpenMMLab. All rights reserved.
import inspect

import torch
import torch.nn as nn
from typing import List

try:
    import torch.nn.functional as F
    import torch.nn.intrinsic as nni
    import torch.nn.intrinsic.qat as nniqat
    import torch.nn.qat as nnqat
    import torch.nn.quantized._reference as nnqr    
    from torch.ao.quantization.backend_config import (BackendConfig,
                                                      BackendPatternConfig,
                                                      DTypeConfig,
                                                      ObservationType)
    from torch.ao.quantization.fuser_method_mappings import (
        fuse_linear_bn, reverse2, reverse3, reverse_sequential_wrapper2)
except ImportError:
    from mmrazor.utils import get_placeholder
    F = get_package_placeholder('torch>=1.13')
    nni = get_package_placeholder('torch>=1.13')
    nniqat = get_package_placeholder('torch>=1.13')
    nnqat = get_package_placeholder('torch>=1.13')
    nnqr = get_package_placeholder('torch>=1.13')    
    BackendConfig = get_placeholder('torch>=1.13')
    BackendPatternConfig = get_placeholder('torch>=1.13')
    DTypeConfig = get_placeholder('torch>=1.13')
    ObservationType = get_placeholder('torch>=1.13')
    fuse_linear_bn = get_placeholder('torch>=1.13')
    reverse2 = get_placeholder('torch>=1.13')
    reverse3 = get_placeholder('torch>=1.13')
    reverse_sequential_wrapper2 = get_placeholder('torch>=1.13')

from mmengine.utils import import_modules_from_strings
from mmrazor.models.architectures.dynamic_ops import (BigNasConv2d,
                                                      DynamicBatchNorm2d,
                                                      DynamicLinear)
from mmrazor.structures.quantization.backend_config import get_openvino_backend_config

custom_imports = 'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_fused'
dynamic_fused = import_modules_from_strings(custom_imports)
custom_imports = 'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_qconv'
dynamic_qconv = import_modules_from_strings(custom_imports)
custom_imports = 'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_qconv_fused'
dynamic_qconv_fused = import_modules_from_strings(custom_imports)


def fuse_conv_bn(is_qat, conv, bn):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."

    fused_module_class_map = {
        BigNasConv2d: dynamic_fused.DynamicConvBn2d,
    }

    if is_qat:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        if fused_module_class is not None:
            return fused_module_class(conv, bn)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn)))
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)

def fuse_conv_bn_relu(is_qat, conv, bn, relu):
    r"""Given the conv and bn modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        conv: Module instance of type conv2d/conv3d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> r1 = nn.ReLU(inplace=False)
        >>> # xdoctest: +SKIP
        >>> m2 = fuse_conv_bn_relu(m1, b1, r1)
    """
    assert(conv.training == bn.training == relu.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    fused_module : Optional[Type[nn.Sequential]] = None
    if is_qat:
        map_to_fused_module_train = {
            BigNasConv2d: dynamic_fused.DynamicConvBnReLU2d,
        }
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(conv, bn, relu)
        else:
            raise NotImplementedError("Cannot fuse train modules: {}".format((conv, bn, relu)))
    else:
        map_to_fused_module_eval = {
            BigNasConv2d: dynamic_fused.DynamicConvReLU2d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError("Cannot fuse eval modules: {}".format((conv, bn, relu)))


def _get_dynamicconv_configs(dtype_configs):
    """Return all configs related to conv modules and ops."""
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT

    # (1) Single conv modules/functions
    # -----------------------------------
    # conv module
    conv_configs.append(
        BackendPatternConfig(BigNasConv2d).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            BigNasConv2d).set_qat_module(dynamic_qconv.DynamicQConv2d))
    # conv qat module
    conv_configs.append(
        BackendPatternConfig(dynamic_qconv.DynamicQConv2d).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            BigNasConv2d))

    # (2) Conv + relu
    # -----------------
    # 2.1 conv module + relu fusion configs
    # conv relu fusion, conv module + relu module
    conv_configs.append(
        BackendPatternConfig(
            (torch.nn.ReLU,
                BigNasConv2d)).set_dtype_configs(dtype_configs)  # noqa: E131
        .set_fuser_method(
            reverse_sequential_wrapper2(
                dynamic_fused.DynamicConvReLU2d)).set_fused_module(
                    dynamic_fused.DynamicConvReLU2d))
    # conv relu fusion, conv module + functional relu
    conv_configs.append(
        BackendPatternConfig(
            (F.relu,
                BigNasConv2d)).set_dtype_configs(dtype_configs)  # noqa: E131
        .set_fuser_method(
            reverse_sequential_wrapper2(
                dynamic_fused.DynamicConvReLU2d)).set_fused_module(
                    dynamic_fused.DynamicConvReLU2d))
    # 2.2 conv module + relu fused module configs
    # conv relu, fused module
    conv_configs.append(
        BackendPatternConfig(dynamic_fused.DynamicConvReLU2d).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            BigNasConv2d).set_qat_module(dynamic_qconv_fused.DynamicQConvReLU2d))
    # conv relu, qat fused module
    conv_configs.append(
        BackendPatternConfig(dynamic_qconv_fused.DynamicQConvReLU2d).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            BigNasConv2d))

    # fused conv relu
    conv_configs.append(
        BackendPatternConfig(dynamic_fused.DynamicConvReLU2d).set_dtype_configs(
            dtype_configs)  # noqa: E131
        .set_qat_module(dynamic_qconv_fused.DynamicQConvReLU2d))

    conv_configs.append(
        BackendPatternConfig(dynamic_qconv_fused.DynamicQConvReLU2d).set_dtype_configs(
            dtype_configs)  # noqa: E131
        .set_root_module(BigNasConv2d))

    # (3) Conv + batchnorm (+ relu)
    # -------------------------------
    # 3.1 conv bn fusion configs
    # conv + bn fusion
    conv_configs.append(
        BackendPatternConfig(
            (DynamicBatchNorm2d,
                BigNasConv2d)).set_dtype_configs(dtype_configs)  # noqa: E131
        .set_fuser_method(reverse2(fuse_conv_bn)).set_fused_module(
            dynamic_fused.DynamicConvBn2d))
    # conv + bn + relu module fusion
    conv_configs.append(
        BackendPatternConfig(
            (nn.ReLU,
                (DynamicBatchNorm2d,
                BigNasConv2d))).set_dtype_configs(dtype_configs)  # noqa: E131
        .set_fuser_method(reverse3(fuse_conv_bn_relu)).set_fused_module(
            dynamic_fused.DynamicConvBnReLU2d))
    # conv + bn + relu functional fusion
    conv_configs.append(
        BackendPatternConfig(
            (F.relu,
                (DynamicBatchNorm2d,
                BigNasConv2d))).set_dtype_configs(dtype_configs)  # noqa: E131
        .set_root_module(BigNasConv2d).set_fuser_method(
            reverse3(fuse_conv_bn_relu)).set_fused_module(
                dynamic_fused.DynamicConvBnReLU2d))
    # TODO: we can add fusion for torch.relu as well

    # 3.2 conv + bn (+ relu) fused module configs
    # fused conv bn
    conv_configs.append(
        BackendPatternConfig(dynamic_fused.DynamicConvBn2d).set_dtype_configs(
            dtype_configs)  # noqa: E131
        .set_qat_module(dynamic_qconv_fused.DynamicQConvBn2d))

    # fused conv bn relu
    conv_configs.append(
        BackendPatternConfig(dynamic_fused.DynamicConvBnReLU2d).set_dtype_configs(
            dtype_configs)  # noqa: E131
        .set_qat_module(dynamic_qconv_fused.DynamicQConvBnReLU2d))

    # conv bn, qat fused module
    conv_configs.append(
        BackendPatternConfig(dynamic_qconv_fused.DynamicQConvBn2d).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            BigNasConv2d))
    # conv bn relu, qat fused module
    conv_configs.append(
        BackendPatternConfig(dynamic_qconv_fused.DynamicQConvBnReLU2d).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            BigNasConv2d))

    return conv_configs

def _get_dynamiclinear_configs(
        dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    """Return all configs related to linear modules and ops."""
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    linear_configs: List[BackendPatternConfig] = []
    custom_imports = 'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_qlinear'
    dynamic_qlinear = import_modules_from_strings(custom_imports)

    # (1) Single linear modules/functions
    # -------------------------------------
    # linear module
    linear_configs.append(
        BackendPatternConfig(DynamicLinear).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            DynamicLinear).set_qat_module(dynamic_qlinear.DynamicQLinear))
    # linear qat module
    linear_configs.append(
        BackendPatternConfig(dynamic_qlinear.DynamicQLinear).set_observation_type(
            observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs).set_root_module(
            DynamicLinear))

    # # (2) Linear + relu
    # # -------------------
    # # 2.1 linear module + relu fusion config
    # # linear relu, linear module + relu module
    # linear_configs.append(
    #     BackendPatternConfig(
    #         (torch.nn.ReLU,
    #          torch.nn.Linear)).set_dtype_configs(dtype_configs)  # noqa: E131
    #     .set_fuser_method(reverse_sequential_wrapper2(
    #         nni.LinearReLU)).set_fused_module(nni.LinearReLU))
    # # linear relu, linear module + functional relu
    # linear_configs.append(
    #     BackendPatternConfig(
    #         (torch.nn.functional.relu,
    #          torch.nn.Linear)).set_dtype_configs(dtype_configs)  # noqa: E131
    #     .set_fuser_method(reverse_sequential_wrapper2(
    #         nni.LinearReLU)).set_fused_module(nni.LinearReLU))

    # # 2.2 linear module + relu, fused module configs
    # # linear relu, fused module
    # linear_configs.append(
    #     BackendPatternConfig(nni.LinearReLU).set_observation_type(
    #         observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs).set_root_module(
    #         torch.nn.Linear).set_reference_quantized_module(
    #             nnqr.Linear).set_qat_module(nniqat.LinearReLU))
    # # linear relu, qat fused module
    # linear_configs.append(
    #     BackendPatternConfig(nniqat.LinearReLU).set_observation_type(
    #         observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs).set_root_module(
    #         torch.nn.Linear).set_reference_quantized_module(nnqr.Linear))
    # # 2.3 functional linear + relu configs
    # # linear relu, functional linear + relu module
    # linear_configs.append(
    #     BackendPatternConfig(
    #         (torch.nn.ReLU,
    #          F.linear)).set_observation_type(observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs))
    # # linear relu, functional linear + functional relu
    # linear_configs.append(
    #     BackendPatternConfig(
    #         (F.relu,
    #          F.linear)).set_observation_type(observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs))

    # # (3) Linear + batchnorm
    # # ------------------------
    # # 3.1 linear bn fusion
    # linear_configs.append(
    #     BackendPatternConfig(
    #         (nn.BatchNorm1d,
    #          nn.Linear)).set_dtype_configs(dtype_configs)  # noqa: E131
    #     .set_fuser_method(reverse2(fuse_linear_bn)).set_fused_module(
    #         nni.LinearBn1d))

    # # 3.2 linear bn fused
    # # linear bn, fused module
    # linear_configs.append(
    #     BackendPatternConfig(nni.LinearBn1d).set_observation_type(
    #         observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs).set_root_module(
    #         torch.nn.Linear).set_reference_quantized_module(
    #             nnqr.Linear).set_qat_module(nniqat.LinearBn1d))
    # # linear bn, qat fused module
    # linear_configs.append(
    #     BackendPatternConfig(nniqat.LinearBn1d).set_observation_type(
    #         observation_type)  # noqa: E131
    #     .set_dtype_configs(dtype_configs).set_root_module(
    #         torch.nn.Linear).set_reference_quantized_module(nnqr.Linear))
    return linear_configs


def get_mutableopenvino_backend_config() -> BackendConfig:
    """Return the `BackendConfig` for the openvino backend.

    Note:
        Learn more about BackendConfig, please refer to:
        https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/backend_config # noqa: E501
    """
    # dtype configs
    weighted_op_qint8_dtype_config = DTypeConfig(
        input_dtype=torch.quint8,
        output_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        bias_dtype=torch.float,
    )
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]

    mutableopenvino_config = get_openvino_backend_config()
    mutableopenvino_config.set_name('mutableopenvino') \
        .set_backend_pattern_configs(_get_dynamicconv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_dynamiclinear_configs(linear_dtype_configs))

    return mutableopenvino_config


def get_mutableopenvino_backend_config_dict():
    """Return the `BackendConfig` for the openvino backend in dictionary
    form."""
    return get_mutableopenvino_backend_config().to_dict()


__all__ = [
    'get_mutableopenvino_backend_config',
    'get_mutableopenvino_backend_config_dict',
]
