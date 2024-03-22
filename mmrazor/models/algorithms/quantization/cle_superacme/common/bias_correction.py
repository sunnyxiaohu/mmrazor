# /usr/bin/env python3.5
# -*- mode: python -*-
"""  holds common code for bias correction """

from mmrazor.models.algorithms.quantization.cle_superacme.common.defs import ActivationType



class ConvBnInfoType:
    """
    Type for hoding convs with bn info and activation types
    Activation types supported are Relu and Relu6
    """
    def __init__(self,
                 input_bn=None,
                 output_bn=None,
                 in_activation_type: ActivationType = ActivationType.no_activation,
                 out_activation_type: ActivationType = ActivationType.no_activation):
        """
        :param input_bn: Reference to Input BatchNorm to layer
        :param output_bn: Reference to Output BatchNorm to layer
        :param in_activation_type: Type of Activation
        :param out_activation_type: Type of Activation
        """

        self.input_bn = input_bn
        self.output_bn = output_bn
        self.in_activation_type = in_activation_type
        self.out_activation_type = out_activation_type


class ConvBnPatternHandler:
    """
    common handler for matched patterns for bias correction and batchnorm fold.
    """

    def __init__(self):
        self.conv_linears_with_bn_dict = {}

    def get_conv_linear_bn_info_dict(self):
        """
        returns the dictionary created
        :return: dictionary of convs/linears with bn and activation info
        """
        return self.conv_linears_with_bn_dict

    def __call__(self, *args, **kwargs):
        """
         custom pattern match handler that keeps a dictionary of convs/linears with bn and activation info.
        """

        _, op_subset = args

        bn_activation_info = ConvBnInfoType()

        activation_type = ActivationType.no_activation
        conv_op = None
        bn_op = None
        convolution_types = ['Conv1d', 'Conv2D', 'DepthwiseConv2dNative', 'Conv', 'ConvTranspose']
        linear_types = ['Dense', 'Gemm']
        bn_types = ['FusedBatchNormV3', 'FusedBatchNorm', 'BatchNormalization']

        for op in op_subset:
            if op.type in convolution_types + linear_types:
                conv_op = op
                if conv_op.get_module() in self.conv_linears_with_bn_dict.keys():
                    bn_activation_info = self.conv_linears_with_bn_dict[conv_op.get_module()]
            elif op.type in bn_types:
                bn_op = op
            elif op.type in ['Relu6', 'Clip']:
                activation_type = ActivationType.relu6
            elif op.type in ['Relu']:
                activation_type = ActivationType.relu

        if len(op_subset) >= 2:
            if op_subset[0].type in bn_types:
                bn_activation_info.input_bn = bn_op
                bn_activation_info.in_activation_type = activation_type
            # we do not match linear layers with preceding bn for bias correction
            elif op_subset[0].type in convolution_types + linear_types:
                bn_activation_info.output_bn = bn_op
                bn_activation_info.out_activation_type = activation_type
            # in tf linear layer has two ops together [flatten/reshape -- dense] , check for len 3
            elif len(op_subset) >= 3 and op_subset[1].type in ['Dense']:
                bn_activation_info.output_bn = bn_op
                bn_activation_info.out_activation_type = activation_type

        self.conv_linears_with_bn_dict[conv_op.get_module()] = bn_activation_info
