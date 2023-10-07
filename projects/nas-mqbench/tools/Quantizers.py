from ppq.core import (OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates)
from ppq.IR import BaseGraph, Operation
from ppq.quantization.quantizer import BaseQuantizer


class MyInt8Quantizer(BaseQuantizer):

    def __init__(self, graph: BaseGraph, per_channel: bool = True, 
                 sym: bool = True, power_of_2: bool = True, 
                 num_of_bits: int = 8) -> None:
        """ Generalized int8 quantizer. """
        assert 16 >= num_of_bits >= 2, 'Unacceptable bit-width.'
        
        self.num_of_bits = num_of_bits
        self.power_of_2  = power_of_2
        self.per_channel = per_channel
        self.symmetric   = sym

        if sym:
            self.quant_min = -pow(2, num_of_bits - 1)
            self.quant_max = pow(2, num_of_bits - 1) - 1
            self.policy    = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.SYMMETRICAL)
        else:
            self.quant_min = 0
            self.quant_max = pow(2, num_of_bits) - 1
            self.policy    = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR + 
                QuantizationProperty.LINEAR + 
                QuantizationProperty.ASYMMETRICAL)

        if power_of_2:
            self.policy = QuantizationPolicy(
                self.policy._policy + 
                QuantizationProperty.POWER_OF_2)

        super().__init__(graph, True)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        """
        We Provide a Reference Quantizer Here:

            1. Only Conv, ConvTranspose, MatMul, Gemm will be quantized.

            2. Only Input Variables(including weight) of them will be quantized.

            3. Bias of those op are not going to be quantized.
        """

        OQC = self.create_default_quant_config(
            op=operation, num_of_bits=8,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            observer_algorithm='percentile',
            policy=self.policy
        )

        if operation.type in {'Conv', 'ConvTranspose', 'MatMul', 'Gemm'}:
            # disable output quantization of them
            OQC.output_quantization_config[0].state = QuantizationStates.FP32

            if operation.num_of_input == 3: # has bias
                # disable quantization of bias
                OQC.input_quantization_config[-1].state = QuantizationStates.FP32

            # modify calibration method of parameter(for higher accuracy)
            OQC.input_quantization_config[1].observer_algorithm = 'minmax'

            # for both SYMMETRICAL and ASYMMETRICAL quantization,
            # weight should always be quantized symmetrically.
            OQC.input_quantization_config[1].quant_min = - pow(2, self.num_of_bits - 1)
            OQC.input_quantization_config[1].quant_max = pow(2, self.num_of_bits - 1) - 1
            OQC.input_quantization_config[1].policy = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR +
                QuantizationProperty.LINEAR + 
                QuantizationProperty.SYMMETRICAL +
                (QuantizationProperty.POWER_OF_2 if self.power_of_2 else 0))

            # Per-channel Variation
            if self.per_channel:
                OQC.input_quantization_config[1].policy = QuantizationPolicy(
                    QuantizationProperty.PER_CHANNEL + 
                    QuantizationProperty.LINEAR + 
                    QuantizationProperty.SYMMETRICAL +
                    (QuantizationProperty.POWER_OF_2 if self.power_of_2 else 0))
                OQC.input_quantization_config[1].channel_axis = 0

                if operation.type == 'ConvTranspose':
                    OQC.input_quantization_config[1].channel_axis = 1
            return OQC

        else: # type not support
            raise TypeError(f'Op type {operation.type} is not supported by this quantizer.')

    def quant_operation_types(self) -> set:
        return 'Conv', 'ConvTranspose', 'MatMul', 'Gemm'

    def stat(self) -> dict:
        return {
            'Power-of-2': self.power_of_2,
            'Per-Channel': self.per_channel,
            'Symmetric': self.symmetric,
            'Linear': True
        }


class MyFP8Quantizer(BaseQuantizer):

    def __init__(self, graph: BaseGraph, calibration: str = 'constant') -> None:
        """ 
            Typical FP8 quantizer.

            Asymmetrical quantizing is not adaptable to FP8 Quantizer.

            PerChannel quantizing is not adaptable to FP8 Quantizer.

            FP8 Quantizer is born with power-of-2 policy.(They do not need a scale.)
        """
        self.calibration = calibration
        super().__init__(graph, True)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        """
        We Provide a Reference Quantizer Here:

            1. Only Conv, ConvTranspose, MatMul, Gemm will be quantized.

            2. Only Input Variables(including weight) of them will be quantized.

            3. Bias of those op are not going to be quantized.
        """
        OQC = self.create_default_quant_config(
            op=operation, num_of_bits=8,
            quant_min= - 448,
            quant_max= + 448,
            observer_algorithm=self.calibration,
            policy=QuantizationPolicy(
                QuantizationProperty.PER_TENSOR + 
                QuantizationProperty.FLOATING + 
                QuantizationProperty.SYMMETRICAL +
                QuantizationProperty.POWER_OF_2
            ),
            exponent_bits=4
        )

        if operation.type in {'Conv', 'ConvTranspose', 'MatMul', 'Gemm'}:
            # disable output quantization of them
            OQC.output_quantization_config[0].state = QuantizationStates.FP32

            if operation.num_of_input == 3: # has bias
                # disable quantization of bias
                OQC.input_quantization_config[-1].state = QuantizationStates.FP32

            return OQC

        else: # type not support
            raise TypeError(f'Op type {operation.type} is not supported by this quantizer.')

    def quant_operation_types(self) -> set:
        return 'Conv', 'ConvTranspose', 'MatMul', 'Gemm'

    def stat(self) -> dict:
        return {
            'Power-of-2': True,
            'Per-Channel': False,
            'Symmetric': True,
            'Linear': False
        }

