import torch
import torch.utils.data as torch_data
from mmrazor.models.algorithms.quantization.cle_superacme.cross_layer_equalization import equalize_model
from mmrazor.models.algorithms.quantization.cle_superacme.meta.utils import get_device


def apply_cross_layer_equalization(model: torch.nn.Module, input_shape: tuple):
    """
    Applies CLE on the model and calculates model accuracy on quantized simulator
    Applying CLE on the model inplace consists of:
        Batch Norm Folding
        Cross Layer Scaling
        High Bias Fold
    Converts any ReLU6 into ReLU.

    :param model: the loaded model
    :param input_shape: the shape of the input to the model
    :return:
    """
    equalize_model(model, input_shape)
    
