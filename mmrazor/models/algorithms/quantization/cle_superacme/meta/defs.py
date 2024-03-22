# /usr/bin/env python3.5
# -*- mode: python -*-

""" Common type definitions that are used across AIMET """

from enum import Enum
from typing import List, Optional, Union

import torch.utils.data

from mmrazor.models.algorithms.quantization.cle_superacme.common.defs import GreedySelectionParameters, TarRankSelectionParameters, RankSelectScheme


class ModuleCompRatioPair:
    """
    Pair of torch.nn.module and a compression-ratio

    :ivar module: Module of type torch.nn.module
    :ivar comp_ratio: Compression ratio. Compression ratio is the ratio of cost of compressed model
            to cost of the original model.
    """

    def __init__(self, module: torch.nn.Module, comp_ratio: float):
        self.module = module
        self.comp_ratio = comp_ratio


class OpToIOTensors:
    """
    Data class to store the input and output tensor names of an operation as a lists.
    """
    def __init__(self, node_inputs: List[str], node_outputs: List[str]):
        """
        :param node_inputs: name of inputs to the node
        :param node_outputs: name of output from the node
        """

        self.inputs = node_inputs
        self.outputs = node_outputs


class SpatialSvdParameters:
    """ Configuration parameters for spatial svd compression """

    class ManualModeParams:
        """
        Configuration parameters for manual-mode spatial svd compression
        """

        def __init__(self, list_of_module_comp_ratio_pairs: List[ModuleCompRatioPair]):
            """
            :param list_of_module_comp_ratio_pairs: List of (module, comp-ratio) pairs
            """
            self.list_of_module_comp_ratio_pairs = list_of_module_comp_ratio_pairs

    class AutoModeParams:
        """
        Configuration parameters for auto-mode compression
        """

        def __init__(self, greedy_select_params: GreedySelectionParameters,
                     modules_to_ignore: Optional[List[torch.nn.Module]] = None):
            """
            :param greedy_select_params: Params for greedy comp-ratio selection algorithm
            :param modules_to_ignore: List of modules to ignore (None indicates nothing to ignore)
            """
            self.greedy_params = greedy_select_params
            self.modules_to_ignore = [] if modules_to_ignore is None else modules_to_ignore

    class Mode(Enum):
        """ Mode enumeration """

        manual = 1
        """ Manual mode """

        auto = 2
        """ Auto mode """

    def __init__(self, mode: Mode, params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        """
        :param mode: Either auto mode or manual mode
        :param params: Parameters for the mode selected
        :param multiplicity: The multiplicity to which ranks/input channels will get rounded. Default: 1
        """
        self.mode = mode
        self.mode_params = params
        self.multiplicity = multiplicity


class ChannelPruningParameters:
    """ Configuration parameters for channel pruning compression """

    class ManualModeParams:
        """
        Configuration parameters for manual-mode channel pruning compression
        """

        def __init__(self, list_of_module_comp_ratio_pairs: List[ModuleCompRatioPair]):
            """
            :param list_of_module_comp_ratio_pairs: List of (module, comp-ratio) pairs
            """
            self.list_of_module_comp_ratio_pairs = list_of_module_comp_ratio_pairs

    class AutoModeParams:
        """
        Configuration parameters for auto-mode compression
        """

        def __init__(self, greedy_select_params: GreedySelectionParameters,
                     modules_to_ignore: Optional[List[torch.nn.Module]] = None):
            """
            :param greedy_select_params: Params for greedy comp-ratio selection algorithm
            :param modules_to_ignore: List of modules to ignore (None indicates nothing to ignore)
            """
            self.greedy_params = greedy_select_params
            self.modules_to_ignore = [] if modules_to_ignore is None else modules_to_ignore

    class Mode(Enum):
        """ Mode enumeration """

        manual = 1
        """ Manual mode: User specifies comp-ratio per layer """

        auto = 2
        """ Auto mode: AIMET computes optimal comp-ratio per layer """

    def __init__(self, data_loader: torch.utils.data.DataLoader, num_reconstruction_samples: int,
                 allow_custom_downsample_ops: bool,
                 mode: Mode, params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        self.data_loader = data_loader
        self.num_reconstruction_samples = num_reconstruction_samples
        self.allow_custom_downsample_ops = allow_custom_downsample_ops
        self.mode = mode
        self.mode_params = params
        self.multiplicity = multiplicity


class WeightSvdParameters:
    """ Configuration parameters for weight svd compression """

    class ManualModeParams:
        """
        Configuration parameters for manual-mode weight svd compression
        """

        def __init__(self, list_of_module_comp_ratio_pairs: List[ModuleCompRatioPair]):
            """
            :param list_of_module_comp_ratio_pairs: List of (module, comp-ratio) pairs
            """
            self.list_of_module_comp_ratio_pairs = list_of_module_comp_ratio_pairs

    class AutoModeParams:
        """
        Configuration parameters for auto-mode compression
        """

        def __init__(self,
                     rank_select_scheme: RankSelectScheme,
                     select_params: Union[GreedySelectionParameters,
                                          TarRankSelectionParameters],
                     modules_to_ignore: Optional[List[torch.nn.Module]] = None):
            """
            :param rank_select_scheme: supports two options greedy and tar
            :param select_params: Params for greedy/TAR comp-ratio selection algorithm
            :param modules_to_ignore: List of modules to ignore (None indicates nothing to ignore)
            """
            self.rank_select_scheme = rank_select_scheme
            self.select_params = select_params
            self.modules_to_ignore = [] if modules_to_ignore is None else modules_to_ignore

    class Mode(Enum):
        """ Mode enumeration """

        manual = 1
        """ Manual mode """

        auto = 2
        """ Auto mode """

    def __init__(self, mode: Mode, params: Union[ManualModeParams, AutoModeParams], multiplicity=1):
        """
        :param mode: Either auto mode or manual mode
        :param params: Parameters for the mode selected
        :param multiplicity: The multiplicity to which ranks/input channels will get rounded. Default: 1
        """
        self.mode = mode
        self.mode_params = params
        self.multiplicity = multiplicity


class PassThroughOp(torch.nn.Module):
    """
    This is a pass-through op, used for purpose of making an op a no-op
    """
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(inputx):
        """
        Forward pass for passthrough op
        """
        return inputx
