# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Set

from torch import nn

from mmrazor.models.mutables.mutable_channel import MutableChannel

MUTABLE_CFG_TYPE = Dict[str, Any]
MUTABLE_CFGS_TYPE = Dict[str, MUTABLE_CFG_TYPE]


class DynamicOP(ABC):
    accepted_mutable_keys: Set[str] = set()

    @abstractmethod
    def to_static_op(self) -> nn.Module:
        ...

    @classmethod
    def parse_mutable_cfgs(
            cls, mutable_cfgs: MUTABLE_CFGS_TYPE) -> MUTABLE_CFGS_TYPE:
        parsed_mutable_cfgs = dict()

        for mutable_key in mutable_cfgs.keys():
            if mutable_key in cls.accepted_mutable_keys:
                parsed_mutable_cfgs[mutable_key] = copy.deepcopy(
                    mutable_cfgs[mutable_key])
        if len(parsed_mutable_cfgs) == 0:
            raise ValueError(
                f'Expected mutable keys: {cls.accepted_mutable_keys}, '
                f'but got: {list(mutable_cfgs.keys())}')

        return parsed_mutable_cfgs


class ChannelDynamicOP(DynamicOP):

    @property
    @abstractmethod
    def mutable_in(self) -> MutableChannel:
        ...

    @property
    def mutable_out(self) -> MutableChannel:
        ...
