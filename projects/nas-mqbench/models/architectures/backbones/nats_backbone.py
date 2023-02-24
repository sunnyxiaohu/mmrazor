import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmrazor.registry import MODELS

try:
    from mmcls.models.backbones.base_backbone import BaseBackbone
except ImportError:
    from mmrazor.utils import get_placeholder
    BaseBackbone = get_placeholder('mmcls')

try:
    import xautodl
except ImportError:
    from mmrazor.utils import get_placeholder
    xautodl = get_placeholder('xautodl')

try:
    import nats_bench
except ImportError:
    from mmrazor.utils import get_placeholder
    nats_bench = get_placeholder('nats_bench')


@MODELS.register_module()
class NATSBackbone(BaseBackbone):

    def __init__(
        self,
        benchmark_api: Union[Dict, object],
        arch_index: str,
        dataset: str,
        seed: Optional[int] = None,
        hp: Optional[str] = '12',
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='Kaiming', layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        super(NATSBackbone, self).__init__(init_cfg)
        if isinstance(benchmark_api, Dict):
            benchmark_api = nats_bench.create(**benchmark_api)
        self.benchmark_api = benchmark_api
        self.arch_index = arch_index
        self.dataset = dataset
        self.seed = seed
        self.hp = hp

        arch_cfg = benchmark_api.get_net_config(arch_index, dataset)
        from xautodl.models import get_cell_based_tiny_net
        self.nats_model = get_cell_based_tiny_net(arch_cfg)
        # Note that when in `val` mode, we have to call `init_weight` manually.
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        # Load the pre-trained weights.
        params = self.benchmark_api.get_net_param(
            self.arch_index, self.dataset, seed=self.seed, hp=self.hp)
        self.nats_model.load_state_dict(params, strict=True)

    def forward(self, x):
        outs = self.nats_model(x)
        return outs
