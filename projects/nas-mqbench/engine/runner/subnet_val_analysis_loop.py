# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from mmengine.evaluator import Evaluator
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.structures import export_fix_subnet
from mmrazor.registry import LOOPS
from mmrazor.engine import SubnetValLoop


def draw_params_boxplot(model, filename, topk_params = 10):
    candidate_params = [(name, param) for name, param in model.named_parameters() if param.dim() >= 2]
    candidate_params = candidate_params[:topk_params]
    cols = int(np.sqrt(len(candidate_params)))
    rows = len(candidate_params) // cols + (len(candidate_params) % cols > 0)
    fig, ax = plt.subplots(rows, cols, figsize=(20, 30))
    for idx, (name, param) in enumerate(candidate_params):
        # params axis: out_channel, in_channel, kernel_size, kernel_size
        data = param.flatten(start_dim=1).detach().cpu().numpy()
        row, col = idx // cols, idx % cols
        ax[row, col].boxplot(data)
        ax[row, col].set_title(f'{name}')
        ax[row, col].set_xlabel('Channel Index')
        ax[row, col].set_ylabel('Value')
    # plt.show()
    fig.savefig(filename)
    plt.close(fig)


@LOOPS.register_module()
class SubnetValAnalysisLoop(SubnetValLoop):
    """Loop for subnet validation in NAS with BN re-calibration.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
        evaluate_fixed_subnet (bool): Whether to evaluate a fixed subnet only
            or not. Defaults to False.
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to 4096.
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to dict(type='mmrazor.ResourceEstimator').
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        fp16: bool = False,
        evaluate_fixed_subnet: bool = False,
        calibrate_sample_num: int = 4096,
        estimator_cfg: Optional[Dict] = dict(type='mmrazor.ResourceEstimator'),
        topk_params: Optional[int] = 10,
        params_modes: Optional[tuple] = ('org', 'fuse_conv_bn', 'cle'),
    ) -> None:
        super().__init__(runner, dataloader, evaluator, fp16, evaluate_fixed_subnet,
            calibrate_sample_num, estimator_cfg)
        self.topk_params = topk_params
        self.input_shapes = (1, ) + tuple(self.dataloader.dataset[0]['inputs'].shape)
        self.params_modes = params_modes

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')

        all_metrics = dict()

        if self.evaluate_fixed_subnet:
            metrics = self._evaluate_once()
            all_metrics.update(add_prefix(metrics, 'fix_subnet'))
        elif hasattr(self.model, 'sample_kinds'):
            for kind in ['max', 'min']:  # self.model.sample_kinds:
                if kind == 'max':
                    self.model.mutator.set_max_choices()
                    metrics = self._evaluate_once(kind=kind)
                    all_metrics.update(add_prefix(metrics, 'max_subnet'))
                elif kind == 'min':
                    self.model.mutator.set_min_choices()
                    metrics = self._evaluate_once(kind=kind)
                    all_metrics.update(add_prefix(metrics, 'min_subnet'))
                elif 'random' in kind:
                    self.model.mutator.set_choices(
                        self.model.mutator.sample_choices())
                    metrics = self._evaluate_once(kind=kind)
                    all_metrics.update(add_prefix(metrics, f'{kind}_subnet'))
        # import pdb; pdb.set_trace()
        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')

    def _evaluate_once(self, kind='') -> Dict:
        """Evaluate a subnet once with BN re-calibration."""
        if self.calibrate_sample_num > 0:
            self.calibrate_bn_statistics(self.runner.train_dataloader,
                                         self.calibrate_sample_num)
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        if self.runner.distributed:
            model = self.runner.model.module
        else:
            model = self.runner.model
        _, sliced_model = export_fix_subnet(
            model, slice_weight=True)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        resource_metrics = self.estimator.estimate(sliced_model)
        # import pdb; pdb.set_trace()
        for mode in self.params_modes:
            filename = f'{self.runner.log_dir}/subnet-{kind}_{mode}_params_boxplot_ep{self.runner.epoch}.png'
            if mode == 'fuse_conv_bn':
                from mqbench.cle_superacme.batch_norm_fold import fold_all_batch_norms
                folded_pairs = fold_all_batch_norms(sliced_model, self.input_shapes)
            elif mode == 'cle':
                from mqbench.cle_superacme.cle import apply_cross_layer_equalization
                apply_cross_layer_equalization(model=sliced_model, input_shape=self.input_shapes)
            draw_params_boxplot(sliced_model, filename, topk_params=self.topk_params)

        metrics.update(resource_metrics)
        # if kind == 'max' or kind == '':
        #     metrics.update(dict(model=sliced_model))

        return metrics
