from typing import Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmrazor.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


@METRICS.register_module()
class Rank1(BaseMetric):
    r"""Rank1 evaluation metric.
    """
    default_prefix: Optional[str] = 'rank1'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.sample_idx_identical_mapping = {}

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            pred_label = data_sample['pred_label']
            gt_label = data_sample['gt_label']
            if 'score' in pred_label:
                result['pred_score'] = pred_label['score'].cpu()
            else:
                result['pred_label'] = pred_label['label'].cpu()
            result['gt_label'] = gt_label['label'].cpu()
            if data_sample['dataset_name'] not in self.sample_idx_identical_mapping:
                self.sample_idx_identical_mapping[data_sample['dataset_name']] = data_sample['sample_idx_identical_mapping']
            else:
                assert self.sample_idx_identical_mapping[data_sample['dataset_name']] == data_sample['sample_idx_identical_mapping']
            result['dataset_name'] = data_sample['dataset_name']
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        dataset_names = list(self.sample_idx_identical_mapping)
        for dataset_name in dataset_names:
            # concat
            dataset_results = []
            for res in results:
                if res['dataset_name'] == dataset_name:
                    dataset_results.append(res)
            target = torch.cat([res['gt_label'] for res in dataset_results])
            assert 'pred_score' in dataset_results[0]
            pred = torch.stack([res['pred_score'] for res in dataset_results])

            try:
                rank1 = self.calculate(pred, target, self.sample_idx_identical_mapping[dataset_name])
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                    '`test_evaluator` fields in your config file.')
            metrics[dataset_name] = rank1
        metrics['avg'] = sum(metrics.values()) / len(metrics)
        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        sample_idx_identical_mapping: Dict,
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the rank1.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            sample_idx_identical_mapping (Dict): The mapping from sample_idx
                to identical_idx.

        Returns:
            float: Rank1.
        """

        pred = to_tensor(pred)
        pred = pred / torch.linalg.norm(pred)
        target = to_tensor(target).to(torch.int64)
        cos_matrix = torch.matmul(pred, pred.t())
        num = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        all_nums = 0.0
        correct_nums = 0.0
        for idx in range(num):
            cosi = cos_matrix[idx] - 0.5
            identical_idxs = sample_idx_identical_mapping[idx]
            pred_idxs = cosi.argsort(descending=True)[:len(identical_idxs)]
            pred_idxs = pred_idxs.numpy().tolist()
            all_nums += len(identical_idxs)
            correct_nums += len(set(identical_idxs) & set(pred_idxs))
        results = 100.0 * correct_nums / all_nums
        return results
