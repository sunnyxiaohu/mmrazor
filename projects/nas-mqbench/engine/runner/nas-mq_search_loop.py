# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       disable_fake_quant, enable_observer)
except ImportError:
    from mmrazor.utils import get_placeholder

    disable_observer = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')

from mmengine import fileio
from mmengine.device import get_device
from mmengine.dist import broadcast_object_list
from mmengine.evaluator import Evaluator
from mmengine.runner import EpochBasedTrainLoop
from mmengine.utils import is_list_of
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS, MODELS, MODEL_WRAPPERS, TASK_UTILS
from mmrazor.structures import (Candidates, convert_fix_subnet,
                                export_fix_subnet)
from mmrazor.utils import SupportRandomSubnet
from mmrazor.engine.runner.utils import (CalibrateBNMixin,
                                         check_subnet_resources)


@LOOPS.register_module()
class NASMQSearchLoop(EpochBasedTrainLoop, CalibrateBNMixin):
    """Loop for evolution searching.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        max_epochs (int): Total searching epochs. Defaults to 20.
        max_keep_ckpts (int): The maximum checkpoints of searcher to keep.
            Defaults to 3.
        resume_from (str, optional): Specify the path of saved .pkl file for
            resuming searching.
        num_candidates (int): The length of candidate pool. Defaults to 50.
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to -1.
        constraints_range (Dict[str, Any]): Constraints to be used for
            screening candidates. Defaults to dict(flops=(0, 330)).
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to None.
        predictor_cfg (dict, Optional): Used for building a metric predictor.
            Defaults to None.
        score_key (str): Specify one metric in evaluation results to score
            candidates. Defaults to 'accuracy_top-1'.
        init_candidates (str, optional): The candidates file path, which is
            used to init `self.init_candidates`. Its format is usually in .pkl
            format and has `all_candidates` key. Defaults to None.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 mq_model: Optional[Dict] = None,
                 mq_model_wrapper_cfg: Optional[Dict] = None,
                 mq_calibrate_dataloader: Optional[Dict] = None,
                 mq_calibrate_steps: int = 16,
                 mq_init_candidates: Optional[str] = None,
                 max_epochs: int = 20,
                 max_keep_ckpts: int = 3,
                 resume_from: Optional[str] = None,
                 num_candidates: int = 50,
                 calibrate_sample_num: int = -1,
                 constraints_range: Dict[str, Any] = dict(flops=(0., 330.)),
                 estimator_cfg: Optional[Dict] = None,
                 predictor_cfg: Optional[Dict] = None,
                 score_key: str = 'accuracy/top1',
                 score_indicator: str = 'score',
                 dump_subnet: bool = False) -> None:
        super().__init__(runner, dataloader, max_epochs)
        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
        else:
            warnings.warn(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.')

        self.mq_model = mq_model
        self.mq_model_wrapper_cfg = mq_model_wrapper_cfg
        self.mq_calibrate_steps = mq_calibrate_steps
        if isinstance(mq_calibrate_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.mq_calibrate_dataloader = self.runner.build_dataloader(
                mq_calibrate_dataloader, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.mq_calibrate_dataloader = mq_calibrate_dataloader
        self.score_indicator = score_indicator
        Candidates._indicators = tuple(set(Candidates._indicators + (score_indicator, )))
        self.num_candidates = num_candidates
        self.constraints_range = constraints_range
        self.calibrate_sample_num = calibrate_sample_num
        self.score_key = score_key
        self.max_keep_ckpts = max_keep_ckpts
        self.resume_from = resume_from
        self.dump_subnet = dump_subnet

        if mq_init_candidates is None:
            self.mq_init_candidates = Candidates()
        else:
            self.mq_init_candidates = fileio.load(mq_init_candidates)['all_candidates']
            assert isinstance(self.mq_init_candidates, Candidates), 'please use the \
                correct init candidates file'
            assert len(self.mq_init_candidates) == self.num_candidates * self._max_epochs

        self.candidates = Candidates()
        self.all_candidates = Candidates()
        self.export_candidates = Candidates()

        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model

        # initialize estimator
        estimator_cfg = dict() if estimator_cfg is None else estimator_cfg
        if 'type' not in estimator_cfg:
            estimator_cfg['type'] = 'mmrazor.ResourceEstimator'
        self.estimator = TASK_UTILS.build(estimator_cfg)

        # initialize predictor
        self.use_predictor = False
        self.predictor_cfg = predictor_cfg
        if self.predictor_cfg is not None:
            self.predictor_cfg['score_key'] = self.score_key
            self.predictor_cfg['search_groups'] = \
                self.model.mutator.search_groups
            self.predictor = TASK_UTILS.build(self.predictor_cfg)

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_train')
        if self.predictor_cfg is not None:
            self._init_predictor()

        if self.resume_from:
            self._resume()

        while self._epoch < self._max_epochs:
            self.run_epoch()
            self._save_searcher_ckpt()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch.

        Steps:
            1. Sample some new candidates from the supernet. Then Append them
                to the candidates, Thus make its number equal to the specified
                number.
            2. Validate these candidates(step 1) and update their scores.
        """
        self.sample_candidates()
        self.update_candidates_scores()

        export_candidates = []
        subnets = self.candidates.subnets
        for idx, (subnet, item) in enumerate(zip(subnets, self.candidates.data)):
            self.model.mutator.set_choices(subnet)
            slice_weight = True if self.dump_subnet else False
            export_subnet, sliced_model = \
                export_fix_subnet(self.model, slice_weight=slice_weight)
            export_subnet = convert_fix_subnet(export_subnet)
            export_item = {}
            export_item[str(export_subnet)] = item[str(subnet)]
            export_candidates.append(export_item)

            if self.dump_subnet and self.runner.rank == 0:
                timestamp_subnet = time.strftime('%Y%m%d_%H%M', time.localtime())
                model_name = f'subnet_{len(self.all_candidates) + idx}_{timestamp_subnet}'
                torch.save({
                    'state_dict': sliced_model.state_dict(),
                    'meta': {}
                }, osp.join(self.runner.work_dir, f'{model_name}.pth'))
                fileio.dump(export_subnet,
                            osp.join(self.runner.work_dir, f'{model_name}.yaml'))
                self.runner.logger.info(f'Subnet {model_name} saved in '
                                        f'{self.runner.work_dir}')

        self.export_candidates.extend(Candidates(export_candidates))
        self.all_candidates.extend(self.candidates)
        self.candidates = Candidates()
        # TODO(shiguang): draw pareto front figure
        self._epoch += 1

    def sample_candidates(self) -> None:
        """Update candidate pool contains specified number of candicates."""
        if len(self.mq_init_candidates) > 0:
            start_index = self._epoch * self.num_candidates
            end_index = start_index + self.num_candidates
            self.candidates = Candidates(
                self.mq_init_candidates.data[start_index:end_index])
            return
        candidates_resources = []
        init_candidates = len(self.candidates)
        idx = 0
        if self.runner.rank == 0:
            while len(self.candidates) < self.num_candidates:
                idx += 1
                if idx == 1:
                    candidate = self.model.mutator.sample_choices('max')
                elif idx == 2:
                    candidate = self.model.mutator.sample_choices('min')
                else:
                    candidate = self.model.mutator.sample_choices()
                is_pass, result = self._check_constraints(
                    random_subnet=candidate)
                if is_pass:
                    self.candidates.append(candidate)
                    candidates_resources.append(result)
            self.candidates = Candidates(self.candidates.data)
        else:
            self.candidates = Candidates([dict(a=0)] * self.num_candidates)

        if len(candidates_resources) > 0:
            self.candidates.update_resources(
                candidates_resources,
                start=len(self.candidates.data) - len(candidates_resources))
            assert init_candidates + len(
                candidates_resources) == self.num_candidates

        # broadcast candidates to val with multi-GPUs.
        broadcast_object_list(self.candidates.data)

    def update_candidates_scores(self) -> None:
        """Validate candicate one by one from the candicate pool, and update
        top-k candicates."""
        for i, candidate in enumerate(self.candidates.subnets):
            self.model.mutator.set_choices(candidate)
            if self.mq_model is None:
                metrics = self._val_candidate(use_predictor=self.use_predictor)
            else:
                metrics = self._mq_val_candidate(use_predictor=self.use_predictor)
            score = round(metrics[self.score_key], 2) \
                if len(metrics) != 0 else 0.
            self.candidates.set_resource(i, score, self.score_indicator)
            indicators_str = ''
            for indicator in Candidates._indicators:
                indicators_str += f' {indicator.capitalize()}: {self.candidates.resources(indicator)[i]}'
            self.runner.logger.info(
                f'Epoch:[{self._epoch + 1}/{self._max_epochs}] '
                f'Candidate:[{i + 1}/{self.num_candidates}]'
                f'{indicators_str}')

    def _resume(self):
        """Resume searching."""
        if self.runner.rank == 0:
            searcher_resume = fileio.load(self.resume_from)
            for k in searcher_resume.keys():
                setattr(self, k, searcher_resume[k])
            epoch_start = int(searcher_resume['_epoch'])
            self.runner.logger.info('#' * 100)
            self.runner.logger.info(f'Resume from epoch: {epoch_start}')
            self.runner.logger.info('#' * 100)

    @torch.no_grad()
    def _val_candidate(self, use_predictor: bool = False) -> Dict:
        """Run validation.

        Args:
            use_predictor (bool): Whether to use predictor to get metrics.
                Defaults to False.
        """
        if use_predictor:
            assert self.predictor is not None
            metrics = self.predictor.predict(self.model)
        else:
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_statistics(self.runner.train_dataloader,
                                             self.calibrate_sample_num)
            self.runner.model.eval()
            for data_batch in self.dataloader:
                outputs = self.runner.model.val_step(data_batch)
                self.evaluator.process(
                    data_samples=outputs, data_batch=data_batch)
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        return metrics

    @torch.no_grad()
    def _mq_val_candidate(self, use_predictor: bool = False) -> Dict:
        """Run validation.

        Args:
            use_predictor (bool): Whether to use predictor to get metrics.
                Defaults to False.
        """
        if use_predictor:
            assert self.predictor is not None
            metrics = self.predictor.predict(self.model)
        else:
            if self.calibrate_sample_num > 0:
                self.calibrate_bn_statistics(self.runner.train_dataloader,
                                             self.calibrate_sample_num)
            fix_subnet, sliced_model = \
                export_fix_subnet(self.model.architecture, slice_weight=True)
            mq_model = copy.deepcopy(self.mq_model)
            mq_model['architecture'] = sliced_model
            mq_model = MODELS.build(mq_model)
            mq_model = mq_model.to(get_device())
            default_args = dict(
                module=mq_model, device_ids=[int(os.environ['LOCAL_RANK'])])
            mq_model = MODEL_WRAPPERS.build(
                self.mq_model_wrapper_cfg, default_args=default_args)

            mq_model.eval()
            mq_model.apply(enable_fake_quant)
            mq_model.apply(enable_observer)
            for idx, data_batch in enumerate(self.mq_calibrate_dataloader):
                if idx == self.mq_calibrate_steps:
                    break
                mq_model.calibrate_step(data_batch)
            mq_model.apply(enable_fake_quant)
            mq_model.apply(disable_observer)
            for data_batch in self.dataloader:
                outputs = mq_model.val_step(data_batch)
                self.evaluator.process(
                    data_samples=outputs, data_batch=data_batch)
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            
        return metrics

    def _save_searcher_ckpt(self) -> None:
        """Save searcher ckpt, which is different from common ckpt.

        It mainly contains the candicate pool, the top-k candicates with scores
        and the current epoch.
        """
        if self.runner.rank == 0:
            save_for_resume = dict()
            save_for_resume['_epoch'] = self._epoch
            for k in ['export_candidates', 'all_candidates']:
                save_for_resume[k] = getattr(self, k)
            fileio.dump(
                save_for_resume,
                osp.join(self.runner.work_dir,
                         f'search_epoch_{self._epoch}.pkl'))

            if self.max_keep_ckpts > 0:
                cur_ckpt = self._epoch + 1
                redundant_ckpts = range(1, cur_ckpt - self.max_keep_ckpts)
                for _step in redundant_ckpts:
                    ckpt_path = osp.join(self.runner.work_dir,
                                         f'search_epoch_{_step}.pkl')
                    if osp.isfile(ckpt_path):
                        os.remove(ckpt_path)

    def _check_constraints(
            self, random_subnet: SupportRandomSubnet) -> Tuple[bool, Dict]:
        """Check whether is beyond constraints.

        Returns:
            bool, result: The result of checking.
        """
        if random_subnet in self.all_candidates.subnets or (
                        random_subnet in self.candidates.subnets):
            return False, dict()

        is_pass, results = check_subnet_resources(
            model=self.model,
            subnet=random_subnet,
            estimator=self.estimator,
            constraints_range=self.constraints_range)

        return is_pass, results

    def _init_predictor(self):
        """Initialize predictor, training is required."""
        if self.predictor.handler_ckpt:
            self.predictor.load_checkpoint()
            self.runner.logger.info(
                f'Loaded Checkpoints from {self.predictor.handler_ckpt}')
        else:
            self.runner.logger.info('No predictor checkpoints found. '
                                    'Start pre-training the predictor.')
            if isinstance(self.predictor.train_samples, str):
                self.runner.logger.info('Find specified samples in '
                                        f'{self.predictor.train_samples}')
                train_samples = fileio.load(self.predictor.train_samples)
                self.candidates = train_samples['subnets']
            else:
                self.runner.logger.info(
                    'Without specified samples. Start random sampling.')
                temp_num_candidates = self.num_candidates
                self.num_candidates = self.predictor.train_samples

                assert self.use_predictor is False, (
                    'Real evaluation is required when initializing predictor.')
                self.sample_candidates()
                self.update_candidates_scores()
                self.num_candidates = temp_num_candidates

            inputs = []
            for candidate in self.candidates.subnets:
                inputs.append(self.predictor.model2vector(candidate))
            inputs = np.array(inputs)
            labels = np.array(self.candidates.scores)
            self.predictor.fit(inputs, labels)
            if self.runner.rank == 0:
                predictor_dir = self.predictor.save_checkpoint(
                    osp.join(self.runner.work_dir, 'predictor'))
                self.runner.logger.info(
                    f'Predictor pre-trained, saved in {predictor_dir}.')
            self.use_predictor = True
            self.candidates = Candidates()
