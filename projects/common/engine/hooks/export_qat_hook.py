# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Sequence,Union,List,Callable
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.structures import BaseDataElement
from mmengine.dist import is_main_process
from mmengine.hooks import CheckpointHook
import logging
from mmengine.logging import print_log
from mmengine.utils import is_list_of, is_seq_of
from math import inf
from mmengine.fileio import FileClient, get_file_backend
import os.path as osp
from mmengine.dist import master_only

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class ExportQATHook(Hook):
    """Estimate model resources periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Defaults to -1, which means "never".
        by_epoch (bool): Saving qatonnx by epoch or by iteration.
            Default to True.

    Example:
    >>> add the `ExportQATHook` in custom_hooks as follows:
        custom_hooks = [
            dict(type='mmrazor.ExportQATHook',
                 interval=1,
                 by_epoch=True,
                 )
        ]
    """
    out_dir: str

    priority = 'VERY_LOW'

    # logic to save best checkpoints
    # Since the key for determining greater or less is related to the
    # downstream tasks, downstream repositories may need to overwrite
    # the following inner variables accordingly.

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 save_param_scheduler: bool = True,
                 out_dir: Optional[Union[str, Path]] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 save_best: Union[str, List[str], None] = None,
                 rule: Union[str, List[str], None] = None,
                 greater_keys: Optional[Sequence[str]] = None,
                 less_keys: Optional[Sequence[str]] = None,
                 file_client_args: Optional[dict] = None,
                 filename_tmpl: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 published_keys: Union[str, List[str], None] = None,
                 **kwargs) -> None:
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_param_scheduler = save_param_scheduler
        self.out_dir = out_dir  # type: ignore
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs

        if file_client_args is not None:
            print_log(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead',
                logger='current',
                level=logging.WARNING)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

        self.file_client_args = file_client_args
        self.backend_args = backend_args

        if filename_tmpl is None:
            if self.by_epoch:
                self.filename_tmpl = 'epoch_{}.onnx'
            else:
                self.filename_tmpl = 'iter_{}.onnx'
        else:
            self.filename_tmpl = filename_tmpl

        # save best logic
        assert (isinstance(save_best, str) or is_list_of(save_best, str)
                or (save_best is None)), (
                    '"save_best" should be a str or list of str or None, '
                    f'but got {type(save_best)}')

        if isinstance(save_best, list):
            if 'auto' in save_best:
                assert len(save_best) == 1, (
                    'Only support one "auto" in "save_best" list.')
            assert len(save_best) == len(
                set(save_best)), ('Find duplicate element in "save_best".')
        else:
            # convert str to list[str]
            if save_best is not None:
                save_best = [save_best]  # type: ignore # noqa: F401
        self.save_best = save_best

        # rule logic
        assert (isinstance(rule, str) or is_list_of(rule, str)
                or (rule is None)), (
                    '"rule" should be a str or list of str or None, '
                    f'but got {type(rule)}')
        if isinstance(rule, list):
            # check the length of rule list
            assert len(rule) in [
                1,
                len(self.save_best)  # type: ignore
            ], ('Number of "rule" must be 1 or the same as number of '
                f'"save_best", but got {len(rule)}.')
        else:
            # convert str/None to list
            rule = [rule]  # type: ignore # noqa: F401

        if greater_keys is None:
            self.greater_keys = self._default_greater_keys
        else:
            if not isinstance(greater_keys, (list, tuple)):
                greater_keys = (greater_keys, )  # type: ignore
            assert is_seq_of(greater_keys, str)
            self.greater_keys = greater_keys  # type: ignore

        if less_keys is None:
            self.less_keys = self._default_less_keys
        else:
            if not isinstance(less_keys, (list, tuple)):
                less_keys = (less_keys, )  # type: ignore
            assert is_seq_of(less_keys, str)
            self.less_keys = less_keys  # type: ignore

        if self.save_best is not None:
            self.is_better_than: Dict[str, Callable] = dict()
            self._init_rule(rule, self.save_best)
            if len(self.key_indicators) == 1:
                self.best_onnx_path: Optional[str] = None
            else:
                self.best_onnx_path_dict: Dict = dict()
                
    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.work_dir

        # If self.file_client_args is None, self.file_client will not
        # used in CheckpointHook. To avoid breaking backward compatibility,
        # it will not be removed util the release of MMEngine1.0
        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = self.file_client

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(
                self.out_dir, basename)  # type: ignore  # noqa: E501

        runner.logger.info(f'onnx will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                if 'best_onnx' not in runner.message_hub.runtime_info:
                    self.best_onnx_path = None
                else:
                    self.best_onnx_path = runner.message_hub.get_info(
                        'best_onnx')
            else:
                for key_indicator in self.key_indicators:
                    best_onnx_name = f'best_onnx_{key_indicator}'
                    if best_onnx_name not in runner.message_hub.runtime_info:
                        self.best_onnx_path_dict[key_indicator] = None
                    else:
                        self.best_onnx_path_dict[
                            key_indicator] = runner.message_hub.get_info(
                                best_onnx_name)
    def is_in_list(self,a:str,b:List[str])->bool:
         flag =False
         for value in b:
             if a in value:
                 flag =True
         return flag   

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Estimate model resources after every n val epochs.

        Args:
            runner (Runner): The runner of the training process.
        """
        # if not self.by_epoch:
        #     return

        # if self.every_n_epochs(runner, self.interval):
        #     self.export_onnx_with_cliprange(runner)
        if len(metrics) == 0:
            runner.logger.warning(
                'Since `metrics` is an empty dict, the behavior to save '
                'the best onnx will be skipped in this evaluation.')
            return
        if self.is_in_list('qat',list(metrics.keys())):
            self._save_best_onnx(runner, metrics)

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence[BaseDataElement]] = None) \
            -> None:
        """Estimate model resources after every n val iters.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.by_epoch:
            return

        if self.every_n_train_iters(runner, self.interval):
            self.export_onnx_with_cliprange(runner)
            
    @torch.no_grad()
    @master_only
    def export_onnx_with_cliprange(self, runner,best_onnx_path) -> None:
        model = runner.model.module if runner.distributed else runner.model
        # model = copy.deepcopy(model)
        observed_model=model.get_deploy_model()
        model.quantizer.export_onnx(observed_model,model.dummy_input.cuda(),best_onnx_path)
        runner.logger.info(f'Export onnx with clipranges.json for deploy for mnn')

    def _save_best_onnx(self, runner, metrics) -> None:
        """Save the current checkpoint and delete outdated checkpoint.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics.
        """
        if not self.save_best:
            return

        if self.by_epoch:
            onnx_filename = self.filename_tmpl.format(runner.epoch)
            cur_type, cur_time = 'epoch', runner.epoch
        else:
            onnx_filename = self.filename_tmpl.format(runner.iter)
            cur_type, cur_time = 'iter', runner.iter

        # handle auto in self.key_indicators and self.rules before the loop
        if 'auto' in self.key_indicators:
            self._init_rule(self.rules, [list(metrics.keys())[0]])

        # save best logic
        # get score from messagehub
        for key_indicator, rule in zip(self.key_indicators, self.rules):
            key_score = metrics[key_indicator]

            if len(self.key_indicators) == 1:
                best_score_key = 'best_score'
                runtime_best_onnx_key = 'best_onnx'
                best_onnx_path = self.best_onnx_path
            else:
                best_score_key = f'best_score_{key_indicator}'
                runtime_best_onnx_key = f'best_onnx_{key_indicator}'
                best_onnx_path = self.best_onnx_path_dict[key_indicator]

            if best_score_key not in runner.message_hub.runtime_info:
                best_score = self.init_value_map[rule]
            else:
                best_score = runner.message_hub.get_info(best_score_key)

            if key_score is None or not self.is_better_than[key_indicator](
                    key_score, best_score):
                continue

            best_score = key_score
            runner.message_hub.update_info(best_score_key, best_score)

            if best_onnx_path and \
               self.file_client.isfile(best_onnx_path) and \
               is_main_process():
                self.file_client.remove(best_onnx_path)
                runner.logger.info(
                    f'The previous best onnx {best_onnx_path} '
                    'is removed')
                best_onnx_path_symbolic = best_onnx_path.replace('.onnx', '_superacme_symbolic.onnx')
                best_onnx_path_json = best_onnx_path.replace('.onnx','_superacme_clip_ranges.json')
                self.file_client.remove(best_onnx_path_symbolic)
                self.file_client.remove(best_onnx_path_json)

            best_onnx_name = f'best_{key_indicator}_{onnx_filename}'
            # Replace illegal characters for filename with `_`
            best_onnx_name = best_onnx_name.replace('/', '_')
            if len(self.key_indicators) == 1:
                self.best_onnx_path = self.file_client.join_path(  # type: ignore # noqa: E501
                    self.out_dir, best_onnx_name)
                runner.message_hub.update_info(runtime_best_onnx_key,
                                               self.best_onnx_path)
            else:
                self.best_onnx_path_dict[
                    key_indicator] = self.file_client.join_path(  # type: ignore # noqa: E501
                        self.out_dir, best_onnx_name)
                runner.message_hub.update_info(
                    runtime_best_onnx_key,
                    self.best_onnx_path_dict[key_indicator])
            self.export_onnx_with_cliprange(runner,self.best_onnx_path)
            runner.logger.info(
                f'The best onnx with {best_score:0.4f} {key_indicator} '
                f'at {cur_time} {cur_type} is saved to {best_onnx_name}.')
            
    def _init_rule(self, rules, key_indicators) -> None:
        """Initialize rule, key_indicator, comparison_func, and best score. If
        key_indicator is a list of string and rule is a string, all metric in
        the key_indicator will share the same rule.

        Here is the rule to determine which rule is used for key indicator when
        the rule is not specific (note that the key indicator matching is case-
        insensitive):

        1. If the key indicator is in ``self.greater_keys``, the rule
            will be specified as 'greater'.
        2. Or if the key indicator is in ``self.less_keys``, the rule
            will be specified as 'less'.
        3. Or if any one item in ``self.greater_keys`` is a substring of
            key_indicator, the rule will be specified as 'greater'.
        4. Or if any one item in ``self.less_keys`` is a substring of
            key_indicator, the rule will be specified as 'less'.

        Args:
            rule (List[Optional[str]]): Comparison rule for best score.
            key_indicator (List[str]): Key indicator to determine
                the comparison rule.
        """
        if len(rules) == 1:
            rules = rules * len(key_indicators)

        self.rules = []
        for rule, key_indicator in zip(rules, key_indicators):

            if rule not in self.rule_map and rule is not None:
                raise KeyError('rule must be greater, less or None, '
                               f'but got {rule}.')

            if rule is None and key_indicator != 'auto':
                # `_lc` here means we use the lower case of keys for
                # case-insensitive matching
                key_indicator_lc = key_indicator.lower()
                greater_keys = {key.lower() for key in self.greater_keys}
                less_keys = {key.lower() for key in self.less_keys}

                if key_indicator_lc in greater_keys:
                    rule = 'greater'
                elif key_indicator_lc in less_keys:
                    rule = 'less'
                elif any(key in key_indicator_lc for key in greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator_lc for key in less_keys):
                    rule = 'less'
                else:
                    raise ValueError('Cannot infer the rule for key '
                                     f'{key_indicator}, thus a specific rule '
                                     'must be specified.')
            if rule is not None:
                self.is_better_than[key_indicator] = self.rule_map[rule]
            self.rules.append(rule)

        self.key_indicators = key_indicators