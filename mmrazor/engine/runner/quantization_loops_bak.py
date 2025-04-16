# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop
from mmengine.dist import get_dist_info

try:
    from torch.ao.quantization import (disable_observer, enable_fake_quant,
                                       enable_observer, disable_fake_quant)
    from torch.nn.intrinsic.qat import freeze_bn_stats
except ImportError:
    from mmrazor.utils import get_placeholder
    disable_fake_quant = get_placeholder('torch>=1.13')
    disable_observer = get_placeholder('torch>=1.13')
    enable_fake_quant = get_placeholder('torch>=1.13')
    enable_observer = get_placeholder('torch>=1.13')
    freeze_bn_stats = get_placeholder('torch>=1.13')

from mmengine.dist import all_reduce_params, is_distributed
from torch.utils.data import DataLoader

from mmrazor.models import register_torch_fake_quants, register_torch_observers
from mmrazor.models.fake_quants import (enable_param_learning,
                                        enable_static_estimate, enable_val,
                                        enable_static_observation)
from mmrazor.registry import LOOPS

TORCH_observers = register_torch_observers()
TORCH_fake_quants = register_torch_fake_quants()


class QuantizationHook:
    """ 量化 Hook 记录每层量化前后的数据差异 """
    def __init__(self):
        self.diffs = {}  # 记录每个层的误差
        self.hook_handles = {}  # 存储 hook 句柄，便于移除

    def hook_fn(self, module, input, output, module_name):
        """
        Hook 计算量化误差
        Args:
            module: 当前 FakeQuantize 模块
            input: 进入 forward 的数据 (Tuple)
            output: forward 计算后的输出数据
            module_name: 该层的名称
        """
        X = input[0].detach()  # 量化前数据
        X_quantized = output.detach()  # 量化后数据

        # 计算误差
        l1_error = torch.abs(X - X_quantized).mean()  # L1 误差
        l2_error = torch.norm(X - X_quantized, p=2)  # L2 误差
        # 计算余弦相似度（Cosine Similarity）
        cosine_sim = F.cosine_similarity(X.view(1, -1), X_quantized.view(1, -1)).mean()

        # 记录误差信息
        self.diffs[module_name] = {'L1': l1_error.item(), 'L2': l2_error.item(), 'Cosine': cosine_sim.item()}

        # 打印信息
        print(f"[Quantization Hook - {module_name}] L1: {l1_error.item():.6f}, L2: {l2_error.item():.6f}, Cosine: {cosine_sim.item():.6f}")

    def attach(self, model):
        """ 遍历 `model` 并为所有 `QNotationFakeQuantize` 层注册 Hook """
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'QNotationFakeQuantize':
                # 注册 Hook，并传入该层的名称
                handle = module.register_forward_hook(
                    lambda mod, inp, out, n=name: self.hook_fn(mod, inp, out, n)
                )
                self.hook_handles[name] = handle  # 记录句柄

    def remove(self):
        """ 移除所有 Hook """
        for name, handle in self.hook_handles.items():
            handle.remove()
        self.hook_handles.clear()


class ModelQuantizationEvaluator:
    """ 计算整个模型的量化误差（L1, L2, Cosine） """

    def __init__(self, model):
        self.model = model
        self.fp32_outputs = {}  # 存储浮点推理的每层输出
        self.quant_outputs = {}  # 存储量化推理的每层输出
        self.hook_handles = {}

    def hook_fn(self, module, input, output, module_name, mode):
        """
        Hook 记录每层前向输出。
        Args:
            module_name: 当前层的名称
            mode: "fp32" or "quant"
        """
        if mode == "fp32":
            self.fp32_outputs[module_name] = output.detach().cpu()
        elif mode == "quant":
            self.quant_outputs[module_name] = output.detach().cpu()

    def attach_hooks(self, mode):
        """ 遍历 `model` 并注册 Hook """
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'QNotationFakeQuantize':
                # 注册 Hook，区分 FP32 和量化模式
                handle = module.register_forward_hook(
                    lambda mod, inp, out, n=name, m=mode: self.hook_fn(mod, inp, out, n, m)
                )
                self.hook_handles[name] = handle

    def remove_hooks(self):
        """ 移除所有 Hook """
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()

    def compare_outputs(self):
        """ 计算每层的量化误差（L1, L2, Cosine Similarity） """
        diffs = {}
        for name in self.fp32_outputs.keys():
            X_fp32 = self.fp32_outputs[name]
            X_quant = self.quant_outputs[name]

            # 计算误差
            l1_error = torch.abs(X_fp32 - X_quant).mean()
            l2_error = torch.norm(X_fp32 - X_quant, p=2)
            cosine_sim = F.cosine_similarity(X_fp32.view(1, -1), X_quant.view(1, -1)).mean()

            diffs[name] = {'L1': l1_error.item(), 'L2': l2_error.item(), 'Cosine': cosine_sim.item()}
            print(f"[{name}] L1: {l1_error:.6f}, L2: {l2_error:.6f}, Cosine: {cosine_sim:.6f}")

        return diffs

    def evaluate(self, data_batch):
        """ 运行完整的量化误差评估流程 """
        # 2️⃣ 关闭量化，记录 FP32 结果
        self.model.apply(disable_fake_quant)
        self.attach_hooks(mode="fp32")
        with torch.no_grad():
            _  = self.model.val_step(data_batch)
        self.remove_hooks()

        # 3️⃣ 开启量化，记录量化结果
        self.model.apply(enable_fake_quant)
        self.attach_hooks(mode="quant")
        with torch.no_grad():
           _  = self.model.val_step(data_batch)
        self.remove_hooks()

        # 4️⃣ 计算误差
        return self.compare_outputs()


@LOOPS.register_module()
class QATEpochBasedLoop(EpochBasedTrainLoop):
    """`EpochBasedLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        disable_observer_begin (int): The number of total epochs to update
            observers. Defaults to -1, which means observers are enabled
            all the time.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats. Defaults to -1, which means no need to freeze bn.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            calibrate_dataloader: Union[DataLoader, Dict] = None,
            val_begin: int = 1,
            val_interval: int = 1,
            disable_observer_begin: int = -1,
            freeze_bn_begin: int = -1,
            is_first_batch: bool = True,
            calibrate_steps: int = -1,
            onnx_node_tensor_translate_mapping = None,
            onnx_node_debug_mode = False,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval, dynamic_intervals)
        self._is_first_batch = is_first_batch
        self.disable_observer_begin = disable_observer_begin
        self.freeze_bn_begin = freeze_bn_begin
        self.calibrate_steps = calibrate_steps
        self.calibrate_dataloader = self._build_calibrate_dataloader(calibrate_dataloader)
        self.onnx_node_tensor_translate_mapping = onnx_node_tensor_translate_mapping
        self.onnx_node_debug_mode = onnx_node_debug_mode

    def _build_calibrate_dataloader(self, dataloader):
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            cali_dataloader = self.runner.build_dataloader(
                dataloader, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            cali_dataloader = self.dataloader
        return cali_dataloader

    def export_ptq(self):
        # import pdb; pdb.set_trace()
        if self.runner.distributed:
            rank, world_size = get_dist_info()
            if rank==0:
                observed_model = self.runner.model.module.get_deploy_model()
                self.dummy_input = torch.randn(self.runner.model.module.input_shapes)
                self.runner.model.module.quantizer.export_onnx(
                    observed_model, self.dummy_input.cuda(), os.path.join(self.runner.work_dir,'ptq.onnx'),
                    onnx_node_tensor_translate_mapping=self.onnx_node_tensor_translate_mapping,
                    debug_mode=self.onnx_node_debug_mode)
        else:
            observed_model = self.runner.model.get_deploy_model()
            self.dummy_input = torch.randn(self.runner.model.input_shapes)
            self.runner.model.quantizer.export_onnx(
                observed_model, self.dummy_input.cuda(), os.path.join(self.runner.work_dir,'ptq.onnx'),
                onnx_node_tensor_translate_mapping=self.onnx_node_tensor_translate_mapping,
                debug_mode=self.onnx_node_debug_mode)

    @property
    def is_first_batch(self):
        return self._epoch == 0 and self._is_first_batch

    def prepare_for_run_epoch(self):
        """Toggle the state of the observers and fake quantizers before qat
        training."""
        if self.is_first_batch and self.calibrate_steps != -1:
            # lsq observer init
            # import pdb; pdb.set_trace()  #, TDL: whether need to turn to `eval` mode?
            self.runner.model.eval()
            self.runner.model.apply(disable_fake_quant)
            self.runner.model.apply(enable_observer)
            print_log('Start calibration...', logger='current')
            for idx, data_batch in enumerate(self.calibrate_dataloader):
                if idx == self.calibrate_steps:
                    break
                _ = self.runner.model.calibrate_step(data_batch)

            if self.runner.distributed:
                all_reduce_params(
                    self.runner.model.parameters(), op='mean')
                all_reduce_params(self.runner.model.buffers(), op='mean')
            self.runner.model.sync_qparams(src_mode='predict')
            print_log('Finish calibration!', logger='current')

            self.runner.save_checkpoint(self.runner.work_dir, 'ptq.pth')
            print_log('save ptq checkpoin after calibration!')
            self.export_ptq()
            self.prepare_for_val()
            self.runner.val_loop.run()
            self.runner.model.train()

        self.runner.model.apply(enable_fake_quant)

        # The initialized _epoch equals to 0 so _epoch + 1
        # equal to the current epoch
        if (self.disable_observer_begin > 0
                and self._epoch + 1 >= self.disable_observer_begin):
            self.runner.model.apply(disable_observer)
        else:
            self.runner.model.apply(enable_observer)

        if (self.freeze_bn_begin > 0
                and self._epoch + 1 >= self.freeze_bn_begin):
            self.runner.model.apply(freeze_bn_stats)

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_fake_quant)
        self.runner.model.apply(disable_observer)

    def run(self):
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs:
            self.prepare_for_run_epoch()
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.model.sync_qparams(src_mode='loss')
        # Make sure the registered buffer such as `observer_enabled` is
        # correct in the saved checkpoint.
        self.prepare_for_val()
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class LSQEpochBasedLoop(QATEpochBasedLoop):
    """`EpochBasedLoop` for `LEARNED STEP SIZE QUANTIZATION`

    Paper: Learned Step Size Quantization. <https://arxiv.org/abs/1902.08153>

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats. Defaults to -1, which means no need to freeze bn.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            calibrate_dataloader:Union[DataLoader, Dict] = None,
            val_begin: int = 1,
            val_interval: int = 1,
            freeze_bn_begin: int = -1,
            is_first_batch: bool = True,
            calibrate_steps: int = -1,
            calibrate_open_fakequant: bool = False,
            onnx_node_tensor_translate_mapping = None,
            onnx_node_debug_mode = False,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(
            runner,
            dataloader,
            max_epochs,
            val_begin,
            val_interval,
            freeze_bn_begin=freeze_bn_begin,
            dynamic_intervals=dynamic_intervals)

        self._is_first_batch = is_first_batch
        self.distributed = is_distributed()
        self.calibrate_steps = calibrate_steps
        self.calibrate_dataloader = self._build_calibrate_dataloader(calibrate_dataloader)
        self.calibrate_open_fakequant = calibrate_open_fakequant
        self.onnx_node_tensor_translate_mapping = onnx_node_tensor_translate_mapping
        self.onnx_node_debug_mode = onnx_node_debug_mode

    def _build_calibrate_dataloader(self, dataloader):
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            cali_dataloader = self.runner.build_dataloader(
                dataloader, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            cali_dataloader = self.dataloader
        return cali_dataloader

    def export_ptq(self):
        # import pdb; pdb.set_trace()
        if self.runner.distributed:
            rank, world_size = get_dist_info()
            if rank==0:
                observed_model = self.runner.model.module.get_deploy_model()
                self.dummy_input = torch.randn(self.runner.model.module.input_shapes)
                self.runner.model.module.quantizer.export_onnx(
                    observed_model, self.dummy_input.cuda(), os.path.join(self.runner.work_dir,'ptq.onnx'),
                    onnx_node_tensor_translate_mapping=self.onnx_node_tensor_translate_mapping,
                    debug_mode=self.onnx_node_debug_mode)
        else:
            observed_model = self.runner.model.get_deploy_model()
            self.dummy_input = torch.randn(self.runner.model.input_shapes)
            self.runner.model.quantizer.export_onnx(
                observed_model, self.dummy_input.cuda(), os.path.join(self.runner.work_dir,'ptq.onnx'),
                onnx_node_tensor_translate_mapping=self.onnx_node_tensor_translate_mapping,
                debug_mode=self.onnx_node_debug_mode)

    def prepare_for_run_epoch(self):
        """Toggle the state of the observers and fake quantizers before qat
        training."""
        if (self.freeze_bn_begin > 0
                and self._epoch + 1 >= self.freeze_bn_begin):
            self.runner.model.apply(freeze_bn_stats)

        if self.is_first_batch and self.calibrate_steps != -1:
            # lsq observer init
            # import pdb; pdb.set_trace(), TDL: whether need to turn to `eval` mode?
            self.runner.model.eval()
            if self.calibrate_open_fakequant:
                self.runner.model.apply(enable_static_estimate)
            else:
                self.runner.model.apply(enable_static_observation)
            print_log('Start calibration...', logger='current')
            for idx, data_batch in enumerate(self.calibrate_dataloader):
                if idx == self.calibrate_steps:
                    break
                _ = self.runner.model.calibrate_step(data_batch)
            if self.distributed:
                all_reduce_params(
                    self.runner.model.parameters(), op='mean')
                all_reduce_params(self.runner.model.buffers(), op='mean')
            self.runner.model.sync_qparams(src_mode='predict')
            self.runner.model.apply(enable_param_learning)
            print_log('Finish calibration!', logger='current')
            self.runner.save_checkpoint(self.runner.work_dir,'ptq.pth')
            print_log('save ptq checkpoin after calibration!')
            self.prepare_for_val()
            # self.export_ptq()
            self.runner.val_loop.run()
            self._is_first_batch = False
            self.runner.model.train()

        self.runner.model.apply(enable_param_learning)

    def prepare_for_val(self):
        """Toggle the state of the observers and fake quantizers before
        validation."""
        self.runner.model.apply(enable_val)

    @property
    def is_first_batch(self):
        return self._epoch == 0 and self._is_first_batch

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            if self.is_first_batch:
                # lsq observer init
                self.runner.model.apply(enable_static_estimate)

            self.run_iter(idx, data_batch)

            if self.is_first_batch:
                # In the first batch, scale in LearnableFakeQuantize is
                # calculated through lsq observer. As the values of `scale` of
                # different observers in different rank are usually different,
                # we have to sync the `scale` here.
                if self.distributed:
                    all_reduce_params(
                        self.runner.model.parameters(), op='mean')

                # Change back to param learning mode
                self._is_first_batch = False
                self.runner.model.apply(enable_param_learning)

        self.runner.model.sync_qparams(src_mode='loss')
        # Make sure the registered buffer such as `observer_enabled` is
        # correct in the saved checkpoint.
        self.prepare_for_val()
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class QATValLoop(ValLoop):
    """`ValLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 only_qat: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        self.only_qat = only_qat
        if self.runner.distributed:
            assert hasattr(self.runner.model.module, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.module.data_preprocessor
            self.architecture = self.runner.model.module.architecture
            self.architecture.data_preprocessor = data_preprocessor

        else:
            assert hasattr(self.runner.model, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.data_preprocessor
            self.architecture = self.runner.model.architecture
            self.architecture.data_preprocessor = data_preprocessor

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.runner.model)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        qat_metrics = dict()
        for key, value in metrics.items():
            qat_key = 'qat.' + key
            ori_key = 'original.' + key
            qat_metrics[qat_key] = value
            self.runner.message_hub.log_scalars.pop(f'val/{ori_key}', None)

        self.runner.call_hook('after_val_epoch', metrics=qat_metrics)

        if not self.only_qat:
            self.runner.call_hook('before_val_epoch')
            self.runner.model.eval()
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch, self.architecture)

            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            qat_metrics = dict()
            for key, value in metrics.items():
                qat_key = 'qat.' + key
                ori_key = 'original.' + key
                qat_metrics[ori_key] = value
                self.runner.message_hub.log_scalars.pop(f'val/{qat_key}', None)

            self.runner.call_hook('after_val_epoch', metrics=qat_metrics)

        self.runner.call_hook('after_val')
        return qat_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], model):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        # quant_hook = QuantizationHook()
        # quant_hook.attach(self.runner.model.module.qmodels.predict)
        # evaluator = ModelQuantizationEvaluator(model)
        # quant_errors = evaluator.evaluate(data_batch)
        # import pdb; pdb.set_trace()
        # # 输出全局误差信息
        # print(quant_errors)
        outputs = model.val_step(data_batch)
        # print(quant_hook.diffs)
        # quant_hook.remove()
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class PTQLoop(TestLoop):
    """`TestLoop` for Post Training Quantization.

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool, optional): Enable FP16 training mode. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 calibrate_dataloader: Union[DataLoader, Dict],
                 calibrate_steps=32,
                 fp16: bool = False,
                 only_val=False):
        super().__init__(runner, dataloader, evaluator, fp16)
        if isinstance(calibrate_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                calibrate_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

        self.calibrate_steps = calibrate_steps
        self.only_val = only_val

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        self.runner.model.eval()

        if not self.only_val:
            self.runner.model.apply(enable_fake_quant)
            self.runner.model.apply(enable_observer)

            print_log('Star calibratiion...')
            for idx, data_batch in enumerate(self.dataloader):
                if idx == self.calibrate_steps:
                    break
                self.run_iter(idx, data_batch)
            print_log('Finish calibratiion!')

            self.runner.model.apply(enable_fake_quant)
            self.runner.model.apply(disable_observer)

            save_dir = os.path.join(self.runner.work_dir,
                                    self.runner.timestamp)
            self.runner.save_checkpoint(
                save_dir,
                'model_ptq.pth',
                file_client_args=None,
                save_optimizer=False,
                save_param_scheduler=False)
            print_log(f'Quantized model is saved in {save_dir}')

        print_log('Start Evaluating quantized model...')
        self.runner.model.apply(enable_fake_quant)
        self.runner.model.apply(disable_observer)
        metricts = self.runner.val_loop.run()
        self.runner.call_hook('after_test_epoch', metrics=metricts)
        self.runner.call_hook('after_test')

        return metricts

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        _ = self.runner.model.calibrate_step(data_batch)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=None)
