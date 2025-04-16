from typing import Dict, Optional
import copy
import torch
import matplotlib.pyplot as plt
import os
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmrazor.models.quantizers.native_quantizer import SUPPORT_QAT_MODULES
from mmrazor.models.task_modules.tracer.fx.graph_utils import _get_attrs


SUPPORT_WEIGHT_MODULES = (torch.nn.Conv2d, torch.nn.Linear)


@HOOKS.register_module()
class QATAnalysisHook(Hook):
    def __init__(self, output_dir='qat_analysis', visualize=True):
        self.output_dir = output_dir
        self.visualize = visualize
        self.activation_store = {}

    def before_run(self, runner) -> None:
        model = runner.model
        model = model.module if is_model_wrapper(model) else model
        self.is_qat = hasattr(model, 'get_deploy_model')
        self.register_hooks(model)
        mkdir_or_exist(os.path.join(runner.work_dir, runner.timestamp, self.output_dir))

    def register_hooks(self, model):
        """注册 forward hook 以捕获激活分布"""
        for name, module in model.named_modules():
            cond = False
            if self.is_qat and isinstance(module, SUPPORT_QAT_MODULES) and '.predict' in name:
                cond = True
            elif not self.is_qat and isinstance(module, SUPPORT_WEIGHT_MODULES):
                cond = True
            if cond:
                self.activation_store[name] = []
                module.register_forward_hook(self._hook_fn(name))

    def _hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activation_store[name].append(output.detach().cpu().numpy().flatten())
        return hook

    def analy_once(self, runner: Runner, deploy=False):
        model = runner.model
        model = model.module if is_model_wrapper(model) else model
        if self.is_qat:
            deploy_model = model.get_deploy_model()
        else:
            from mmrazor.models.utils import fuse_conv_bn
            deploy_model = copy.deepcopy(model)
            deploy_model = fuse_conv_bn(deploy_model)

        for name, module in model.named_modules():
            cond = False
            if self.is_qat and isinstance(module, SUPPORT_QAT_MODULES) and '.predict' in name:
                cond = True
            elif not self.is_qat and isinstance(module, SUPPORT_WEIGHT_MODULES):
                cond = True
            if cond:
                # 1. weight analysis
                weight = module.weight.detach().cpu().numpy().flatten()
                deploy_name = name.replace('qmodels.predict.', '') if self.is_qat else name
                deploy_weight = _get_attrs(deploy_model, deploy_name).weight.detach().cpu().numpy().flatten()
                filename = os.path.join(runner.work_dir, runner.timestamp, self.output_dir, f'{name}')
                all_activations = None
                # 2. activation analysis
                if name in self.activation_store and self.activation_store[name]:
                    all_activations = torch.cat([torch.tensor(a) for a in self.activation_store[name]], dim=0).numpy()
                    self.activation_store[name] = []
                self._plot_distribution(weight, filename, deploy_values=deploy_weight, act_values=all_activations)

    def after_val_epoch(self, runner: Runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        # import pdb; pdb.set_trace()
        self.analy_once(runner)

    def after_test_epoch(self, runner: Runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        # import pdb; pdb.set_trace()
        self.analy_once(runner)

    def _plot_distribution(self, values, filename, deploy_values=None, act_values=None):
        if self.visualize and len(values) > 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(2, 2, 1)
            plt.hist(values, bins=50, alpha=0.75, color='blue', edgecolor='black')
            plt.xlabel('Weight Values')
            plt.ylabel('Frequency')
            plt.title('Distribution of Weight Values')
            plt.grid(True)
            min_val, max_val = values.min(), values.max()
            plt.axvline(min_val, color='blue', linestyle='dashed', linewidth=1)
            plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=1)
            plt.text(min_val, plt.ylim()[1] * 0.9, f'Min: {min_val:.4f}', color='blue')
            plt.text(max_val, plt.ylim()[1] * 0.9, f'Max: {max_val:.4f}', color='blue')

            if deploy_values is not None:
                plt.subplot(2, 2, 2)
                plt.hist(deploy_values, bins=50, alpha=0.75, color='red', edgecolor='black')
                plt.xlabel('Deployed Weight Values')
                plt.ylabel('Frequency')
                plt.title('Distribution of Deployed Weight Values')
                plt.grid(True)
                min_val, max_val = deploy_values.min(), deploy_values.max()
                plt.axvline(min_val, color='red', linestyle='dashed', linewidth=1)
                plt.axvline(max_val, color='red', linestyle='dashed', linewidth=1)
                plt.text(min_val, plt.ylim()[1] * 0.9, f'Min: {min_val:.4f}', color='red')
                plt.text(max_val, plt.ylim()[1] * 0.9, f'Max: {max_val:.4f}', color='red')                

            if act_values is not None:
                plt.subplot(2, 2, 3)
                plt.hist(act_values, bins=50, alpha=0.75, color='green', edgecolor='black')
                plt.xlabel('Activation Values')
                plt.ylabel('Frequency')
                plt.title('Distribution of Activation Values')
                plt.grid(True)
                min_val, max_val = act_values.min(), act_values.max()
                plt.axvline(min_val, color='green', linestyle='dashed', linewidth=1)
                plt.axvline(max_val, color='green', linestyle='dashed', linewidth=1)
                plt.text(min_val, plt.ylim()[1] * 0.9, f'Min: {min_val:.4f}', color='green')
                plt.text(max_val, plt.ylim()[1] * 0.9, f'Max: {max_val:.4f}', color='green')                 
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{filename}.png'))
            plt.close()
