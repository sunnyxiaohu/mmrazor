import argparse
from copy import deepcopy
import os
from collections import OrderedDict
from pathlib import Path
import re
import glob
import torch
from torch.ao.quantization import FakeQuantizeBase

import mmengine
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmrazor.utils import register_all_modules

from analysis import graphwise_analysis


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='NAS-MQBench analysis model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')        
    # the margin between quant results and dequant results is very large (> 10 points.)
    args = parser.parse_args()
    return args


def analysis_weight(model):
    for name, param in model.named_parameters():
        # params axis: out_channel, in_channel, kernel_size, kernel_size
        # import pdb; pdb.set_trace()
        if len(param.size()) < 2:
            continue
        param = torch.abs(param)
        tmin_val, tmax_val = torch.aminmax(param)
        cmin_val, cmax_val = torch.aminmax(param.flatten(start_dim=1), dim=1)
        # import pdb; pdb.set_trace()
        print(f'{name}, (T, CMean, CVar) of min_val: ({tmin_val}, {torch.mean(cmin_val)}, {torch.var(cmin_val)}), '
              f'max_val: ({tmax_val}, {torch.mean(cmax_val)}, {torch.var(cmax_val)})')

    for name, module in model.named_modules():
        if isinstance(module, FakeQuantizeBase) and 'activation_post_process' not in name:
            print(f'{name}: Mean: {torch.mean(module.scale)}, Var: {torch.var(module.scale)}')  # , {module.scale}')
            if 'layer1.0.downsample' in name:
                module.disable_fake_quant()


def main():
    register_all_modules(False)
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = ''
    method = 'snr'  # 'cosine'
    runner = Runner.from_cfg(cfg)
    runner.test()
    analysis_weight(runner.model)
    only_act_metrics = runner.val_loop.run()
    runner.call_hook('after_test_epoch', metrics=only_act_metrics)
    runner.call_hook('after_test')    
    import pdb; pdb.set_trace()     
    results = graphwise_analysis(
        runner.model, runner.val_dataloader, method, step=1, verbose=True, topk=1)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
