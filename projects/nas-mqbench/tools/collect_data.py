import argparse
from copy import deepcopy
import os
from collections import OrderedDict
from pathlib import Path
import re

import torch

from mmengine.config import Config
from mmengine.runner import Runner

from mmrazor.utils import register_all_modules

from analysis import graphwise_analysis


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='NAS-MQBench collect data')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('xrange', type=str, help='xrange for arch_index')
    parser.add_argument('data_root', type=str)
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--store',
        action='store_true',
        default=False,
        help='enable collect and store data')
    parser.add_argument(
        '--analysis',
        action='store_true',
        default=False,
        help='analysis the collapse results')
    # the margin between quant results and dequant results is very large (> 10 points.)
    args = parser.parse_args()
    return args


def main():
    register_all_modules(False)
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    
    from nats_bench import create, pickle_save, pickle_load, ArchResults, ResultsCount

    # Create the API instance for the size search space in NATS
    benchmark_api = create(**cfg.model.architecture.backbone.benchmark_api)
    dataset = cfg.model.architecture.backbone.dataset
    hp = cfg.model.architecture.backbone.hp
    is_random = cfg.model.architecture.backbone.seed

    total_archs = len(benchmark_api)
    match = re.search('(.+)_8xb16_.+_xrange([0-9]+)-([0-9]+)_seed([0-9]+)', args.data_root.split('/')[-1])
    setname = match.group(1)
    start, end = match.group(2), match.group(3)
    start, end = int(start), int(end)
    assert start >= 0 and end < total_archs and start <= end    
    this_seed = int(match.group(4))

    filenames = os.listdir(args.data_root)
    filenames = list(
        filter(lambda x: os.path.isdir(
            os.path.join(args.data_root, x)), filenames))
    assert len(filenames) == (end - start), f'Miss match xrange: {len(filenames)} vs {(end - start)}'
    filenames.sort()
    regex1 = 'arch_index=([0-9]+),'
    regex2 = 'accuracy/top1: ([0-9]+.[0-9]+)'
    for filename in filenames:
        path = os.path.join(args.data_root, filename, filename + '.log')
        arch_index = None
        accuracy = None
        with open(path) as f:
            for line in f:
                match = re.search(regex1, line)
                if match and arch_index is None:
                    arch_index = int(match.group(1))
                match = re.search(regex2, line)
                if match and accuracy is None:
                    accuracy = float(match.group(1))
        assert arch_index is not None and accuracy is not None and start <= arch_index <= end, \
            f'{path}, arch_index: {arch_index}, accuracy: {accuracy}'
        # backbone_config = benchmark_api.get_net_config(arch_index, cfg.model.architecture.backbone.dataset)
        info = benchmark_api.get_more_info(arch_index, dataset, hp=hp,is_random=is_random)
        print(arch_index, info['test-accuracy'], accuracy)

        if args.store:
            # collect and store data.
            benchmark_api.clear_params(arch_index, hp=cfg.model.architecture.backbone.hp)
            import pdb; pdb.set_trace()
            archresult = benchmark_api.query_meta_info_by_index(arch_index, hp=hp)
            xresult = deepcopy(archresult.query(dataset, seed=is_random))

            # update results
            new_dataset = f'{dataset}_{setname}'
            xresult.seed = this_seed
            xresult.name = new_dataset
            accs, losses, times = {}, {}, {}
            for iepoch in range(xresult.epochs):
                accs[f'{setname}@{iepoch}'] = 0
                losses[f'{setname}@{iepoch}'] = 0
                times[f'{setname}@{iepoch}'] = 0
            accs[f'{setname}@{iepoch}'] = accuracy
            xresult.update_eval(accs, losses, times)
            archresult.update(new_dataset, this_seed, xresult)
            to_save_data = OrderedDict({hp: archresult.state_dict()})
            nas_mqbench = 'NASMQ_' + cfg.model.architecture.backbone.benchmark_api.file_path_or_dict.split('/')[-1]
            full_save_dir = Path(args.work_dir) / nas_mqbench
            full_save_dir.mkdir(parents=True, exist_ok=True)
            pickle_save(to_save_data, str(full_save_dir / f'{arch_index:06d}.pickle'))
            # archresult.get_metrics(new_dataset, setname, is_random=this_seed)
            # archresult.get_metrics(dataset, 'ori-test', is_random=is_random)

        if args.analysis and info['test-accuracy'] - accuracy > 10:
            int8_cfg = cfg = Config.fromfile(args.config)
            int8_cfg.model.architecture.backbone.arch_index = arch_index
            int8_cfg.work_dir = os.path.join(args.work_dir, 'analysis')
            method = 'snr'  # 'cosine'
            # int8_cfg.val_dataloader.batch_size=4
            int8_runner = Runner.from_cfg(int8_cfg)
            int8_runner.test()
            results = graphwise_analysis(
                int8_runner.model, int8_runner.val_dataloader, method, step=1, verbose=True, topk=1)
            import pdb; pdb.set_trace()
            print()


if __name__ == '__main__':
    main()
