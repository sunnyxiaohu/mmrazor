import argparse
from copy import deepcopy
import os
from collections import OrderedDict
from pathlib import Path
import re
import glob
from nats_bench import create, pickle_save, pickle_load, ArchResults, ResultsCount
import torch

import mmengine
from mmengine.config import Config, DictAction
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
        '--xrange',
        help='If specified, it will only collect data in xrange.')
    parser.add_argument(
        '--force_index',
        help='If specified, it will force collect data, and set the default value as 0.',
        default=None)
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


def main():
    register_all_modules(False)
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Create the API instance for the size search space in NATS
    benchmark_api = create(**cfg.model.architecture.backbone.benchmark_api)
    force_index = None if args.force_index is None else args.force_index.split(',')
    collect_subset(cfg, benchmark_api, args.data_root,
                   store=args.store, analysis=args.analysis, work_dir=args.work_dir,
                   xran=args.xrange, force_index=force_index)

def valid_check():
    pass

def _get_exp_args(path):
    match = re.search('(.+)_8xb16_(.+)_calib.+_hp([0-9]+)_seed([0-9]+)', path)
    setname = match.group(1)
    this_dataset = match.group(2)
    this_hp = match.group(3)
    this_seed = int(match.group(4))
    return setname, this_dataset, this_hp, this_seed

def get_quantize_result(index_list, root_dir='', force_index=None):
    results = {}
    for index in index_list:
        index_path = os.path.join(root_dir, index)
        exps = list(
                    filter(lambda x: os.path.isdir(
                        os.path.join(index_path, x)), os.listdir(index_path)))
        assert len(exps) == 1
        try:
            exp_path = os.path.join(index_path, exps[0], f'{exps[0]}.json')
            rst = mmengine.load(exp_path)
        except FileNotFoundError:
            if force_index is not None and index in force_index:
                rst = {'accuracy/top1': 0.0}
                print(f'{exp_path} not found, use default value 0')
            else:
                raise FileNotFoundError
        results[int(index)] = rst['accuracy/top1']
    return results

def collect_subset(cfg, benchmark_api, subset_root,
                   store=False, analysis=False, work_dir='', xran=None,
                   force_index=None):
    dataset = cfg.model.architecture.backbone.dataset
    hp = cfg.model.architecture.backbone.hp
    is_random = cfg.model.architecture.backbone.seed

    total_archs = len(benchmark_api)
    setname, this_dataset, this_hp, this_seed = _get_exp_args(subset_root.split('/')[-1])
    assert this_hp == hp and dataset == this_dataset
    filenames = os.listdir(subset_root)
    filenames = list(
        filter(lambda x: os.path.isdir(
            os.path.join(subset_root, x)), filenames))
    assert len(filenames) == total_archs, f'{len(filenames)} vs {total_archs}'
    filenames.sort()
    quantize_result = get_quantize_result(filenames, root_dir=subset_root, force_index=force_index)

    if not store and not analysis:
        return

    if xran is not None:
        xran = [int(x) for x in xran.split('-')]
        xran = range(xran[0], xran[1])

    for arch_index, accuracy in quantize_result.items():
        if xran is not None and arch_index not in xran:
            continue
        # backbone_config = benchmark_api.get_net_config(arch_index, cfg.model.architecture.backbone.dataset)
        info = benchmark_api.get_more_info(arch_index, dataset, hp=hp, is_random=is_random)
        benchmark_api.clear_params(arch_index, hp=hp)
        print(arch_index, info['test-accuracy'], accuracy)

        if store:
            # collect and store data.
            # import pdb; pdb.set_trace()
            arch2infos_dict = benchmark_api.arch2infos_dict[arch_index]
            # cfg.model.architecture.backbone.benchmark_api.file_path_or_dict = work_dir
            # benchmark_apixxx = create(**cfg.model.architecture.backbone.benchmark_api)
            # benchmark_apixxx._prepare_info(arch_index)
            # arch2infos_dict = benchmark_apixxx.arch2infos_dict[arch_index]
            to_save_data = OrderedDict(
                {hp: arch2infos_dict[hp].state_dict() for hp in benchmark_api._avaliable_hps})

            # archresult = benchmark_api.query_meta_info_by_index(arch_index, hp=hp)
            archresult = arch2infos_dict[hp]
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

            to_save_data[hp] = archresult.state_dict()
            # nas_mqbench = 'NASMQ_' + cfg.model.architecture.backbone.benchmark_api.file_path_or_dict.split('/')[-1]
            full_save_dir = Path(work_dir)  #  / nas_mqbench
            full_save_dir.mkdir(parents=True, exist_ok=True)
            pickle_save(to_save_data, str(full_save_dir / f'{arch_index:06d}.pickle'))

        if analysis and info['test-accuracy'] - accuracy > 10:
            int8_cfg = cfg
            int8_cfg.model.architecture.backbone.arch_index = arch_index
            int8_cfg.work_dir = os.path.join(work_dir, 'analysis')
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
