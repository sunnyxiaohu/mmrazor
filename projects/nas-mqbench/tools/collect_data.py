import argparse
from copy import deepcopy
import os
from collections import OrderedDict
from pathlib import Path
import re
from nats_bench import create, pickle_save, pickle_load, ArchResults, ResultsCount
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
    subnet_root_list = os.listdir(args.data_root)
    for subnet_root in subnet_root_list:
        print(f'Handle subnet: {subnet_root}')
        collect_subset(cfg, benchmark_api, os.path.join(args.data_root, subnet_root),
                       store=args.store, analysis=args.analysis, work_dir=args.work_dir)

def valid_check():
    pass

def _get_setname_xrange_seed(path):
    match = re.search('(.+)_8xb16_.+_xrange([0-9]+)-([0-9]+)_seed([0-9]+)', path)
    setname = match.group(1)
    start, end = int(match.group(2)), int(match.group(3))
    this_seed = int(match.group(4))
    return setname, start, end, this_seed

def get_quantize_result(filename_dir_list, start, end, root_dir=''):
    if len(filename_dir_list) != (end - start):
        print(f'Missmatch filenames and xrange: {len(filename_dir_list)} vs [{start}, {end}]')
        import pdb; pdb.set_trace()
    results = {}
    regex1 = 'arch_index=([0-9]+),'
    regex2 = 'accuracy/top1: ([0-9]+.[0-9]+)'    
    for filename in filename_dir_list:
        path = os.path.join(root_dir, filename, filename + '.log')
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
        
        if arch_index is None or accuracy is None or arch_index < start or arch_index >= end or arch_index in results:
            print(f'{path}, arch_index: {arch_index}, accuracy: {accuracy}, xrange: [{start}, {end}]')
            import pdb; pdb.set_trace()
            continue
        results[arch_index] = accuracy
    assert len(results) == (end - start)        
    return results

def collect_subset(cfg, benchmark_api, subnet_root, store=False, analysis=False, work_dir=''):
    dataset = cfg.model.architecture.backbone.dataset
    hp = cfg.model.architecture.backbone.hp
    is_random = cfg.model.architecture.backbone.seed

    total_archs = len(benchmark_api)
    setname, start, end, this_seed = _get_setname_xrange_seed(subnet_root.split('/')[-1])
    assert start >= 0 and end < total_archs and start <= end

    filenames = os.listdir(subnet_root)
    filenames = list(
        filter(lambda x: os.path.isdir(
            os.path.join(subnet_root, x)), filenames))

    filenames.sort()
    quantize_result = get_quantize_result(filenames, start, end, root_dir=subnet_root)

    if not store and not analysis:
        return

    for arch_index, accuracy in quantize_result.items():
        # backbone_config = benchmark_api.get_net_config(arch_index, cfg.model.architecture.backbone.dataset)
        info = benchmark_api.get_more_info(arch_index, dataset, hp=hp,is_random=is_random)
        benchmark_api.clear_params(arch_index, hp=hp)
        print(arch_index, info['test-accuracy'], accuracy)

        if store:
            # collect and store data.
            # import pdb; pdb.set_trace()
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
            # nas_mqbench = 'NASMQ_' + cfg.model.architecture.backbone.benchmark_api.file_path_or_dict.split('/')[-1]
            full_save_dir = Path(work_dir)  #  / nas_mqbench
            full_save_dir.mkdir(parents=True, exist_ok=True)
            pickle_save(to_save_data, str(full_save_dir / f'{arch_index:06d}.pickle'))
            # archresult.get_metrics(new_dataset, setname, is_random=this_seed)
            # archresult.get_metrics(dataset, 'ori-test', is_random=is_random)

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
