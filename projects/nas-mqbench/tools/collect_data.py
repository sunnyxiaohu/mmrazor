import argparse
import os
import os.path as osp

import re

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmrazor.utils import register_all_modules


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='NAS-MQBench collect data')
    parser.add_argument('config', help='test config file path')        
    parser.add_argument('xrange', type=str, help='xrange for arch_index')
    parser.add_argument('data_root', type=str)
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    args = parser.parse_args()
    return args


def main():
    # register_all_modules(False)
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config, import_custom_modules=False)
    
    from nats_bench import create

    # Create the API instance for the size search space in NATS
    benchmark_api = create(**cfg.model.architecture.backbone.benchmark_api)

    total_archs = len(benchmark_api)
    start, end = args.xrange.split('-')
    start, end = int(start), int(end)
    assert start >= 0 and end < total_archs and start <= end

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
        info = benchmark_api.get_more_info(
            arch_index, cfg.model.architecture.backbone.dataset,
            hp=cfg.model.architecture.backbone.hp,
            is_random=cfg.model.architecture.backbone.seed)
        print(arch_index, info['test-accuracy'], accuracy)
        benchmark_api.clear_params(arch_index, hp=cfg.model.architecture.backbone.hp)
        # import pdb; pdb.set_trace()



if __name__ == '__main__':
    main()
