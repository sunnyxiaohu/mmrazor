# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import torch

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmrazor.utils import register_all_modules


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMRazor test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('xrange', type=str, help='xrange for arch_index')
    parser.add_argument('--resume-index', type=int, help='resume index')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
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
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    register_all_modules(False)
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = 'none'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # cfg.load_from = None if args.checkpoint == 'none' else args.checkpoint

    from nats_bench import create

    # Create the API instance for the size search space in NATS
    benchmark_api = create(**cfg.model.architecture.backbone.benchmark_api)

    total_archs = len(benchmark_api)
    start, end = args.xrange.split('-')
    start, end = int(start), int(end)
    assert start >= 0 and end < total_archs and start <= end
    index = args.resume_index if args.resume_index else start
    assert index >= start and index <= end
    cfg.work_dir = cfg.work_dir + f'_xrange{start}-{end}_seed{cfg.randomness.get("seed")}'

    while(index < end):
        backbone_config = benchmark_api.get_net_config(index, cfg.model.architecture.backbone.dataset)
        cfg.model.architecture.backbone.arch_index = index
        # cfg.model.architecture.backbone.benchmark_api = benchmark_api
        # build the runner from config
        runner = Runner.from_cfg(cfg)
        # start testing
        runner.test()
        index += 1


if __name__ == '__main__':
    main()
