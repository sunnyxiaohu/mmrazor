import argparse
from copy import deepcopy
import os
import os.path as osp

import re

from mmengine.config import Config
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
    this_seed = 120

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
        info = benchmark_api.get_more_info(arch_index, dataset, hp=hp,is_random=is_random)
        print(arch_index, info['test-accuracy'], accuracy)
        benchmark_api.clear_params(arch_index, hp=cfg.model.architecture.backbone.hp)
        # import pdb; pdb.set_trace()
        archresult = benchmark_api.query_meta_info_by_index(arch_index)
        xresult = deepcopy(archresult.query(dataset, seed=is_random))

        # update results
        setname = 'per-tensor_histogram'
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
        archresult.get_metrics(new_dataset, setname, is_random=this_seed)


        from typing import Any
        import torch
        import ppq.lib as PFL
        from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
        from ppq.api import ENABLE_CUDA_KERNEL, dump_torch_to_onnx, load_onnx_graph
        from ppq.quantization.optim import (ParameterQuantizePass,
                                            RuntimeCalibrationPass)

        from Quantizers import MyFP8Quantizer, MyInt8Quantizer

        # 1. build pytorch module.
        import xautodl
        arch_cfg = benchmark_api.get_net_config(arch_index, dataset)
        from xautodl.models import get_cell_based_tiny_net
        model = get_cell_based_tiny_net(arch_cfg)
        import pdb; pdb.set_trace()
        # 2. convert pytorch module to onnx with a sample input.
        ONNX_FILE_NAME = 'Tmp.onnx'
        sample_input = torch.rand(size=[1, 3, 32, 32])  #.cuda()
        dump_torch_to_onnx(
            model=model, onnx_export_file=ONNX_FILE_NAME, 
            inputs=sample_input, input_shape=None)

        # 3. create quantizer.
        PerChannel  = True
        Symmetrical = True
        PowerOf2    = False
        FP8         = False
        import ppq
        ppq.core.common.FORMATTER_REPLACE_BN_TO_CONV = False
        graph = load_onnx_graph(onnx_import_file=ONNX_FILE_NAME)
        if not FP8:
            quantizer = MyInt8Quantizer(
                graph=graph, 
                per_channel=PerChannel, 
                sym=Symmetrical, 
                power_of_2=PowerOf2)
        else:
            quantizer = MyFP8Quantizer()


        # 4. quantize operations, only conv and gemm will be quantized by this script.
        for op in graph.operations.values():
            if op.type in {'Conv', 'Gemm'}:
                platform = TargetPlatform.INT8 if not FP8 else TargetPlatform.FP8
            else: platform = TargetPlatform.FP32

            quantizer.quantize_operation(
                op_name = op.name, platform=platform)


        # 5. build quantization pipeline
        pipeline = PFL.Pipeline([
            ParameterQuantizePass(),
            RuntimeCalibrationPass(),

            # LearnedStepSizePass is a network retraining procedure.
            # LearnedStepSizePass(steps=500)
        ])


        # 6. run quantization, measure quantization loss.
        # rewrite the definition of dataloader with real data first!
        dataloader = [torch.rand(size=[1, 3, 32, 32]) for _ in range(32)]

        def collate_fn(batch: Any):
            return batch  #.cuda()

        # with ENABLE_CUDA_KERNEL():
        executor = TorchExecutor(graph=graph, device='cpu')
        executor.tracing_operation_meta(inputs=sample_input)

        pipeline.optimize(
            graph=graph, dataloader=dataloader, verbose=True, 
            calib_steps=32, collate_fn=collate_fn, executor=executor)

        error_dict = graphwise_error_analyse(
            graph=graph, running_device='cpu', 
            dataloader=dataloader, collate_fn=collate_fn)

            # quantization error are stored in error_dict
            # error_dict = {op name(str): error(float)}        


        # if info['test-accuracy'] - accuracy > 10:
        data_batch = {'inputs': torch.rand(size=[1, 3, 32, 32])}
        fp32_cfg = Config.fromfile('/home/GML/mmrazor/projects/nas-mqbench/configs/ptq_fp32_nats_8xb16_cifar10.py')
        fp32_cfg.model.backbone.arch_index = arch_index
        fp32_cfg.launcher = 'none'
        fp32_cfg.work_dir = 'work_dirs/fp32'
        fp32_runner = Runner.from_cfg(fp32_cfg)
        fp32_runner.model.eval()
        data_batch = next(iter(fp32_runner.test_dataloader))
        data_batch1 = deepcopy(data_batch)
        import pdb; pdb.set_trace()
        fp32_outputs = fp32_runner.model.test_step(data_batch1)
        # fp32_runner.test()

        int8_cfg = Config.fromfile('/home/GML/mmrazor/projects/nas-mqbench/configs/ptq_per-tensor_minmax_nats_8xb16_cifar10_calib32xb16.py')
        int8_cfg.model.architecture.backbone.arch_index = arch_index
        int8_cfg.launcher = 'none'
        int8_cfg.work_dir = 'work_dirs/int8'
        int8_runner = Runner.from_cfg(int8_cfg)
        int8_runner.test()  # for calibrate
        data_batch2 = deepcopy(data_batch)
        int8_outputs = int8_runner.model.test_step(data_batch2)
        import pdb; pdb.set_trace()
        print()                 


if __name__ == '__main__':
    main()
