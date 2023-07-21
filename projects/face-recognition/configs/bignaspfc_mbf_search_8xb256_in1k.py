_base_ = ['./bignaspfc_mbf_supernet_8xb256_wf42m.py']

_base_.custom_imports.imports += [
        'projects.face-recognition.engine.runner.face_evolution_search_loop',
]

# For speed accelerate
_base_.val_dataloader.batch_size=256
_base_.val_dataloader.num_workers=4
_base_.val_dataloader.pin_memory=True
train_cfg = dict(
    _delete_=True,
    type='mmrazor.FaceEvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=20,
    num_candidates=32,
    top_k=10,
    num_mutation=16,
    num_crossover=16,
    mutate_prob=0.1,
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_sample_num=40960,
    estimator_cfg=dict(
        type='mmrazor.HERONResourceEstimator',
        heronmodel_cfg=dict(
            work_dir='work_dirs/bignaspfc_mbf_search_8xb256_in1k',
            ptq_json='projects/commons/heron_files/face_config_ptq.json',
            HeronCompiler = '/alg-data/ftp-upload/private/wangshiguang/HeronRT/HeronRT_v0.8.0_2023.06.15/tool/HeronCompiler',
            HeronProfiler = '/alg-data/ftp-upload/private/wangshiguang/HeronRT/HeronRT_v0.8.0_2023.06.15/tool/HeronProfiler'
        )),
    # baseline: fix_subnet.flops: 1373.3300  fix_subnet.params: 1.2040  fix_subnet.latency: 0.0000  fix_subnet.heron_bandwidth: 2.0604  fix_subnet.heron_latency: 0.6129  fix_subnet.heron_params: 1.2291
    # max: heron_bandwidth: 3.66 heron_params: 1.87; min: heron_bandwidth: 1.68 heron_params: 0.86
    constraints_range=dict(heron_params=1.3, heron_bandwidth=4, heron_latency=0.55),
    score_indicator=f'rank1/avg_{_base_.sample_ratio}',
    score_key='rank1/avg')

# model = dict(
#     architecture=dict(
#         backbone=dict(fp16=False),
#         head=dict(_delete_=True),
#         input_resizer_cfg=dict(_delete_=True)
#     ))
# _base_.model.architecture.head=None
_base_.model.architecture.input_resizer_cfg=None
_base_.model.architecture.backbone.fp16=False
_base_.model_wrapper_cfg.exclude_module=None

randomness = dict(diff_rank_seed=True)
