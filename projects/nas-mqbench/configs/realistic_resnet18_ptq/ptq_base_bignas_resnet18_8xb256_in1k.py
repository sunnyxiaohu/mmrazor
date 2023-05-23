_base_ = [
    '../realistic_resnet18_fp32/bignas_resnet18_supernet_16xb128_in1k.py'
]

custom_imports = dict(
    imports=[
        'projects.nas-mqbench.models.architectures.backbones.searchable_resnet',
        'projects.nas-mqbench.engine.runner.nas-mq_search_loop'
    ],
    allow_failed_imports=False)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.NASMQSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=2,
    num_candidates=5,
    calibrate_sample_num=32,
    constraints_range=dict(flops=(0., 7000.)),
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
