_base_ = [
    '../realistic_resnet18_fp32/bignas_resnet18_supernet_16xb128_in1k.py'
]

_base_.custom_imports.imports += [
        'projects.nas-mqbench.engine.runner.nas-mq_search_loop'
]

train_cfg = dict(
    _delete_=True,
    type='mmrazor.NASMQSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=5,
    num_candidates=20,
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_sample_num=32,
    constraints_range=dict(flops=(0., 7000.)),
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
