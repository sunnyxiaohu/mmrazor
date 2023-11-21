_base_ = [
    '../realistic_resnet50d_fp32/bignas_resnet50d_supernet_16xb128_in1k.py'
]

_base_.custom_imports.imports += [
        'projects.nas-mqbench.engine.runner.nas-mq_search_loop'
]

train_cfg = dict(
    _delete_=True,
    type='mmrazor.NASMQSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=1,
    num_candidates=5,
    calibrate_sample_num=32,
    constraints_range=dict(flops=(0., 30000.)),
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
