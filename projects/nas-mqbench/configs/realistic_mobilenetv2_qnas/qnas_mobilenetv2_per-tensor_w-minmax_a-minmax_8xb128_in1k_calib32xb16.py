_base_ = [
    './qnas_mobilenetv2_supernet_8xb128_in1k.py'
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
    calibrate_sample_num=4096,
    constraints_range=dict(flops=(0., 7000.)),
    mq_init_candidates='work_dirs/ptq_base_bignas_mobilenetv2_8xb256_in1k/search_epoch_1.pkl',
    score_indicator='per-tensor_qnas',
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
