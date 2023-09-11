_base_ = [
    './qnas_resnet18_supernet_8xb256_in1k.py'
]
_base_.custom_imports.imports += [
    'projects.nas-mqbench.models.task_modules.estimators.counters.op_counters.dynamic_qlayer_counter',
]
train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=5,
    num_candidates=50,
    num_mutation=25,
    num_crossover=25,
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_sample_num=4096,
    constraints_range=dict(flops=(0., 7000.)),
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
_base_.model.architecture.quantizer.nested_quant_bits_in_layer = True