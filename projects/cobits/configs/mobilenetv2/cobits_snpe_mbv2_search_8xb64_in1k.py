_base_ = [
    './cobits_snpe_mbv2_supernet_8xb64_in1k.py'
]

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEvolutionSearchLoop',
    solve_mode='ilp',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=1,
    num_candidates=10,
    num_mutation=0,
    num_crossover=0,
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_sample_num=65536,
    # w4a4: Flops: 5394.053 Params: 19.011
    # w3a3: Flops: 3373.459 Params: 16.822
    constraints_range=dict(flops=(0., 5395)),
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
_base_.model.architecture.quantizer.nested_quant_bits_in_layer = True
