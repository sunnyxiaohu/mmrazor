_base_ = [
    './cobits_snpe_yolox_s_supernet_8xb8_coco.py'
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
    calibrate_sample_num=200,
    # w4a4: Flops: 229102.333 Params: 35.696
    # w5a5: Flops: 345179.696 Params: 44.605
    # w6a6: Flops: 487052.029 Params: 53.513
    constraints_range=dict(flops=(0., 487052.)),
    estimator_cfg=dict(type='mmrazor.ResourceEstimator', input_shape=(1, 3, 640, 640)),
    score_key='coco/bbox_mAP')

val_cfg = dict(_delete_=True)
_base_.model.architecture.quantizer.nested_quant_bits_in_layer = True
