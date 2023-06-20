_base_ = [
    './ptq_base_bignas_resnet18_8xb256_in1k.py'
]

calibrate_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=_base_.val_dataloader.dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.MinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

mq_model = dict(
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.supernet.data_preprocessor,
    architecture=None,  # _base_.model,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict'),
    quantizer=dict(
        type='mmrazor.TensorRTQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

mq_model_wrapper_cfg = dict(type='mmrazor.MMArchitectureQuantDDP', )

train_cfg = dict(
    mq_model=mq_model,
    mq_model_wrapper_cfg=mq_model_wrapper_cfg,
    mq_calibrate_dataloader=calibrate_dataloader,
    mq_calibrate_steps=32,
    mq_init_candidates='work_dirs/ptq_base_bignas_resnet18_8xb256_in1k/search_epoch_1.pkl',
    score_indicator='per-tensor_w-minmax_a-minmax')
