_base_ = [
    '../realistic_resnet50d_fp32/bignas_resnet50d_subnet_8xb256_in1k.py'
    # 'mmrazor::nas/mmcls/onceforall/ofa_mobilenet_subnet_8xb256_in1k.py'
]

calibrate_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=_base_.val_dataloader.dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=calibrate_dataloader,
    calibrate_steps=32,
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
_base_.model.init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'work_dirs/bignas_resnet50d_search_8xb128_in1k/subnet_20230519_0915.pth')
model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.data_preprocessor,
    architecture=_base_.model,
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

model_wrapper_cfg = dict(_delete_=True, type='mmrazor.MMArchitectureQuantDDP', )
default_hooks = dict(checkpoint=None)
val_cfg = dict(_delete_=True)
