_base_ = [
    '../realistic_resnet18_fp32/bignas_resnet18_subnet_8xb256_in1k.py'
]

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
)

_base_.model.init_cfg = dict(
        type='Pretrained',
        prefix='architecture.',
        checkpoint=  # noqa: E251
        'work_dirs/bignas_resnet18_search_8xb128_in1k/subnet_20230519_0915.pth')
model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.data_preprocessor,
    architecture=_base_.model,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict', 'loss'),
    quantizer=dict(
        type='mmrazor.TensorRTQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

optim_wrapper = dict(
    optimizer=dict(lr=0.008),
    paramwise_cfg=dict(
        # custom_keys={
        # 'qmodels': dict(lr_mult=0.1)},
        bypass_duplicate=True
    ))

max_epochs = 20 # _base_.max_epochs
# learning policy
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=max_epochs,
    by_epoch=True,
    begin=0,
    end=max_epochs)

model_wrapper_cfg = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=5,
    freeze_bn_begin=1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(
    checkpoint=dict(save_best=None),
    sync=dict(type='SyncBuffersHook'))
