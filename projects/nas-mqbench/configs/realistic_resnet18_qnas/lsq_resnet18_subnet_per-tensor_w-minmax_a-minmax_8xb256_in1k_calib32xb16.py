_base_ = [
    '../realistic_resnet18_fp32/bignas_resnet18_subnet_8xb256_in1k.py'
]

_base_.custom_imports.imports += [
    'projects.nas-mqbench.models.quantizers.mutable_quantizer',
]

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
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
        type='mmrazor.MutableOpenVINOQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.conv1',
            'head.fc'
        ],
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

optim_wrapper = dict(
    optimizer=dict(lr=0.08),
    paramwise_cfg=dict(
        # custom_keys={
        # 'qmodels': dict(decay_mult=5)},
        bypass_duplicate=True
    ))

max_epochs = 100
warm_epochs = 1
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        # about 2500 iterations for ImageNet-1k
        end=warm_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs-warm_epochs,
        by_epoch=True,
        begin=warm_epochs,
        end=max_epochs,
    ),
]

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
