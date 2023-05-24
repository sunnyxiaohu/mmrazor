_base_ = ['mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py',]
custom_imports = dict(
    imports=[
        'projects.common.engine.runner.superacme_quantization_loops',
        'projects.superacme_backend.models.algorithms.quantization.superacme_architecture',
        'projects.superacme_backend.models.quantizers.superacme_quantizer',
    ],
    allow_failed_imports=False)
mv2 = _base_.model
float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'  # noqa: E501

dataset_type = 'ImageNet'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQPerChannelObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=False),
    a_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SuperAcmeArchitectureQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=1000,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    architecture=mv2,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.SuperAcmeQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    _delete_=True, type='ConstantLR', factor=1.0, by_epoch=True)

model_wrapper_cfg = dict(
    type='mmrazor.SuperAcmeArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.SuperAcmeLSQEpochBasedLoop',
    max_epochs=10,
    val_interval=1,
    freeze_bn_begin=1)
val_cfg = dict(_delete_=True, type='mmrazor.SuperAcmeQATValLoop')

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))
custom_hooks = [dict(type='ExportQATHook', interval=1, by_epoch=True,save_best='qat.accuracy/top1')]