_base_ = [
    'mmcls::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.face-recognition.models.algorithms.spos_partialfc',
        'projects.face-recognition.models.losses.margin_loss',
        'projects.face-recognition.models.heads.partialfc_head',
        'projects.face-recognition.models.backbones.mobilefacenet',
        'projects.face-recognition.datasets.mx_face_dataset',
    ],
    allow_failed_imports=False
)

# model settings
architecture = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    data_preprocessor=dict(
        # num_classes=1000,
        # RGB format normalization parameters
        mean=[128, 128, 128],
        std=[128, 128, 128],
        # convert image from BGR to RGB
        to_rgb=True,
    ),
    # TODO: replace model config
    backbone=dict(
        type='mmrazor.MobileFaceNet',
        fp16=True,
        num_features=512),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.PartialFCHead',
        embedding_size=512,
        num_classes=2059906,
        sample_rate=0.2,
        fp16=True,
        loss=dict(type='mmrazor.CombinedMarginLoss', s=64, m1=1.0, m2=0.0, m3=0.4, interclass_filtering_threshold=0.0),
    )
)

model = dict(
    type='mmrazor.SPOSPartialFC',
    architecture=architecture,
    mutator=dict(type='mmrazor.NasMutator'))

model_wrapper_cfg = dict(
    type='mmrazor.SPOSPartialFCDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)


# dataset settings
dataset_type = 'mmrazor.MXFaceDataset'
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
# TODO(shiguang): handle data_processor in deploy pipeline.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=512,
    num_workers=3,
    # prefetch_factor=1,
    drop_last=True,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root='/alg-data2/datasets/cv_dirty/fanxiao/fanxiao/webface260m/',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

# TODO(shiguang): handle validation.
val_dataloader = None
val_evaluator = None  # dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# TODO(shiguang): handle fp16 properly.
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0005,
        by_epoch=True,
        begin=0,
        end=2,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='PolyLR',
        power=2.0,
        by_epoch=True,
        begin=2,
        end=10,
        # update by iter
        convert_to_iter_based=True,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
val_cfg = None  # dict()
test_cfg = None  # dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=512)

# _base_.default_hooks.logger.interval = 10
