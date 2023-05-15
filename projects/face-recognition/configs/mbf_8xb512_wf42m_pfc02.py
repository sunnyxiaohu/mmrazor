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
        'projects.face-recognition.evaluation.match_rank',
        'projects.face-recognition.engine.runner.loopx',
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
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        # convert image from BGR to RGB
        to_rgb=True,
    ),
    # TODO: replace model config
    backbone=dict(
        type='mmrazor.MobileFaceNet',
        # init_cfg=dict(type='Pretrained', checkpoint='/alg-data/ftp-upload/private/wangshiguang/projects/Epoch_3.pt'),
        num_features=256,
        fp16=False,
        scale=1,
        blocks=(2, 4, 6, 2)),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.PartialFCHead',
        # init_cfg=dict(type='Pretrained', checkpoint='/home/wangshiguang/Archface/arcface_torch/out_dir_256/checkpoint'),
        embedding_size=256,
        num_classes=2059906,
        sample_rate=0.2,
        fp16=False,
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
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
train_dataloader = dict(
    batch_size=256,
    num_workers=3,
    drop_last=True,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/data/webface260m/',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
# TODO(shiguang): handle data_processor in deploy pipeline.
mdataset_type = 'mmrazor.MatchFaceDataset'
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackClsInputs',
         meta_keys=('sample_idx_identical_mapping', 'dataset_name')),
]
val_dataloader = dict(
    batch_size=128,
    num_workers=1,
    dataset=dict(type='ConcatDataset', datasets=[
        dict(type=mdataset_type,
             dataset_name='120P',
             data_root='/mnt/data/face_data/OV_test/face_recognition/120P',
             key_file='120p.key',
             data_prefix=dict(img_path='norm_facex'),
             pipeline=test_pipeline),
        dict(type=mdataset_type,
             dataset_name='baby_500p',
             data_root='/mnt/data/face_data/OV_test/face_recognition/baby_500p',
             key_file='baby_500p.key',
             data_prefix=dict(img_path='norm_face'),
             pipeline=test_pipeline),
        dict(type=mdataset_type,
             dataset_name='glint1k',
             data_root='/mnt/data/face_data/OV_test/face_recognition/glint1k',
             key_file='glint1k_val.key',
             data_prefix=dict(img_path='images'),
             pipeline=test_pipeline),
        dict(type=mdataset_type,
             dataset_name='mask_369p',
             data_root='/mnt/data/face_data/OV_test/face_recognition/mask_369p',
             key_file='mask_369p.key',
             data_prefix=dict(img_path='norm_face'),
             pipeline=test_pipeline),
        dict(type=mdataset_type,
             dataset_name='menjin_20p',
             data_root='/mnt/data/face_data/OV_test/face_recognition/menjin_20p',
             key_file='menjin_20p.key',
             data_prefix=dict(img_path='norm_face'),
             pipeline=test_pipeline),
        dict(type=mdataset_type,
             dataset_name='xm14',
             data_root='/mnt/data/face_data/OV_test/face_recognition/xm14',
             key_file='xm14.key',
             data_prefix=dict(img_path='norm_112'),
             pipeline=test_pipeline),
    ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='mmrazor.Rank1')

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# TODO(shiguang): handle fp16 properly.
# optimizer
optim_wrapper = {
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',  # loss_scale=dict(growth_interval=100),
    'constructor': 'mmrazor.SeparateOptimWrapperConstructor',
    'architecture.backbone': dict(
        optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4),
        clip_grad=dict(type='norm', max_norm=5)),
    'architecture.head': dict(
        optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4),
    )}

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
# Note that: Use LoopX is a little faster than EpochBasedTrainLoop
train_cfg = dict(type='mmrazor.EpochBasedTrainLoopX', max_epochs=10, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)

# _base_.default_hooks.logger.interval = 10
