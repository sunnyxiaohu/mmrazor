_base_ = [
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.person-detection.models.backbones.csp_darknet',
        'projects.person-detection.models.necks.yolox_pafpn_qat',
        'projects.person-detection.engine.hooks.old_sync_norm_hook',
        'projects.person-detection.engine.hooks.spos_yolox_mode_switch_hook'
    ],
    allow_failed_imports=False
)

img_scale =(640, 384) 
test_img_scale=(640, 384) 
yolox_img_scale = (384, 640) 
# model settings
architecture = dict(
    _scope_='mmdet',
    type='YOLOX',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)],
    ),
    # TODO: replace model config
    backbone=dict(
        type='mmrazor.CSPDarknet',
        arch='P5',
        deepen_factor=0.33,
        widen_factor=0.25,
        use_depthwise=False,
        act_cfg=dict(type='ReLU'),
        #init_cfg=None
        ),
    neck=dict(
        type='mmrazor.YOLOXPAFPNQAT',
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=False,
        act_cfg=dict(type='ReLU')
        ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        in_channels=64,
        feat_channels=64,
        use_depthwise=False,
        act_cfg=dict(type='ReLU')
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
)

model = dict(
    type='mmrazor.SPOS',
    architecture=architecture,
    mutator=dict(type='mmrazor.NasMutator'),
    )

# dataset settings
dataset_type = 'CocoDataset'
train_pipeline = [
    dict(type='Mosaic', img_scale=yolox_img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 2.0),
        border=(-yolox_img_scale[0]//2, -yolox_img_scale[1]//2)),
    dict(
        type='MixUp',
        img_scale=yolox_img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Pad',
        pad_to_square=False,
        size = yolox_img_scale,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs'),
    
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        metainfo = {'classes':'person'},
        data_root='/mnt/data/tju_coco_voc_person/images/',
        ann_file='/mnt/data/tju_coco_voc_person/annotations/sum_train_20230506.json',
        #'/mnt/data/tju_coco_voc_person/annotations/coco_person_train.json', 
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=None),
    pipeline=train_pipeline
)

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset
)

img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=False,
        size = yolox_img_scale,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo = {'classes':'person'},
        ann_file='/mnt/data/dhd_campus_test_images/annotations/tj_campus_test_20230129.json',
        data_prefix=dict(img='/mnt/data/dhd_campus_test_images/images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        ),
   
)

custom_hooks = [
    dict(type='mmrazor.SPOS_YOLOXModeSwitchHook', num_last_epochs=15),
    dict(type='mmrazor.Old_SyncNormHook',num_last_epochs=15, interval=10),
    dict(
        type='EMAHook',
        begin_epoch = 185,
        momentum=0.0001
        )
]

#val_evaluator = dict(type='CocoMetric',metric='bbox')
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/mnt/data/dhd_campus_test_images/annotations/tj_campus_test_20230129.json',
    metric=['bbox'],
    format_only=False)
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/mnt/data/dhd_campus_test_images/annotations/tj_campus_test_20230129.json',
    metric=['bbox'],
    format_only=True,
    outfile_prefix='./work_dirs/yolox_test')
# TODO(shiguang): handle fp16 properly.
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', 
        lr=0.01, 
        momentum=0.95, 
        weight_decay=5e-4,
        nesterov=True,
        ),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=None,
    )

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0005,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type = 'CosineAnnealingLR',
        begin = 5,
        end = 185,
        by_epoch=True,
        eta_min_ratio = 0.05,
        # update by iter
        convert_to_iter_based=True,
    ),
    dict(
        type='LinearLR',
        start_factor=1.0,
        by_epoch=True,
        begin=185,
        end=200,
        # update by iter
        convert_to_iter_based=True,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10, dynamic_intervals=[(185, 1)])
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10))
val_cfg = dict()
test_cfg = dict()
