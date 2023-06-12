_base_ = [
    'mmdet::yolox/yolox_s_8xb8-300e_coco.py'
]

custom_imports = dict(
    imports=[
        'projects.common.engine.runner.superacme_quantization_loops',
        'projects.superacme_backend.models.algorithms.quantization.superacme_architecture',
        'projects.superacme_backend.models.quantizers.superacme_quantizer',
        'projects.detection-quant.models.backbones.mobilenet_v2_mmclass',
        'projects.detection-quant.models.necks.yolox_pafpn_qat',
    ],
    allow_failed_imports=False
)
#dataset
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
dataset_type = 'CocoDataset'
data_root = '/mnt/data/voc2coco/'#voc2coco:/alg-data/ftp-upload/datasets/voc2coco/
float_checkpoint='/home/wangcheng/code/gitlab/main_dev_observers/mmrazor_superacme/projects/detection-quant/weights/yolox100_787_formmrazor.pth'
#model
img_scale = (640, 640)
test_img_scale=(416,416)
widen_factor=1.0

default_channels=[32, 96, 320]
neck_in_chanels = [int(ch*widen_factor) for ch in default_channels]

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
                random_size_range=(320, 640),
                size_divisor=32,
                interval=10)],
    ),
    backbone=dict(
        type='mmrazor.MobileNetV2_mmclass',
        widen_factor=widen_factor,
        out_indices=(2, 4, 6),
        act_cfg=dict(type='ReLU6'),
        init_cfg=None),
    neck=dict(
        type='mmrazor.YOLOXPAFPNQAT',
        in_channels=neck_in_chanels,
        out_channels=neck_in_chanels[1],
        num_csp_blocks=1,
        use_depthwise=True),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=len(classes),
        in_channels=neck_in_chanels[1],
        feat_channels=neck_in_chanels[1],
        use_depthwise=True),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))



train_pipeline = [
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
backend_args = None 

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/trainvoc_annotations.json',
        data_prefix=dict(img='jpeg/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=test_img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo = {'classes':classes},
        ann_file='annotations/testvoc2_annotations.json',
        data_prefix=dict(img='jpeg/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/testvoc2_annotations.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator


# optimizer
# default 8 gpu
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=1e-6, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=None)
# learning policy
max_epochs = 100
num_last_epochs = 15
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=1e-6 * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]


global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=False),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

model = dict(
    _delete_=True,
    type='mmrazor.SuperAcmeArchitectureQuant',
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
    architecture=architecture,
    float_checkpoint=float_checkpoint,
    input_shapes =(1, 3, 416, 416),
    quantizer=dict(
        type='mmrazor.SuperAcmeQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.predict_by_feat',  # noqa: E501
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.loss_by_feat',
            ])))


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
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    # dict(
    #     type='EMAHook',
    #     ema_type='ExpMomentumEMA',
    #     momentum=0.0001,
    #     update_buffers=True,
    #     priority=49),
    dict(type='ExportQATHook', interval=1, by_epoch=True,save_best='qat.coco/bbox_mAP_50')]

