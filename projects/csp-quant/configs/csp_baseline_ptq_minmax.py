_base_ = [
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'projects.common.engine.runner.superacme_quantization_loops',
        'projects.superacme_backend.models.algorithms.quantization.superacme_architecture',
        'projects.superacme_backend.models.quantizers.superacme_quantizer',
        'projects.detection-quant.models.backbones.mobilenet_v2_mmclass',
        'projects.csp-quant.models.backbones.csp_darknet',
        'projects.csp-quant.models.necks.yolox_pafpn_qat',
        'projects.quant_observers.models.observers.mse',
    ],
    allow_failed_imports=False
)

float_checkpoint = '/home/wangcheng/code/gitlab/main_dev_observers/mmrazor_superacme/projects/csp-quant/configs/best_mean_map50_epoch_198.pth'

img_scale =(640, 384) 
test_img_scale=(640, 384) 
yolox_img_scale = (384, 640) 



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
        data_root='/mnt/data/persondata/tju_coco_voc_person/images/',
        ann_file='/mnt/data/persondata/tju_coco_voc_person/annotations/coco_person_train.json',
        #'/mnt/data/persondata/tju_coco_voc_person/annotations/coco_person_train.json', 
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
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10, dynamic_intervals=[(185, 1)])
optim_wrapper ={}
val_cfg ={}
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
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo = {'classes':'person'},
        ann_file='/mnt/data/persondata/dhd_campus_test_images/annotations/tj_campus_test_20230129.json',
        data_prefix=dict(img='/mnt/data/persondata/dhd_campus_test_images/images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        ),
   
)

#val_evaluator = dict(type='CocoMetric',metric='bbox')
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/mnt/data/persondata/dhd_campus_test_images/annotations/tj_campus_test_20230129.json',
    metric=['bbox'],
    format_only=False)
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
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
    # train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
)

test_cfg = dict(
    type='mmrazor.SuperAcmePTQLoop',
    calibrate_dataloader=val_dataloader,
    calibrate_steps=900,
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True),
)


model = dict(
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
    input_shapes =(1, 3, 384, 640),
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

