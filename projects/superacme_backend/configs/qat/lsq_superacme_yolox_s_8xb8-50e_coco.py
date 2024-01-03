_base_ = [
    'mmdet::yolox/yolox_s_8xb8-300e_coco.py'
]

# float_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
float_checkpoint = '/alg-data/models/pretrained_models/yolox_s_8xb8-300e_coco/epoch_300.pth

_base_.train_dataloader.dataset.pipeline = [
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=_base_.img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

# optimizer
# default 8 gpu
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=2e-6, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0., bypass_duplicate=True),
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=False),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=False, zero_point_trainable=True),
)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)],
    ),
    architecture=_base_.model,  # architecture,
    float_checkpoint=float_checkpoint,
    input_shapes =(1, 3, 640, 640),
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
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# learning policy
max_epochs = 50  # 100
warm_epochs = 1
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.025,
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
# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=1,
    calibrate_steps=100,
    freeze_bn_begin=-1)

val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')
test_cfg = val_cfg

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))
custom_hooks = []
# custom_hooks = [
#     dict(type='YOLOXModeSwitchHook', num_last_epochs=299, priority=48),]
#     # dict(type='SyncNormHook', num_last_epochs=299, interval=10, priority=48)]

test_cfg = dict(
    _delete_=True,
    type='mmrazor.SubnetExportValLoop',
    evaluate_fixed_subnet=True,
    calibrate_sample_num=0,
    is_supernet=False,
    estimator_cfg=dict(
        type='mmrazor.HERONResourceEstimator',
        heronmodel_cfg=dict(
            type='mmrazor.HERONModelWrapperDet',
            is_quantized=True,
            work_dir='work_dirs/lsq_superacme_yolox_s_8xb8-50e_coco',
            mnn_quant_json='projects/commons/heron_files/det_config_qat.json',
            # Uncomment and adjust `num_infer` for QoR
            num_infer=None,
            infer_metric=_base_.test_evaluator,
            # Use netron to checkout outputs_mapping.
            # outputs_mapping = {
            #     '403': '~multi_level_conv_cls.0~Conv_output_0',
            #     '418': '~multi_level_conv_cls.1~Conv_output_0',
            #     '433': '~multi_level_conv_cls.2~Conv_output_0',
            #     '405': '~multi_level_conv_obj.0~Conv_output_0',
            #     '420': '~multi_level_conv_obj.1~Conv_output_0',
            #     '435': '~multi_level_conv_obj.2~Conv_output_0',
            #     '404': '~multi_level_conv_reg.0~Conv_output_0',
            #     '419': '~multi_level_conv_reg.1~Conv_output_0',
            #     '434': '~multi_level_conv_reg.2~Conv_output_0'
            # }
        )))