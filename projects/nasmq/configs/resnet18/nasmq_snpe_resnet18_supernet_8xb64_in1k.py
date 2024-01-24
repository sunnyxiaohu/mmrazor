_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py', './resnet18_supernet.py']

custom_imports = dict(
    imports =['projects.nasmq.models.architectures.backbones.searchable_resnet'])

data_preprocessor = dict(
    type='mmcls.ClsDataPreprocessor',
    mean=[123.675, 116.28, 103.53,],
    std=[58.395, 57.12, 57.375,],
    to_rgb=True)
architecture = dict(
    _scope_='mmcls',
    type='mmrazor.SearchableImageClassifier',
    backbone=_base_.nas_backbone,
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.DynamicLinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
    ),
    connect_head=dict(connect_with_backbone='backbone.last_mutable_channels'))
global_qconfig = dict(
    w_observer=dict(type='mmrazor.BatchLSQObserver'),
    a_observer=dict(type='mmrazor.BatchLSQObserver'),
    w_fake_quant=dict(type='mmrazor.DynamicBatchLearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.DynamicBatchLearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=False, zero_point_trainable=True, extreme_estimator=1, param_share_mode=4),
    a_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=False, zero_point_trainable=True, extreme_estimator=1, param_share_mode=4)
)
# Make sure that the architecture and qmodels have the same data_preprocessor.
qmodel = dict(
    _scope_='mmcls',
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=data_preprocessor,
    architecture=architecture,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict', 'loss'),
    quantizer=dict(
        type='mmrazor.SNPEQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.conv1',
            'head.fc'
        ],
        w_bits=[3,4,5,6],
        a_bits=[3,4,5,6],
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_module_classes=[
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_conv.BigNasConv2d',
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_function.DynamicInputResizer',
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_linear.DynamicLinear',
                'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_norm.DynamicBatchNorm2d',
            ],
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='QNAS',
    num_random_samples=2,
    architecture=qmodel,
    mutator=dict(type='mmrazor.NasMutator'))

train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001, nesterov=True),
    paramwise_cfg=dict(
        bypass_duplicate=True
    ),)

model_wrapper_cfg = dict(
    type='mmrazor.QNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# learning policy
max_epochs = 50  #25
warm_epochs = 1
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
train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=5,
    qat_begin=10,
    freeze_bn_begin=-1)

# total calibrate_sample_num = 256 * 8 * 2
val_cfg = dict(_delete_=True, type='mmrazor.QNASValLoop', calibrate_sample_num=65536, quant_bits=[3,4,5,6])
# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'),
                     checkpoint=dict(save_best=None, max_keep_ckpts=1))

# _base_.train_dataloader.dataset.pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', scale=224, backend='pillow'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
#     dict(type='PackClsInputs'),
# ]
# _base_.train_dataloader.batch_size = 256
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(edge='short', scale=256, type='ResizeEdge', backend='pillow'),
#     dict(crop_size=224, type='CenterCrop'),
#     dict(type='PackClsInputs')
# ]
# _base_.val_dataloader.dataset.pipeline = test_pipeline
# _base_.test_dataloader.dataset.pipeline = test_pipeline
# optim_wrapper = dict(
#     _delete_=True,
#     optimizer=dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True))
