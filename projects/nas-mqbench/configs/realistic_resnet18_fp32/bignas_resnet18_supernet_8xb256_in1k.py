_base_ = [
    'mmcls::_base_/default_runtime.py',
    # TODO(shiguang): check augmentation and optim
    'mmrazor::_base_/settings/imagenet_bs2048_ofa.py',
    './bignas_resnet18_supernet.py',
]

custom_imports = dict(
    imports=[
        'projects.nas-mqbench.models.architectures.backbones.searchable_resnet',
        'projects.nas-mqbench.engine.runner.subnet_val_analysis_loop',
        'projects.nas-mqbench.engine.optim.independent_constructor',
        'projects.nas-mqbench.engine.optim.param_scheduler'
    ],
    allow_failed_imports=False)

supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    data_preprocessor=_base_.data_preprocessor,
    backbone=_base_.nas_backbone,
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=560,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    input_resizer_cfg=_base_.input_resizer_cfg,
    connect_head=dict(connect_with_backbone='backbone.last_mutable_channels'),
)

model = dict(
    _scope_='mmrazor',
    type='BigNAS',
    drop_path_rate=0.0,
    num_random_samples=2,
    # Note index start from 1
    backbone_dropout_stages=[3, 4],
    architecture=supernet,
    distiller=dict(
        type='ConfigurableDistiller',
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kd=dict(type='KDSoftCELoss', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kd=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=False)))),
    mutator=dict(type='mmrazor.NasMutator'))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# optim_wrapper = dict(
#     # constructor='mmrazor.IndependentOptimWrapperConstructor',
#     type='AmpOptimWrapper',)
    #  clip_grad=dict(type='value', clip_value=1.0))
    # clip_grad=dict(type='norm', max_norm=1.0))

# _base_.param_scheduler += [
#     # # # weight_decay
#     # dict(
#     #     type='CosineAnnealingParamScheduler',
#     #     param_name='weight_decay',
#     #     T_max=max_epochs,
#     #     by_epoch=True,
#     #     begin=0,
#     #     end=max_epochs,
#     #     eta_min_ratio=0.1),
#     dict(
#         type='mmrazor.QRangeParamScheduler',
#         param_name='weight_decay',
#         verbose=True,
#         monitor='max_subnet.model'),    
# ]

default_hooks = dict(
    # param_scheduler=dict(priority='NORMAL'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best=None))

train_dataloader = dict(batch_size=256, pin_memory=True)
# params_modes=('org', 'fuse_conv_bn', 'cle')
val_cfg = dict(type='SubnetValAnalysisLoop', params_modes=('org', ), topk_params=10, calibrate_sample_num=4096)
