_base_ = [
    'mmcls::_base_/default_runtime.py',
    # TODO(shiguang): check augmentation and optim
    'mmrazor::_base_/settings/imagenet_bs2048_ofa.py',
    './bignas_resnet18_supernet.py',
]

custom_imports = dict(
    imports=[
        'projects.nas-mqbench.models.architectures.backbones.searchable_resnet'
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
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=False)))),
    mutator=dict(type='mmrazor.NasMutator'))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

optim_wrapper = dict(
    type='AmpOptimWrapper',)
    #  clip_grad=dict(type='value', clip_value=1.0))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto'))

train_dataloader = dict(batch_size=128, pin_memory=True)
