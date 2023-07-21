_base_ = [
    '../realistic_resnet18_fp32/bignas_resnet18_supernet_8xb256_in1k.py',
]

_base_.custom_imports.imports += [
    'projects.nas-mqbench.models.algorithms.qnas',
    'projects.nas-mqbench.models.quantizers.mutable_quantizer',
    'projects.nas-mqbench.engine.runner.qnas_loops',
    'projects.nas-mqbench.models.fake_quants.batch_lsq',
]

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

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQObserver'),
    # a_observer=dict(type='mmrazor.LSQObserver'),
    a_observer=dict(type='mmrazor.MinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    # a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.BatchLearnableFakeQuantize', zero_point_trainable=True, extreme_estimator=0),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)
# Make sure that the architecture and qmodels have the same data_preprocessor.
qmodel = dict(
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.data_preprocessor,
    architecture=supernet,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict', 'loss'),
    quantizer=dict(
        type='mmrazor.MutableOpenVINOQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.conv1',
            'head.fc'
        ],
        # quant_bits=[4, 8, 32],
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            # skipped_module_names=['input_resizer'],
            skipped_module_classes=[
                # 'mmrazor.models.architectures.dynamic_ops.bricks.dynamic_container.DynamicSequential',
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
    drop_path_rate=0.0,
    num_random_samples=2,
    # Note index start from 1
    backbone_dropout_stages=[3, 4],
    architecture=qmodel,
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
    qat_distiller=dict(
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

optim_wrapper = dict(
    paramwise_cfg=dict(
        # custom_keys={
        # 'architecture.qmodels': dict(lr_mult=0.1)},
        bypass_duplicate=True
    ),
    type='AmpOptimWrapper',)

model_wrapper_cfg = dict(
    type='mmrazor.QNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEpochBasedLoop',
    max_epochs=_base_.max_epochs,
    val_interval=5,
    qat_begin=81,
    freeze_bn_begin=-1)

# total calibrate_sample_num = 256 * 8 * 2
val_cfg = dict(_delete_=True, type='mmrazor.QNASValLoop', calibrate_sample_num=4096)
# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(
    checkpoint=dict(save_best=None),
    sync=dict(type='SyncBuffersHook'))
