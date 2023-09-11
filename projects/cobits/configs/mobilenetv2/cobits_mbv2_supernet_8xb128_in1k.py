_base_ = [
    './mobilenet-v2_8xb128-warmup-lbs-coslr-nwd_in1k.py',
]

custom_imports = dict(
    imports = [
    'projects.nas-mqbench.models.algorithms.qnas',
    'projects.nas-mqbench.models.quantizers.mutable_quantizer',
    'projects.nas-mqbench.engine.runner.qnas_loops',
    'projects.nas-mqbench.models.architectures.dynamic_qops.dynamic_lsq',
    'projects.nas-mqbench.models.observers.batch_lsq',
])

_base_.data_preprocessor.type = 'mmcls.ClsDataPreprocessor'
_base_.model.backbone.conv_cfg = dict(type='mmrazor.BigNasConv2d')
_base_.model.backbone.norm_cfg = dict(type='mmrazor.DynamicBatchNorm2d')
_base_.model.init_cfg = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'work_dirs/pretrained_models/mobilenet-v2_8xb128-warmup-lbs-coslr-nwd_in1k/20230906_112051/epoch_250.pth')

global_qconfig = dict(
    w_observer=dict(type='mmrazor.BatchLSQObserver'),
    a_observer=dict(type='mmrazor.BatchLSQObserver'),
    w_fake_quant=dict(type='mmrazor.DynamicBatchLearnableFakeQuantize'),
    # a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.DynamicBatchLearnableFakeQuantize'),
    # w_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=True),
    # a_qscheme=dict(qdtype='quint8', bit=4, is_symmetry=True),
    w_qscheme=dict(qdtype='qint8', bit=4, is_symmetry=True, zero_point_trainable=True, extreme_estimator=1, residual_mode=0, param_share_mode=5),
    a_qscheme=dict(qdtype='quint8', bit=4, is_symmetry=True, zero_point_trainable=True, extreme_estimator=1, residual_mode=0, param_share_mode=5)
)
# Make sure that the architecture and qmodels have the same data_preprocessor.
qmodel = dict(
    _scope_='mmcls',
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.data_preprocessor,
    architecture=_base_.model,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict', 'loss'),
    quantizer=dict(
        type='mmrazor.MutableOpenVINOQuantizer',
        quant_bits_skipped_module_names=[
            'backbone.conv1.conv',
            'head.fc'
        ],
        quant_bits=[4, 6, 8],
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
    optimizer=dict(lr=0.02, weight_decay=0.00001),
    paramwise_cfg=dict(
        # custom_keys={
        # 'architecture.qmodels': dict(lr_mult=0.1)},
        bypass_duplicate=True
    ),)

model_wrapper_cfg = dict(
    type='mmrazor.QNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEpochBasedLoop',
    max_epochs=100,
    val_interval=5,
    qat_begin=1,
    freeze_bn_begin=-1)

# total calibrate_sample_num = 256 * 8 * 2
val_cfg = dict(_delete_=True, type='mmrazor.QNASValLoop', calibrate_sample_num=65536, quant_bits=[4, 8])
# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(
    checkpoint=dict(save_best=None, max_keep_ckpts=1),
    sync=dict(type='SyncBuffersHook'))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]