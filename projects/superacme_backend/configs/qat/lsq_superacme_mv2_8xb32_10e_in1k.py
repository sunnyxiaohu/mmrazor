_base_ = ['mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py',]
mv2 = _base_.model
float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'  # noqa: E501

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/mnt/data/imagenet')
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/mnt/data/imagenet')
)
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/mnt/data/imagenet')
)
global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQPerChannelObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=False),
    a_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=1000,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    architecture=mv2,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.SuperAcmeQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    _delete_=True, type='ConstantLR', factor=1.0, by_epoch=True)

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=10,
    val_interval=1,
    freeze_bn_begin=1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))

test_cfg = dict(
    _delete_=True,
    type='mmrazor.SubnetExportValLoop',
    evaluate_fixed_subnet=True,
    calibrate_sample_num=0,
    is_supernet=False,
    estimator_cfg=dict(
        type='mmrazor.HERONResourceEstimator',
        heronmodel_cfg=dict(
            type='mmrazor.HERONModelWrapper',
            is_quantized=True,
            work_dir='work_dirs/lsq_superacme_mv2_8xb32_10e_in1k',
            mnn_quant_json='projects/commons/heron_files/config_qat.json',
            # Uncomment and adjust `num_infer` for QoR
            num_infer=1000,
            infer_metric=_base_.test_evaluator
        )))
