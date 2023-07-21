_base_ = [
    'mmcls::_base_/default_runtime.py',
    './bignaspfc_mbf_supernet.py',
]

custom_imports = dict(
    imports=[
        'projects.commons.models.task_modules.estimators.ov_estimator',
        'projects.commons.models.task_modules.estimators.heron_estimator',
        'projects.commons.engine.runner.subnet_ov_val_loop',
        'projects.face-recognition.models.algorithms.bignas_partialfc',
        'projects.face-recognition.models.losses.margin_loss',
        'projects.face-recognition.models.heads.partialfc_head',
        'projects.face-recognition.models.backbones.searchable_mobilefacenet',
        'projects.face-recognition.datasets.mx_face_dataset',
        'projects.face-recognition.evaluation.match_rank',
        'projects.face-recognition.engine.runner.loopx',
        'projects.face-recognition.engine.optimizers.optimizer_constructor',
    ],
    allow_failed_imports=False)


# model settings
supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    data_preprocessor=dict(
        # num_classes=1000,
        # RGB format normalization parameters
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        # convert image from BGR to RGB
        to_rgb=True,
    ),
    backbone=_base_.nas_backbone,
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.PartialFCHead',
        embedding_size=_base_.embedding_size,
        num_classes=2059906,
        sample_rate=0.2,
        fp16=False,
        loss=dict(
            type='mmrazor.CombinedMarginLoss',
            s=64,
            m1=1.0,
            m2=0.0,
            m3=0.4,
            interclass_filtering_threshold=0.0),
    ),
    input_resizer_cfg=_base_.input_resizer_cfg,
    # connect_head=dict(connect_with_backbone='backbone.last_mutable_channels'),
)

model = dict(
    _scope_='mmrazor',
    type='BigNASPartialFC',
    num_random_samples=2,
    architecture=supernet,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            feat=dict(type='mmrazor.ModuleOutputs', source='backbone.features.layers.2')),
        teacher_recorders=dict(
            feat=dict(type='mmrazor.ModuleOutputs', source='backbone.features.layers.2')),
        distill_losses=dict(
            loss_feat=dict(type='mmrazor.L2Loss', loss_weight=1, teacher_detach=True)),
        loss_forward_mappings=dict(
            loss_feat=dict(
                s_feature=dict(
                    from_student=True,
                    # TODO(shiguang): connector='loss_s4_sfeat',
                    recorder='feat'),
                t_feature=dict(
                    from_student=False, recorder='feat')))),
    mutator=dict(type='mmrazor.NasMutator'))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASPartialFCDDP',
    exclude_module='architecture.head',
    broadcast_buffers=False,
    find_unused_parameters=True)

# dataset settings
dataset_type = 'mmrazor.MXFaceDataset'
train_pipeline = [
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
train_dataloader = dict(
    batch_size=256,
    num_workers=3,
    drop_last=True,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/data/webface260m/',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

mdataset_type = 'mmrazor.MatchFaceDataset'
sample_ratio = 1.0
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='PackClsInputs',
        meta_keys=('sample_idx_identical_mapping', 'dataset_name')),
]
val_dataloader = dict(
    batch_size=128,
    num_workers=1,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=mdataset_type,
                dataset_name='120P',
                sample_ratio=sample_ratio,
                data_root='/mnt/data/face_data/OV_test/face_recognition/120P',
                key_file='120p.key',
                data_prefix=dict(img_path='norm_facex'),
                pipeline=val_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='baby_500p',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/baby_500p',
                key_file='baby_500p.key',
                data_prefix=dict(img_path='norm_face'),
                pipeline=val_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='glint1k',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/glint1k',
                key_file='glint1k_val.key',
                data_prefix=dict(img_path='images'),
                pipeline=val_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='mask_369p',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/mask_369p',
                key_file='mask_369p.key',
                data_prefix=dict(img_path='norm_face'),
                pipeline=val_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='menjin_20p',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/menjin_20p',
                key_file='menjin_20p.key',
                data_prefix=dict(img_path='norm_face'),
                pipeline=val_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='xm14',
                sample_ratio=sample_ratio,
                data_root='/mnt/data/face_data/OV_test/face_recognition/xm14',
                key_file='xm14.key',
                data_prefix=dict(img_path='norm_112'),
                pipeline=val_pipeline),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='mmrazor.Rank1')

# If you want standard test, please manually configure the test dataset
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # Since `data_preprocessor` will not be composed into onnx model when using `tensor` mode for foward,
    # We have to move the corresonding transforms to here.
    dict(type='Normalize', mean=127.5, std=127.5),
    dict(
        type='PackClsInputs',
        meta_keys=('sample_idx_identical_mapping', 'dataset_name')),
]


test_dataloader = dict(
    batch_size=128,
    num_workers=1,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=mdataset_type,
                dataset_name='120P',
                sample_ratio=sample_ratio,
                data_root='/mnt/data/face_data/OV_test/face_recognition/120P',
                key_file='120p.key',
                data_prefix=dict(img_path='norm_facex'),
                pipeline=test_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='baby_500p',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/baby_500p',
                key_file='baby_500p.key',
                data_prefix=dict(img_path='norm_face'),
                pipeline=test_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='glint1k',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/glint1k',
                key_file='glint1k_val.key',
                data_prefix=dict(img_path='images'),
                pipeline=test_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='mask_369p',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/mask_369p',
                key_file='mask_369p.key',
                data_prefix=dict(img_path='norm_face'),
                pipeline=test_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='menjin_20p',
                sample_ratio=sample_ratio,
                data_root=
                '/mnt/data/face_data/OV_test/face_recognition/menjin_20p',
                key_file='menjin_20p.key',
                data_prefix=dict(img_path='norm_face'),
                pipeline=test_pipeline),
            dict(
                type=mdataset_type,
                dataset_name='xm14',
                sample_ratio=sample_ratio,
                data_root='/mnt/data/face_data/OV_test/face_recognition/xm14',
                key_file='xm14.key',
                data_prefix=dict(img_path='norm_112'),
                pipeline=test_pipeline),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_evaluator = val_evaluator

# optimizer
optim_wrapper = {
    'constructor':
    'mmrazor.FaceSeparateOptimWrapperConstructor',
    'architecture.backbone':
    dict(
        type='AmpOptimWrapper',
        loss_scale=dict(growth_interval=100),
        optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4),
        clip_grad=dict(type='norm', max_norm=5)),
    'architecture.head':
    dict(
        type='AmpOptimWrapper',
        loss_scale=dict(growth_interval=100),
        optimizer=dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4), )
}

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0005,
        by_epoch=True,
        begin=0,
        end=2,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='PolyLR',
        power=2.0,
        by_epoch=True,
        begin=2,
        end=10,
        # update by iter
        convert_to_iter_based=True,
    )
]

# train, val, test setting
# Note that: Use LoopX is a little faster than EpochBasedTrainLoop
train_cfg = dict(
    type='mmrazor.EpochBasedTrainLoop', max_epochs=10, val_interval=1)
test_cfg = dict(
    type='mmrazor.SubnetOVValLoop',
    evaluate_fixed_subnet=True,
    calibrate_sample_num=40960,
    estimator_cfg=dict(
        type='mmrazor.HERONResourceEstimator',
        heronmodel_cfg=dict(
            work_dir='work_dirs/mbf_8xb512_wf42m_pfc02',
            ptq_json='projects/commons/heron_files/face_config_ptq.json',
            HeronCompiler = '/alg-data/ftp-upload/private/wangshiguang/HeronRT/HeronRT_v0.8.0_2023.06.15/tool/HeronCompiler',
            HeronProfiler = '/alg-data/ftp-upload/private/wangshiguang/HeronRT/HeronRT_v0.8.0_2023.06.15/tool/HeronProfiler'
        )))
    # estimator_cfg=dict(
    #     type='mmrazor.OVResourceEstimator',
    #     ovmodel_cfg=dict(
    #         work_dir='work_dirs/mbf_8xb256_wf42m_pfc02',
    #         qdef_file_dir='projects/commons/ov_qdefs',
    #         qfnodes='qdef_ifm_q8.qfnodes',
    #         qfops='qdef_q8.qfops',
    #         ifmq='q8',
    #         # Uncomment and adjust `num_infer` for QoR
    #         # infer_metric=test_evaluator,
    #         num_infer=200,
    #         num_calib=5)))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)

# _base_.default_hooks.logger.interval = 10

val_cfg = dict(type='mmrazor.SubnetValLoop', calibrate_sample_num=40960)
