_base_ = ['./mbf_8xb256_wf42m_pfc02.py']

# model settings
teacher = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    data_preprocessor=dict(
        # num_classes=1000,
        # RGB format normalization parameters
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        # convert image from BGR to RGB
        to_rgb=True,
    ),
    # TODO: replace model config
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'work_dirs/mbf_scale1.5_8xb256_wf42m_pfc02/epoch_10.pth',  # noqa: E501
            prefix='architecture.backbone.'),        
        type='mmrazor.MobileFaceNet',
        num_features=256,
        fp16=False,
        scale=1.2,
        blocks=(2, 4, 6, 2)),
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.PartialFCHead',
        embedding_size=256,
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
    ))

model = dict(
    type='mmrazor.SPOSPartialFC',
    architecture=_base_.architecture,
    teacher=teacher,
    teacher_ckpt=None,
    distiller=dict(
        type='mmrazor.ConfigurableDistiller',
        student_recorders=dict(
            feat=dict(type='mmrazor.ModuleOutputs', source='backbone.features.layers.2')),
        teacher_recorders=dict(
            feat=dict(type='mmrazor.ModuleOutputs', source='backbone.features.layers.2')),
        distill_losses=dict(
            loss_feat=dict(type='mmrazor.L2Loss', loss_weight=20)),
        loss_forward_mappings=dict(
            loss_feat=dict(
                s_feature=dict(
                    from_student=True,
                    # TODO(shiguang): connector='loss_s4_sfeat',
                    recorder='feat'),
                t_feature=dict(
                    from_student=False, recorder='feat')))),
            # TODO(shiguang): KL loss
            # loss_feat=dict(
            #     preds_S=dict(from_student=True, recorder='feat'),
            #     preds_T=dict(from_student=False, recorder='feat')))
    mutator=dict(type='mmrazor.NasMutator'))

custom_hooks = [dict(type='mmrazor.StopDistillHook', stop_epoch=7)]
