_base_ = ['./default_runtime.py', './c3k2_v19_test1_pfkvnsf_qat_t04_dataset.py']


# float_checkpoint='/alg-secure-data/yangjunpei/newnettrain/dev_keypoint_old/mmdetection_superacme/work_dirs/c3k2_v19_test1_pfkvnsf_float_t05/best_coco_sum_of_average_PR95_epoch_10.pth'
float_checkpoint='/alg-secure-data/yangjunpei/newnettrain/dev_keypoint_old/mmdetection_superacme/work_dirs/c3k2_v19_test1_pfkvnsf_float_t08/best_coco_sum_of_average_PR95_epoch_16.pth'

# change to your script_workdir
script_workdir='work_dirs/c3k2_v19_test1_pfkvnsf_qat_t04_wxax/results'
script_results_path=f'{script_workdir}/results_new.json'

custom_imports = dict(
    imports=[
        'projects.person_det.models.backbones.csp_darknet_common',
        'projects.person_det.models.necks.yolox_pafpn_common',
        'projects.person_det.datasets.cross_domin_batchsample',
        'projects.person_det.datasets.cross_domain_dataset',
        'projects.person_det.datasets.samplers.group_sampler',
        'projects.person_det.engine.hooks.filter_negative_image_hook',
        'projects.person_det.evaluation.metrics.coco_superacme_metric_',
        'projects.person_det.engine.schedulers.yolox_warmup',
        'projects.person_det.engine.schedulers.yolox_cosineannealinglr',
        'projects.PVN_det_qat.mmrazor.heron_estimator',
        # 'projects.quant_observers.models.observers.mse',
    ], 
    allow_failed_imports=False
)

# training settings
max_epochs = 50
num_last_epochs = 49
interval = 1

load_from=None
resume=False

neck_in_chanels=[64, 128, 256]
neck_out_channels=64
head_feat_channels=64
act_cfg=dict(type='Swish')

# model settings
data_preprocessor=dict(
    type='DetDataPreprocessor',
    pad_size_divisor=32,
    batch_augments=[
        dict(
            type='BatchSyncRandomResize',
            random_size_range=(384, 576), 
            size_divisor=32,
            interval=10)
    ])

floatmodel=dict(
    type='YOLOX',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='CSPDarknetCommon', arch="heron", deepen_factor=0.33, widen_factor=0.25, act_cfg=act_cfg, use_depthwise=False,
                  arch_ovewrite=[[128, 128, 3, True, False], [128, 256, 9, True, False],[256, 512, 12, True, False], [512, 1024, 3, False, False]],
                  struct_param = [
                    {"firstconv":"FirstConvCommon", "param":{"mid_channels":32, "kernel_size":3, "rm2conv":False}},
                    {"block":"FusedIBN_CSPLayer", "param":{"expand_ratio":1,"num_blocks":1}},
                    {"block":"C3k2Layer", "param":{"expand_ratio":0.5, "num_blocks":1, "usec3k":False}},
                    {"block":"C3k2Layer", "param":{"expand_ratio":0.5, "num_blocks":2, "usec3k":True}},
                    {"block":"C3k2SppfLayer", "param":{"expand_ratio":0.25, "num_blocks":2, "usec3k":True}},
                  ]),
    neck=dict(
        type='YOLOXPAFPNCommon',
        in_channels=neck_in_chanels,
        out_channels=neck_out_channels,
        num_csp_blocks=1,
        reduce_conv_kernel=1,
        act_cfg=act_cfg, use_depthwise=False,
        struct_param = [
            [{"neck_down_2":"C3k2Layer", "param":{"expand_ratio":0.5, "num_blocks":1, "usec3k":True}},
             {"neck_down_1":"C3k2Layer", "param":{"expand_ratio":0.5, "num_blocks":1, "usec3k":True}},
            ],
            [{"neck_up_0":"C3k2Layer", "param":{"expand_ratio":0.5, "num_blocks":1, "usec3k":True}},
             {"neck_up_1":"C3k2Layer", "param":{"expand_ratio":0.5, "num_blocks":1, "usec3k":True}},
            ]
        ]),
    bbox_head=dict(
        type='YOLOXKeypointHead',
        kp_num=5,
        kp_conv_num=1,
        mind_cls_index=(0,1,2,3,4,5),
        mind_cls_cls_weight=(1.5, 0.85, 1.1, 3.0, 1.0, 1.0),
        mind_cls_bbox_weight=(1.5, 0.85, 1.1, 3.0, 1.0, 1.0),
        loss_kps=dict(type='SoftWingLoss', omega1=10.0, omega2=20.0, epsilon=0.5, loss_weight=1.0),
        use_kps_weight=True,
        num_classes=len(_base_.metaclass['classes']),
        in_channels=64,
        feat_channels=64,
        stacked_convs=1,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=act_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.4),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
        negcls=dict(use_negcls=True,
                    type='pred_score',
                    negcls_th=0.45,
                    negcls_weight=0.1,
                    negcls_iouth=0.01,
                    begin_ep=max_epochs-num_last_epochs)
        ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5, ignore_iof_thr=0.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

base_lr = 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam', lr=base_lr, weight_decay=1e-5),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.,bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[5,10,15],
        gamma=0.1,)
]

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMSEObserver'),
    a_observer=dict(type='mmrazor.EMAMSEObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=4, is_symmetry=True, is_symmetric_range=False),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=False, averaging_constant=0.1),
)

model = dict(
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=data_preprocessor,
    architecture=floatmodel,
    float_checkpoint=float_checkpoint,
    use_cle = True,
    cle_float_checkpoint=None,
    #cle_float_checkpoint= '/alg-secure-ftp/upload/wangshiguang/projects/PVN/mmdetection_v2.23.0/cle.pth',
    input_shapes =(1, 3, 480, 800),
    quantizer=dict(
        type='mmrazor.SuperAcmeQuantizer',
        global_qconfig=global_qconfig,
        quant_bits_skipped_module_names=[
            ['call_module', 'backbone.stem.conv_focus.conv', [4], None],
            ['call_function', 'sigmoid', None, [4]],
            ['call_function', 'mul', None, [4, 4]],
            ['call_module', 'backbone.stem.conv.conv', [8], [4]], # t1
            # ['call_function', 'sigmoid_1', None, [4]],
            # ['call_function', 'mul_1', None, [4, 4]],
            # ['call_module', 'backbone.stage1.0.conv', [8], [4]], # t3
            # 'backbone.stage1.1.blocks.0.expand_conv.conv',
            # 'backbone.stage1.1.blocks.0.linear_conv.conv',
            # 'backbone.stage2.0.conv',
            # 'backbone.stage3.0.conv', (x) input and output's datatype of sigmoid must be equal.
        ],
        use_cle=True,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.yoloxkeypoint_head.YOLOXKeypointHead.predict_by_feat',  # noqa: E501
                'mmdet.models.dense_heads.yoloxkeypoint_head.YOLOXKeypointHead.loss_by_feat',
            ])))

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    type='mmrazor.LSQEpochBasedLoop',
    calibrate_steps=50,
    max_epochs=max_epochs,
    val_interval=interval,
    val_begin=5,
    freeze_bn_begin=-1,
    calibrate_dataloader = _base_.cali_dataloader
    )

val_cfg = dict( type='mmrazor.QATValLoop', only_qat=True)

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'),
                     checkpoint=dict(interval=1,max_keep_ckpts=100,save_best='qat.coco/sum_of/average_PR95',rule='greater'))

train_dataset = _base_.train_dataset
weak_aug_pipeline = _base_.weak_aug_pipeline
test_pipeline = _base_.test_pipeline

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='FilterNegativeImageHook', 
            train_datasets=train_dataset, 
            train_pipeline=weak_aug_pipeline,
            test_pipeline=test_pipeline,
            num_last_epochs=num_last_epochs,
            batch_size=24,
            num_work=8,
            filter_th=0.01,
            ann_image_ratio=0.25,
            sampler_cfg=dict(
                sampler=dict(type='DistributedGroupSampler', samples_per_gpu=15),
                batch_sampler =dict(type='CrossMultiDomainBatchSampler', add_domain_num=4, batch_size=[15, 3, 3, 1, 2]),),
                ),
    dict(type='SyncNormHook', priority=48),
]

randomness = dict(deterministic=False, seed=None)

# Pay attention !!! for offline test gen jsonfiles so we comment this test_config,if you want deploy for heron please open it 
val_evaluator = dict(
    type='CocoSuperAcmeMetric',
    ann_file=[ds['ann_file'] for ds in _base_.val_dataset['datasets']],
    metric='bbox',
    ignore_min_area=(14*26, 8*8, 26*26, 14*26, 14*26, 14*26, 1),
    superAcme = True,
    scale=(1920, 1080), 
    keep_ratio=True,
    script_cmd=f'bash /alg-secure-data/yangjunpei/shiguangdebug/v3quant/detection_evalution/runcommand_pfkvnsm.sh {script_workdir} {script_workdir} {script_results_path}',
    script_results_path=script_results_path,
    outfile_prefix=f'{script_workdir}/',
    backend_args=None)

test_evaluator = dict(
    type='CocoSuperAcmeMetric',
    ann_file=[ds['ann_file'] for ds in _base_.test_dataset['datasets']],
    metric='bbox',
    ignore_min_area=(14*26, 8*8, 26*26, 14*26, 14*26, 14*26, 1),
    superAcme = True,
    scale=(1920, 1080), 
    keep_ratio=True,
    script_cmd=f'bash /alg-secure-data/yangjunpei/shiguangdebug/v3quant/detection_evalution/runcommand_pfkvnsm.sh {script_workdir} {script_workdir} {script_results_path}',
    script_results_path=script_results_path,
    outfile_prefix=f'{script_workdir}/',
    backend_args=None)

test_cfg = val_cfg
test_cfg = dict(
    type='mmrazor.SubnetExportValLoop',
    evaluate_fixed_subnet=True,
    calibrate_sample_num=0,
    estimator_cfg=dict(
        type='mmrazor.HERONResourceEstimator',
        heronmodel_cfg=dict(
            type='mmrazor.YoloxHERONModelWrapper',
            is_quantized=True,
            build_args='--saveSimpleModel 1 ', # --withNetInfo 1 
            profiler_args='-b 1.5 -f 0.5 -L ',
            work_dir='./work_dirs/c3k2_v19_test1_pfkvnsf_qat_t04_wxax',
            mnn_quant_json='projects/pvnfs/configs/config_qat.json',
            # Uncomment and adjust `num_infer` for QoR
            num_infer=0,
            infer_metric=test_evaluator,
            # Use netron to checkout outputs_mapping.
            outputs_mapping = {
            },
            # onnx_node_debug_mode=True,
            onnx_node_tensor_translate_mapping = {
            '/multi_level_conv_cls.0/Conv_output_0':'5860',
            '/multi_level_conv_obj.0/Conv_output_0':'5870',
            '/multi_level_conv_reg.0/Conv_output_0':'5865',
            '/multi_level_conv_cls.1/Conv_output_0':'5887',
            '/multi_level_conv_obj.1/Conv_output_0':'5897',
            '/multi_level_conv_reg.1/Conv_output_0':'5892',
            '/multi_level_conv_cls.2/Conv_output_0':'5914',
            '/multi_level_conv_obj.2/Conv_output_0':'5924',
            '/multi_level_conv_reg.2/Conv_output_0':'5919',
            '/multi_level_conv_kps.0.1/Conv_output_0':'5920',
            '/multi_level_conv_kps.1.1/Conv_output_0': '5921',
            '/multi_level_conv_kps.2.1/Conv_output_0': '5922'            
            }
        )))


