_base_ = [
    'mmcls::resnet/resnet18_8xb16_cifar10.py'
]

custom_imports = dict(
    imports=[
        'projects.nas-mqbench.models.architectures.backbones.nats_backbone'
    ],
    allow_failed_imports=False)

model = dict(
    _delete_=True,
    type='mmcls.ImageClassifier',
    data_preprocessor = dict(
        num_classes=10,
        mean=[125.3, 123.0, 113.9],
        std=[63.0, 62.1, 66.7],
        to_rgb=False),
    backbone=dict(
        type='mmrazor.NATSBackbone',
        benchmark_api=dict(    
            file_path_or_dict='/mnt/lustre/wangshiguang/sftp-src/NATS-tss-v1_0-3ffb9-full-min',
            search_space='tss',
            fast_mode=True,
            verbose=True),
        arch_index=1017,  #  1017, 1452, 1990
        dataset='cifar10',
        seed=111,
        hp='12',
    ),
    head=dict(
        type='mmcls.ClsHead'
    ))

