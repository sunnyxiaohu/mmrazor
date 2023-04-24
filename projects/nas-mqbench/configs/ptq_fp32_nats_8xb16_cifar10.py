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
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=10,
        mean=[125.3, 123.0, 113.9],
        std=[63.0, 62.1, 66.7],
        to_rgb=False),
    backbone=dict(
        type='mmrazor.NATSBackbone',
        benchmark_api=dict(
            file_path_or_dict='/alg-data/ftp-upload/private/wangshiguang/datasets/NATS/NATS-tss-v1_0-3ffb9-full',
            # file_path_or_dict='/alg-data/ftp-upload/private/wangshiguang/datasets/NATS/NATS-sss-v1_0-50262-full',
            search_space='tss',
            fast_mode=True,
            verbose=True),
        arch_index=1452,  # 1017, 1452, 1990
        dataset='cifar10',
        # NATS-tss: (hp, seed) -> (12, 111), (200, 777|888|999)
        # NATS-sss: (hp, seed) -> (01|12|90, 777)
        seed=777,
        hp='200',  # 12, 200
    ),
    head=dict(
        type='mmcls.ClsHead'
    ))

_base_.val_dataloader.batch_size = 64
# For memory saving.
_base_.train_dataloader.num_workers = 1
_base_.val_dataloader.num_workers = 1
_base_.test_dataloader.num_workers = 1
