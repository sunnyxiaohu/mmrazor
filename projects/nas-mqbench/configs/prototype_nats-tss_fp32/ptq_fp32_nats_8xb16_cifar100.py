_base_ = [
    'mmcls::resnet/resnet50_8xb16_cifar100.py'
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
        num_classes=100,
        mean=[129.3, 124.1, 112.4],
        std=[68.2, 65.4, 70.4],
        to_rgb=False),
    backbone=dict(
        type='mmrazor.NATSBackbone',
        benchmark_api=dict(
            file_path_or_dict='/alg-data/ftp-upload/private/wangshiguang/datasets/NATS/NATS-tss-v1_0-3ffb9-full',
            search_space='tss',
            fast_mode=True,
            verbose=True),
        arch_index=1452,  # 1017, 1452, 1990
        dataset='cifar100',
        # (hp, seed) -> (12, 111), (200, 777|888|999)
        seed=111,
        hp='12',
    ),
    head=dict(
        type='mmcls.ClsHead'
    ))

_base_.val_dataloader.batch_size = 128
_base_.train_dataloader.num_workers = 2
_base_.val_dataloader.num_workers = 2
_base_.test_dataloader.num_workers = 2
