_base_ = [
    'mmcls::resnet/resnet18_8xb16_cifar10.py'
]

custom_imports = dict(
    imports=[
        'projects.nas-mqbench.models.architectures.backbones.nats_backbone',
        'projects.nas-mqbench.datasets.imagenet16'
    ],
    allow_failed_imports=False)

# dataset settings
dataset_type = 'mmrazor.ImageNet16'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_prefix='data/ImageNet16/',
        class_nums=120))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_prefix='data/ImageNet16/',
        class_nums=120))

test_dataloader = val_dataloader


model = dict(
    _delete_=True,
    type='mmcls.ImageClassifier',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=120,
        mean=[122.68, 116.66, 104.01],
        std=[63.22, 61.26, 65.09],
        to_rgb=False),
    backbone=dict(
        type='mmrazor.NATSBackbone',
        benchmark_api=dict(
            file_path_or_dict='/alg-data/ftp-upload/private/wangshiguang/datasets/NATS/NATS-tss-v1_0-3ffb9-full',
            search_space='tss',
            fast_mode=True,
            verbose=True),
        arch_index=1017,  # 1017, 1452, 1990
        dataset='ImageNet16-120',
        # (hp, seed) -> (12, 111), (200, 777|888|999)
        seed=777,
        hp='200',
    ),
    head=dict(
        type='mmcls.ClsHead'
    ))

_base_.val_dataloader.batch_size = 128
_base_.train_dataloader.num_workers = 2
_base_.val_dataloader.num_workers = 2
_base_.test_dataloader.num_workers = 2
