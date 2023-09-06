_base_ = [
    'mmcls::_base_/models/resnet18.py', 'mmcls::_base_/datasets/imagenet_bs64.py',
    'mmcls::_base_/schedules/imagenet_bs2048_coslr.py',
    'mmcls::_base_/default_runtime.py'
]
model = dict(
    _scope_='mmcls',
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
    ))
_base_.train_dataloader.dataset.pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='PackClsInputs'),
]
_base_.train_dataloader.batch_size = 256
optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(type='SGD', lr=0.8, momentum=0.9, weight_decay=0.0001, nesterov=True))
