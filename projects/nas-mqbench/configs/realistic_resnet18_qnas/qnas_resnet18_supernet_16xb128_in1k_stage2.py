_base_ = [
    './qnas_resnet18_supernet_16xb128_in1k.py',
]

optim_wrapper = dict(
    optimizer=dict(lr=0.008),
    paramwise_cfg=dict(
        _delete_=True,
        bias_decay_mult=0.0, norm_decay_mult=0.0,
        # custom_keys={
        # 'architecture.qmodels': dict(lr_mult=0.1)},
        bypass_duplicate=True
    ))

max_epochs = 20

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QNASEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=5,
    qat_begin=1,
    freeze_bn_begin=-1)

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=max_epochs,
    by_epoch=True,
    begin=0,
    end=max_epochs)
