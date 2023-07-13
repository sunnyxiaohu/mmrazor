_base_ = [
    './qnas_resnet18_supernet_16xb128_in1k.py',
]

optim_wrapper = dict(
    optimizer=dict(lr=0.016),
    paramwise_cfg=dict(
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
warm_epochs = 1
_base_.param_scheduler[0].end = warm_epochs
_base_.param_scheduler[1].T_max = max_epochs - warm_epochs
_base_.param_scheduler[1].begin = warm_epochs
_base_.param_scheduler[1].end = max_epochs
