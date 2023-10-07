_base_ = [
    './qnas_mobilenetv2_supernet_8xb128_in1k.py',
]

optim_wrapper = dict(
    optimizer=dict(lr=0.02),
    paramwise_cfg=dict(
        # custom_keys={
        # 'architecture.qmodels': dict(lr_mult=0.1)},
        bypass_duplicate=True
    ))

max_epochs = 100

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

train_dataloader = dict(batch_size=64)
