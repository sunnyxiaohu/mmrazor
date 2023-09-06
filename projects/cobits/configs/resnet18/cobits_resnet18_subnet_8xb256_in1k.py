_base_ = './cobits_resnet18_supernet_8xb256_in1k.py'

_base_.model.fixed_subnet = 'work_dirs/cobits_resnet18_search_8xb256_in1k/min_subnet_20230905_0510.yaml'

train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.004,
        momentum=0.9,
        weight_decay=0.0001,
        _scope_='mmcls'))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.025,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=99, by_epoch=True, begin=1, end=100)
]
val_cfg = dict(evaluate_fixed_subnet=True)
# TODO(shiguang): need fix `to_static_op` by `get_deploy`