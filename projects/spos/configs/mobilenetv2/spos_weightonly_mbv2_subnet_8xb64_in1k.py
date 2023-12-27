_base_ = './spos_weightonly_mbv2_supernet_8xb64_in1k.py'

global_qconfig = dict(
    w_observer=dict(type='mmrazor.LSQObserver'),
    a_observer=dict(type='mmrazor.LSQObserver'),
    w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
    w_qscheme=dict(qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)
_base_.qmodel.quantizer.global_qconfig = global_qconfig
model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.qmodel,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='work_dirs/cobits_resnet18_search_8xb256_in1k/best_fix_subnet.yaml',
    # You can load the checkpoint of supernet instead of the specific
    # subnet by modifying the `checkpoint`(path) in the following `init_cfg`
    # with `init_weight_from_supernet = True`.
    init_weight_from_supernet=False,
    init_cfg=None)
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint=  # noqa: E251
    #     'work_dirs/cobits_resnet18_search_8xb256_in1k/subnet_20230904_1400.pth',  # noqa: E501
    #     prefix='architecture.'))

train_dataloader = dict(batch_size=64)
optim_wrapper = dict(optimizer=dict(lr=0.0002))

# learning policy
max_epochs = 75
warm_epochs = 1
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.025,
        by_epoch=True,
        begin=0,
        # about 2500 iterations for ImageNet-1k
        end=warm_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs-warm_epochs,
        by_epoch=True,
        begin=warm_epochs,
        end=max_epochs,
    ),
]

model_wrapper_cfg = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.LSQEpochBasedLoop',
    max_epochs=max_epochs,
    val_interval=5,
    is_first_batch=False,
    freeze_bn_begin=-1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')

# Make sure the buffer such as min_val/max_val in saved checkpoint is the same
# among different rank.
default_hooks = dict(sync=dict(type='SyncBuffersHook'))
