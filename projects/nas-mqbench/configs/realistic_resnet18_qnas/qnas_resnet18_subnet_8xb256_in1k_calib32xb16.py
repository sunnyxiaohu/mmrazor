# _base_ = './qnas_resnet18_supernet_16xb128_in1k.py'

# model = dict(
#     _delete_=True,
#     _scope_='mmrazor',
#     type='sub_model',
#     cfg=_base_.qmodel,
#     # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
#     fix_subnet='work_dirs/qnas_resnet18_search_8xb256_in1k/best_fix_subnet.yaml',
#     # You can load the checkpoint of supernet instead of the specific
#     # subnet by modifying the `checkpoint`(path) in the following `init_cfg`
#     # with `init_weight_from_supernet = True`.
#     init_weight_from_supernet=False,
#     # init_cfg=None)
#     init_cfg=dict(
#         type='Pretrained',
#         checkpoint=  # noqa: E251
#         'work_dirs/bignas_resnet18_search_8xb128_in1k/subnet_20230519_0915.pth',  # noqa: E501
#         prefix='architecture.'))

# model_wrapper_cfg = None
# find_unused_parameters = True

# # test_cfg = dict(evaluate_fixed_subnet=True)
# test_cfg = dict(_delete_=True)
# val_cfg = dict(_delete_=True)

# # default_hooks = dict(checkpoint=None)

# TODO(shiguang): need fix `to_static_op` by `get_deploy`