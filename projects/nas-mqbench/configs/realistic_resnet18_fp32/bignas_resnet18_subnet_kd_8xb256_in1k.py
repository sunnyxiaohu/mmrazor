_base_ = [
    'bignas_resnet18_subnet_8xb256_in1k.py'
]

student = _base_.model

teacher = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='work_dirs/bignas_resnet18_search_8xb256_in1k/best_fix_subnet.yaml',
    # You can load the checkpoint of supernet instead of the specific
    # subnet by modifying the `checkpoint`(path) in the following `init_cfg`
    # with `init_weight_from_supernet = True`.
    init_weight_from_supernet=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'work_dirs/bignas_resnet18_search_8xb256_in1k/subnet_20230519_0915.pth',  # noqa: E501
        prefix=None))

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=_base_.data_preprocessor,
    architecture=student,
    teacher=teacher,
    teacher_ckpt=None,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=3)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
