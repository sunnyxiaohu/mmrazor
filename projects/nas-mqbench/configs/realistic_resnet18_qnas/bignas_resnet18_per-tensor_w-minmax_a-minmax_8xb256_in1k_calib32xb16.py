_base_ = [
    '../realistic_resnet18_fp32/bignas_resnet18_supernet_16xb128_in1k.py'
]

_base_.custom_imports.imports += [
        'projects.nas-mqbench.engine.runner.nas-mq_search_loop'
]

calibrate_dataloader = dict(
    batch_size=16,
    num_workers=1,
    dataset=_base_.val_dataloader.dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.MinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

mq_model = dict(
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=_base_.supernet.data_preprocessor,
    architecture=None,  # _base_.model,
    float_checkpoint=None,
    forward_modes=('tensor', 'predict'),
    quantizer=dict(
        type='mmrazor.TensorRTQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

mq_model_wrapper_cfg = dict(type='mmrazor.MMArchitectureQuantDDP', )


train_cfg = dict(
    _delete_=True,
    mq_model=mq_model,
    mq_model_wrapper_cfg=mq_model_wrapper_cfg,
    mq_calibrate_dataloader=calibrate_dataloader,
    mq_calibrate_steps=32,
    mq_init_candidates='work_dirs/bignas_resnet18_per-tensor_w-minmax_a-minmax_8xb256_in1k_calib32xb16/search_epoch_5.pkl',    
    type='mmrazor.NASMQSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=5,
    num_candidates=20,
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_sample_num=4096,
    constraints_range=dict(flops=(0., 7000.)),
    score_indicator='score',
    score_key='accuracy/top1')

val_cfg = dict(_delete_=True)
model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'work_dirs/qnas_resnet18_supernet_16xb128_in1k/epoch_100.pth',  # noqa: E501
        prefix='architecture.'))
