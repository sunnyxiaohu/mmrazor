_base_ = [
    'mmcls::mobilenet_v2/mobilenet-v2_8xb32_in1k.py',
]
custom_imports = dict(
    imports=[
        'projects.common.engine.runner.superacme_quantization_loops',
        'projects.quant_observers.models.observers.quantile',
    ],
    allow_failed_imports=False)

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/mnt/data/imagenet')
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/mnt/data/imagenet')
)
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        data_root='/mnt/data/imagenet')
)

test_cfg = dict(
    type='mmrazor.SuperAcmePTQLoop',
    calibrate_dataloader=val_dataloader,
    calibrate_steps=32,
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.EMAQuantileObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=False),
    a_qscheme=dict(qdtype='quint8', bit=8, is_symmetry=True),
)

float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'  # noqa: E501

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmcls.ClsDataPreprocessor',
        num_classes=1000,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True),
    architecture=_base_.model,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)
