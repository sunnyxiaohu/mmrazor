_base_ = [
    './ptq_fp32_nats_8xb16_cifar10.py'
]

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=_base_.train_dataloader,
    calibrate_steps=32,
)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.MinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    architecture=_base_.model,
    float_checkpoint=None,
    forward_modes={'predict'},
    calibrate_mode='predict',
    quantizer=dict(
        type='mmrazor.NativeQuantizer',
        global_qconfig=global_qconfig,
        # no_observer_modules=['xautodl.models.cell_operations.ResNetBasicblock'],
        no_observer_names=[
            'backbone.nats_model.classifier',
            # 'backbone.nats_model.cells.11.conv_a.op.1',
            # 'backbone.nats_model.cells.11.conv_b.op.1',
            # 'backbone.nats_model.cells.11.downsample.0',
            # 'backbone.nats_model.cells.11.downsample.1'
            # 'backbone.nats_model.cells.5.conv_a.op.1',
            # 'backbone.nats_model.cells.5.conv_b.op.1',
            # 'backbone.nats_model.cells.5.downsample.0',
            # 'backbone.nats_model.cells.5.downsample.1'            
        ],
        no_observer_names_regex=[
            # 'backbone.nats_model.cells.4',
            # 'backbone.nats_model.cells.5',
            # 'backbone.nats_model.cells.6',
            # 'backbone.nats_model.cells.7',
            # 'backbone.nats_model.cells.8',
            # 'backbone.nats_model.cells.9',
            # 'backbone.nats_model.cells.10',
            # 'backbone.nats_model.cells.11',
            # 'backbone.nats_model.cells.14',
            'backbone.nats_model.cells.15',
            'backbone.nats_model.cells.16'
        ],
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))
