arch_setting = dict(
    # parsed by np.arange
    channels_mult=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 'categorical'],
)

nas_backbone=dict(
    _scope_='mmrazor',
    type = 'QNASResNet',
    depth=18,
    arch_setting=arch_setting)
