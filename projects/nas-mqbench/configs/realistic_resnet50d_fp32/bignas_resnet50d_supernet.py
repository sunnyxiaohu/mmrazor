# search space
arch_setting = dict(
    num_blocks_add=[
        [-1, 0, 1, 'categorical'],
        [-2, -1, 0, 'categorical'],
        [-2, -1, 0, 'categorical'],
        [-1, 0, 1, 'categorical'],
    ],
    expand_ratio_mult=[
        [0.2, 0.25, 0.35, 'categorical'],
        [0.2, 0.25, 0.35, 'categorical'],
        [0.2, 0.25, 0.35, 'categorical'],
        [0.2, 0.25, 0.35, 'categorical']
    ],
    out_channels_mult=[
        [0.65, 0.8, 1.0, 'categorical'],
        [0.65, 0.8, 1.0, 'categorical'],
        [0.65, 0.8, 1.0, 'categorical'],
        [0.65, 0.8, 1.0, 'categorical'],
    ])

input_resizer_cfg = dict(
    input_sizes=[[160, 160], [192, 192], [224, 224]])

nas_backbone = dict(
    type='mmrazor.BigNASResNetD',
    depth=50,
    arch_setting=arch_setting,
    norm_cfg=dict(type='DynamicBatchNorm2d'),
    fine_grained_mode=True
    )
