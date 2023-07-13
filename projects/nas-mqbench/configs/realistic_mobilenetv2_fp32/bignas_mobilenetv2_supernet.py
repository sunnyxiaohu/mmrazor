# search space
arch_setting = dict(
    # parsed by np.arange
    num_blocks_add=[  # [min_num_blocks, max_num_blocks, step]
        [0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [0, 1, 'categorical'],
    ],
    expand_ratio_add=[  # [min_expand_ratio, max_expand_ratio, step]
        [0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
    ],
    out_channels_mult=[  # [min_channel_mult, max_channel_mult, step]
        [1.0, 'categorical'],
        [1.0, 'categorical'],
        [1.0, 'categorical'],
        [0.9, 1.0, 1.1, 'categorical'],
        [0.9, 1.0, 1.1, 'categorical'],
        [0.9, 1.0, 1.1, 'categorical'],
        [0.9, 1.0, 1.1, 'categorical'],
    ])

input_resizer_cfg = dict(
    input_sizes=[[160, 160], [192, 192], [224, 224]])

nas_backbone = dict(
    type='mmrazor.BigNASMobileNetV2',
    arch_setting=arch_setting,
    norm_cfg=dict(type='DynamicBatchNorm2d'),
    fine_grained_mode=True
    )
