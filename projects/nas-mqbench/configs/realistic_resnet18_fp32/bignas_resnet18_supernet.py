# search space
arch_setting = dict(
    # parsed by np.arange
    num_blocks_add=[  # [min_num_blocks, max_num_blocks, step]
        [-1, 2, 1],
        [-1, 2, 1],
        [-1, 2, 1],
        [-1, 2, 1],
    ],
    expand_ratio_mult=[  # [min_expand_ratio, max_expand_ratio, step]
        [0.9, 1.1, 0.1],
        [0.9, 1.1, 0.1],
        [0.9, 1.1, 0.1],
        [0.9, 1.1, 0.1],
    ],
    out_channels_mult=[  # [min_channel_mult, max_channel_mult, step]
        [0.9, 1.1, 0.1],
        [0.9, 1.1, 0.1],
        [0.9, 1.1, 0.1],
        [0.9, 1.1, 0.1],
    ])

input_resizer_cfg = dict(
    input_sizes=[[160, 160], [192, 192], [224, 224]])

nas_backbone = dict(
    type='mmrazor.BigNASResNet',
    depth=18,
    arch_setting=arch_setting,
    norm_cfg=dict(type='DynamicBatchNorm2d'),
    fine_grained_mode=True   
    )
