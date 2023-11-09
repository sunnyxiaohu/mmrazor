# search space
arch_setting = dict(
    # parsed by np.arange
    num_blocks_add=[  # [min_num_blocks, max_num_blocks, step]
        [0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical'],
        [-1, 0, 1, 'categorical']
    ],
    out_channels_mult=[  # [min_expand_ratio, max_expand_ratio, step]
        [0.6, 1.0, 1.4, 'categorical'],
        [0.6, 0.8, 1.0, 1.2, 1.4, 'categorical'],
        [0.6, 0.8, 1.0, 1.2, 1.4, 'categorical'],
        [0, 'categorical']
    ],
    mid_channels_mult=[  # [min_channel_mult, max_channel_mult, step]
        [0.6, 0.8, 1.0, 1.2, 1.4, 'categorical'],
        [0.6, 0.8, 1.0, 1.2, 1.4, 'categorical'],
        [0.7, 0.8, 1.0, 1.2, 1.3, 'categorical'],
        [0.7, 0.8, 1.0, 1.2, 1.3, 'categorical'],
    ],
    ds_channels_mult=[  # [min_expand_ratio, max_expand_ratio, step]
        [0.6, 0.8, 1.0, 1.2, 1.4, 'categorical'],
        [0.7, 0.8, 1.0, 1.2, 1.3, 'categorical'],
        [0.7, 0.8, 1.0, 1.2, 1.3, 'categorical'],
        [0, 'categorical']
    ])

embedding_size = 256
nas_backbone=dict(
    type='mmrazor.SearchableMobileFaceNet',
    num_features=embedding_size,
    arch_setting=arch_setting)

input_resizer_cfg = dict(
    input_sizes=[[108, 108], [112, 112], [120, 120]])
