_MIDDLE_MUTABLE = dict(
    _scope_='mmrazor',
    type='OneShotMutableOP',
    candidates=dict(
        fusedibn_k3e3l2 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=3,
            num_blocks=2,
            ),
        fusedibn_k3e3l3 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=3,
            num_blocks=3,
            ),
        fusedibn_k3e3l4 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=3,
            num_blocks=4,
            ),
        fusedibn_k3e4l2 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=4,
            num_blocks=2,
            ),
        fusedibn_k3e4l3 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=4,
            num_blocks=3,
            ),
        fusedibn_k3e4l4 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=4,
            num_blocks=4,
            ),
        fusedibn_k3e6l2 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=6,
            num_blocks=2,
            ),
        fusedibn_k3e6l3 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=6,
            num_blocks=3,
            ),
        fusedibn_k3e6l4 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=6,
            num_blocks=4,
            ),
        tucker_k3c25e25l2 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.25,
            num_blocks=2,
            ),
        tucker_k3c25e25l3 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.25,
            num_blocks=3,
            ),
        tucker_k3c25e25l4 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.25,
            num_blocks=4,
            ),
        tucker_k3c25e75l2 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.75,
            num_blocks=2,
            ),
        tucker_k3c25e75l3 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.75,
            num_blocks=3,
            ),
        tucker_k3c25e75l4 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.75,
            num_blocks=4,
            ),
        tucker_k3c75e25l2 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.25,
            num_blocks=2,
            ),
        tucker_k3c75e25l3 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.25,
            num_blocks=3,
            ),
        tucker_k3c75e25l4 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.25,
            num_blocks=4,
            ),
        tucker_k3c75e75l2 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.75,
            num_blocks=2,
            ),
        tucker_k3c75e75l3 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.75,
            num_blocks=3,
            ),
        tucker_k3c75e75l4 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.75,
            num_blocks=4,
            ),
        csp_l2e025_identity=dict(
            type='CSPLayer',
            num_blocks=2,
            expand_ratio=0.25,
            add_identity=True,
        ),
        csp_l2e050_identity=dict(
            type='CSPLayer',
            num_blocks=2,
            expand_ratio=0.5,
            add_identity=True,
        ),
        csp_l2e075_identity=dict(
            type='CSPLayer',
            num_blocks=2,
            expand_ratio=0.75,
            add_identity=True,
        ),
        csp_l3e025_identity=dict(
            type='CSPLayer',
            num_blocks=3,
            expand_ratio=0.25,
            add_identity=True,
        ),
        csp_l3e050_identity=dict(
            type='CSPLayer',
            num_blocks=3,
            expand_ratio=0.5,
            add_identity=True,
        ),
        csp_l3e075_identity=dict(
            type='CSPLayer',
            num_blocks=3,
            expand_ratio=0.75,
            add_identity=True,
        ),
        csp_l4e025_identity=dict(
            type='CSPLayer',
            num_blocks=4,
            expand_ratio=0.25,
            add_identity=True,
        ),
        csp_l4e050_identity=dict(
            type='CSPLayer',
            num_blocks=4,
            expand_ratio=0.5,
            add_identity=True,
        ),
        csp_l4e075_identity=dict(
            type='CSPLayer',
            num_blocks=4,
            expand_ratio=0.75,
            add_identity=True,
        ),
        ))

_FIRST_MUTABLE_LAST = dict(
    _scope_='mmrazor',
    type='OneShotMutableOP',
    candidates=dict(
        tucker_k3c75e75l1 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.75,
            num_blocks=1,
            ),
        tucker_k3c25e25l1 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.25,
            num_blocks=1,
            ),
        tucker_k3c25e75l1 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.25,
            expand_ratio=0.75,
            num_blocks=1,
            ),
        tucker_k3c75e25l1 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.75,
            expand_ratio=0.25,
            num_blocks=1,
            ),
        tucker_k3c50e50l1 = dict(
            type='Tucker_CSPLayer',
            compress_ratio=0.5,
            expand_ratio=0.5,
            num_blocks=1,
            ),
        fusedibn_k3e3l1 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=3,
            num_blocks=1,
            ),
        fusedibn_k3e2l1 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=2,
            num_blocks=1,
            ),
        fusedibn_k3e1l1 = dict(
            type='FusedIBN_CSPLayer',
            expand_ratio=1,
            num_blocks=1,
            ),
        csp_l1e025_noidentity=dict(
            type='CSPLayer',
            num_blocks=1,
            expand_ratio=0.25,
            add_identity=False,
        ),
        csp_l1e050_noidentity=dict(
            type='CSPLayer',
            num_blocks=1,
            expand_ratio=0.5,
            add_identity=False,
        ),
        csp_l1e075_noidentity=dict(
            type='CSPLayer',
            num_blocks=1,
            expand_ratio=0.75,
            add_identity=False,
        ),
        csp_l1e025_identity=dict(
            type='CSPLayer',
            num_blocks=1,
            expand_ratio=0.25,
            add_identity=True,
        ),
        csp_l1e050_identity=dict(
            type='CSPLayer',
            num_blocks=1,
            expand_ratio=0.5,
            add_identity=True,
        ),
        csp_l1e075_identity=dict(
            type='CSPLayer',
            num_blocks=1,
            expand_ratio=0.75,
            add_identity=True,
        ),
        ))

arch_setting =[
    [64, 128, 3, True, False,_FIRST_MUTABLE_LAST], 
    [128, 256, 9, True, False,_MIDDLE_MUTABLE],
    [256, 512, 9, True, False,_MIDDLE_MUTABLE], 
    [512, 1024, 3, False, False,_FIRST_MUTABLE_LAST]
    ],

nas_backbone = dict(
    _scope_='mmrazor',
    type='Searchable_CSPDarknet',
    arch_setting=arch_setting,
    focusconv_use='calibration_c16',
    deepen_factor=0.33, 
    widen_factor=0.5, 
    act_cfg=dict(type='ReLU')
    )
