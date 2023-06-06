_base_ = ['./mbf_8xb512_wf42m_pfc02.py']

custom_imports = dict(
    imports=[
        'projects.commons.models.task_modules.estimators.ov_estimator',
    ],
    allow_failed_imports=False)

model = dict(norm_training=True)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=20,
    num_candidates=50,
    top_k=10,
    num_mutation=25,
    num_crossover=25,
    mutate_prob=0.1,
    constraints_range=dict(ov_ddr_bandwidth=10,ov_npu_time=0.012,params=2.0),
    estimator_cfg = dict(
        type='mmrazor.OVEstimator',
        task_type='cls',
        input_shape=(1,3,112,112),
        ov_file_path = '/home/chenzhixuan/project/mmrazor_superacme_main_dev_facerecog/projects/commons/models/task_modules/estimators/temp_files/',
        input_img_prefix = 'demo.jpg',
        onnx_file_prefix = 'temp.onnx' ,
        qfnodes_def_file_prefix = 'default_ifm_uq8.qfnodes',
        val_root_prefix = 'mini_val2017',
        val_annFile_prefix = 'mini_instances_val2017.json',
        ovm_file_preifx = 'noadd0.ovm'
        ),
    score_key='rank1/avg',
    )
