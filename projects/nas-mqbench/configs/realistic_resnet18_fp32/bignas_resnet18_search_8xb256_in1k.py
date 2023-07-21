_base_ = ['./bignas_resnet18_supernet_8xb256_in1k.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=1,
    num_candidates=5,
    top_k=5,
    num_mutation=2,
    num_crossover=3,
    mutate_prob=0.1,
    calibrate_sample_num=32,
    constraints_range=dict(flops=(0., 7000.)),
    score_key='accuracy/top1')
