#!/usr/bin/env bash
WORK_DIRS=$1
START=$2
END=$3
PORT=${PORT:-29500}


for IDX in $(seq ${START} $(($END - 1))); do
    FILES=$(find ${WORK_DIRS} -type f -name "*subnet_${IDX}_*.yaml")

    if [ ! ${#FILES[@]} -eq 1 ]; then
        continue
    fi
    PORT=${PORT} bash tools/dist_train.sh projects/nas-mqbench/configs/realistic_resnet18_fp32/bignas_resnet18_subnet_8xb256_in1k.py 8 \
    --cfg-options optim_wrapper.optimizer.weight_decay=0.0001 train_dataloader.batch_size=256 randomness.seed=777 \
    model.fix_subnet=${FILES} --work-dir ${WORK_DIRS}_subnet${IDX}

done
