#!/usr/bin/env bash
WORK_DIRS=$1
OUT_DIR=$2
START=$3
END=$4
PORT=${PORT:-29500}


for IDX in $(seq ${START} $(($END - 1))); do
    FILES=$(find ${WORK_DIRS} -type f -name "*subnet_${IDX}_*.yaml")

    if [ ! ${#FILES[@]} -eq 1 ]; then
        continue
    fi
    # PORT=${PORT} bash tools/dist_test.sh projects/nas-mqbench/configs/realistic_resnet18_ptq/ptq_per-tensor_w-minmax_a-minmax_subnet_8xb256_in1k_calib32xb16.py none 8 \
    PORT=${PORT} bash tools/dist_test.sh projects/nas-mqbench/configs/realistic_resnet18_ptq/ptq_per-channel_w-minmax_a-minmax_subnet_8xb256_in1k_calib32xb16.py none 8 \
    --cfg-options model.architecture.init_cfg.checkpoint=${WORK_DIRS}/subnet${IDX}/float32/epoch_100.pth \
    model.architecture.fix_subnet=${FILES} randomness.seed=777 \
    --work-dir=${WORK_DIRS}/subnet${IDX}/${OUT_DIR}
done
