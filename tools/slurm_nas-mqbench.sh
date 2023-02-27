#!/usr/bin/env bash

set -x

PARTITION=$1
CONFIG=$2
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
PY_ARGS=${@:3}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name='nas-mqbench' \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/nas-mqbench.py ${CONFIG} ${PY_ARGS}
