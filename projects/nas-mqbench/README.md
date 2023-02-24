# Install [NATS-Bench](https://github.com/D-X-Y/NATS-Bench.git)

# Install [AutoDL-Project](https://github.com/D-X-Y/AutoDL-Projects)

# Apply PTQ
## Get accuracy of float32
``GPUS_PER_NODE=1 GPUS=1 sh tools/slurm_test.sh caif_gml ptq projects/nas-mqbench/configs/ptq_fp32_nats_8xb16_cifar10.py none --work-dir work_dirs/ptq_nats --launcher none``
## Get accuracy of int8
``GPUS_PER_NODE=1 GPUS=1 sh tools/slurm_test.sh caif_gml ptq projects/nas-mqbench/configs/ptq_openvino_nats_8xb16_cifar10_calib32xb16.py none --work-dir work_dirs/ptq_nats --launcher none``

# Re-create NAS-MQBench from scratch
