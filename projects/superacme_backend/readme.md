# 基于SuperACME后端部署的QAT例子

## Usage
### Setup Environment
torch >= 1.3.1
集群上准备有参考的conda环境，放在`/alg-ftp/ftp-upload/private/wangshiguang/envs/pt1.13`
`mmcls`和`mmdet`采用对应的`main_dev`分支

### QAT量化训练
```bash
PORT=29513 bash tools/dist_train.sh projects/superacme_backend/configs/qat/lsq_superac
me_mv2_8xb32_10e_in1k.py 8 --cfg-options randomness.seed=777
```

### QAT部署与精度对齐
```bash
bash tools/dist_test.sh projects/superacme_backend/configs/qat/lsq_superac
me_mv2_8xb32_10e_in1k.py $CKPT_FROM_QAT_TRAINING 1
```

## Results
### MBv2 on ImageNet
| framwork | bit-width | Q-Algorithm | dataset      | Top-1 | ddr_io | sram_io | params | ddr_occp | sram_occp | fps      |
| ---      | ---       | ---         | ---          | ---   | ---    | ---     | ---    | ---      |  ---      | ---      |
| torch    | FP32      | -           | imagenet(5w) | 71.86 | -      | -       | -      | -        |-          | -        | 
| mmrazor | W8A8       | LSQ         | imagenet(5w) | 70.4  |  |  |  | | | |
| mmrazor | W8A8       | LSQ         | imagenet(1k) | 69.5  |  |  |  | | | |
| sann    | W8A8  | from-mmrazor-LSQ | imagenet(1k) | 70.2 | 3.7103 | 1.3398 | 3.4932 | 4.6416 | 0.5742 | 1004.5927 |

### Yolox-S on COCO
| framwork | bit-width | Q-Algorithm | dataset      | bbox_mAP | ddr_io | sram_io | params | ddr_occp | sram_occp | fps      |
| ---      | ---       | ---         | ---          | ---      | ---    | ---     | ---    | ---      |  ---      | ---      |
| torch    | FP32      | -           | coco(5k)     | 40.4     | -      | -       | -      | -        |-          | -        | 
| mmrazor | W8A8       | LSQ         | coco(5k) | 39.7     |  |  |  | | | |
| sann    | W8A8  | from-mmrazor-LSQ | coco(5k) | 39.6 | 43.7138 | 10.5530 | 8.6823 | 21.1238| 0.9766 | 80.2595 |
