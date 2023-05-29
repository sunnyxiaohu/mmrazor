## 启动训练
 (注：此配置为单机八卡配置, 注意总batch_size和lr的对应关系, 此配置下3090约需36h)
 ```
 bash tools/dist_train.sh projects/face-recognition/configs/mbf_8xb256_wf42m_pfc02.py 8
 ```

## 数据
1. 公开数据集webface260m存储在`/alg-data2/datasets/cv_dirty/fanxiao/fanxiao/webface260m`, 为了加速训练，部分机器上也有该数据的备份，如192.168.8.70。

2. 测试数据存储在`/alg-data/ftp-upload/datasets/face_data/OV_test/face_recognition/`, 为了加速训练，部分机器上也有该数据的备份，如192.168.8.70。

## 复现说明
1. 复现基于[archface_torch](http://192.168.8.60/system/algorithm/arcface_torch/-/tree/mbf_baseline), 在原有配置的基础上，将batch_size和embedding_size都调整为256的结果，相比调整之前，精度和速度差异很小。

2. 几个注意点
- `grad_clip`时仅作用在`backbone.parameters()`上，不作用在`head.parameters()`，所以需要实现两个`optim_wrapper`。
- train dataset 和 test dataset的逻辑不一致, train dataset的数据是已经转为RGB的raw数据, test dataset的数据需要从图片中读取，默认读取为BGR，但是data_preprocessor的逻辑是一致的，所以train dataset的数据在读取后要先统一转到BGR。
- `PartialFC`不能包在DDP里，因为每张卡上的参数不能做同步，具体实现可以看`SPOSPartialFCDDP`。这里要注意在实现PartialFC时，前向的ALLGather和普通的all_gather一致，但反向时每张卡的梯度要接收所有其他卡的梯度做平均。

2. 重点说明。由于PartialFC的存在，整体FC的优化被分在了各个RANK上，所以存checkpoint时要按每个GPU存一个checkpoint来存。
- MMengine原来的Runner紧支持在master rank存checkpoint，所以更完整的Runner复现在``engine/runner/runnx.py``
- 为了和原来tools/train.py里的逻辑保持一致，而整个训练的时间一共也才1天多，所以默认没有支持完整的Resume逻辑，如果要支持完整的Resume逻辑，在train.py中将`Runner`替换为上述文件中的`FaceRunner`即可

3. 关于速度，由于代码实现的差异（主要为Dataloader部分），复现代码相比之前单次实验要慢5小时左右。

4. 关于精度，原仓库的精度评测方式为：先转换至onnx，然后再评测；本仓库可直接支持在训练过程中作为ValLoop，以torch模型为输入直接评测结果。另外本仓库对待测评数据做了一次筛选（有些数据丢失），评测结果更准确。

## 精度
Rank1结果
|  TrainSetting     | 120P     | baby_500p | glint1k | mask_369p | menjin_20p | xm14 |  Avg |
|-------|----------|-----------|---------|-----------|------------|------|-----|
| 原仓库BaseLine: wbf42m+mbf1M_ReLu+10ep         | 85.1 (84.54) |   92.72 (92.49)       | 96.72 (96.82) | 81.89 (82.36) |   99.67 (99.72)          |   93.53 (93.43)        |  91.6 (91.56)  |
| 本仓库复现结果 | 89.89 |   94.28 | 84.26 |   85.30    | 99.52   | 96.74  |   91.67  |

