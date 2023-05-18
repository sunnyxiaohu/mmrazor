## 复现说明
1. 几个注意点
- `grad_clip`时仅作用在`backbone.parameters()`上，不作用在`head.parameters()`，所以需要实现两个`optim_wrapper`。这里后续可以做一下实验，我感觉就算作用在全局model上应该也是可以的
- train dataset 和 test dataset的逻辑不一致, train dataset的数据是已经转为RGB的raw数据, test dataset的数据需要从图片中读取，默认读取为BGR，但是data_preprocessor的逻辑是一致的，所以train dataset的数据在读取后要先统一转到BGR。
- `PartialFC`不能包在DDP里，因为每张卡上的参数不能做同步，具体实现可以看`SPOSPartialFCDDP`。这里要注意在实现PartialFC时，前向的ALLGather和普通的all_gather一致，但反向时每张卡的梯度要接收所有其他卡的梯度做平均。

2. 重点说明。由于PartialFC的存在，整体FC的优化被分在了各个RANK上，所以存checkpoint时要按每个GPU存一个checkpoint来存。
- MMengine原来的Runner紧支持在master rank存checkpoint，所以更完整的Runner复现在``engine/runner/runnx.py``
- 为了和原来tools/train.py里的逻辑保持一致，而整个训练的时间一共也才1天多，所以默认没有支持完整的Resume逻辑，如果要支持完整的Resume逻辑，在train.py中将`Runner`替换为上述文件中的`FaceRunner`即可

## 精度
