# simple introduction

1. 因为yolox中有些数据读取相关的函数没法被正确的trace，因此mmdet里有些代码需要修改下，替换后才能trace整个网络。相关代码包含在detection-quant下的models

2. 仅针对mobilenetv2 backbone的yolox做了一些PTQ和Qat的实验验证，并且对齐了mnn中的精度

(superacme) | type | model | bits | calibrate_steps | w_observer | w_hyp | a_observer | a_hyp | metric(ap50) | cost_time(s) | other | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmrazor | PTQ | mv2+yolox | w8a8 | 32 | PerChannelMinMaxObserver |  | MovingAverageMinMaxObserver |  | 77.9 | 28.84 |  |  |
| mnn |  |  |  |  |  |  |  |  | 77.8 |  |  |  |
| mmrazor | PTQ | mv2+yolox | w8a8 | 32 | PerChannelMinMaxObserver |  | MovingAverageMinMaxObserver |  | 77.0(-0.9) |  |  | disable_fakequant |
| mnn |  |  |  |  |  |  |  |  | 76.9（-0.1） |  |  | disable_fakequant |
| mmrazor | PTQ | mv2+yolox | w8a8 | 32 | MinMaxObserver |  | MinMaxObserver |  | 76.8 | 29.43 |  |  |
| mmrazor | PTQ | mv2+yolox | w8a8 | 225 | MinMaxObserver |  | MinMaxObserver |  | 76.8 | 121.94 |  | disable_fakequant |
| mnn |  |  |  |  |  |  |  |  | 77.0(+0.2) |  |  | disable_fakequant |