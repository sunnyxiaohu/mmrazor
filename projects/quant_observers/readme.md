### Observer Support Matrix:

| observer(from)                              | Symmetrical | Asymmetrical | Per-channel | Per-tensor | weight | activation | two phase |
| ------------------------------------------- | ----------- | ------------ | ----------- | ---------- | ------ | ---------- | --------- |
| FasterMSEObserver(ppq)                      | ✔           | ✔            |             | ✔          |        | ✔          | ✔         |
| KLObserver(ppq)                             | ✔           |              |             | ✔          |        | ✔          | ✔         |
| TensorRTHistogramBasedMSEObserver(tensorrt) | ✔           |              |             | ✔          | ✔      | ✔          |           |
| EMAQuantileObserver(mqbench)                | ✔           | ✔            |             | ✔          | ✔      | ✔          |           |
| MSEObserver(mqbench)                        | ✔           | ✔            |             | ✔          | ✔      | ✔          |           |
| PerChannelMSEObserver(mqbench)              | ✔           |              | ✔           |            | ✔      |            |           |

### Some experiments

| backend   | model | bits | calibrate_steps | w_observer               | w_hyp | a_observer                        | a_hyp             | metric(top1) | cost_time(s) | note     |
| --------- | ----- | ---- | --------------- | ------------------------ | ----- | --------------------------------- | ----------------- | ------------ | ------------ | -------- |
| openvino  | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | MovingAverageMinMaxObserver       |                   | 70.284       | 11.54        | baseline |
| openvino  | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | EMAQuantileObserver               |                   | 71.016       | 469.01       |          |
| openvino  | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | MSEObserver                       | iter=95,step=0.01 | 71.068       | 291.26       |          |
| openvino  | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | TensorRTHistogramBasedMSEObserver |                   |              |              | 不支持激活u8  |
| superacme | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | TensorRTHistogramBasedMSEObserver |                   | 71.106       | 1272.07      |          |
| openvino  | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | KLObserver                        | hist_bins=4096    |              | 30.51        | 不支持激活u8  |
| superacme | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | KLObserver                        | hist_bins=4096    | 70.95        | 38.16        |          |
| openvino  | mv2   | w8a8 | 32              | PerChannelMinMaxObserver |       | FasterMSEObserver                 |                   | 71.056       | 69.33        |          |
