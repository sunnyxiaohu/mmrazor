### align with mnn exp

| framwork | dataset | quant_type | metric | original | detail |
| --- | --- | --- | --- | --- | --- |
| mmrazor | imagenet(5w) | ptq | 69.978/89.406 |  |  |
| mnn | imagenet(5w) | from mmrazor | 69.98/89.29 |  |  |
| mmrazor | imagenet(5w) | qat+lsq+epoch15 | 70.3900/89.5100 |  | update_weight_with_fakequant=True +no cliprange fakequant weight+std=58.64 |
| mnn | imagenet(5w) | from mmrazor | 70.45/89.47 |  |  |