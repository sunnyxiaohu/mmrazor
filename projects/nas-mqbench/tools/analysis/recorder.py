from functools import partial

import torch
from .measure import (torch_cosine_similarity, torch_mean_square_error,
                      torch_snr_error)


class MeasureRecorder():
    """Helper class for collecting data."""
    def __init__(self, measurement: str = 'cosine', reduce: str = 'mean') -> None:
        self.num_of_elements = 0
        self.measure         = 0
        if reduce not in {'mean', 'max'}:
            raise ValueError(f'PPQ MeasureRecorder Only support reduce by mean or max, however {reduce} was given.')

        if str(measurement).lower() == 'cosine':
            measure_fn = partial(torch_cosine_similarity, reduction=reduce)
        elif str(measurement).lower() == 'mse':
            measure_fn = partial(torch_mean_square_error, reduction=reduce)
        elif str(measurement).lower() == 'snr':
            measure_fn = partial(torch_snr_error, reduction=reduce)
        else:
            raise ValueError('Unsupported measurement detected. '
                f'PPQ only support mse, snr and consine now, while {measurement} was given.')

        self.measure_fn = measure_fn
        self.reduce     = reduce

    def update(self, y_pred: torch.Tensor, y_real: torch.Tensor):
        elements = y_pred.shape[0]
        if elements != y_real.shape[0]:
            raise Exception(
                'Can not update measurement, cause your input data do not share a same batchsize. '
                f'Shape of y_pred {y_pred.shape} - against shape of y_real {y_real.shape}')
        result = self.measure_fn(y_pred=y_pred, y_real=y_real).item()

        if self.reduce == 'mean':
            self.measure = self.measure * self.num_of_elements + result * elements
            self.num_of_elements += elements
            self.measure /= self.num_of_elements

        if self.reduce == 'max':
            self.measure = max(self.measure, result)
            self.num_of_elements += elements
