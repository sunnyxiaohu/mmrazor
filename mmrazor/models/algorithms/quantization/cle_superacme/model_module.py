# /usr/bin/env python3.5
""" Abstract ModelModule class """

from abc import ABC
from enum import Enum

class ModelApi(Enum):
    """ Enum differentiating between Pytorch or Tensorflow """
    pytorch = 0
    tensorflow = 1
    keras = 2


class ModelModule(ABC):
    """ Abstract ModelModule class to represent either a pytorch module or a Tensorflow op """

    def __init__(self, model_module):
        self._model_module = model_module

    def get_module(self):
        """ Getter for module """
        return self._model_module


class PytorchModelModule(ModelModule):
    """ Pytorch ModelModule class to represent a module inside a Pytorch model """

    def __init__(self, model_module):
        super().__init__(model_module)
        self._api = ModelApi.pytorch


class TfModelModule(ModelModule):
    """ Tensorflow ModelModule class to represent an op inside a Tensorflow model """

    def __init__(self, model_module):
        super().__init__(model_module)
        self._api = ModelApi.tensorflow


class KerasModelModule(ModelModule):
    """ Keras ModelModule class to represent an op inside a Keras model """

    def __init__(self, model_module):
        super().__init__(model_module)
        self._api = ModelApi.keras
