# /usr/bin/env python3.5

""" Modules for functional elementwise ops """
import torch
import torch.nn


class Add(torch.nn.Module):
    """ Add module for a functional add"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for add op
        """
        return x + y


class Subtract(torch.nn.Module):
    """ Subtract module for a functional subtract"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for subtract op
        """
        return x - y


class Multiply(torch.nn.Module):
    """ Multiply module for a functional multiply"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for multiply op
        """
        return x * y


class Divide(torch.nn.Module):
    """ Divide module for a functional divide"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for divide op
        """
        return torch.div(x, y)


class Concat(torch.nn.Module):
    """ Concat module for a functional concat"""
    def __init__(self, axis: int = 0):
        super(Concat, self).__init__()
        self._axis = axis

    # pylint:disable=arguments-differ
    def forward(self, *x) -> torch.Tensor:
        """
        Forward-pass routine for cat op
        """
        return torch.cat(x, dim=self._axis)


class MatMul(torch.nn.Module):
    """ MatMul module for a functional matmul"""
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass routine for matmul op
        """
        return torch.matmul(x, y)
