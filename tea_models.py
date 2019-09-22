#!/usr/bin/env python3
# coding: utf-8

"""This module contains classes (models) and methods for exploring linear and non-linear autoencoders

"""
import numpy as np
import pandas as pd

from torch import nn, optim, tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

class LinearMLP(nn.Module):
    """A pytorch module to build a simple linear multilayer perceptron"""

    def __init__(self, input_dim = 194, hidden_dim = 256, output_dim = 1643):
        """
        Parameters
        ----------
        input_dim : int
            The number of input features
        hidden_dim : int
            The number of features in the hidden layer
        output_dim : int
            The number of output features
        """
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
  
    def forward(self, X):
        X = self.input(X)
        return (self.output(X))
