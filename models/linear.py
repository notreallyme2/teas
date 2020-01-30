#!/usr/bin/env python3
# coding: utf-8

"""This module contains classes (models) and methods for exploring linear and non-linear autoencoders

"""
import numpy as np
import pandas as pd

from torch import nn, optim, tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

class ModelUtilities:
    """This class holds methods that are common to every torch class / model in this module"""
    
    def update_batch(self, X, Y, optimizer, criterion, train = True):
        """update_batch takes a model, data, a learning rate and a boolean indicating whether this update 
        should be treated as a training run (i.e. the model's weights should be updated) 
        or not.  

        Parameters
        ----------
        model : torch.nn.mnodule
            The model to be updated
        X : torch.FloatTensor
            The input data (i.e feature matrix)
        Y : torch.FloatTensor
            The target matrix)
        lr : float
            The learning rate to be passed to the optimizer
        train : bool
            Should the weights be updated (default = True)
        """
        Y_hat = self.forward(X)
        loss = criterion(Y_hat, Y)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

class LinearMLP(nn.Module, ModelUtilities):
    """A pytorch module to build a simple linear multilayer perceptron"""

    def __init__(self, input_dim = 100, hidden_dim = 512, output_dim = 1000):
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

class MiniLinearMLP(nn.Module, ModelUtilities):
    """A pytorch module to build an even simpler linear multilayer perceptron"""

    def __init__(self, input_dim = 100, output_dim = 1000):
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
        self.input_output = nn.Linear(input_dim, output_dim)
  
    def forward(self, X):
        return (self.input_output(X))


class LinearAE(nn.Module, ModelUtilities):
    """A pytorch module to build a simple linear autoencoder"""

    def __init__(self, input_dim = 100, hidden_dim = 512):
        """
        Parameters
        ----------
        input_dim : int
            The number of input (and output) features
        hidden_dim : int
            The number of features in the hidden layer
        """
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_dim)
  
    def forward(self, X):
        X = self.input(X)
        return (self.output(X))

    def Z_from_X(self, X):
        return self.input(X)

    def X_from_Z(self, Z):
        return self.output(Z)

class LinearFEA(nn.Module, ModelUtilities):
    """A pytorch module to build a linear forward-embedding autoencoder"""

    def __init__(self, input_dim = 100, hidden_dim = 512, output_dim = 1000):
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
        self.predict_Y = nn.Linear(hidden_dim, output_dim)
        self.reconstruct_X = nn.Linear(hidden_dim, input_dim)
  
    def forward(self, X):
        Z = self.input(X)
        Y_hat = self.predict_Y(Z)
        X_tilde = self.reconstruct_X(Z)
        return Y_hat, X_tilde

class LinearTEA(nn.Module, ModelUtilities):
    """A pytorch module to build a linear target-embedding autoencoder"""

    def __init__(self, input_dim=100, hidden_dim = 256, output_dim=1000):
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
        self.input_X = nn.Linear(input_dim, hidden_dim)
        self.input_Y = nn.Linear(output_dim, hidden_dim)
        self.predict_Y = nn.Linear(hidden_dim, output_dim)
  
    def forward(self, X, Y):
        Z_from_X = self.input_X(X)
        Z_from_Y = self.input_Y(Y)
        Y_hat = self.predict_Y(Z_from_Y)
        return Y_hat, Z_from_Y, Z_from_X
    
    def predict_Y_from_X(self, X):
        """Make a prediction of Y from X only. For inference use."""
        Z_from_X = self.input_X(X)
        Y_hat = self.predict_Y(Z_from_X)
        return Y_hat
 