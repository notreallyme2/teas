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

def update_lmlp_batch(model, X, Y, lr, train = True):
    """update_batch takes a model, data, a learning rate and a boolean indicating whether this update 
    should be treated as a training run (i.e. the model's weights should be updated) 
    or not. This function is not for production use, as it has a number of hidden parameters (e.g. optimizer).  
    
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
    Y_hat = model(X)
    loss = loss_func(Y_hat, Y)
    if train:
        opt = optim.Adam(model.parameters(), lr)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item()

class LinearAE(nn.Module):
    """A pytorch module to build a simple linear autoencoder"""

    def __init__(self, input_dim = 194, hidden_dim = 256):
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
    
def update_lae_batch(model, X, lr, train = True):
    """update_batch takes a model, data, a learning rate and a boolean indicating whether this update 
    should be treated as a training run (i.e. the model's weights should be updated) 
    or not. This function is not for production use, as it has a number of hidden parameters (e.g. optimizer).  
    
    Parameters
    ----------
    model : torch.nn.mnodule
        The model to be updated
    X : torch.FloatTensor
        The input data (i.e feature matrix)
    lr : float
        The learning rate to be passed to the optimizer
    train : bool
        Should the weights be updated (default = True)
    """
    X_tilde = model(X)
    loss = mse(X_tilde, X)
    if train:
        opt = optim.Adam(model.parameters(), lr)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item()