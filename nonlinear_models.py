#!/usr/bin/env python3
# coding: utf-8

"""This module contains classes (models) and methods for exploring linear and non-linear autoencoders

"""
import numpy as np
import pandas as pd

from torch import nn, optim, tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

class NonLinearMLP(nn.Module):
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
        X = F.relu(self.input(X))
        return (self.output(X))

    def update_batch(self, X, Y, optimizer, criterion, train = True):
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
        Y_hat = self.forward(X)
        loss = criterion(Y_hat, Y)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

class NonLinearAE(nn.Module):
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
        X = F.relu(self.input(X))
        return (self.output(X))

    def update_batch(self, X, optimizer, criterion, train = True):
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
        X_tilde = self.forward(X)
        loss = criterion(X_tilde, X)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

class LinearFEA(nn.Module):
    """A pytorch module to build a linear forward-embedding autoencoder"""

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
        self.predict_Y = nn.Linear(hidden_dim, output_dim)
        self.reconstruct_X = nn.Linear(hidden_dim, input_dim)
  
    def forward(self, X):
        Z = self.input(X)
        Y_hat = self.predict_Y(Z)
        X_tilde = self.reconstruct_X(Z)
        return Y_hat, X_tilde
    
    def update_batch(self, X, Y, optimizer, criterion, train = True):
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
        Y_hat, X_tilde = self.forward(X)
        loss = criterion(X, X_tilde, Y, Y_hat)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()