#!/usr/bin/env python3
# coding: utf-8

"""This module contains classes (models) and methods for exploring linear autoencoders

"""
import numpy as np
import pandas as pd

from torch import nn, optim, tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

class LinearMLP(nn.Module):
    """A pytorch module to build a simple linear multilayer perceptron"""

    def __init__(self, arch = [100, 100]):
        """
        Parameters
        ----------
        arch : list[int]
            The architecture of the MLP. Each element in the list is the number of nodes in a layer. E.g. arch = [100, 50, 100] creates an MLP with 100 inputs and outpus and a 50 node hidden layer
        """
        super().__init__()
        self.arch = arch
        layers = []
        for i in range(len(self.arch) - 1):
            layers.append(nn.Linear(self.arch[i], self.arch[i+1]))
        self.net = nn.Sequential(*layers)
  
    def forward(self, X):
        return (self.net(X))

    def update_batch(self, X, Y, optimizer, criterion, train = True):
            """update_batch takes data, an optimizer, a loss function and a boolean indicating whether this update should be treated as a training run (i.e. the model's weights should be updated) 
            or not.  

            Parameters
            ----------
            model : torch.nn.mnodule
                The model to be updated
            X : torch.FloatTensor
                The input data (i.e feature matrix)
            Y : torch.FloatTensor
                The target matrix)
            optimizer : torch.optim
                The optimizer to be used
            criterion : torch.nn.Module
                The loss function
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

class LinearAE(nn.Module):
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

    def update_batch(self, X, optimizer, criterion, train = True):
        """update_batch takes data, an optimizer, a loss function and a boolean indicating whether this update should be treated as a training run (i.e. the model's weights should be updated) 
        or not. 

        Parameters
        ----------
        model : torch.nn.mnodule
            The model to be updated
        X : torch.FloatTensor
            The input data (i.e feature matrix)
        optimizer : torch.optim
            The optimizer to be used
        criterion : torch.nn.Module
            The loss function
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
    
    def update_batch(self, X, Y, optimizer, criterion, train = True):
        """update_batch takes data, an optimizer, a loss function and a boolean indicating whether this update should be treated as a training run (i.e. the model's weights should be updated) 
        or not. 
        
        Parameters
        ----------
        model : torch.nn.mnodule
            The model to be updated
        X : torch.FloatTensor
            The input data (i.e feature matrix)
        Y : torch.FloatTensor
            The target matrix)
        optimizer : torch.optim
            The optimizer to be used
        criterion : torch.nn.Module
            The loss function
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

class LinearTEA(nn.Module):
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
        """ Forward pass through the model

        Parameters
        ----------
        X : tensor
        Y : tensor
        """
        Z_from_X = self.input_X(X)
        Z_from_Y = self.input_Y(Y)
        Y_hat = self.predict_Y(Z_from_Y)
        return Y_hat, Z_from_Y, Z_from_X
    
    def predict_Y_from_X(self, X):
        """Make a prediction of Y from X. For inference use"""
        Z_from_X = self.input_X(X)
        Y_hat = self.predict_Y(Z_from_X)
        return Y_hat
    
    def update_batch(self, X, Y, optimizer, criterion, train = True):
        """update_batch takes data, an optimizer, a loss function and a boolean indicating whether this update should be treated as a training run (i.e. the model's weights should be updated) 
        or not.  

        Parameters
        ----------
        model : torch.nn.mnodule
            The model to be updated
        X : torch.FloatTensor
            The input data (i.e feature matrix)
        Y : torch.FloatTensor
            The target matrix)
        optimizer : torch.optim
            The optimizer to be used
        criterion : torch.nn.Module
            The loss function
        train : bool
            Should the weights be updated (default = True)
        """
        Y_hat, Z, Z_hat = self.forward(X, Y)
        loss = criterion(Y, Y_hat, Z, Z_hat)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()