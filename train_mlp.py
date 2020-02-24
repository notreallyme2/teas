# train a simple MLP on the synthetic sklearn data

import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn, optim, tensor, FloatTensor
from torch.utils.data import Dataset, TensorDataset, DataLoader

from data.skl_synthetic import load_skl_data
from models.linear import LinearMLP, LinearAE, LinearFEA
from plotting import plot_losses, plot_predicted_vs_actual

import matplotlib.pyplot as plt

def main(*args, **kwargs):
    torch.manual_seed(123)

    # load data, set model paramaters 
    # ---------------------------------

    home = Path.home()
    path_for_data = home/"teas-data/sklearn/"
    if not os.path.exists(path_for_data):
        raise ValueError("No data. By default, this script uses synthetic data that you can generate by running skl_synthetic.py. Otherwise please modify this script")
    if os.path.exists(path_for_data):
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = map(FloatTensor, load_skl_data(path_for_data))

    batch_size = 128
    train_ds = TensorDataset(X_train, Y_train)
    valid_ds = TensorDataset(X_valid, Y_valid)
    test_ds = TensorDataset(X_test, Y_test)
    train_dl = DataLoader(train_ds, batch_size)
    valid_dl = DataLoader(valid_ds, batch_size)
    test_dl = DataLoader(test_ds, batch_size)

    # these give us some shape values for later
    X, Y = next(iter(train_ds))
    input_dim = X.shape[0]
    hidden_dim = 512
    output_dim = Y.shape[0]

    # first, train and benchmark a simple Linear MLP 
    lmlp_model = LinearMLP([input_dim, 512, output_dim])

    # train the linear MLP
    epochs = 10
    lr = 1e-2
    opt = optim.Adam(lmlp_model.parameters(), lr)
    mse = nn.MSELoss()
    train_loss, valid_loss = [], []

    print("Training a linear MLP")
    for e in tqdm(range(epochs)):
        this_train_loss = np.mean([lmlp_model.update_batch(X, Y, opt, mse) for X, Y in train_dl])
        this_valid_loss = np.mean([lmlp_model.update_batch(X, Y, opt, mse, train=False) for X, Y in valid_dl])
        train_loss.append(this_train_loss)
        valid_loss.append(this_valid_loss)

    plot_losses(epochs, train_loss, valid_loss)

    # visualise predicted vs. actual
    # pull out a random row
    idx = np.random.randint(0, X_valid.shape[1])
    X, Y = valid_ds[idx]
    Y_hat = lmlp_model(X)
    # Y_hat vs. Y
    plot_predicted_vs_actual(Y, Y_hat, idx)

    # test losses
    test_pred = []
    pred_error = []
    mse = nn.MSELoss()
    for X, Y in test_ds:
        Y_hat = lmlp_model(X)
        test_pred.append(Y_hat.detach().numpy())
        pred_error.append(mse(Y_hat, Y).detach().numpy())
    print("Final test MSE loss on prediction task (linear MLP): {}".format(np.mean(pred_error)))

if __name__ == "__main__":
    main()