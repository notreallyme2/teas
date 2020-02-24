# feature-embedding autoencoders
# linear, FEAs (as described in the accompanying draft) deployed on synthetic data.

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

# def main(*args, **kwargs):
torch.manual_seed(123)

# load data, set model paramaters 
# ---------------------------------

# create sklearn data if needed
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
# now, a FEAs model
# we will pretrain a simple linear autencoder, then use those weights to initialise a FEAs model

lae_model = LinearAE(input_dim, hidden_dim)

# train the linear AE
epochs = 10
lr = 1e-2
opt = optim.Adam(lae_model.parameters(), lr)
mse = nn.MSELoss()
train_loss, valid_loss = [], []

print("Training a linear autoencoder; the weights will be used in the FEA")

for e in tqdm(range(epochs)):
    this_train_loss = np.mean([lae_model.update_batch(X, opt, mse) for X, _ in train_dl])
    this_valid_loss = np.mean([lae_model.update_batch(X, opt, mse, train=False) for X, _ in valid_dl])
    train_loss.append(this_train_loss)
    valid_loss.append(this_valid_loss)

plot_losses(epochs, train_loss, valid_loss)

# visualise predicted vs. actual
# generate predictions on a random row
idx = np.random.randint(0, X_valid.shape[1])
X, _ = valid_ds[idx]
X_tilde = lae_model(X)
plot_predicted_vs_actual(X, X_tilde, idx, title="Predicted X vs. observed X (linear AE)")

# now use these weights in a FEA
class joint_loss(nn.Module):
    """
    Parameters
    ----------
    lambda_ : float
        Weighting in the joint joss. 
        Higher lambda_ favours lower reconstruction loss.
    """
    def __init__(self, lambda_=0.5):
        super().__init__()
        self.X = X
        self.Y = Y
        self.lambda_ = lambda_
        
    def forward(self, X, X_tilde, Y, Y_hat):
        mse = nn.MSELoss()
        return ( ((1 - self.lambda_) * mse(Y_hat, Y)) + (self.lambda_ * mse(X_tilde, X)) )

lfea_model = LinearFEA(input_dim, hidden_dim, output_dim)

# copy the weights and biases from the trained AE
lfea_model.input.load_state_dict(lae_model.input.state_dict(), strict=True)
lfea_model.reconstruct_X.load_state_dict(lae_model.output.state_dict(), strict=True)

# now train the linear FEA
epochs = 100
lr = 6e-4
opt = optim.Adam(lfea_model.parameters(), lr)
criterion = joint_loss(lambda_= 0.10)
train_loss, valid_loss = [], []

print("Now training the linear FEA")
for e in tqdm(range(epochs)):
    this_train_loss = np.mean([lfea_model.update_batch(X, Y, opt, criterion) for X, Y in train_dl])
    this_valid_loss = np.mean([lfea_model.update_batch(X, Y, opt, criterion, train=False) for X, Y in valid_dl])
    train_loss.append(this_train_loss)
    valid_loss.append(this_valid_loss)

plot_losses(epochs, train_loss, valid_loss)

# final validation loss on predicting Y
# validation losses
test_pred = []
pred_error = []
mse = nn.MSELoss()
for X, Y in valid_ds:
    Y_hat, _ = lfea_model(X)
    test_pred.append(Y_hat.detach().numpy())
    pred_error.append(mse(Y_hat, Y).detach().numpy())
print("Final validation MSE loss on prediction task (linear FEA): {}".format(np.mean(pred_error)))

# results for different values of lambda:
# 
# | Lambda   | Final validation loss  |
# | :------- | :------------------    |
# | 0.10     | 0.014358229003846645   |
# | 0.25     | 0.014377197250723839   |
# | 0.50     | 0.014417761005461216   |
# | 0.75     | 0.01452191174030304    |
# | 0.90     | 0.014881270006299019   ||
# 
# the best performing model(s) favour Y_hat loss.

# visualise predicted vs. actual
# pick a row, generate predictions
idx = 10
X, Y = test_ds[idx]
Y_hat, X_tilde = lfea_model(X)
# Y vs Y_hat
plot_predicted_vs_actual(Y, Y_hat, idx, title = "Predicted Y vs. observed Y")
# X_tilde vs. X
plot_predicted_vs_actual(X, X_tilde, idx, title = "Predicted X vs. observed X")

# test losses
test_pred = []
pred_error = []
mse = nn.MSELoss()
for X, Y in test_ds:
    Y_hat, _ = lfea_model(X)
    test_pred.append(Y_hat.detach().numpy())
    pred_error.append(mse(Y_hat, Y).detach().numpy())
print("Final test MSE loss on prediction task: {}".format(np.mean(pred_error)))

# if __name__ == "__main__":
#     main()