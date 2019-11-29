import typing
import os
from pathlib import Path
import fire
import numpy as np
from sklearn.datasets import make_regression


def make_skl_dataset(n = 10000, p = 100, len_y = 1000, x_rank=1, noise=.1, train_split = .6, valid_split = .2):
    """ Generates a synthetic dataset suitable for training and assessing a TEAs model, and saves it locally for subsequent use.
    The function defaults to a preset seed so the datset is identical every time.
    
    Parameters
    ----------
    n : int
        Number of rows / observations
    p : int
        Number of features in the input matrix (X)
    len_y : int
        Number of features in Y
    x_rank : float
        A weird number used by sklearn.make_regression. It should be an integer, but values < 1 work. Changing this changes the degree of compressibility in X
    noise : float
        The amount of noise to inject (this is the SD of a Gaussian distribution)
    train_split : float
        The fraction of data used for the training datasets
    valid_split : float
        The fraction of data used for the validation dataset. The fraction of data used for testing = 1 - train_split _ valid_split
    """
    data = make_regression(n_samples = n, 
                           n_features = p, 
                           n_informative = p, 
                           n_targets = len_y, 
                           effective_rank = x_rank, 
                           noise = .1,
                           random_state = 123)
    
    train_split_idx = int(n//(1/train_split) + 1)
    valid_split_idx = int(train_split_idx + n//(1/valid_split))
    
    X_train, X_valid, X_test = (
        data[0][:train_split_idx, :], 
        data[0][train_split_idx:valid_split_idx, :],
        data[0][valid_split_idx:, :])
    
    Y_train, Y_valid, Y_test = (
        data[1][:train_split_idx, :], 
        data[1][train_split_idx:valid_split_idx, :],
        data[1][valid_split_idx:, :])
    
    home = Path.home()
    path_for_data = home/"teas-data/sklearn/"
    if not os.path.exists(path_for_data):
        os.makedirs(path_for_data)
        
    for (data, file_name) in [(X_train, "X_train.csv"), 
                          (X_valid, "X_valid.csv"), 
                          (X_test, "X_test.csv"),
                          (Y_train, "Y_train.csv"), 
                          (Y_valid, "Y_valid.csv"), 
                          (Y_test, "Y_test.csv")]:
        np.savetxt(path_for_data/file_name, data, delimiter=",")
        
def load_skl_data(path_for_data):
    """Loads and returns a pregenerated sklearn dataset from `data_path`
    """
    X_train = np.loadtxt(path_for_data/"X_train.csv", delimiter=",")
    X_valid = np.loadtxt(path_for_data/"X_valid.csv", delimiter=",")
    X_test = np.loadtxt(path_for_data/"X_test.csv", delimiter=",")
    Y_train = np.loadtxt(path_for_data/"Y_train.csv", delimiter=",")
    Y_valid = np.loadtxt(path_for_data/"Y_valid.csv", delimiter=",")
    Y_test = np.loadtxt(path_for_data/"Y_test.csv", delimiter=",")
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

# use fire to create a CLI
if __name__ == '__main__':
  fire.Fire(make_skl_dataset)
