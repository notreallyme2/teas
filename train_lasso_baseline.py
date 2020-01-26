"""
Train a lasso baseline model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from data.skl_synthetic import *

def main(*args, **kwargs):
    home = Path.home()
    path_for_data = home/"teas-data/sklearn/"
    if not os.path.exists(path_for_data):
        make_skl_dataset()
    if os.path.exists(path_for_data):
        X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_skl_data(path_for_data)

    # visualise the data by plotting a random feature, and check compressibility
    # this should probably be in either plotting.py or the data module
    idx = np.random.randint(0, X_train.shape[1])
    plt.boxplot(X_train[:,idx]);
    plt.xlabel("Feature number: {}".format(idx));
    plt.show()

    pca = PCA(n_components=16)
    plt.plot( pca.fit(X_train).explained_variance_ratio_ )
    plt.title("X_train variance explained");
    plt.show()

    plt.plot( pca.fit(Y_train).explained_variance_ratio_ )
    plt.title("Y_train variance explained");
    plt.ylim(0, 0.2);
    plt.show()

    # search for optimal alpha
    print("Searching for optimal alpha")
    alphas = np.linspace(5e-5, 10e-4, num=10)
    valid_losses = []
    for alpha in alphas:
        print("Computing validation loss for alpha = {}".format(alpha))
        lasso_model = Lasso(alpha).fit(X_train, Y_train)
        Y_hat = lasso_model.predict(X_valid)
        valid_losses.append(mean_squared_error(Y_hat, Y_valid))

    # plot losses
    plt.plot(alphas, valid_losses)
    plt.title("Validation loss by alpha")
    plt.show()

    print("Best alpha: {}".format(np.argmin(valid_losses)))
    print("Training lasso model")
    lasso_model = Lasso(alphas[np.argmin(valid_losses)]).fit(X_train, Y_train)

    # look at the model coefficients
    plt.plot(lasso_model.coef_[1])
    plt.show()

    Y_hat_valid = lasso_model.predict(X_valid)

    # plot predicted vs. observed for a random row
    idx = np.random.randint(0, X_valid.shape[0])
    plt.scatter(Y_valid[idx], Y_hat_valid[idx])
    plt.xlabel("Observed")
    plt.xlabel("Predicted")
    plt.title("Predicted vs. observed (valid) for row {}".format(idx));

    Y_hat_test = lasso_model.predict(X_test)
    print("Final test loss (mse): {}".format(mean_squared_error(Y_hat_test, Y_test)))

if __name__ == "__main__":
    main()