import matplotlib.pyplot as plt

def plot_losses(epochs, train_loss, valid_loss):
    """Plot the losses from a training run"""
    plt.plot([e for e in range(epochs)], train_loss, 'b')
    plt.plot([e for e in range(epochs)], valid_loss, 'r')
    plt.title("Losses (training loss in blue)")
    plt.show()

def plot_predicted_vs_actual(var, var_hat, idx, title = "Predicted Y vs. observed Y"):
    """Plot predicted (var_hat) vs actual (var) for a row (or column) of data (idx)

    Paramaters
    ----------
    var : actual
    var_hat : predicted
    idx : the index of the row
    title : title of the plot
    """
    plt.scatter(var.detach().numpy(), var_hat.detach().numpy())
    plt.title(title);
    plt.xlabel("Feature {} observed".format(idx));
    plt.ylabel("Predicted");
    plt.show()