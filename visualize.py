import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_loss(log_history):

    epoch_losses = defaultdict(list)
    eval_loss = []
    epochs = []

    for log in log_history:
        if "loss" in log and "epoch" in log and "eval_loss" not in log:
            epoch_losses[int(log["epoch"])].append(log["loss"])

        if "eval_loss" in log:
            eval_loss.append(log["eval_loss"])
            epochs.append(int(log["epoch"]))

    train_loss = [np.mean(epoch_losses[e]) for e in epochs]

    plt.figure()
    plt.plot(epochs, train_loss, marker='o', label="Train Loss")
    plt.plot(epochs, eval_loss, marker='o', label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()

    plt.show()
