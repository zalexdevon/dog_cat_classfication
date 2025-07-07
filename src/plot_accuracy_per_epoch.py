import matplotlib.pyplot as plt


def plot_accuracy_per_epoch(history):

    train_accuracies = history["accuracy"]
    train_accuracies = [item * 100 for item in train_accuracies]

    val_accuracies = history["val_accuracy"]
    val_accuracies = [item * 100 for item in val_accuracies]

    num_epochs = len(train_accuracies)
    epochs = list(range(1, num_epochs + 1))

    plt.plot(epochs, train_accuracies, label="train", color="gray")
    plt.plot(epochs, val_accuracies, label="val", color="blue")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
