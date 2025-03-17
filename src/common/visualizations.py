import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from typing import Optional


def plot_training_validation(
    training_losses: np.ndarray,
    training_accuracies: np.ndarray,
    validation_losses: np.ndarray,
    validation_accuracies: np.ndarray,
):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(15)

    epochs = range(training_losses.shape[0])


    ax[0].plot(epochs, training_losses, label="Training")
    ax[0].plot(epochs, validation_losses, label="Validation")

    ax[0].set_title("Loss progression")
    ax[0].set_xlabel("Epoch")
    ax[0].set_xticks(epochs)
    ax[0].set_ylabel("Loss")
    ax[0].legend()


    ax[1].plot(epochs, training_accuracies, label="Training")
    ax[1].plot(epochs, validation_accuracies, label="Validation")

    ax[1].set_title("Accuracy progression")
    ax[1].set_xlabel("Epoch")
    ax[1].set_xticks(epochs)
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

def plot_confusion_matrix(predicted: pd.DataFrame, actual: pd.DataFrame):
    precision, recall, f1_score, support = precision_recall_fscore_support(actual, predicted)
    matrix = confusion_matrix(actual, predicted)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Gammas", "Protons"])
    display.plot(values_format="")

    print(f"Gammas: precision={precision[0]:.3f}, recall={recall[0]:.3f}, f1={f1_score[0]:.3f} | Protons: precision={precision[1]:.3f}, recall={recall[1]:.3f}, f1={f1_score[1]:.3f}")

def print_training_progress(
    current_epoch: int,
    max_epochs: int,
    epoch_time_seconds: Optional[int],
    train_loss_accuracy: (float, float),
    validation_loss_accuracy: (float, float),
):
    epoch_msg = f"Epoch {current_epoch + 1}/{max_epochs}"
    time_msg = f" (took {epoch_time_seconds:.1f}s)" if epoch_time_seconds is not None else " "
    train_msg = f"TRAIN(loss|acc): {train_loss_accuracy[0]:.2f}|{100 * train_loss_accuracy[1]:.2f}%"
    validation_msg = f"VALID(loss|acc): {validation_loss_accuracy[0]:.2f}|{100 * validation_loss_accuracy[1]:.2f}%"

    print(f"{epoch_msg}{time_msg} - {train_msg} - {validation_msg}")

def print_test_results(loss: float, accuracy: float):
    print(f"Test data evaluation - LOSS: {loss:.2f} - ACC: {100 * accuracy:.3f}%")
