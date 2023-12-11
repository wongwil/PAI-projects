import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn

def ece(predicted_probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 30) -> float:
    """
    Computes the Expected Calibration Error (ECE).
    Many options are possible; in this implementation, we provide a simple version.

    Using a uniform binning scheme on the full range of probabilities, zero
    to one, we bin the probabilities of the predicted label only (ignoring
    all other probabilities). For the ith bin, we compute the avg predicted
    probability, p_i, and the bin's total accuracy, a_i.
    We then compute the ith calibration error of the bin, |p_i - a_i|.
    The final returned value is the weighted average of calibration errors of each bin.

    :param predicted_probabilities: Predicted probabilities, float array of shape (num_samples, num_classes)
    :param labels: True labels, int tensor of shape (num_samples,) with each entry in {0, ..., num_classes - 1}
    :param n_bins: Number of bins for histogram binning
    :return: ECE score as a float
    """
    num_samples, num_classes = predicted_probabilities.shape
    if np.min(labels) == -1:
        # Treat ambiguous as its own class (which is never predicted)
        labels[labels == -1] = num_classes
        num_classes += 1
        predicted_probabilities = np.concatenate((predicted_probabilities, np.zeros((num_samples, 1))), axis=1)

    # Predictions are the classes with highest probability
    predictions = np.argmax(predicted_probabilities, axis=1)
    prediction_confidences = predicted_probabilities[range(num_samples), predictions]

    # Use uniform bins on the range of probabilities, i.e. closed interval [0.,1.]
    bin_upper_edges = np.histogram_bin_edges([], bins=n_bins, range=(0.0, 1.0))
    bin_upper_edges = bin_upper_edges[1:]  # bin_upper_edges[0] = 0.

    probs_as_bin_num = np.digitize(prediction_confidences, bin_upper_edges)
    sums_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins, weights=prediction_confidences)
    sums_per_bin = sums_per_bin.astype(np.float32)

    total_per_bin = (
        np.bincount(probs_as_bin_num, minlength=n_bins) + np.finfo(sums_per_bin.dtype).eps
    )  # division by zero
    avg_prob_per_bin = sums_per_bin / total_per_bin

    onehot_labels = np.eye(num_classes)[labels]
    accuracies = onehot_labels[range(num_samples), predictions]  # accuracies[i] is 0 or 1
    accuracies_per_bin = np.bincount(probs_as_bin_num, weights=accuracies, minlength=n_bins) / total_per_bin

    prob_of_being_in_a_bin = total_per_bin / float(num_samples)

    ece_ret = np.abs(accuracies_per_bin - avg_prob_per_bin) * prob_of_being_in_a_bin
    ece_ret = np.sum(ece_ret)
    return float(ece_ret)


def cost_function(predicted_ys: torch.Tensor, actual_ys: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cost of predicting `predicted_ys` for actual classes `actual_ys`.

    Predicted and actual ys are in {-1, 0, 1, 2, 3, 4, 5}, where -1 corresponds to "don't know".
    Predicting -1 always incurs a fixed cost, even for ambiguous samples.
    Wrongly predicting a class in {0, 1, 2, 3, 4, 5} incurs a larger fixed cost.
    Note that predicting any class in {0, 1, 2, 3, 4, 5} for ambiguous samples counts as wrong.
    """
    assert predicted_ys.size() == actual_ys.size()

    COST_WRONG = 3
    COST_UNSURE = 1

    num_predictions = predicted_ys.size(0)
    unsure_pred_mask = (predicted_ys == -1)
    num_unsure_predictions = unsure_pred_mask.float().sum()
    num_wrong_predictions = (predicted_ys[~unsure_pred_mask] != actual_ys[~unsure_pred_mask]).float().sum()
    return (COST_UNSURE * num_unsure_predictions + COST_WRONG * num_wrong_predictions) / num_predictions


def draw_reliability_diagram(out, title="Reliability Diagram", xlabel="Confidence", ylabel="Accuracy"):
    """Draws a reliability diagram into a subplot."""
    fig, ax = plt.subplots()
    accuracies = out["calib_accuracy"]
    confidences = out["calib_confidence"]
    counts = out["p"]
    bins = out["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size / 2.0

    widths = bin_size
    alphas = 0.3

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.0
    colors[:, 1] = 60 / 255.0
    colors[:, 2] = 60 / 255.0
    colors[:, 3] = alphas

    gap_plt = ax.bar(
        positions,
        np.abs(accuracies - confidences),
        bottom=np.minimum(accuracies, confidences),
        width=widths,
        edgecolor=colors,
        color=colors,
        linewidth=1,
        label="Gap",
    )

    acc_plt = ax.bar(
        positions,
        0,
        bottom=accuracies,
        width=widths,
        edgecolor="black",
        color="black",
        alpha=1.0,
        linewidth=3,
        label="Accuracy",
    )

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[gap_plt, acc_plt])
    fig.tight_layout()
    return fig


def calc_calibration_curve(predicted_probabilities: np.ndarray, labels: np.ndarray, num_bins: int) -> plt.figure:
    """
    Calculates a calibration curve to be used in reliability diagrams and returns an ECE
    """

    num_samples, num_classes = predicted_probabilities.shape
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    confidences = predicted_probabilities[range(num_samples), predicted_classes]
    bins = np.linspace(start=0, stop=1, num=num_bins + 1)

    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    accuracies = predicted_classes == labels

    calib_confidence = np.zeros(num_bins, dtype=np.float32)
    calib_accuracy = np.zeros(num_bins, dtype=np.float32)
    ratios = np.zeros(num_bins, dtype=np.float32)
    ece = 0.0

    for bin_i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            calib_confidence[bin_i] = avg_confidence_in_bin
            calib_accuracy[bin_i] = accuracy_in_bin
            ratios[bin_i] = prop_in_bin

    return {
        "calib_confidence": calib_confidence,
        "calib_accuracy": calib_accuracy,
        "p": ratios,
        "bins": bins,
        "ece": ece,
    }


def setup_seeds() -> None:
    """
    Globally fixes seeds in case manual seeding is missing somewhere
    """
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # You do not need to use a GPU, and the code is only tested on CPU.
    # However, if you still use CUDA, the following code makes execution deterministic.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    torch.backends.cudnn.benchmark = False
