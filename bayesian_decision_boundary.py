#!/usr/bin/env python3.5
import matplotlib_handle_display  # Must be imported before anything matplotlib related
import numpy as np
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def plot_3d_results(x_mean, x_std, x_volume, y, unique_y, clusters, fignum=0, pause=True,
                    window_title=None):
    """Docstring for plot_3d_results."""
    # Create some necessary variables for plotting
    plt.ion()
    color_mask = ["r", "b"]
    legend_label = ["label 0", "label 1"]
    # Plot the original points with original labels
    fig = plt.figure(fignum)
    if window_title is not None:
        fig.canvas.set_window_title("Figure {} - {}".format(fignum, window_title))
    ax = fig.add_subplot(211, projection='3d')
    for label in unique_y:
        idx = (y == label)
        ax.scatter(x_mean[idx], x_std[idx], x_volume[idx], color=color_mask[label],
                   label=legend_label[label])
    ax.set_xlabel("Mean")
    ax.set_ylabel("Std Dev")
    ax.set_zlabel("Volume")
    ax.legend()
    ax.set_title("Original labels")
    # Plot the original points and color according to the cluster
    ax = fig.add_subplot(212, projection='3d')
    for label in unique_y:
        idx = (clusters == label)
        ax.scatter(x_mean[idx], x_std[idx], x_volume[idx], color=color_mask[label],
                   label=legend_label[label])
    ax.set_xlabel("Mean")
    ax.set_ylabel("Std Dev")
    ax.set_zlabel("Volume")
    ax.legend()
    ax.set_title("Clusters found")
    plt.show()
    if pause:
        input("Press ENTER to close all figures...")
        plt.close("all")


def get_confusion_matrix(true_labels, pred_labels):
    """Docstring for get_confusion_matrix."""
    errors_vector = np.array(pred_labels != true_labels)
    num_errors = np.sum(errors_vector)
    size_set = pred_labels.size
    accuracy = 1 - num_errors / size_set
    print("  Results: {} errors from {} examples.".format(num_errors, size_set))
    print("  Accuracy: {}".format(accuracy))
    print("  Confusion Matrix (true 0s are col 0, true 1s are col 1):")
    confusion_mat = confusion_matrix(true_labels, pred_labels)
    if len(confusion_mat) < 2:
        if true_labels[0] == 0:
            confusion_mat = np.array([[confusion_mat[0, 0], 0], [0, 0]])
        else:
            confusion_mat = np.array([[0, 0], [0, confusion_mat[0, 0]]])
    print("        {}\n        {}".format(confusion_mat[0], confusion_mat[1]))
    print("  Precision and Recall:")
    num_true_labels = [sum(confusion_mat[0, :]), sum(confusion_mat[1, :])]
    num_pred_labels = [sum(confusion_mat[:, 0]), sum(confusion_mat[:, 1])]
    recall = [confusion_mat[0, 0] / num_true_labels[0], confusion_mat[1, 1] / num_true_labels[1]]
    precision = [confusion_mat[0, 0] / num_pred_labels[0],
                 confusion_mat[1, 1] / num_pred_labels[1]]
    print("    Precision: {}".format(precision))
    print("    Recall: {}".format(recall))
    print("   ", classification_report(true_labels, pred_labels).replace("\n", "\n    "))


def main():
    """Fins bayesian boundary of features dataset."""
    # Load and extract data from dataset
    dataset_name = "features_dataset_mean_std_volume.npz"
    f = np.load(dataset_name)
    x = f["x"]
    y = f["y"]
    unique_y = np.unique(y)
    num_labels = len(unique_y)
    x_mean = x[:, 0]
    x_std = x[:, 1]
    x_volume = x[:, 2]
    print("Shape X: {}".format(x.shape))
    print("Shape Y: {}".format(y.shape))
    print("Metrics: Mean ({}), Std ({}), Volume ({})".format(x_mean.shape, x_std.shape,
                                                             x_volume.shape))

    # Use a Gaussian Mixture model to fit
    g = mixture.GMM(n_components=num_labels)
    g.fit(x)
    # Return an index list of which cluster every sample belongs to
    clusters1 = g.predict(x)
    # Plot
    print("\nGaussian Mixture Model")
    get_confusion_matrix(y, clusters1)
    plot_3d_results(x_mean, x_std, x_volume, y, unique_y, clusters1, fignum=0, pause=False,
                    window_title="Gaussian Mixture Model")

    # Gaussian Naive Bayes
    clf = GaussianNB()
    clf.fit(x, y)
    clusters2 = clf.predict(x)
    # Plot
    print("\nGaussian Naive Bayes")
    get_confusion_matrix(y, clusters2)
    plot_3d_results(x_mean, x_std, x_volume, y, unique_y, clusters2, fignum=1, pause=False,
                    window_title="Gaussian Naive Bayes")

    # SVM
    clf = svm.SVC(degree=1)
    clf.fit(x, y)
    clusters3 = clf.predict(x)
    # Plot
    print("\nSupport Vector Machines")
    get_confusion_matrix(y, clusters3)
    plot_3d_results(x_mean, x_std, x_volume, y, unique_y, clusters3, fignum=2, pause=True,
                    window_title="Support Vector Machines")


if __name__ == "__main__":
    main()
