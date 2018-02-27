#!/usr/bin/env python3.5

import argparse
import itertools
import os
import pickle
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from keras import losses, optimizers
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix

from keras_utils import flexible_neural_net


def plot_accuracy_curve(acc, val_acc=None, title=None, fig_num=0, filename=None, show=True):
    """Plot train and validation history.

    If filename is None, the figure will be shown, otherwise it will be saved with name filename
    """
    # Plot epoch history for accuracy and loss
    if filename is None and show:
        plt.ion()
    fig = plt.figure(fig_num)
    subfig = fig.add_subplot(111)
    x_pts = range(1, len(acc) + 1)
    subfig.plot(x_pts, acc, label="training")
    if val_acc is not None:
        subfig.plot(x_pts, val_acc, label="validation")
    if title is None:
        title = "Model Accuracy History"
    subfig.set_title(title)
    fig.canvas.set_window_title("Figure {} - {}".format(fig_num, title))
    subfig.set_xlabel("Epoch")
    plt.legend()
    if filename is None:
        if show:
            plt.show()
            plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()


def plot_multiple_accuracy_curves(accs, val_accs=None, title=None, fig_num=0, filename=None,
                                  show=True):
    """Plot multiple train and validation histories in several figures.

    If filename is None, the figure will be shown, otherwise it will be saved with name filename
    """
    # Plot epoch history for accuracy and loss
    if filename is None and show:
        plt.ion()
    num_curves = len(accs)
    if num_curves % 5 == 0:
        h = int(num_curves / 5)
        w = 5
    else:
        h = int(np.floor(np.sqrt(num_curves)))
        w = int(np.ceil(np.sqrt(num_curves)))
    fig = plt.figure(fig_num, figsize=(8 * w, 6 * h))
    for i, (acc, val_acc) in enumerate(zip(accs, val_accs)):
        subfig = fig.add_subplot(h, w, i + 1)
        x_pts = range(1, len(acc) + 1)
        subfig.plot(x_pts, acc, label="training")
        if val_acc is not None:
            subfig.plot(x_pts, val_acc, label="validation")
        subfig.set_xlabel("Epoch")
        plt.legend()
    if title is None:
        title = "Model Accuracy History"
    fig.suptitle(title)
    fig.canvas.set_window_title("Figure {} - {}".format(fig_num, title))
    if filename is None:
        if show:
            plt.show()
            plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()


def save_plt_figures_to_pdf(filename, figs=None, dpi=200):
    """Save all matplotlib figures in a single PDF file."""
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    print("PDF file saved in '{}'.".format(filename))


def plot_image(location, title=None, fig_num=0, show=True):
    """Show image saved in location."""
    img = mpimg.imread(location)
    if show:
        plt.ion()
    fig = plt.figure(fig_num, figsize=(8 * 8, 6 * 8), dpi=300)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        fig.suptitle(title)
        fig.canvas.set_window_title("Figure {} - {}".format(fig_num, title))
    if show:
        plt.show()
        plt.ioff()


def transform_curves_to_plot(y_pts, x_pts):
    """Transform lists to format that pyplot will print them.

    Transforms this: [[1,2,3,4], [5,6,7,8]] into [[1,5], [2,6], [3,7], [4,8]]
    """
    new_y_pts = []
    new_x_pts = []
    last_row = max([len(row) for row in y_pts])
    for i, row in enumerate(y_pts):
        for j, y in enumerate(row):
            try:
                new_y_pts[j].append(y)
                new_x_pts[j].append(x_pts[i][j])
            except IndexError:
                new_y_pts.append([y])
                new_x_pts.append([x_pts[i][j]])
        x = x_pts[i][j]
        while j < last_row:
            j += 1
            try:
                new_y_pts[j].append(y)
                new_x_pts[j].append(x)
            except IndexError:
                new_y_pts.append([y])
                new_x_pts.append([x])
    return (new_y_pts, new_x_pts)


def plot_line(y_pts, x_pts=None, y_label=None, x_label=None, title=None, axis=None, style="-",
              color=None, y_scale="linear", x_scale="linear", label=None, fig_num=0, show=True,
              filename=None, n_ticks=None, linewidth=None):
    """Plot one or several 1D or 2D lines or point clouds.

    :param y_pts: y coordinates. A list of list can represent several lines
    :param x_pts: x coordinates. A list of list can represent several lines
    :param y_label: label for y axis
    :param x_label: label for x axis
    :param title: the title of the figure
    :param axis: len4 list [xmin, xmax, ymin, ymax] to pick range we will see
    :param style: ('-': line), ('x': cross), ('o': circle), ('s': squre), ('--': dotted line)...
    :param color: 'r','g','b','c','m','y','k'... If left blank, every curve will take a new color
    :param label: text that will be displayed if we show a legend
    :param fig_num: number of figure
    :param filename: if not None, save image to such location
    :param n_ticks: len2 list [ticks_x, tics_y] to select # ticks in both axis
    :param show: if False, the figure will not be shown. Only used if filename is None
    """
    if filename is None and show:
        plt.ion()
    fig = plt.figure(fig_num)
    if x_pts is None:
        plt.plot(y_pts, color=color, linestyle=style, linewidth=linewidth, label=label)
    else:
        if isinstance(y_pts, list) and isinstance(y_pts[0], list):
            (y_pts, x_pts) = transform_curves_to_plot(y_pts, x_pts)
        plt.plot(x_pts, y_pts, color=color, linestyle=style, linewidth=linewidth, label=label)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    if title is not None:
        plt.title(title)
        fig.canvas.set_window_title("Figure {} - {}".format(fig_num, title))
    if axis is not None:
        plt.axis(axis)
    plt.yscale(y_scale)
    plt.xscale(x_scale)
    if n_ticks is not None:
        if n_ticks[0] is not None:
            plt.locator_params(axis='x', nbins=n_ticks[0])
        if n_ticks[1] is not None:
            plt.locator_params(axis='y', nbins=n_ticks[1])
    if label is not None:
        plt.legend()
    plt.draw()
    if filename is None:
        if show:
            plt.show()
            plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()


def plot_binary_backgorund(y_pts, first_x=0, y_label=None, x_label=None, title=None, axis=None,
                           color0="red", color1="blue", fig_num=0, show=True, filename=None,
                           n_ticks=None):
    """Plot binary data as background colors: 0 as color0, and 1 as color1.

    :param y_pts: y coordinates (only 0s or 1s, please)
    :param x_pts: x coordinates
    :param y_label: label for y axis
    :param x_label: label for x axis
    :param title: the title of the figure
    :param axis: len4 list [xmin, xmax, ymin, ymax] to pick range we will see
    :param color0: color for label 0. 'r','g','b','c','m','y','k'...
    :param color1: color for label 1. 'r','g','b','c','m','y','k'...
    :param fig_num: number of figure
    :param filename: if not None, save image to such location
    :param n_ticks: len2 list [ticks_x, tics_y] to select # ticks in both axis
    :param show: if False, the figure will not be shown. Only used if filename is None
    """
    if filename is None and show:
        plt.ion()
    fig = plt.figure(fig_num)
    new_y = []
    new_x = []
    prev = 0
    for i, y in enumerate(y_pts):
        if y != prev:
            new_y.append(prev)
            prev = y
            new_x.append(i + first_x)
        new_y.append(y)
        new_x.append(i + first_x)
    new_y.append(0)
    new_x.append(new_x[-1])
    plt.fill(new_x, new_y, color=color1, alpha=0.5, label="1s", lw=0)
    new_y = []
    new_x = []
    prev = 1
    for i, y in enumerate(y_pts):
        if y != prev:
            new_y.append(prev)
            prev = y
            new_x.append(i + first_x)
        new_y.append(y)
        new_x.append(i + first_x)
    new_y.append(1)
    new_x.append(new_x[-1])
    plt.fill(new_x, new_y, color=color0, alpha=0.5, label="0s", lw=0)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    if title is not None:
        plt.title(title)
        fig.canvas.set_window_title("Figure {} - {}".format(fig_num, title))
    if axis is not None:
        plt.axis(axis)
    else:
        plt.axis((None, None, 0, 1))
    if n_ticks is not None:
        if n_ticks[0] is not None:
            plt.locator_params(axis='x', nbins=n_ticks[0])
        if n_ticks[1] is not None:
            plt.locator_params(axis='y', nbins=n_ticks[1])
    plt.legend()
    plt.draw()
    if filename is None:
        if show:
            plt.show()
            plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()


def load_organized_dataset(path):
    """Load organized dataset (contains balanced train and test set)."""
    f = np.load(path + "/training_set.npz")
    try:
        x_train = f["x"]
        y_train = f["y"]
    except KeyError:
        x_train = f["arr_0"]
        y_train = f["arr_1"]
    f.close()
    f = np.load(path + "/test_set.npz")
    try:
        x_test = f["x"]
        y_test = f["y"]
    except KeyError:
        x_test = f["arr_0"]
        y_test = f["arr_1"]
    f.close()
    with open(path + "/training_set_patients.pkl", 'rb') as f:
        patients_train = pickle.load(f)
    with open(path + "/test_set_patients.pkl", 'rb') as f:
        patients_test = pickle.load(f)
    with open(path + "/training_set_masks.pkl", 'rb') as f:
        mask_train = pickle.load(f)
    with open(path + "/test_set_masks.pkl", 'rb') as f:
        mask_test = pickle.load(f)
    return (x_train, y_train), (x_test, y_test), (patients_train, mask_train), (patients_test,
                                                                                mask_test)


def get_confusion_matrix(model, x_set, y_set):
    """Docstring for get_confusion_matrix."""
    pred_percents = model.predict(x_set)
    true_labels = np.argmax(y_set, axis=1)
    pred_labels = np.argmax(pred_percents, axis=1)
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
    return accuracy, precision, recall, num_true_labels


def calculate_patients_label(slices_labels, patients):
    """Count number of 1 and 0 labels for every patient and calculate their average label."""
    patient_percentages = []
    prev_patient = ""
    num_patients = 0
    for label, patient in zip(slices_labels, patients):
        if patient == prev_patient:
            patient_percentages[-1] += label
            num_patients += 1
        else:
            if len(patient_percentages) > 0:
                patient_percentages[-1] /= num_patients
            patient_percentages.append(label)
            num_patients = 1
        prev_patient = patient
    patient_percentages[-1] /= num_patients
    return np.array(patient_percentages)


def get_confusion_matrix_for_patient(model, x_set, y_set, patient_set):
    """Docstring for get_confusion_matrix_for_patient."""
    pred_percents = model.predict(x_set)
    pred_labels = np.argmax(pred_percents, axis=1)
    true_labels = np.argmax(y_set, axis=1)
    pred_labels_percentages = calculate_patients_label(pred_labels, patient_set)
    true_labels_percentages = calculate_patients_label(true_labels, patient_set)
    pred_labels = pred_labels_percentages > 0.5
    true_labels = true_labels_percentages > 0.5
    pred_labels = pred_labels.astype(int)
    true_labels = true_labels.astype(int)
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
    return (accuracy, precision, recall, num_true_labels, pred_labels_percentages,
            true_labels_percentages)


def parse_arguments():
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Run single experiment on organized dataset.")
    parser.add_argument('-plcv', '--patient_level_cross_validation', default=False,
                        action="store_true", help="split dataset cross validation at the patient "
                        "level.")
    return parser.parse_args()


def do_cross_validation(layers, optimizer, loss, x_whole, y_whole, patients_whole, num_patients,
                        location="cross_validation_results", patient_level_cv=False):
    """Do cross validation on dataset."""
    # Do 10-fold CV in whole set
    num_folds = 10
    if patient_level_cv:
        # Get splits indices to separate dataset in patients
        num_folds = 11  # 11 because 77 % 11 = 0
        num_patients_per_fold = int(np.ceil(num_patients / num_folds))
        patient_num = 0
        prev_patient = ""
        folds_indices = [0]
        for i, patient in enumerate(patients_whole):
            if patient != prev_patient:
                prev_patient = patient
                patient_num += 1
                if patient_num > num_patients_per_fold:
                    folds_indices.append(i)
                    patient_num = 1
        folds_indices.append(len(patients_whole))
    else:
        size_fold = x_whole.shape[0] / num_folds
    historic_acc = None
    historic_val_acc = None
    tr_all_data_log = {"history_acc": [], "history_val_acc": [], "accuracy": [], "recall0": [],
                       "recall1": [], "precision0": [], "precision1": [], "num_label0": [],
                       "num_label1": [], "num_labels": []}
    all_data_log = {"history_acc": [], "history_val_acc": [], "accuracy": [], "recall0": [],
                    "recall1": [], "precision0": [], "precision1": [], "num_label0": [],
                    "num_label1": [], "num_labels": []}
    pat_all_data_log = {"history_acc": [], "history_val_acc": [], "accuracy": [], "recall0": [],
                        "recall1": [], "precision0": [], "precision1": [], "num_label0": [],
                        "num_label1": [], "num_labels": [], "pred_percentages": [],
                        "true_percentages": []}
    data_splits = []
    weights = None  # This makes sure that the weight for every layer are reset every fold
    for i in range(num_folds):
        print("\n--------------------------------------------------------------------------------")
        print("\nFold {}/{} in Cross Validation Analysis".format(i + 1, num_folds))
        # Split dataset in training and cross-validation sets
        if not patient_level_cv:
            idx0 = int(np.round(i * size_fold))
            idx1 = int(np.round((i + 1) * size_fold))
        else:
            idx0 = folds_indices[i]
            idx1 = folds_indices[i + 1]
        if i != 0:
            data_splits.append(idx0)
        x_train_cv = np.append(x_whole[:idx0], x_whole[idx1:], axis=0)
        y_train_cv = np.append(y_whole[:idx0], y_whole[idx1:], axis=0)
        patients_whole_cv = np.append(patients_whole[:idx0], patients_whole[idx1:], axis=0)
        x_test_cv = x_whole[idx0:idx1]
        y_test_cv = y_whole[idx0:idx1]
        patients_test_cv = patients_whole[idx0:idx1]

        # Train model
        parameters = flexible_neural_net((x_train_cv, y_train_cv), (x_test_cv, y_test_cv),
                                         optimizer, loss, *layers,
                                         batch_size=32, epochs=5, initial_weights=weights,
                                         early_stopping=None, verbose=False,
                                         files_suffix=i, location=location, return_more=True)
        [lTr, aTr], [lTe, aTe], time, location, n_epochs, weights, model, history = parameters

        # Save learning curve
        if historic_acc is None:
            historic_acc = np.array(history.history['acc'])
            historic_val_acc = np.array(history.history['val_acc'])
        else:
            historic_acc += history.history['acc']
            historic_val_acc += history.history['val_acc']

        # Save statistical data for cross val set
        print("Cross Validation Statistics:")
        accuracy, precision, recall, num_labels = get_confusion_matrix(model, x_test_cv, y_test_cv)
        all_data_log["history_acc"].append(history.history['acc'])
        all_data_log["history_val_acc"].append(history.history['val_acc'])
        all_data_log["accuracy"].append(accuracy)
        all_data_log["recall0"].append(recall[0])
        all_data_log["precision0"].append(precision[0])
        all_data_log["recall1"].append(recall[1])
        all_data_log["precision1"].append(precision[1])
        all_data_log["num_label0"].append(num_labels[0])
        all_data_log["num_label1"].append(num_labels[1])
        all_data_log["num_labels"].append(num_labels[1] + num_labels[0])

        # Save statistical data for training set
        print("Training Statistics:")
        accuracy, precision, recall, num_labels = get_confusion_matrix(model, x_train_cv,
                                                                       y_train_cv)
        tr_all_data_log["accuracy"].append(accuracy)
        tr_all_data_log["recall0"].append(recall[0])
        tr_all_data_log["precision0"].append(precision[0])
        tr_all_data_log["recall1"].append(recall[1])
        tr_all_data_log["precision1"].append(precision[1])
        tr_all_data_log["num_label0"].append(num_labels[0])
        tr_all_data_log["num_label1"].append(num_labels[1])
        tr_all_data_log["num_labels"].append(num_labels[1] + num_labels[0])

        # Save patient level data from cross valiation set
        print("Patient Level Statistics")
        params = get_confusion_matrix_for_patient(model, x_test_cv, y_test_cv, patients_test_cv)
        accuracy, precision, recall, num_labels, pred_percentages, true_percentages = params
        pat_all_data_log["accuracy"].append(accuracy)
        pat_all_data_log["recall0"].append(recall[0])
        pat_all_data_log["precision0"].append(precision[0])
        pat_all_data_log["recall1"].append(recall[1])
        pat_all_data_log["precision1"].append(precision[1])
        pat_all_data_log["num_label0"].append(num_labels[0])
        pat_all_data_log["num_label1"].append(num_labels[1])
        pat_all_data_log["num_labels"].append(num_labels[1] + num_labels[0])
        pat_all_data_log["pred_percentages"].extend(pred_percentages)
        pat_all_data_log["true_percentages"].extend(true_percentages)

        # Print feedback
        print("\nAccuracy Training: {}".format(aTr))
        print("Accuracy Test:     {}".format(aTe))
        print("Time taken:        {0:.3f} seconds".format(time))
        print("Location:          {}".format(location))
    # Convert historic_acc into average value
    historic_acc = historic_acc / num_folds
    historic_val_acc = historic_val_acc / num_folds
    # Plot stuff
    show_plots = False
    plt.close("all")
    # Fig 2
    f = 2
    plot_accuracy_curve(historic_acc, historic_val_acc, title="Model Mean Accuracy", fig_num=f,
                        show=show_plots)
    # Fig 1
    f = 1
    plot_multiple_accuracy_curves(all_data_log["history_acc"], all_data_log["history_val_acc"],
                                  title="Model Fold Accuracy History", fig_num=f, show=show_plots)
    # Fig 3
    f = 3
    plot_line(all_data_log["accuracy"], range(1, num_folds + 1), label="Accuracy", fig_num=f,
              show=show_plots)
    plot_line(all_data_log["recall0"], range(1, num_folds + 1), label="Recall 0", fig_num=f,
              show=show_plots)
    plot_line(all_data_log["recall1"], range(1, num_folds + 1), label="Recall 1", fig_num=f,
              show=show_plots)
    plot_line(all_data_log["precision0"], range(1, num_folds + 1), label="Precision 0", fig_num=f,
              show=show_plots)
    plot_line(all_data_log["precision1"], range(1, num_folds + 1), label="Precision 1", fig_num=f,
              title="Cross Validation Accuracy, Recall and Precision", show=show_plots,
              x_label="Cross Validation Fold Number", n_ticks=[10, None])
    # Fig 4
    f = 4
    plot_line(all_data_log["num_label1"], range(1, num_folds + 1), label="Number 1s", fig_num=f,
              show=show_plots)
    plot_line(all_data_log["num_labels"], range(1, num_folds + 1), label="Number 0s and 1s",
              fig_num=f, title="Cross Validation Set Size", axis=[None, None, 0, None],
              x_label="Cross Validation Fold Number", n_ticks=[10, None], show=show_plots)
    # Fig 5
    f = 5
    plot_line(tr_all_data_log["accuracy"], range(1, num_folds + 1), label="Accuracy", fig_num=f,
              show=show_plots)
    plot_line(tr_all_data_log["recall0"], range(1, num_folds + 1), label="Recall 0", fig_num=f,
              show=show_plots)
    plot_line(tr_all_data_log["recall1"], range(1, num_folds + 1), label="Recall 1", fig_num=f,
              show=show_plots)
    plot_line(tr_all_data_log["precision0"], range(1, num_folds + 1), label="Precision 0",
              fig_num=f, show=show_plots)
    plot_line(tr_all_data_log["precision1"], range(1, num_folds + 1), label="Precision 1",
              fig_num=f, title="Training Accuracy, Recall and Precision", show=show_plots,
              x_label="Cross Validation Fold Number", n_ticks=[10, None])
    # Fig 6
    f = 6
    plot_line(tr_all_data_log["num_label1"], range(1, num_folds + 1), label="Number 1s", fig_num=f,
              show=show_plots)
    plot_line(tr_all_data_log["num_labels"], range(1, num_folds + 1), label="Number 0s and 1s",
              fig_num=f, title="Training Set Size", axis=[None, None, 0, None], show=show_plots,
              x_label="Cross Validation Fold Number", n_ticks=[10, None])
    # Fig 7
    f = 7
    plot_line(pat_all_data_log["accuracy"], range(1, num_folds + 1), label="Accuracy", fig_num=f,
              show=show_plots)
    plot_line(pat_all_data_log["recall0"], range(1, num_folds + 1), label="Recall 0", fig_num=f,
              show=show_plots)
    plot_line(pat_all_data_log["recall1"], range(1, num_folds + 1), label="Recall 1", fig_num=f,
              show=show_plots)
    plot_line(pat_all_data_log["precision0"], range(1, num_folds + 1), label="Precision 0",
              fig_num=f, show=show_plots)
    plot_line(pat_all_data_log["precision1"], range(1, num_folds + 1), label="Precision 1",
              fig_num=f, title="Cross Validation Patient Accuracy, Recall and Precision",
              show=show_plots, x_label="Cross Validation Fold Number", n_ticks=[10, None])
    # Fig 8
    f = 8
    plot_line(pat_all_data_log["num_label1"], range(1, num_folds + 1), label="Number 1s",
              fig_num=f, show=show_plots)
    plot_line(pat_all_data_log["num_labels"], range(1, num_folds + 1), label="Number 0s and 1s",
              fig_num=f, title="Cross Validation Patient Set Size", axis=[None, None, 0, None],
              show=show_plots, x_label="Cross Validation Fold Number", n_ticks=[10, None])
    # Fig 9
    f = 9
    plot_binary_backgorund(pat_all_data_log["true_percentages"], fig_num=f, show=show_plots,
                           x_label="Patient Number", title="Label Conviction per Patient")
    plot_line([0.5, 0.5], [0, num_patients], fig_num=f, show=show_plots)
    plot_line(pat_all_data_log["pred_percentages"], np.array(range(0, num_patients)) + 0.5,
              label="Label conviction", color="#00ff00", fig_num=f, show=show_plots,
              axis=(None, None, -0.005, 1.005))
    # Fig 0
    f = 0
    plot_image(location + "/model0.png", fig_num=f, title="Model used", show=show_plots)
    # Fig 10
    f = 10
    patient_changes = []
    prev_label = 1
    prev_patient = ""
    for patient in patients_whole:
        if patient != prev_patient:
            prev_patient = patient
            prev_label = abs(prev_label - 1)
        patient_changes.append(prev_label)
    plot_binary_backgorund(patient_changes, fig_num=f, show=show_plots, x_label="Slice number",
                           title="Dataset patient distribution vs. cross validation dataset split")
    for i, split in enumerate(data_splits):
        split_label = None
        if i == 0:
            split_label = "Cross validation split"
        plot_line([0, 1], [split, split], fig_num=f, label=split_label, color="#00ff00",
                  style="--", linewidth=2, show=show_plots)
    # Save all figures to a PDF called figures.pdf
    save_plt_figures_to_pdf(location + "/figures.pdf")
    if show_plots:
        input("Press ENTER to close figures")
        plt.close("all")
    return all_data_log, tr_all_data_log, pat_all_data_log


def create_layers(input_shape, labels, filters=16, units=16, num_convolutions=1, dropout1=0,
                  dropout2=0):
    """Create list of layers based on some parameters."""
    # Define model
    filters = filters  # [8, 16, 32, 64]
    units = units  # [8, 16, 32, 64]
    activation = "relu"  # ["relu", "tanh"]
    num_conv = num_convolutions  # [1, 2, 3]
    loss = losses.categorical_crossentropy
    optimizer = optimizers.Adam()
    d1 = dropout1  # [0, 0.25, 0.5]
    d2 = dropout2  # [0, 0.25, 0.5]
    dropout1 = [] if d1 <= 0 else [Dropout(d1)]
    dropout2 = [] if d2 <= 0 else [Dropout(d2)]
    convolutional_layers = []
    for n in range(num_conv):
        if n == 0:
            convolutional_layers.append(Conv2D(filters, kernel_size=(3, 3), activation=activation,
                                               input_shape=input_shape))
        else:
            convolutional_layers.append(Conv2D(filters, kernel_size=(3, 3), activation=activation))
        convolutional_layers.append(MaxPooling2D(pool_size=(2, 2)))
    layers = [*convolutional_layers,
              *dropout1,
              Flatten(),
              Dense(units, activation=activation),
              *dropout2,
              Dense(len(labels), activation='softmax')]
    # Save model image
    # model = Sequential()
    # for layer in layers:
    #     model.add(layer)
    # plot_model(model, show_shapes=True, show_layer_names=False, show_params=True,
    #            to_file="model.png")
    return layers, optimizer, loss


def main():
    """Get dataset and train model."""
    # Print when and how job starts
    print("--------------------------------------------------------------------------------------")
    now = datetime.now()
    date_formated = "{} {:02d}:{:02d}:{:02d}".format(now.date(), now.hour, now.minute, now.second)
    print("|  Running: {:<71}  |".format(" ".join(sys.argv)))
    print("|  Time:    {:<71}  |".format(date_formated))
    print("--------------------------------------------------------------------------------------")

    # Parse arguments
    args = parse_arguments()

    # Load dataset
    train_set, test_set, train_aux, test_aux = load_organized_dataset("data/organized")
    (x_train, y_train), (x_test, y_test), = train_set, test_set
    (patients_train, mask_train), (patients_test, mask_test) = train_aux, test_aux
    x_whole = np.append(x_train, x_test, axis=0)
    y_whole = np.append(y_train, y_test, axis=0)
    mask_whole = np.append(mask_train, mask_test, axis=0)
    patients_whole = np.append(patients_train, patients_test, axis=0)
    patients = np.unique(patients_whole)
    input_shape = x_whole.shape[1:]
    num_patients = len(patients)
    labels = np.unique(y_whole)
    y_whole = np_utils.to_categorical(y_whole, len(labels))
    y_train = np_utils.to_categorical(y_train, len(labels))
    y_test = np_utils.to_categorical(y_test, len(labels))

    # Print some information of data
    print("Training set shape:  {}".format(x_train.shape))
    print("Test set shape:      {}".format(x_test.shape))
    print("Whole set shape:     {}".format(x_whole.shape))
    print("Existing labels:     {}".format(labels))
    print("Number of patients:  {}".format(num_patients))
    print("Number of slices:    {}".format(x_whole.shape[0]))

    # Create folder where we will save data
    n = 0
    while True:
        location = "nn{:04d}".format(n)
        if not os.path.exists(location):
            break
        n += 1

    filters = [8, 16, 32]
    units = [16]
    num_convolutions = [1]
    dropout1 = [0]
    dropout2 = [0]
    all_data = []
    for comb in itertools.product(filters, units, num_convolutions, dropout1, dropout2):
        # Create layers list that will define model
        f, u, c, d1, d2 = comb
        layers, optimizer, loss = create_layers(input_shape, labels, filters=f, units=u,
                                                num_convolutions=c, dropout1=d1, dropout2=d2)
        subfolder = "-".join([str(x) for x in comb])
        params = do_cross_validation(layers, optimizer, loss, x_whole, y_whole, patients_whole,
                                     num_patients, location=location + "/" + subfolder,
                                     patient_level_cv=args.patient_level_cross_validation)
        all_data.append(params)


if __name__ == "__main__":
    main()
