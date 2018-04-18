#!/usr/bin/env python3.5
import matplotlib_handle_display  # Must be imported before anything matplotlib related
import numpy as np
import os
import pickle
from prettytable import PrettyTable
from cycler import cycler
from matplotlib import pyplot as plt
from single_experiment_runner import plot_line, save_plt_figures_to_pdf, plot_accuracy_curve
from single_experiment_runner import plot_multiple_accuracy_curves
from single_experiment_runner import plot_multiple_roc_curves, plot_roc_curve
from sklearn.metrics import roc_curve, auc


# Cycle colors from normal line to dotted line (allows to tell 20 plots apart)
plt.rc('axes', prop_cycle=(cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                            '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e',
                                            '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) +
                           cycler('linestyle', ['-'] * 10 + ['--'] * 10)))


def main(filename1=None, filename2=None, filename3=None, destination_folder=None):
    """Load data from 3 files previoulsy saved with single_experiment_runner and save mean.

    This is an even dirtier copy of what single_experiment_runner does at the end.
    """
    # Load all saved files
    if filename1 is None:
        filename1 = ("nn_models1_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_"
                     "0-8192_mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename1, 'rb') as f:
        all_data1 = pickle.load(f)
        print("File 1 loaded from: '{}'".format(filename1))
    if filename2 is None:
        filename2 = ("nn_models2_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_"
                     "0-8192_mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename2, 'rb') as f:
        all_data2 = pickle.load(f)
        print("File 2 loaded from: '{}'".format(filename2))
    if filename3 is None:
        filename3 = ("nn_models3_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_"
                     "0-8192_mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename3, 'rb') as f:
        all_data3 = pickle.load(f)
        print("File 3 loaded from: '{}'".format(filename3))

    if destination_folder is None:
        destination_folder = "medians123_corrected"
    try:
        os.mkdir(destination_folder)
    except FileExistsError:
        pass
    print("Destination folder: '{}'".format(destination_folder))

    # Plot summary of results
    print("\nGenerating global figures...")
    plt.close("all")
    title = "Test"
    show_plots = False
    last_idx = len(all_data1) - 1
    wider_figsize = list(plt.rcParams.get('figure.figsize'))
    wider_figsize[0] += 2.1
    len_x_axis = len(all_data1[0][1]["accuracy"])
    if len_x_axis == 1:
        x_axis = [1024]
        x_scale = "linear"
        x_label = "Number of Patients in Training Set"
    elif len_x_axis == 6:
        x_axis = [1, 2, 3, 4, 5, 6]
        x_scale = "linear"
        x_label = "Cross Validation Fold Number"
    elif len_x_axis == 8:
        x_axis = [1, 2, 3, 4, 5, 6, 7, 8]
        x_scale = "linear"
        x_label = "Cross Validation Fold Number"
    elif len_x_axis == 9:
        x_axis = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        x_scale = "log"
        x_label = "Number of Patients in Training Set"
    x_history = range(1, 50 + 1)
    table_acc = PrettyTable(["Model \\ Training Samples"] + x_axis + ["Average"])
    table_pat_acc = PrettyTable(["Model \\ Training Samples"] + x_axis + ["Average"])
    mean_accuracies = []
    mean_pat_accuracies = []
    combs = []
    for i, (params1, params2, params3) in enumerate(zip(all_data1, all_data2, all_data3)):
        comb1, all_cv1, all_train1, all_pat1, history1, rocs1 = params1
        comb2, all_cv2, all_train2, all_pat2, history2, rocs2 = params2
        comb3, all_cv3, all_train3, all_pat3, history3, rocs3 = params3
        comb = comb1
        combs.append(comb)

        historic_acc = (history1[0] + history2[0] + history3[0]) / 3
        historic_val_acc = (history1[1] + history2[1] + history3[1]) / 3

        all_data_log = {}
        all_data_log["history_acc"] = (np.zeros((len(x_axis), len(historic_acc))) + all_cv1["history_acc"] + all_cv2["history_acc"] + all_cv3["history_acc"]) / 3
        all_data_log["history_val_acc"] = (np.zeros((len(x_axis), len(historic_acc))) + all_cv1["history_val_acc"] + all_cv2["history_val_acc"] + all_cv3["history_val_acc"]) / 3
        all_data_log["accuracy"] = (np.zeros(len(x_axis)) + all_cv1["accuracy"] + all_cv2["accuracy"] + all_cv3["accuracy"]) / 3
        all_data_log["recall0"] = (np.zeros(len(x_axis)) + all_cv1["recall0"] + all_cv2["recall0"] + all_cv3["recall0"]) / 3
        all_data_log["precision0"] = (np.zeros(len(x_axis)) + all_cv1["precision0"] + all_cv2["precision0"] + all_cv3["precision0"]) / 3
        all_data_log["recall1"] = (np.zeros(len(x_axis)) + all_cv1["recall1"] + all_cv2["recall1"] + all_cv3["recall1"]) / 3
        all_data_log["precision1"] = (np.zeros(len(x_axis)) + all_cv1["precision1"] + all_cv2["precision1"] + all_cv3["precision1"]) / 3
        all_data_log["num_label0"] = (np.zeros(len(x_axis)) + all_cv1["num_label0"] + all_cv2["num_label0"] + all_cv3["num_label0"]) / 3
        all_data_log["num_label1"] = (np.zeros(len(x_axis)) + all_cv1["num_label1"] + all_cv2["num_label1"] + all_cv3["num_label1"]) / 3
        all_data_log["num_labels"] = (np.zeros(len(x_axis)) + all_cv1["num_labels"] + all_cv2["num_labels"] + all_cv3["num_labels"]) / 3
        all_data_log["true_cv"] = (np.array(all_cv1["true_cv"]) + all_cv2["true_cv"] + all_cv3["true_cv"]) / 3
        all_data_log["pred_cv"] = (np.array(all_cv1["pred_cv"]) + all_cv2["pred_cv"] + all_cv3["pred_cv"]) / 3

        tr_all_data_log = {}
        tr_all_data_log["accuracy"] = (np.zeros(len(x_axis)) + all_train1["accuracy"] + all_train2["accuracy"] + all_train3["accuracy"]) / 3
        tr_all_data_log["recall0"] = (np.zeros(len(x_axis)) + all_train1["recall0"] + all_train2["recall0"] + all_train3["recall0"]) / 3
        tr_all_data_log["precision0"] = (np.zeros(len(x_axis)) + all_train1["precision0"] + all_train2["precision0"] + all_train3["precision0"]) / 3
        tr_all_data_log["recall1"] = (np.zeros(len(x_axis)) + all_train1["recall1"] + all_train2["recall1"] + all_train3["recall1"]) / 3
        tr_all_data_log["precision1"] = (np.zeros(len(x_axis)) + all_train1["precision1"] + all_train2["precision1"] + all_train3["precision1"]) / 3
        tr_all_data_log["num_label0"] = (np.zeros(len(x_axis)) + all_train1["num_label0"] + all_train2["num_label0"] + all_train3["num_label0"]) / 3
        tr_all_data_log["num_label1"] = (np.zeros(len(x_axis)) + all_train1["num_label1"] + all_train2["num_label1"] + all_train3["num_label1"]) / 3
        tr_all_data_log["num_labels"] = (np.zeros(len(x_axis)) + all_train1["num_labels"] + all_train2["num_labels"] + all_train3["num_labels"]) / 3
        tr_all_data_log["true_cv"] = (np.array(all_train1["true_cv"]) + all_train2["true_cv"] + all_train3["true_cv"]) / 3
        tr_all_data_log["pred_cv"] = (np.array(all_train1["pred_cv"]) + all_train2["pred_cv"] + all_train3["pred_cv"]) / 3

        pat_all_data_log = {}
        pat_all_data_log["accuracy"] = (np.zeros(len(x_axis)) + all_pat1["accuracy"] + all_pat2["accuracy"] + all_pat3["accuracy"]) / 3
        pat_all_data_log["recall0"] = (np.zeros(len(x_axis)) + all_pat1["recall0"] + all_pat2["recall0"] + all_pat3["recall0"]) / 3
        pat_all_data_log["precision0"] = (np.zeros(len(x_axis)) + all_pat1["precision0"] + all_pat2["precision0"] + all_pat3["precision0"]) / 3
        pat_all_data_log["recall1"] = (np.zeros(len(x_axis)) + all_pat1["recall1"] + all_pat2["recall1"] + all_pat3["recall1"]) / 3
        pat_all_data_log["precision1"] = (np.zeros(len(x_axis)) + all_pat1["precision1"] + all_pat2["precision1"] + all_pat3["precision1"]) / 3
        pat_all_data_log["num_label0"] = (np.zeros(len(x_axis)) + all_pat1["num_label0"] + all_pat2["num_label0"] + all_pat3["num_label0"]) / 3
        pat_all_data_log["num_label1"] = (np.zeros(len(x_axis)) + all_pat1["num_label1"] + all_pat2["num_label1"] + all_pat3["num_label1"]) / 3
        pat_all_data_log["num_labels"] = (np.zeros(len(x_axis)) + all_pat1["num_labels"] + all_pat2["num_labels"] + all_pat3["num_labels"]) / 3
        pat_all_data_log["true_cv"] = (np.array(all_pat1["true_cv"]) + all_pat2["true_cv"] + all_pat3["true_cv"]) / 3
        pat_all_data_log["pred_cv"] = (np.array(all_pat1["pred_cv"]) + all_pat2["pred_cv"] + all_pat3["pred_cv"]) / 3
        pat_all_data_log["true_percentages"] = (np.array(all_pat1["true_percentages"]) + all_pat2["true_percentages"] + all_pat3["true_percentages"]) / 3
        pat_all_data_log["pred_percentages"] = (np.array(all_pat1["pred_percentages"]) + all_pat2["pred_percentages"] + all_pat3["pred_percentages"]) / 3

        # Plot stuff
        plt.close("all")
        # Fig 2
        f = 2
        plot_accuracy_curve(historic_acc, historic_val_acc, title="Model Mean Accuracy", fig_num=f,
                            show=show_plots)
        # Fig 1
        f = 1
        title_train = ["Training: {} patients".format(x) for x in x_axis]
        plot_multiple_accuracy_curves(all_data_log["history_acc"], all_data_log["history_val_acc"],
                                      title="Accuracy History  vs.  Dataset Size", fig_num=f,
                                      show=show_plots, labels=title_train)
        # Fig 3
        f = 3
        plot_line(all_data_log["accuracy"], x_axis, label="Accuracy", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(all_data_log["recall0"], x_axis, label="Recall 0", fig_num=f, x_scale="log",
                  show=show_plots, style=".-")
        plot_line(all_data_log["recall1"], x_axis, label="Recall 1", fig_num=f, x_scale="log",
                  show=show_plots, style=".-")
        plot_line(all_data_log["precision0"], x_axis, label="Precision 0", x_scale="log",
                  fig_num=f, show=show_plots, style=".-")
        plot_line(all_data_log["precision1"], x_axis, label="Precision 1", x_scale="log",
                  fig_num=f, title="Test Accuracy, Recall and Precision",
                  show=show_plots, style=".-", x_label="Number of Patients in Training Set")
        # Fig 4
        f = 4
        plot_line(all_data_log["num_label1"], x_axis, label="Number 1s", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(all_data_log["num_labels"], x_axis, label="Number 0s and 1s",
                  fig_num=f, title="Test Set Size", axis=[None, None, 0, None], style=".-",
                  x_label="Number of Patients in Training Set", show=show_plots,  x_scale="log")
        # Fig 5
        f = 5
        plot_line(tr_all_data_log["accuracy"], x_axis, label="Accuracy", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(tr_all_data_log["recall0"], x_axis, label="Recall 0", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(tr_all_data_log["recall1"], x_axis, label="Recall 1", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(tr_all_data_log["precision0"], x_axis, label="Precision 0",
                  fig_num=f, show=show_plots, style=".-", x_scale="log")
        plot_line(tr_all_data_log["precision1"], x_axis, label="Precision 1",
                  fig_num=f, title="Training Accuracy, Recall and Precision", show=show_plots,
                  x_label="Number of Patients in Training Set", style=".-", x_scale="log")
        # Fig 6
        f = 6
        plot_line(tr_all_data_log["num_label1"], x_axis, label="Number 1s", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(tr_all_data_log["num_labels"], x_axis, label="Number 0s and 1s", show=show_plots,
                  fig_num=f, title="Training Set Size", axis=[None, None, 0, None],
                  x_label="Number of Patients in Training Set", style=".-", x_scale="log")
        # Fig 7
        f = 7
        plot_line(pat_all_data_log["accuracy"], x_axis, label="Accuracy", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(pat_all_data_log["recall0"], x_axis, label="Recall 0", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(pat_all_data_log["recall1"], x_axis, label="Recall 1", fig_num=f,
                  show=show_plots, style=".-", x_scale="log")
        plot_line(pat_all_data_log["precision0"], x_axis, label="Precision 0",
                  fig_num=f, show=show_plots, style=".-", x_scale="log")
        plot_line(pat_all_data_log["precision1"], x_axis, label="Precision 1",
                  fig_num=f, title="Test Patient Accuracy, Recall and Precision", x_scale="log",
                  show=show_plots, x_label="Number of Patients in Training Set", style=".-")
        # Fig 8
        f = 8
        plot_line(pat_all_data_log["num_label1"], x_axis, label="Number 1s",
                  fig_num=f, show=show_plots, style=".-", x_scale="log")
        plot_line(pat_all_data_log["num_labels"], x_axis, label="Number 0s and 1s",
                  fig_num=f, title="Test Patient Set Size", axis=[None, None, 0, None], style=".-",
                  show=show_plots, x_label="Number of Patients in Training Set", x_scale="log")
        # Fig 9
        f = 9
        plot_multiple_roc_curves(rocs1, title="ROC Curves 1  vs.  Dataset Size", fig_num=f,
                                 show=show_plots, labels=title_train)
        # Fig 10
        f = 10
        plot_multiple_roc_curves(rocs2, title="ROC Curves 2  vs.  Dataset Size", fig_num=f,
                                 show=show_plots, labels=title_train)
        # Fig 11
        f = 11
        plot_multiple_roc_curves(rocs3, title="ROC Curves 3  vs.  Dataset Size", fig_num=f,
                                 show=show_plots, labels=title_train)
        # Fig 12
        f = 12
        # Put all true_cv and pred_cv in one vector
        true_c, pred_c = None, None
        for tc, pc in zip(all_data_log["true_cv"], all_data_log["pred_cv"]):
            if true_c is None:
                true_c = np.array(tc)
                pred_c = np.array(pc)
            else:
                true_c = np.concatenate([true_c, tc])
                pred_c = np.concatenate([pred_c, pc])
        # Compute ROC curve and ROC area for each class
        mean_fpr = dict()
        mean_tpr = dict()
        mean_roc_auc = dict()
        for i in range(2):  # Only 2 classes
            mean_fpr[i], mean_tpr[i], _ = roc_curve(true_c[:, i], pred_c[:, i])
            mean_roc_auc[i] = auc(mean_fpr[i], mean_tpr[i])
        # Compute micro-average ROC curve and ROC area
        mean_fpr["micro"], mean_tpr["micro"], _ = roc_curve(true_c.ravel(), pred_c.ravel())
        mean_roc_auc["micro"] = auc(mean_fpr["micro"], mean_tpr["micro"])
        # Plot average ROC curve
        plot_roc_curve(mean_fpr, mean_tpr, mean_roc_auc, title="Model Mean ROC Curve", fig_num=f,
                       show=show_plots)
        # Fig 13
        f = 13
        rocs = []
        for ii in range(len(all_data_log["true_cv"])):
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):  # Only 2 classes
                fpr[i], tpr[i], _ = roc_curve(all_data_log["true_cv"][ii][:, i],
                                              all_data_log["pred_cv"][ii][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(all_data_log["true_cv"][ii].ravel(),
                                                      all_data_log["pred_cv"][ii].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            rocs.append((fpr, tpr, roc_auc))
        plot_multiple_roc_curves(rocs, title="ROC Curves  vs.  Dataset Size", fig_num=f,
                                 show=show_plots, labels=title_train)

        # Save all figures to a PDF called figures.pdf
        save_plt_figures_to_pdf(destination_folder + "/" + "-".join([str(x) for x in comb]) + "_figures.pdf")

    plt.close("all")
    for i, (params1, params2, params3) in enumerate(zip(all_data1, all_data2, all_data3)):
        comb1, all_cv1, all_train1, all_pat1, history1, rocs1 = params1
        comb2, all_cv2, all_train2, all_pat2, history2, rocs2 = params2
        comb3, all_cv3, all_train3, all_pat3, history3, rocs3 = params3
        comb = comb1
        combs.append(comb)

        # Calculate average global results
        figsize = wider_figsize if i == 0 else None

        plot_line((np.array(all_cv1["accuracy"]) + all_cv2["accuracy"] + all_cv3["accuracy"]) / 3, x_axis,
                  label=str(comb), fig_num=0, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Accuracy".format(title), legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_cv1["recall0"]) + all_cv2["recall0"] + all_cv3["recall0"]) / 3, x_axis,
                  label=str(comb), fig_num=1, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Recall (0)".format(title), legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_cv1["recall1"]) + all_cv2["recall1"] + all_cv3["recall1"]) / 3, x_axis,
                  label=str(comb), fig_num=2, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Recall (1)".format(title), legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_cv1["precision0"]) + all_cv2["precision0"] + all_cv3["precision0"]) / 3, x_axis,
                  label=str(comb), fig_num=3, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Precision (0)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_cv1["precision1"]) + all_cv2["precision1"] + all_cv3["precision1"]) / 3, x_axis,
                  label=str(comb), fig_num=4, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Precision (1)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)

        mean_accuracies.append(np.around((np.array(all_cv1["accuracy"]) + all_cv2["accuracy"] + all_cv3["accuracy"]) / 3, decimals=6))
        table_acc.add_row([comb] + list(mean_accuracies[-1]) + [np.mean(mean_accuracies[-1])])

        plot_line((np.array(all_train1["accuracy"]) + all_train2["accuracy"] + all_train3["accuracy"]) / 3, x_axis,
                  label=str(comb), fig_num=5, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="Training Accuracy", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["recall0"]) + all_train2["recall0"] + all_train3["recall0"]) / 3, x_axis,
                  label=str(comb), fig_num=6, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="Training Recall (0)", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["recall1"]) + all_train2["recall1"] + all_train3["recall1"]) / 3, x_axis,
                  label=str(comb), fig_num=7, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="Training Recall (1)", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["precision0"]) + all_train2["precision0"] + all_train3["precision0"]) / 3, x_axis,
                  label=str(comb), fig_num=8, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="Training Precision (0)", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["precision1"]) + all_train2["precision1"] + all_train3["precision1"]) / 3, x_axis,
                  label=str(comb), fig_num=9, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="Training Precision (1)", legend_out=(i == last_idx), figsize=figsize)

        plot_line((np.array(all_pat1["accuracy"]) + all_pat2["accuracy"] + all_pat3["accuracy"]) / 3, x_axis,
                  label=str(comb), fig_num=10, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Patient Accuracy".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["recall0"]) + all_pat2["recall0"] + all_pat3["recall0"]) / 3, x_axis,
                  label=str(comb), fig_num=11, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Patient Recall (0)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["recall1"]) + all_pat2["recall1"] + all_pat3["recall1"]) / 3, x_axis,
                  label=str(comb), fig_num=12, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Patient Recall (1)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["precision0"]) + all_pat2["precision0"] + all_pat3["precision0"]) / 3, x_axis,
                  label=str(comb), fig_num=13, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Patient Precision (0)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["precision1"]) + all_pat2["precision1"] + all_pat3["precision1"]) / 3, x_axis,
                  label=str(comb), fig_num=14, show=show_plots, marker=".", x_scale=x_scale, x_label=x_label,
                  title="{} Patient Precision (1)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)

        mean_pat_accuracies.append(np.around((np.array(all_pat1["accuracy"]) + all_pat2["accuracy"] + all_pat3["accuracy"]) / 3, decimals=6))
        table_pat_acc.add_row([comb] + list(mean_pat_accuracies[-1]) + [np.mean(mean_pat_accuracies[-1])])

        plot_line((np.array(history1[0]) + history2[0] + history3[0]) / 3, x_history,
                  label=str(comb), fig_num=15, show=show_plots, figsize=figsize, x_label="Epoch",
                  title="Mean Training Accuracy History ", legend_out=(i == last_idx))
        plot_line((np.array(history1[1]) + history2[1] + history3[1]) / 3, x_history,
                  label=str(comb), fig_num=16, show=show_plots, figsize=figsize, x_label="Epoch",
                  title="Mean Test Accuracy History", legend_out=(i == last_idx))

        for j in range(len(all_cv1["history_val_acc"])):
            test_histories = (np.array(all_cv1["history_val_acc"][j]) + all_cv2["history_val_acc"][j] + all_cv3["history_val_acc"][j]) / 3
            plot_line(test_histories, x_history, legend_out=(i == last_idx), figsize=figsize,
                      fig_num=17 + j, show=show_plots, x_label="Epoch", label=comb,
                      title="Test Accuracy History - {} patients".format(x_axis[j]))

    # Print accuracies
    print("\nTest Accuracies")
    print(table_acc)

    print("\nPatient Accuracies")
    print(table_pat_acc)

    print("\nTest Information")
    mean_accuracies = np.array(mean_accuracies)
    bigger_figsize = list(plt.rcParams.get('figure.figsize'))
    bigger_figsize[0] += 2
    bigger_figsize[1] += 1.5
    fig_num = 17 + len(all_cv1["history_val_acc"])
    plt.figure(fig_num, figsize=bigger_figsize)
    plt.axhline(0.93, color='k', linestyle=':', label="BFT accuracy")
    for i in range(len(x_axis)):
        plot_line(mean_accuracies[:, i], show=show_plots, fig_num=fig_num,
                  label="{} patients training".format(x_axis[i]), y_label="Accuracy",
                  title="Training set size vs. Model",
                  legend_out=(i == len(x_axis) - 1), xticks_labels=combs)
        print("Training: {} patients".format(x_axis[i]))
        print("  {}. Accuracy: {}".format(0, np.mean(mean_accuracies[:, i])))
        for k, j in enumerate(np.argsort(mean_accuracies[:, i])[::-1]):
            if k >= 5:
                break
            print("  {}. Accuracy: {}".format(k + 1, mean_accuracies[j, i]))
            print("     Model:    {}".format(combs[j]))

    print("\nPatient Information")
    mean_pat_accuracies = np.array(mean_pat_accuracies)
    bigger_figsize = list(plt.rcParams.get('figure.figsize'))
    bigger_figsize[0] += 2
    bigger_figsize[1] += 1.5
    fig_num = fig_num + 1
    plt.figure(fig_num, figsize=bigger_figsize)
    plt.axhline(0.93, color='k', linestyle=':', label="BFT accuracy")
    for i in range(len(x_axis)):
        plot_line(mean_pat_accuracies[:, i], show=show_plots, fig_num=fig_num,
                  label="{} patients training".format(x_axis[i]), y_label="Patient Accuracy",
                  title="Patient Accuracies - Training set size vs. Model",
                  legend_out=(i == len(x_axis) - 1), xticks_labels=combs)
        print("Training: {} patients".format(x_axis[i]))
        print("  {}. Patient Accuracy: {}".format(0, np.mean(mean_pat_accuracies[:, i])))
        for k, j in enumerate(np.argsort(mean_pat_accuracies[:, i])[::-1]):
            if k >= 5:
                break
            print("  {}. Patient Accuracy: {}".format(k + 1, mean_accuracies[j, i]))
            print("     Model:            {}".format(combs[j]))

    # Save PDF results
    save_plt_figures_to_pdf("{}/figures_median.pdf".format(destination_folder))

    # Print mean results
    print("Raw Data:")
    print(list((np.array(all_cv1["accuracy"]) + all_cv2["accuracy"] + all_cv3["accuracy"]) / 3))
    print(list((np.array(all_cv1["recall0"]) + all_cv2["recall0"] + all_cv3["recall0"]) / 3))
    print(list((np.array(all_cv1["recall1"]) + all_cv2["recall1"] + all_cv3["recall1"]) / 3))
    print(list((np.array(all_cv1["precision0"]) + all_cv2["precision0"] + all_cv3["precision0"]) / 3))
    print(list((np.array(all_cv1["precision1"]) + all_cv2["precision1"] + all_cv3["precision1"]) / 3))


if __name__ == "__main__":
    # # filename1 = ("nn_models1_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_"
    # #              "0-8192_mos_trimmed2-filters-units-dropout2.pkl")
    # # filename1 = ("nn_models4/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192"
    # #              "_exactly60_augmented-b1024_mos_trimmed2.pkl")
    # filename1 = ("nn_lumpy_no_aug_zero1/results_create_datasets-data-custom_lumpy_dataset_v3_"
    #              "0-8192_zeroed_mos_trimmed2-filters-units-dropout2.pkl")
    # # filename2 = ("nn_models2_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_"
    # #              "0-8192_mos_trimmed2-filters-units-dropout2.pkl")
    # # filename2 = ("nn_models5/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192"
    # #              "_exactly60_augmented-b1024_mos_trimmed2.pkl")
    # filename2 = ("nn_lumpy_no_aug_zero2/results_create_datasets-data-custom_lumpy_dataset_v3_"
    #              "0-8192_zeroed_mos_trimmed2-filters-units-dropout2.pkl")
    # # filename3 = ("nn_models3_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_"
    # #              "0-8192_mos_trimmed2-filters-units-dropout2.pkl")
    # # filename3 = ("nn_models6/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192"
    # #              "_exactly60_augmented-b1024_mos_trimmed2.pkl")
    # filename3 = ("nn_lumpy_no_aug_zero3/results_create_datasets-data-custom_lumpy_dataset_v3_"
    #              "0-8192_zeroed_mos_trimmed2-filters-units-dropout2.pkl")
    # # folder = "medians_nn_lumpy_no_aug_no_zero123"
    # folder = "medians_nn_lumpy_no_aug_zero123"
    #
    # main(filename1=filename1, filename2=filename2, filename3=filename3,
    #      destination_folder=folder)

    target_base_name = None
    source_folder = "experiments"
    destination_subfolder = "mean_results"
    destination_folder = "{}/{}".format(source_folder, destination_subfolder)
    try:
        os.mkdir(destination_folder)
    except FileExistsError:
        pass
    subfolders = sorted(next(os.walk(source_folder))[1])
    if destination_subfolder in subfolders:
        subfolders.remove(destination_subfolder)
    if "old_experiments" in subfolders:
        subfolders.remove("old_experiments")
    for i, subfolder in enumerate(subfolders):
        full_path = "{}/{}".format(source_folder, subfolder)
        pkl_names = [x for x in os.listdir(full_path) if x.endswith(".pkl")]
        if len(pkl_names) != 1:
            raise Exception("Found more than one or none .pkl files in '{}'".format(subfolder))
        full_path = "{}/{}/{}".format(source_folder, subfolder, pkl_names[0])
        if i % 3 == 0:
            base_name = subfolder[:-1]
            filename1 = full_path
        else:
            if subfolder[:-1] != base_name:
                raise Exception("Error in the folders names (see folder '{}')".format(subfolder))
            if i % 3 == 1:
                filename2 = full_path
            else:
                filename3 = full_path
        if target_base_name is not None and base_name != target_base_name:
            continue
        if i % 3 == 2:
            print("\n--------------------------------------------------------------------------\n")
            main(filename1=filename1, filename2=filename2, filename3=filename3,
                 destination_folder="{}/{}".format(destination_folder, base_name))
            input("Press ENTER to see next...")

            print("\n--------------------------------------------------------------------------\n")
