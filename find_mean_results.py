#!/usr/bin/env python3.5
import matplotlib_handle_display  # Must be imported before anything matplotlib related
import numpy as np
import pickle
from prettytable import PrettyTable
from cycler import cycler
from matplotlib import pyplot as plt
from single_experiment_runner import plot_line, save_plt_figures_to_pdf


# Cycle colors from normal line to dotted line (allows to tell 20 plots apart)
plt.rc('axes', prop_cycle=(cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                                            '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e',
                                            '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']) +
                           cycler('linestyle', ['-'] * 10 + ['--'] * 10)))


def main():
    """Load data from 3 files previoulsy saved with single_experiment_runner and save mean.

    This is an even dirtier copy of what single_experiment_runner does at the end.
    """
    # Load all saved files
    filename1 = ("nn_models1_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192"
                 "_mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename1, 'rb') as f:
        all_data1 = pickle.load(f)
    filename2 = ("nn_models2_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192"
                 "_mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename2, 'rb') as f:
        all_data2 = pickle.load(f)
    filename3 = ("nn_models3_corrected/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192"
                 "_mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename3, 'rb') as f:
        all_data3 = pickle.load(f)

    # Plot summary of results
    print("\nGenerating global figures...")
    plt.close("all")
    title = "Test"
    show_plots = False
    last_idx = len(all_data1) - 1
    wider_figsize = list(plt.rcParams.get('figure.figsize'))
    wider_figsize[0] += 2.1
    x_axis = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    x_scale = "log"
    x_label = "Number of Patients in Training Set"
    x_history = range(1, 50 + 1)
    table_acc = PrettyTable(["Model \\ Training Samples"] + x_axis)
    mean_accuracies = []
    combs = []
    for i, (params1, params2, params3) in enumerate(zip(all_data1, all_data2, all_data3)):
        comb1, all_cv1, all_train1, all_pat1, history1, rocs1 = params1
        comb2, all_cv2, all_train2, all_pat2, history2, rocs2 = params2
        comb3, all_cv3, all_train3, all_pat3, history3, rocs3 = params3
        comb = comb1
        combs.append(comb)

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
        table_acc.add_row([comb] + list(mean_accuracies[-1]))

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
    print(table_acc)

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

    # Save PDF results
    save_plt_figures_to_pdf("figures_median.pdf")

    # Print mean results
    print("Raw Data:")
    print(list((np.array(all_cv1["accuracy"]) + all_cv2["accuracy"] + all_cv3["accuracy"]) / 3))
    print(list((np.array(all_cv1["recall0"]) + all_cv2["recall0"] + all_cv3["recall0"]) / 3))
    print(list((np.array(all_cv1["recall1"]) + all_cv2["recall1"] + all_cv3["recall1"]) / 3))
    print(list((np.array(all_cv1["precision0"]) + all_cv2["precision0"] + all_cv3["precision0"]) / 3))
    print(list((np.array(all_cv1["precision1"]) + all_cv2["precision1"] + all_cv3["precision1"]) / 3))


if __name__ == "__main__":
    main()
