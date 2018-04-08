#!/usr/bin/env python3.5
import matplotlib_handle_display  # Must be imported before anything matplotlib related
import numpy as np
import pickle
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


def main(correction):
    """Load data from 3 files previoulsy saved with single_experiment_runner and save mean.

    This is an even dirtier copy of what single_experiment_runner does at the end.
    """
    # Load all saved files
    filename1 = ("nn_models1/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192_"
                 "mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename1, 'rb') as f:
        all_data1 = pickle.load(f)
    filename2 = ("nn_models2/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192_"
                 "mos_trimmed2-filters-units-dropout2.pkl")
    with open(filename2, 'rb') as f:
        all_data2 = pickle.load(f)
    filename3 = ("nn_models3/results_create_datasets-data-custom_lumpy_dataset_v3_0-8192_"
                 "mos_trimmed2-filters-units-dropout2.pkl")
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
    for i, (params1, params2, params3) in enumerate(zip(all_data1, all_data2, all_data3)):
        comb1, all_cv1, all_train1, all_pat1, history1, rocs1 = params1
        comb2, all_cv2, all_train2, all_pat2, history2, rocs2 = params2
        comb3, all_cv3, all_train3, all_pat3, history3, rocs3 = params3
        comb = comb1

        figsize = wider_figsize if i == 0 else None

        plot_line((np.array(all_cv1["accuracy"]) + all_cv2["accuracy"] + all_cv3["accuracy"]) / 3,
                  label=str(comb), fig_num=0, show=show_plots, marker=".",
                  title="{} Accuracy".format(title), legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_cv1["recall0"]) + all_cv2["recall0"] + all_cv3["recall0"]) / 3,
                  label=str(comb), fig_num=1, show=show_plots, marker=".",
                  title="{} Recall (0)".format(title), legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_cv1["recall1"]) + all_cv2["recall1"] + all_cv3["recall1"]) / 3,
                  label=str(comb), fig_num=2, show=show_plots, marker=".",
                  title="{} Recall (1)".format(title), legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_cv1["precision0"]) + all_cv2["precision0"] + all_cv3["precision0"]) / 3,
                  label=str(comb), fig_num=3, show=show_plots, marker=".",
                  title="{} Precision (0)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_cv1["precision1"]) + all_cv2["precision1"] + all_cv3["precision1"]) / 3,
                  label=str(comb), fig_num=4, show=show_plots, marker=".",
                  title="{} Precision (1)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)

        plot_line((np.array(all_train1["accuracy"]) + all_train2["accuracy"] + all_train3["accuracy"]) / 3,
                  label=str(comb), fig_num=5, show=show_plots, marker=".",
                  title="Training Accuracy", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["recall0"]) + all_train2["recall0"] + all_train3["recall0"]) / 3,
                  label=str(comb), fig_num=6, show=show_plots, marker=".",
                  title="Training Recall (0)", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["recall1"]) + all_train2["recall1"] + all_train3["recall1"]) / 3,
                  label=str(comb), fig_num=7, show=show_plots, marker=".",
                  title="Training Recall (1)", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["precision0"]) + all_train2["precision0"] + all_train3["precision0"]) / 3,
                  label=str(comb), fig_num=8, show=show_plots, marker=".",
                  title="Training Precision (0)", legend_out=(i == last_idx), figsize=figsize)
        plot_line((np.array(all_train1["precision1"]) + all_train2["precision1"] + all_train3["precision1"]) / 3,
                  label=str(comb), fig_num=9, show=show_plots, marker=".",
                  title="Training Precision (1)", legend_out=(i == last_idx), figsize=figsize)

        plot_line((np.array(all_pat1["accuracy"]) + all_pat2["accuracy"] + all_pat3["accuracy"]) / 3,
                  label=str(comb), fig_num=10, show=show_plots, marker=".",
                  title="{} Patient Accuracy".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["recall0"]) + all_pat2["recall0"] + all_pat3["recall0"]) / 3,
                  label=str(comb), fig_num=11, show=show_plots, marker=".",
                  title="{} Patient Recall (0)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["recall1"]) + all_pat2["recall1"] + all_pat3["recall1"]) / 3,
                  label=str(comb), fig_num=12, show=show_plots, marker=".",
                  title="{} Patient Recall (1)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["precision0"]) + all_pat2["precision0"] + all_pat3["precision0"]) / 3,
                  label=str(comb), fig_num=13, show=show_plots, marker=".",
                  title="{} Patient Precision (0)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)
        plot_line((np.array(all_pat1["precision1"]) + all_pat2["precision1"] + all_pat3["precision1"]) / 3,
                  label=str(comb), fig_num=14, show=show_plots, marker=".",
                  title="{} Patient Precision (1)".format(title), legend_out=(i == last_idx),
                  figsize=figsize)

        plot_line((np.array(history1[0]) + history2[0] + history3[0]) / 3,
                  label=str(comb), fig_num=15, show=show_plots, figsize=figsize,
                  title="Mean Training Accuracy History ", legend_out=(i == last_idx))
        plot_line((np.array(history1[1]) + history2[1] + history3[1]) / 3,
                  label=str(comb), fig_num=16, show=show_plots, figsize=figsize,
                  title="Mean Test Accuracy History", legend_out=(i == last_idx))
    # Save PDF results
    save_plt_figures_to_pdf("figures_median.pdf")

    # Print mean results
    print((np.array(all_cv1["accuracy"]) + all_cv2["accuracy"] + all_cv3["accuracy"]) / 3)
    print((np.array(all_cv1["recall0"]) + all_cv2["recall0"] + all_cv3["recall0"]) / 3)
    print((np.array(all_cv1["recall1"]) + all_cv2["recall1"] + all_cv3["recall1"]) / 3)
    print((np.array(all_cv1["precision0"]) + all_cv2["precision0"] + all_cv3["precision0"]) / 3)
    print((np.array(all_cv1["precision1"]) + all_cv2["precision1"] + all_cv3["precision1"]) / 3)


if __name__ == "__main__":
    main()
