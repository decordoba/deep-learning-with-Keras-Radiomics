#!/usr/bin/env python3.5
"""Calculate statistics of a 3D dataset."""
import sys
import argparse
import numpy as np
from single_experiment_runner import load_organized_dataset, plot_slices
from single_experiment_runner import limit_number_patients_per_label
from matplotlib import pyplot as plt
from keras.utils import np_utils
sys.path.insert(0, 'create_datasets')
from generate_realizations_of_dataset import save_statistics
from save_datasets import calculate_shared_axis, plot_boxplot, plot_histogram
from save_datasets import save_plt_figures_to_pdf, analyze_data, simple_plot_histogram


def read_dataset(dataset_location, num_patients_per_label=None, slices_plot=False, plot=False):
    """Read and transfrom dataset."""
    train_set, test_set, train_aux, test_aux = load_organized_dataset(dataset_location)
    (x_train, y_train), (x_test, y_test), = train_set, test_set
    (patients_train, mask_train), (patients_test, mask_test) = train_aux, test_aux
    try:
        x_whole = np.append(x_train, x_test, axis=0)
    except ValueError:
        x_whole = x_train + x_test
    try:
        y_whole = np.append(y_train, y_test, axis=0)
    except ValueError:
        y_whole = y_train + y_test
    try:
        mask_whole = np.append(mask_train, mask_test, axis=0)
    except ValueError:
        mask_whole = mask_train + mask_test
    try:
        patients_whole = np.append(patients_train, patients_test, axis=0)
    except ValueError:
        patients_whole = patients_train + patients_test
    analyze_data(x_whole, y_whole, patients_whole, mask_whole, plot_data=plot, dataset_name=None)

    # Remove elements of the dataset if necessary
    if num_patients_per_label is not None:
        params = limit_number_patients_per_label(x_whole, y_whole, mask_whole, patients_whole,
                                                 num_patients_per_label=num_patients_per_label)
        x_whole, y_whole, mask_whole, patients_whole = params
        analyze_data(x_whole, y_whole, patients_whole, mask_whole, plot_data=plot,
                     dataset_name=None)
    plt.close("all")

    patients = np.unique(patients_whole)
    num_patients = len(patients)
    labels = np.unique(y_whole)
    y_whole = np_utils.to_categorical(y_whole, len(labels))

    if slices_plot:
        i = 0
        plt.ion()
        while i < len(x_whole):
            s = "{}/{} - Patient: {} - Label: {}".format(i, len(x_whole), patients_whole[i],
                                                         y_whole[i][1])
            plot_slices(x_whole[i], title=s, fig_num=0)
            print(s)
            r = input("ENTER: next slice, q: quit plot, n: next patient.\n>> ")
            if len(r) > 0 and r[0].lower() == "q":
                break
            elif len(r) > 0 and r[0].lower() == "n":
                p = patients_whole[i]
                while i < len(patients_whole) and patients_whole[i] == p:
                    i += 1
            else:
                i += 1
        plt.close("all")
        plt.ioff()

    # Print some information of data
    try:
        print("Whole set shape:     {}".format(x_whole.shape))
    except AttributeError:
        print("Whole set size:     {}".format(len(x_whole)))
    print("Existing labels:     {}".format(labels))
    print("Number of patients:  {}".format(num_patients))
    try:
        print("Number of slices:    {}".format(x_whole.shape[0]))
    except AttributeError:
        pass

    return x_whole, y_whole, mask_whole, patients_whole


def parse_arguments():
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Calculate several statistics from dataset.")
    parser.add_argument('-p', '--plot', default=False, action="store_true",
                        help="show figures before saving them")
    parser.add_argument('-ps', '--plot_slices', default=False, action="store_true",
                        help="show slices of volume in dataset")
    parser.add_argument('-s', '--size', default=None, type=int,
                        help="max number of patients per label (default: all)")
    parser.add_argument('-d0', '--dataset0', default="organized", type=str,
                        help="location of the dataset for label 0; if dataset1 is None, it will "
                        "compare labels 0 and 1 in dataset0 (default: organized)")
    parser.add_argument('-d1', '--dataset1', default=None, type=str,
                        help="location of the dataset for label 1; if dataset1 is None, it will "
                        "compare labels 0 and 1 in dataset0 (default: None)")
    return parser.parse_args()


def plot_metric(data0, data1, label0="Metrics 0", label1="Metrics 1", label_all="Metrics Total",
                figure=0, plot_data=True, window_histogram="Histogram",
                window_boxplot="Boxplot", simple_histograms=False):
    """Plot histogram and boxplot for label0 and label1."""
    print("Generating figures for: {} ...".format(label_all))
    num_bins = 20
    if plot_data:
        plt.ion()
    xlim = calculate_shared_axis(data0, data1)
    if not simple_histograms:
        plot_histogram(data0, label0, figure, 311, num_bins, xlim, show=plot_data)
        plot_histogram(data1, label1, figure, 312, num_bins, xlim, show=plot_data)
        plot_histogram(data0 + data1, label_all, figure, 313, num_bins, xlim,
                       window_title=window_histogram, show=plot_data)
    else:
        simple_plot_histogram(data0, label0, figure, 311, num_bins, xlim, show=plot_data)
        simple_plot_histogram(data1, label1, figure, 312, num_bins, xlim, show=plot_data)
        simple_plot_histogram(data0 + data1, label_all, figure, 313, num_bins, xlim,
                              window_title=window_histogram, show=plot_data)

    ylim = calculate_shared_axis(data0, data1)
    plot_boxplot(data0, label0, figure + 1, 121, ylim, show=plot_data)
    plot_boxplot(data1, label1, figure + 1, 122, ylim, True, show=plot_data,
                 window_title=window_boxplot)
    if plot_data:
        plt.ioff()


def main(dataset0, dataset1=None, size=None, plot_slices=False, plot=False):
    """Load dataset and print statistics."""
    # Load dataset
    x_whole0, y_whole0, mask_whole0, patients_whole0 = read_dataset(dataset0, size, plot_slices,
                                                                    plot)

    # Calculate statistics
    metrics = [[], []]
    gray_values = [[], []]
    masked_gray_values = [[], []]
    for i, (x, y, m, p) in enumerate(zip(x_whole0, y_whole0, mask_whole0, patients_whole0)):
        label = int(y[1])
        metrics[label].append(save_statistics(x, m))
        gray_values[label].extend(list(x.flatten()))
        mask_positions = np.nonzero(m)
        masked_gray_values[label].extend(list(x[mask_positions]))
    if dataset1 is not None:
        x_whole1, y_whole1, mask_whole1, patients_whole1 = read_dataset(dataset1, size,
                                                                        plot_slices, plot)
        print("Warning! Because dataset1 is not None, all patients in {} are considered to have"
              "label 0 and all patients in {} are considered to have label 1"
              "".format(dataset0, dataset1))
        metrics = [metrics[0] + metrics[1], []]
        gray_values = [gray_values[0] + gray_values[1], []]
        masked_gray_values = [masked_gray_values[0] + masked_gray_values[1], []]
        for i, (x, y, m, p) in enumerate(zip(x_whole1, y_whole1, mask_whole1, patients_whole1)):
            label = 1
            metrics[label].append(save_statistics(x, m))
            gray_values[label].extend(list(x.flatten()))
            mask_positions = np.nonzero(m)
            masked_gray_values[label].extend(list(x[mask_positions]))
    metrics = [np.array(metrics[0]), np.array(metrics[1])]
    medians0 = np.median(metrics[0])
    medians1 = np.median(metrics[1])
    medians_diff = medians0 - medians1
    if dataset1 is None:
        print("Differences between Label 0 and label 1 ({}):".format(dataset0))
    else:
        print("Differences between Label 0 ({}) and label 1 ({}):".format(dataset0, dataset1))
    print(medians_diff)
    return medians_diff


if __name__ == "__main__":
    args = parse_arguments()
    main(args.dataset0, args.dataset1, args.size, args.plot_slices, args.plot)
