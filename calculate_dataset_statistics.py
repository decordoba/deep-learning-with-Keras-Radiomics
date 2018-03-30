#!/usr/bin/env python3.5
"""Calculate statistics of a 3D dataset."""
import os
import sys
import argparse
import numpy as np
from single_experiment_runner import load_organized_dataset, plot_slices
from single_experiment_runner import limit_number_patients_per_label
from matplotlib import pyplot as plt
from keras.utils import np_utils
from scipy.ndimage.morphology import binary_erosion
from skimage import feature
sys.path.insert(0, 'create_datasets')
from save_datasets import calculate_shared_axis, plot_boxplot, plot_histogram
from save_datasets import save_plt_figures_to_pdf, analyze_data, simple_plot_histogram


def get_statistics_mask(mask):
    """Get size box and volume of mask where we can fit all 1s in contour."""
    ones_pos = np.nonzero(mask)
    eroded = binary_erosion(mask)
    outer_mask = mask - eroded
    volume = len(ones_pos[0])
    surface = outer_mask.sum()
    return surface, volume, ones_pos


def get_glcm_statistics(volume):
    """Get statistics realted to GLCM."""
    # very technically, GLCMs are only defined in 2d and there is
    # considerable disagreement as to how to translate them into 3d.
    # the common practice for small, similar objects like yours
    # therefore is to select typical images from the volume. this
    # can be a few slices toward the middle and average or even just
    # use the median slice. Here I am using the median slice
    image_array = volume[:, :, int(volume.shape[2] / 2)]
    # skimage will compute the GLCM for multiple pixel offsets
    # at once; we only need nearest neighbors so the offset is 1
    offsets = np.array([1]).astype(np.int)
    # the values of GLCM statistics from 0, 45, 90, 135 usually
    # are averaged together, especially for textures we expect
    # to be reasonably random
    radians = np.pi * np.arange(4) / 4
    # skimage is kind of stupid about counting so you must make sure
    # that number of levels matches what your data *can* be, not what
    # they are. the problem is that the matrices are too sparse using
    # all levels on small, low contrast images. therefore we downsample
    # the shades to something reasonable. FYI: this is properly done
    # via histogram matching but no MD ever does this correctly. instead
    # they do (again, this is *INCORRECT* (but quite common in the field)):
    LEVELS = 16
    lo, hi = image_array.min(), image_array.max()
    image_array = np.round((image_array - lo) / (hi - lo) * (LEVELS - 1)).astype(np.uint8)
    # Calculate co-matrix
    glcms = feature.greycomatrix(image_array, offsets, radians, LEVELS, symmetric=True,
                                 normed=False)
    # compute the desired GLCM statistic
    dissimil = feature.greycoprops(glcms, prop='dissimilarity')
    # now that you have a GLCM for each offset and each direction, average over direction
    # 0 because there is only one offset
    dissimil = [dissimil[0, angle] for angle in range(radians.size)]
    dissimil = np.mean(dissimil)
    correlation = feature.greycoprops(glcms, prop='correlation')
    correlation = [correlation[0, angle] for angle in range(radians.size)]
    correlation = np.mean(correlation)
    correlation = feature.greycoprops(glcms, prop='correlation')
    correlation = [correlation[0, angle] for angle in range(radians.size)]
    correlation = np.mean(correlation)
    asm = feature.greycoprops(glcms, prop='ASM')
    asm = [asm[0, angle] for angle in range(radians.size)]
    asm = np.mean(asm)
    return dissimil, correlation, asm


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
    parser.add_argument('-d', '--dataset', default="organized", type=str,
                        help="location of the dataset inside the ./data folder "
                        "(default: organized)")
    parser.add_argument('-v', '--verbose', default=False, action="store_true", help="enable "
                        "verbose mode when training")
    parser.add_argument('-dr', '--dry_run', default=False, action="store_true", help="do not "
                        "save pdf with results")
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


def main():
    """Load dataset and print statistics."""
    # Parse arguments
    args = parse_arguments()

    # Load dataset
    dataset_location = args.dataset
    if not os.path.exists(dataset_location) and not dataset_location.startswith("data/"):
        dataset_location = "data/{}".format(dataset_location)
    x_whole, y_whole, mask_whole, patients_whole = read_dataset(dataset_location, args.size,
                                                                args.plot_slices, args.plot)

    # Calculate statistics
    metrics = [{
        "std": [], "mean": [], "median": [], "surface_to_volume": [],
        "glcm_dissimilarity": [], "glcm_correlation": [], "glcm_asm": []
    }, {
        "std": [], "mean": [], "median": [], "surface_to_volume": [],
        "glcm_dissimilarity": [], "glcm_correlation": [], "glcm_asm": []
    }]
    patients = set()
    gray_values = [[], []]
    masked_gray_values = [[], []]
    for i, (x, y, m, p) in enumerate(zip(x_whole, y_whole, mask_whole, patients_whole)):
        if p in patients:
            input("Repeated patient '{}'. This should never happen.".format(p))
        patients.add(p)
        label = int(y[1])
        std_dev = np.std(x)
        mean = np.mean(x)
        median = np.median(x)
        surface, volume, mask_positions = get_statistics_mask(m)
        surf_to_vol = surface / volume
        dissimilarity, correlation, asm = get_glcm_statistics(x)
        gray_values[label].extend(list(x.flatten()))
        masked_gray_values[label].extend(list(x[mask_positions]))
        if args.verbose:
            print("Label:              {}".format(label))
            print("Mean:               {}".format(mean))
            print("Median:             {}".format(median))
            print("Std:                {}".format(std_dev))
            print("Surface to Volume:  {} (S: {}, V: {})".format(surf_to_vol, surface, volume))
            print("GLCM dissimilarity: {}".format(dissimilarity))
            print("GLCM correlation:   {}".format(correlation))
            print("GLCM asm:           {}".format(asm))
            print(" ")
        metrics[label]["std"].append(std_dev)
        metrics[label]["mean"].append(mean)
        metrics[label]["median"].append(median)
        metrics[label]["surface_to_volume"].append(surf_to_vol)
        metrics[label]["glcm_dissimilarity"].append(dissimilarity)
        metrics[label]["glcm_correlation"].append(correlation)
        metrics[label]["glcm_asm"].append(asm)

    f = 0
    plot_metric(metrics[0]["std"], metrics[1]["std"], label0="Std Dev Label 0",
                label1="Std Dev Label 1", label_all="Std Dev Labels 0 and 1",
                figure=f, plot_data=args.plot, window_histogram="Histogram Std Dev",
                window_boxplot="Boxplot Std Dev")
    f = 2
    plot_metric(metrics[0]["mean"], metrics[1]["mean"], label0="Mean Label 0",
                label1="Mean Label 1", label_all="Mean Labels 0 and 1",
                figure=f, plot_data=args.plot, window_histogram="Histogram Mean",
                window_boxplot="Boxplot Mean")
    f = 4
    plot_metric(metrics[0]["median"], metrics[1]["median"], label0="Median Label 0",
                label1="Median Label 1", label_all="Median Labels 0 and 1",
                figure=f, plot_data=args.plot, window_histogram="Histogram Median",
                window_boxplot="Boxplot Median")
    f = 6
    plot_metric(metrics[0]["surface_to_volume"], metrics[1]["surface_to_volume"],
                label0="Surface to Volume Ratio Label 0",
                label1="Surface to Volume Ratio Label 1",
                label_all="Surface to Volume Ratio Labels 0 and 1",
                figure=f, plot_data=args.plot,
                window_histogram="Histogram Surface to Volume Ratio",
                window_boxplot="Boxplot Surface to Volume Ratio")
    f = 8
    plot_metric(metrics[0]["glcm_dissimilarity"], metrics[1]["glcm_dissimilarity"],
                label0="GLCM Dissimilarity Label 0",
                label1="GLCM Dissimilarity Label 1",
                label_all="GLCM Dissimilarity Labels 0 and 1",
                figure=f, plot_data=args.plot,
                window_histogram="Histogram GLCM Dissimilarity",
                window_boxplot="Boxplot GLCM Dissimilarity")
    f = 10
    plot_metric(metrics[0]["glcm_correlation"], metrics[1]["glcm_correlation"],
                label0="GLCM Correlation Label 0",
                label1="GLCM Correlation Label 1",
                label_all="GLCM Correlation Labels 0 and 1",
                figure=f, plot_data=args.plot,
                window_histogram="Histogram GLCM Correlation",
                window_boxplot="Boxplot GLCM Correlation")
    f = 12
    plot_metric(metrics[0]["glcm_asm"], metrics[1]["glcm_asm"],
                label0="GLCM ASM Label 0",
                label1="GLCM ASM Label 1",
                label_all="GLCM ASM Labels 0 and 1",
                figure=f, plot_data=args.plot,
                window_histogram="Histogram GLCM ASM",
                window_boxplot="Boxplot GLCM ASM")
    if not args.dry_run:
        print("Saving figures ...")
        save_plt_figures_to_pdf("{}/statistics.pdf".format(dataset_location), verbose=True)
    if args.plot:
        input("Press ENTER to close all figures and continue.")
    plt.close("all")

    # Create figures of intensities that will be saved and/or plotted
    f = 14
    plot_metric(masked_gray_values[0], masked_gray_values[1],
                label0="Tumor Intensities Label 0",
                label1="Tumor Intensities Label 1",
                label_all="Tumor Intensities Labels 0 and 1",
                figure=f, plot_data=args.plot,
                window_histogram="Histogram Intensities",
                window_boxplot="Boxplot Intensities",
                simple_histograms=True)
    f = 16
    plot_metric(gray_values[0], gray_values[1],
                label0="Whole Box Intensities Label 0",
                label1="Whole Box Intensities Label 1",
                label_all="Whole Box Intensities Labels 0 and 1",
                figure=f, plot_data=args.plot,
                window_histogram="Histogram Intensities",
                window_boxplot="Boxplot Intensities",
                simple_histograms=True)
    if not args.dry_run:
        print("Saving figures ...")
        save_plt_figures_to_pdf("{}/intensities.pdf".format(dataset_location), verbose=True)
    if args.plot:
        input("Press ENTER to close all figures and continue.")
        plt.close("all")


if __name__ == "__main__":
    main()
