#!/usr/bin/env python3.5

import sys
from time import clock
from datetime import timedelta, datetime
import argparse
from importlib import import_module


if __name__ == "__main__":
    # Available experiments names
    experiments = ["MyFirstExperiment", "MyFirstExperimentContinued", "SingleExperiment",
                   "MyFirstExperimentShort"]

    parser = argparse.ArgumentParser(description="Create and run experiments with modular_keras, and save results neatly")
    parser.add_argument('-d', '--dataset', choices=["mnist", "cifar10", "cifar100"], default="mnist",
                        help="Name of the dataset to try. Default is 'mnist'.")
    parser.add_argument('-e', '--experiment', choices=range(len(experiments)), type=int, default=0,
                        help="Experiement architecture (0 to {}). Default is 0.".format(len(experiments) - 1))
    parser.add_argument('-f', '--folder', type=str, help="Name of the folder where the results will be saved.", default=None)
    parser.add_argument('-ne', '--number_epochs', type=int, help="Maximum number of epochs before termination. Default is 100.", default=100)
    parser.add_argument('-dr', '--data_reduction', type=int, default=None,
                        help="Number by which to divide the data used. For example, dr=3 means only 1/3 of the data will be used. Default is 1.")

    args = parser.parse_args()

    # Put this here so it runs before importing TensorFlow
    print("---------------------------------------------------------------------------------------")
    now = datetime.now()
    print("|  Running: {:<72}  |".format(" ".join(sys.argv)))
    print("|  Time:    {:<72}  |".format("{} {:02d}:{:02d}:{:02d}".format(now.date(), now.hour,
                                                                          now.minute, now.second)))
    print("---------------------------------------------------------------------------------------")
    print("Arguments used:")
    for arg in args._get_kwargs():
        print("    {} : {}".format(arg[0], arg[1]))
    print()

    # Imports that load the TensorFlow backend (slow, should only happen if we are going to use it)
    modular_NN = import_module("modular_neural_network")
    experiment = getattr(modular_NN, experiments[args.experiment])  # Only import experiment used
    keras_datasets = import_module("keras.datasets")
    data = getattr(keras_datasets, args.dataset).load_data  # Only import dataset used
    from results_plotter import plot_results
    from results_observer import observe_results
    from keras_experiments import experiments_runner

    t = clock()
    folder = experiments_runner(data, experiment, folder=args.folder,
                                data_reduction=args.data_reduction, epochs=args.number_epochs)
    print("\nTotal Time Taken to perform Experiment: {} s\n\n".format(timedelta(seconds=clock() - t)))

    t = clock()
    results = plot_results(folder=folder, height_keys=["accTr", "accTe"], plot_mode=0,
                           static_z_scale=True, secondary_plot=None, save_without_prompt=True)
    print("\nTotal Time Taken to plot Results: {} s\n\n".format(timedelta(seconds=clock() - t)))

    t = clock()
    for subfolder in results:
        full_path = folder + "/" + subfolder
        observe_results(data, folder=full_path, filename=full_path + "/confusion_test.png",
                        observe_training=0, mode=0, data_reduction=args.data_reduction,
                        misclassified_wizard=False)
        observe_results(data, folder=full_path, filename=full_path + "/confusion_train.png",
                        observe_training=1, mode=0, data_reduction=args.data_reduction,
                        misclassified_wizard=False)
        observe_results(data, folder=full_path, filename=full_path + "/confusion_all.png",
                        observe_training=2, mode=0, data_reduction=args.data_reduction,
                        misclassified_wizard=False)
    print("\nTotal Time Taken to Check Errors: {} s".format(timedelta(seconds=clock() - t)))