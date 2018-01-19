#!/usr/bin/env python3.5

import sys
from time import clock
from datetime import timedelta, datetime
import argparse
from importlib import import_module
from dataset_loader import load_dataset


def get_experiment_names(library):
    # Get all classes that can be experiments inside a module
    library += ".py"
    with open(library, "r") as f:
        lines = f.readlines()
    # Classes that inherit from Experiment (or from any class ending in *Experiment)
    classes = [line for line in lines if line.lstrip().startswith("class") and
               line.rstrip().endswith("Experiment):")]
    class_names = [cl.split(" ", 2)[1].split("(", 1)[0] for cl in classes]
    return class_names


if __name__ == "__main__":
    # Available experiments names
    experiments = get_experiment_names("modular_neural_network")
    datasets = ["mnist", "cifar10", "cifar100", "radiomics1", "radiomics2", "radiomics3"]

    # Parser to allow fancy command arguments input
    parser = argparse.ArgumentParser(description="Create and run experiments with modular_keras, "
                                                 "and save results neatly")
    parser.add_argument('-d', '--dataset', choices=datasets, default=datasets[0], metavar="DATASET",
                        help="Name of the dataset to try. Options are: {}. Default is '{}'."
                             "".format(datasets, datasets[0]))
    parser.add_argument('-e', '--experiment', choices=experiments, default=experiments[0],
                        help="Experiement architecture. Options are: {}. "
                             "Default is '{}'.".format(experiments, experiments[0]),
                        metavar="EXPERIMENT")
    parser.add_argument('-f', '--folder', type=str, default=None,
                        help="Name of the folder where the results are saved. If not set, the "
                             "folder is named with the current date & time.")
    parser.add_argument('-ne', '--number_epochs', type=int, default=100,
                        help="Maximum number of epochs before termination. Default is 100.")
    parser.add_argument('-es', '--early_stopping', metavar="NUMBER_EPOCHS", type=int, default=10,
                        help="After how many epochs without any improvement should the training be"
                             "interrupted. Set it to 0 to disable early_stopping. Default is 10.")
    parser.add_argument('-dr', '--data_reduction', type=int, default=None,
                        help="Number by which to divide the data used. For example, dr=3 means "
                             "only 1/3 of the data is used. Default is 1.")
    parser.add_argument('--log', default=False, action="store_true",
                        help="Enable this option to redirect stdout to a log file.")

    # Parses inputs, if argument is -h or --help, prints help and exits
    args = parser.parse_args()

    original_stdout = sys.stdout
    log_file = None
    if args.log:
        log_file = open("log.log", "a")
        sys.stdout = log_file

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
    print(" ")

    # Imports that load the TensorFlow backend (slow, should only happen if we are going to use it)
    modular_NN = import_module("modular_neural_network")
    experiment = getattr(modular_NN, args.experiment)  # Only import experiment used
    data = load_dataset(args.dataset)
    from results_plotter import plot_results
    from results_observer import observe_results
    from keras_experiments import experiments_runner

    # Run all experiments (according to the chosen experiment, performed over the chosen dataset)
    # and save results into folder with chosen folder name. #epochs and dr can also be set
    t = clock()  # Start measure of time taken
    folder = experiments_runner(data, experiment, folder=args.folder,
                                data_reduction=args.data_reduction, epochs=args.number_epochs,
                                early_stopping=args.early_stopping)
    print("\nTime Taken to perform Experiment: {} s\n\n".format(timedelta(seconds=clock() - t)))

    # Parse created folder and save all existing combinations of figures for accTr and accTe.
    # Set save_without_prompt to False to see the dialog and see and modify the figures before
    # saving them. Take a look at results_plotter for more settings
    t = clock()  # Start measure of time taken
    results = plot_results(folder=folder, height_keys=["accTr", "accTe"], plot_mode=0,
                           static_z_scale=True, secondary_plot=None, save_without_prompt=True)
    print("\nTime Taken to plot Results: {} s\n\n".format(timedelta(seconds=clock() - t)))

    # Parse every folder and create the confusion matrix for the training, test and both. Set the
    # misclassified_wizard boolean to True to see the figures before saving them and to also see
    # the wizard to watch the misclassified examples. Read results_observer for more settings
    t = clock()  # Start measure of time taken
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
    print("\nTime Taken to Check Errors: {} s".format(timedelta(seconds=clock() - t)))

    if args.log:
        sys.stdout = original_stdout
        log_file.close()