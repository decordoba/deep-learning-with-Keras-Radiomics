#!/usr/bin/env python3.5

import sys
import os
import yaml
import numpy as np
from keras_plot import plot_3D_bar_graph, plot_colormap, plot_graph_grid  # 'Library' by Daniel


def search_results(folder=None, pause_in_every_result=True):
    """
    From the selected folder (or current folder if folder == None), filter results according to
    result.yaml
    """

    # Navigate to folder and load result.yaml
    if folder is not None:
        os.chdir(folder)
    with open("results.yaml") as f:
        try:
            result = yaml.load(f)
        except yaml.YAMLError as YamlError:
            print("There was an error parsing 'results.yaml'. Plotting aborted.")
            print(YamlError)
            if folder is not None:
                os.chdir("./..")
            return

    # Extract params from result, which will save all possible values for every key in params
    parameters = {}
    for sample_key in result:
        sample = result[sample_key]
        for key in sample["params"]:
            if not key in parameters:
                parameters[key] = set()
            parameters[key].add(sample["params"][key])

    # params_keys will hold all the plotable keys
    params_keys = sorted(parameters.keys())
    params_dict = dict(zip(params_keys, range(len(params_keys))))
    params_values = []

    # Ask the users what variables to filter
    print("Select the search values for every parameter. Leave blank to ignore parameter.")
    for key in params_keys:
        val = input("Value for {}. Possible values: {}\n>> ".format(key, parameters[key]))
        params_values.append(val)

    # Print results
    for sample_key in result:
        sample = result[sample_key]
        sample_filtered = False
        for key in sample["params"]:
            val = params_values[params_dict[key]]
            if val == "":
                continue
            if val != sample["params"][key]:
                sample_filtered = True
                break
        if not sample_filtered:
            print("Sample {}:".format(sample_key))
            print("Params:")
            for key in sample["params"]:
                print("{}: {}".format(key, sample["params"][key]))
            print("Result:")
            for key in sample["result"]:
                print("{}: {}".format(key, sample["result"][key]))
            if pause_in_every_result:
                input("Press ENTER to see next result")
            print(" ")
    if folder is not None:
        os.chdir("./..")


if __name__ == "__main__":
    folder = None
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    """
    Expects:
        py search_results.py
        py search_results.py folder
    """