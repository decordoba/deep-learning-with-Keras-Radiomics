#!/usr/bin/env python3.5

import sys
import os
import yaml


def filter_by_params(parameters, result, params_keys, params_values, params_dict,
                     pause_in_every_result=True):
    # Ask the users what variables to filter
    print("Select the search values for every parameter.")
    print("Leave blank (press ENTER) to ignore a parameter.")
    for i, key in enumerate(params_keys):
        if len(parameters[key]) <= 20 and len(parameters[key]) > 1:
            possible_values = [str(x) for x in parameters[key]]
            val = None
            while val != "" and val not in possible_values:
                print("{}. Value for {}. Possible values: {}".format(i + 1, key,
                                                                     sorted(list(parameters[key]))))
                val = input(">> ").strip()
                if val != "" and val not in possible_values:
                    print("Impossible value {}. Leave blank to ignore {}.".format(val, key))
        else:
            val = ""
        params_values.append(val.strip())

    # Print results
    for sample_key in result:
        sample = result[sample_key]
        sample_filtered = False
        for key in sample["params"]:
            val = params_values[params_dict[key]]
            if val == "":
                continue
            if val != str(sample["params"][key]):
                sample_filtered = True
                break
        if not sample_filtered:
            print("Sample {}:".format(sample_key))
            print("---------------------------------------------------")
            for key in sorted(sample["params"]):
                print("  {:>20}: {}".format(key, sample["params"][key]))
            print("---------------------------------------------------")
            for key in sorted(sample["result"]):
                print("  {:>20}: {}".format(key, sample["result"][key]))
            print("---------------------------------------------------")
            if pause_in_every_result:
                input("Press ENTER to see next result")
            print(" ")


def filter_by_result(parameters, result, result_keys, pause_in_every_result=True):
    # Ask the users what variables to filter
    print("Select the result variable that will be used to filter.")
    print("0. Exit")
    for i, key in enumerate(result_keys):
        print("{}. {}", format(i + 1, key))
    num = -1
    while num < 0 or num > len(result_keys):
        try:
            num = int(input(">> "))
        except ValueError:
            num = -1
    if num == 0:
        return
    param = result_keys[num - 1]
    print("Select what experiments you want to find.")
    print("0. Exit\n1. With MAX {}\n2. With MIN {}".format(param, param))
    num = -1
    while num < 0 or num > 2:
        try:
            num = int(input(">> "))
        except ValueError:
            num = -1
    if num == 0:
        return
    sign = (num == 1)
    print("Select how many experiments you want to see.")
    num = -1
    while num < 0 or num > 2:
        try:
            num = int(input(">> "))
        except ValueError:
            num = -1
    if num == 0:
        return

    # Get sorted results
    sample_keys = []
    sample_values = []
    for sample_key in result:
        sample = result[sample_key]
        sample_keys.append(sample_key)
        sample_values.append(sample["result"][param])
    sorted_keys = [x for _, x in sorted(zip(sample_values, sample_keys), reversed=sign)]

    # Print results
    for sample_key in sorted_keys[:num]:
        sample = result[sample_key]
        print("Sample {}:".format(sample_key))
        print("---------------------------------------------------")
        for key in sorted(sample["params"]):
            print("  {:>20}: {}".format(key, sample["params"][key]))
        print("---------------------------------------------------")
        for key in sorted(sample["result"]):
            print("  {:>20}: {}".format(key, sample["result"][key]))
        print("---------------------------------------------------")
        if pause_in_every_result:
            input("Press ENTER to see next result")
        print(" ")


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
    if folder is not None:
        os.chdir("./..")

    # Extract params from result, which will save all possible values for every key in params
    parameters = {}
    for sample_key in result:
        sample = result[sample_key]
        for key in sample["params"]:
            if key not in parameters:
                parameters[key] = set()
            parameters[key].add(sample["params"][key])

    # save all params and results required
    params_keys = sorted(parameters.keys())
    params_dict = dict(zip(params_keys, range(len(params_keys))))
    params_values = []
    result_keys = sorted(result[sample_key]["result"].keys())

    while True:
        print("Select what to do:")
        print("0. Exit\n1. Filter by params\n2. Filter by result")
        num = -1
        while num < 0 or num > 2:
            try:
                num = int(input(">> "))
            except ValueError:
                num = -1
        print(" ")
        if num == 0:
            break
        elif num == 1:
            filter_by_params(parameters, result, params_keys, params_values, params_dict,
                             pause_in_every_result=pause_in_every_result)
        else:
            filter_by_result(parameters, result, params_keys, params_values, params_dict,
                             pause_in_every_result=pause_in_every_result)


if __name__ == "__main__":
    folder = None
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    search_results(folder)

    """
    Expects:
        py search_results.py
        py search_results.py folder
    """