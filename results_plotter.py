#!/usr/bin/env python3.5

import sys
import os
import yaml
from keras_plot import plot_3D_bar_graph, plot_colormap  # 'Library' by Daniel


def plot_results(folder=None, height_keys=["accTr", "accTe"], plot_mode=0, ordered_results=True):
    """
    From the selected folder (or current folder if folder == None), plot result found in
    result.yaml
    Different plot modes will plot the same data differently. Right now, only 3 modes are
    implemented: 0 (3D bars graph), 1 (2D colormap) or 2 (3D bars graph but seen from top)
    """
    # Navigate to folder and load result.yaml
    if folder is not None:
        os.chdir(folder)
    with open("results.yaml") as f:
        try:
            result = yaml.load(f)
        except yaml.YAMLError as YamlError:
            print("There was an error parsing 'results.yaml'. Plotting aborted")
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

    # Remove keys which always hold the same value, or which (almost) vary every sample, we don't want to plot them
    for key in list(parameters.keys()):
        if len(parameters[key]) < 2 or len(parameters[key]) > len(result) * 0.95:
            del parameters[key]

    # params_keys will hold all the plotable keys
    params_keys = sorted(parameters.keys())

    # Ask the users what two variables they want to plot
    print("Number of samples: {}\n".format(len(result)))
    idx1 = None
    while idx1 is None:
        for i, key in enumerate(params_keys):
            print("{}. {}".format(i + 1, key))
        print("Choose the first parameter (X) to plot (type number + ENTER):")
        num = -1
        while num <= 0 or num > len(params_keys):
            try:
                num = int(input(">> "))
            except ValueError:
                num = -1
        idx0 = num - 1
        print("Parameter 1 selected: {}\n".format(params_keys[idx0]))
        for i, key in enumerate(params_keys):
            if i == idx0:
                continue
            print("{}. {}".format(i + 1, key))
        print("Choose the second parameter (Y) to plot (type number + ENTER) or enter 0 to start over:")
        num = -1
        while num < 0 or num > len(params_keys) or num == idx0 + 1:
            try:
                num = int(input(">> "))
            except ValueError:
                num = -1
        if num == 0:
            print()
            continue
        idx1 = num - 1
        print("Parameter 2 selected: {}\n".format(params_keys[idx1]))

    # Prepare variables to plot
    keyX = params_keys[idx0]  # name of key x axis
    keyY = params_keys[idx1]  # name of key y axis

    # Remove X and Y labels from params_keys
    if idx1 > idx0:
        del params_keys[idx1]
        del params_keys[idx0]
    else:
        del params_keys[idx0]
        del params_keys[idx1]

    # Save all info from result in 2D array
    plotsX = []
    plotsY = []
    plotsZ = []
    comb_to_fig = {}
    fig_to_comb = {}
    comb_list = []
    for sample_key in result:
        comb = ()
        for key in params_keys:
            comb += (result[sample_key]["params"][key],)
        if not comb in comb_to_fig:
            comb_to_fig[comb] = len(plotsX)
            fig_to_comb[len(plotsX)] = comb
            comb_list.append(comb)
            plotsX.append([])
            plotsY.append([])
            plotsZ.append([])
        plotsX[comb_to_fig[comb]].append(result[sample_key]["params"][keyX])
        plotsY[comb_to_fig[comb]].append(result[sample_key]["params"][keyY])
        for i, keyZ in enumerate(height_keys):
            plotsZ[comb_to_fig[comb]][i].append(result[sample_key]["result"][keyZ])


    # Plot a figure for every combination (I know, there is a lot of repeated code, sorry)
    if ordered_results:
        # Results are plotted in a logical order
        comb_list = sorted(comb_list)
        for comb in comb_list:
            title = ""
            filename = "{}={} {}={} {}={} ".format("metric", height_key, "X", keyX, "Y", keyY)
            for j, param in enumerate(params_keys):
                title += "{}: {}\n".format(param, comb[j])
                filename += "{}={} ".format(param, comb[j])
            filename += "{}={} ".format("plot_mode", plot_mode)
            x = plotsX[comb_to_fig[comb]]
            y = plotsY[comb_to_fig[comb]]
            z = plotsZ[comb_to_fig[comb]]
            if plot_mode == 0:
                plot_3D_bar_graph(x, y, z, axis_labels=(keyX, keyY, height_key), title=title,
                                  filename=filename, bird_view=False, orthogonal_projection=False)
            elif plot_mode == 1:
                plot_3D_bar_graph(x, y, z, axis_labels=(keyX, keyY, height_key), title=title,
                                  filename=filename, bird_view=True, orthogonal_projection=True)
            else:
                title = "Metric: {}\n".format(height_key) + title
                plot_colormap(x, y, z, axis_labels=(keyX, keyY), title=title,
                              filename=filename)
    else:
        # Results will be plotted in original order (order in results.yaml)
        for i, (x, y, z) in enumerate(zip(plotsX, plotsY, plotsZ)):
            title = ""
            filename = "{}={} {}={} {}={} ".format("metric", height_key, "X", keyX, "Y", keyY)
            for j, param in enumerate(params_keys):
                title += "{}: {}\n".format(param, fig_to_comb[i][j])
                filename += "{}={} ".format(param, fig_to_comb[i][j])
            filename += "{}={} ".format("plot_mode", plot_mode)
            if plot_mode == 0:
                plot_3D_bar_graph(x, y, z, axis_labels=(keyX, keyY, height_key), title=title,
                                  filename=filename, bird_view=False, orthogonal_projection=False)
            elif plot_mode == 1:
                plot_3D_bar_graph(x, y, z, axis_labels=(keyX, keyY, height_key), title=title,
                                  filename=filename, bird_view=True, orthogonal_projection=True)
            else:
                title = "Metric: {}\n".format(height_key) + title
                plot_colormap(x, y, z, axis_labels=(keyX, keyY), title=title,
                              filename=filename)


if __name__ == "__main__":
    folder = None
    metric = None
    mode = 0
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    if len(sys.argv) == 3:
        mode = int(sys.argv[2])
    elif len(sys.argv) > 3:
        metric = sys.argv[2]
        mode = int(sys.argv[3])
    if metric is not None:
        plot_results(folder=folder, height_key=metric, plot_mode=mode)
    else:
        plot_results(folder=folder, plot_mode=mode)
