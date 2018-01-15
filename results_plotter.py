#!/usr/bin/env python3.5

import sys
import os
import yaml
import numpy as np
from keras_plot import plot_3D_bar_graph, plot_colormap, plot_graph_grid  # 'Library' by Daniel


def plot_results(folder=None, height_keys=["accTr", "accTe"], plot_mode=0,
                 shared_color_scale=True, static_z_scale=False, save_without_prompt=False,
                 secondary_plot=["train_accuracy_history", "test_accuracy_history"]):
    """
    From the selected folder (or current folder if folder == None), plot result found in
    result.yaml
    Different plot modes will plot the same data differently. Right now, only 3 modes are
    implemented: 0 (3D bars graph), 1 (2D colormap) or 2 (3D bars graph but seen from top).
    If secondary_plot, the histories (saved in result.yaml in every subfolder) will be shown too.
    If !shared_color_scale, every plot will have a unique scale. Else, the scale will be shared.
    If static_z_scale, the color scale and zaxis used will be the same for every experiment.
    """
    # Make sure height keys is a list
    if not isinstance(height_keys, list):
        height_keys = [height_keys]

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
            if key not in parameters:
                parameters[key] = set()
            parameters[key].add(sample["params"][key])

    # Remove keys which always hold the same value, or which (almost) vary every sample, we don't want to plot them
    for key in list(parameters.keys()):
        if len(parameters) <= 2:
            break
        if len(parameters[key]) < 2 or len(parameters[key]) > len(result) * 0.95:
            del parameters[key]

    # params_keys will hold all the plotable keys
    params_keys = sorted(parameters.keys())

    # # for visualization purposes, it can be useful to change the order of parameters
    # tmp = params_keys[0]
    # params_keys[0] = params_keys[1]
    # params_keys[1] = tmp

    # Ask the users what two variables they want to plot
    print("Number of samples: {}\n".format(len(result)))
    idx1 = None
    if len(parameters) <= 2 or save_without_prompt:
        idx0 = 0
        idx1 = 1
        if len(parameters) <= 1:
            idx1 = 0
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
        print("Parameter X selected: {}\n".format(params_keys[idx0]))
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
        print("Parameter Y selected: {}\n".format(params_keys[idx1]))

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
    locations = []
    comb_to_fig = {}
    comb_list = []
    Zs = []
    for sample_key in result:
        comb = ()
        for key in params_keys:
            comb += (result[sample_key]["params"][key],)
        if not comb in comb_to_fig:
            comb_to_fig[comb] = len(plotsX)
            comb_list.append(comb)
            plotsX.append([])
            plotsY.append([])
            plotsZ.append([[] for _ in height_keys])
            locations.append([])
        plotsX[comb_to_fig[comb]].append(result[sample_key]["params"][keyX])
        plotsY[comb_to_fig[comb]].append(result[sample_key]["params"][keyY])
        for i, keyZ in enumerate(height_keys):
            Zpt = result[sample_key]["result"][keyZ]
            plotsZ[comb_to_fig[comb]][i].append(Zpt)
            Zs.append(Zpt)
        try:
            locations[comb_to_fig[comb]].append(result[sample_key]["result"]["location"])  # technically using the sample_key would also work, but this is better practice
        except KeyError:
            locations[comb_to_fig[comb]].append(sample_key)  # in my early work, I did not save location, so this supports earlier versions

    Z20 = np.percentile(Zs, 20)
    Z80 = np.percentile(Zs, 80)
    minZ = min(Zs)
    maxZ = max(Zs)
    color_scale = None
    zlim = None
    if shared_color_scale and static_z_scale:
        color_scale = (Z20, Z80)  # used to select color of bars
        zlim = (minZ, maxZ)  # used to select z_scale

    # Results are plotted in a logical order
    pt_view = None
    orthog_proj = False
    if plot_mode != 2:
        elev0 = 50
        azim0 = 45
        orthog_proj = False
        if plot_mode == 1:
            elev0 = 90
            azim0 = 90
            orthog_proj = True
        elif plot_mode == 3:
            elev0 = 60
            azim0 = 69
        pt_view = [(elev0, azim0) for _ in height_keys]
    comb_list = sorted(comb_list)
    for comb in comb_list:
        suptitle = ""
        filename = "{}={} {}={} {}={} ".format("metric", height_keys, "X", keyX, "Y", keyY)
        for j, param in enumerate(params_keys):
            suptitle += "{}: {}\n".format(param, comb[j])
            filename += "{}={} ".format(param, comb[j])
        filename += "{}={} ".format("plot_mode", plot_mode)
        x = plotsX[comb_to_fig[comb]]
        y = plotsY[comb_to_fig[comb]]
        zs = plotsZ[comb_to_fig[comb]]
        sub_result = None
        if secondary_plot is not None:
            sub_result = [[] for _ in secondary_plot]
            for location in locations[comb_to_fig[comb]]:
                with open(location + "/result.yaml") as f:
                    try:
                        tmp = yaml.load(f)
                        for i, secondary_key in enumerate(secondary_plot):
                            sub_result[i].append(tmp[secondary_key])
                    except yaml.YAMLError as YamlError:
                        print("There was an error parsing '{}/result.yaml'. Secondary plotting aborted.".format(location))
                        print(YamlError)
                        sub_result = None
                if sub_result is None:
                    break
        if sub_result is not None:
            subaxis_labels = None  # ("Epochs", "Accuracy") but what if user wants to see loss? Better leave it as None
            scaleX = None
            scaleY = None
            for i, (z_history, legend_label) in enumerate(zip(sub_result, secondary_plot)):
                scaleX, scaleY = plot_graph_grid(x, y, z_history, subaxis_labels=subaxis_labels,
                                                 fig_num=1, axis_labels=(keyX, keyY),
                                                 suptitle=suptitle, filename=None, scaleY=scaleY,
                                                 scaleX=scaleX, fig_clear=i==0,
                                                 legend_label=legend_label, invert_yaxis=True)
        if shared_color_scale and not static_z_scale:
            color_scale = (np.min(zs), np.max(zs))
        for i, (height_key, z) in enumerate(zip(height_keys, zs)):
            subplot_position = 100 + len(height_keys) * 10 + i + 1
            figsize = (len(height_keys), 1)
            fn = filename if i + 1 == len(height_keys) else None
            title = "{}: {}\n".format("metric", height_key)
            if plot_mode == 0 or plot_mode == 1 or plot_mode == 3:
                tmp_view = plot_3D_bar_graph(x, y, z, axis_labels=(keyX, keyY, height_key),
                                             suptitle=suptitle, title=title, filename=fn,
                                             fig_clear=i==0, orthogonal_projection=orthog_proj,
                                             view_azim=pt_view[i][1], view_elev=pt_view[i][0],
                                             subplot_position=subplot_position, figsize=figsize,
                                             global_colorbar=shared_color_scale, fig_num=0,
                                             color_scale=color_scale, invert_yaxis=True,
                                             invert_xaxis=True, zlim=zlim,
                                             save_without_prompt=save_without_prompt)
                if tmp_view is not None:
                    pt_view = tmp_view
            else:
                plot_colormap(x, y, z, axis_labels=(keyX, keyY), title=title, suptitle=suptitle,
                              fig_clear=i==0, filename=fn, subplot_position=subplot_position,
                              global_colorbar=shared_color_scale, color_scale=color_scale,
                              figsize=figsize, fig_num=0, save_without_prompt=save_without_prompt)
            # TODO: check why some plots are flat (in test_0703)

    if folder is not None:
        os.chdir("./..")
    return result


if __name__ == "__main__":
    folder = None
    mode = 0
    metric = None  # if left None, the metrics used will be ["accTr", "accTe"]
    static_z_scale = False
    secondary_plot = False
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    if len(sys.argv) > 2:
        mode = int(sys.argv[2])
        if mode >= 12:
            static_z_scale = True
            mode = 3
        else:
            if mode >= 6:
                mode -= 6
                secondary_plot = True
            if mode >= 3:
                mode -= 3
                static_z_scale = True
    if len(sys.argv) > 3:
        metric = sys.argv[3]

    if secondary_plot:
        if metric is not None:
            plot_results(folder=folder, height_keys=metric, plot_mode=mode,
                         static_z_scale=static_z_scale)
        else:
            plot_results(folder=folder, plot_mode=mode, static_z_scale=static_z_scale)
    else:
        if metric is not None:
            plot_results(folder=folder, height_keys=metric, plot_mode=mode,
                         static_z_scale=static_z_scale, secondary_plot=None)
        else:
            plot_results(folder=folder, plot_mode=mode, static_z_scale=static_z_scale,
                         secondary_plot=None)

    """
    Expects:
        py results_plotter.py
        py results_plotter.py folder
        py results_plotter.py folder mode
        py results_plotter.py folder mode metric

    * MODE: 0,1,2... !secondary_plot, !static_z_scale
            3,4,5... !secondary_plot,  static_z_scale
            6,7,8...  secondary_plot, !static_z_scale
            9,A,B...  secondary_plot,  static_z_scale
            C...     !secondary_plot,  static_z_scale, mode 3 (the secret mode)
    """