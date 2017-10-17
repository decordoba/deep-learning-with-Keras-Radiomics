import matplotlib as mpl
import os
AGG = False
r1 = os.system('python3 -c "import matplotlib.pyplot as plt;plt.figure()"')  # Linux
r2 = os.system('py -c "import matplotlib.pyplot as plt;plt.figure()"')  # Windows
# This line allows mpl to run with no DISPLAY defined
if r1 != 0 and r2 != 0:
    print("$DISPLAY not detected, matplotlib set to use 'Agg' backend")
    mpl.use('Agg')
    AGG = True
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors as mpl_colors
import math


def plot_images(images, fig_num=0, labels=None, label_description="Label", labels2=None,
                label2_description="Label", show_errors_only=False, cmap="Greys", no_axis=True,
                invert_colors=False, title=None):
    """
    Show all images in images list, one at a time, waiting for an ENTER to show the next one
    If q + ENTER is pressed, the function is terminated
    """
    plt.ion()  # Allows plots to be non-blocking
    fig = plt.figure(fig_num)
    fig.clear()
    factor = 1 if not invert_colors else -1
    for i, img in enumerate(images):
        if show_errors_only and labels is not None and labels2 is not None and labels[i] == labels2[i]:
            continue
        try:
            if cmap is None:
                plt.imshow(img * factor)
            else:
                plt.imshow(img * factor, cmap=cmap)
        except TypeError:
            img = img[:, :, 0]
            if cmap is None:
                plt.imshow(img * factor)
            else:
                plt.imshow(img * factor, cmap=cmap)
        if labels is not None:
            if labels2 is None:
                title = "{} = {}".format(label_description, labels[i])
            else:
                title = "{} = {} , {} = {}".format(label_description, labels[i],
                                                   label2_description, labels2[i])
            plt.title(title, fontsize="xx-large")
        elif title is not None:
            # Title is only used if labels is None
            plt.title(title, fontsize="xx-large")
        if no_axis:
            plt.yticks([])
            plt.xticks([])
        plt.pause(0.001)
        s = input("Press ENTER to see the next image, or Q (q) to continue:  ")
        if len(s) > 0 and s[0].lower() == "q":
            break
    fig.clear()
    plt.close()  # Hide plotting window
    plt.ioff()  # Make plots blocking again

def plot_all_images(images, fig_num=0, filename=None, labels=None, label_description="Label",
                    labels2=None, label2_description="Label", cmap="Greys", no_axis=True,
                    title=None, max_cols=5):
    """
    Show several images with labels at the same time (in the same figure)
    """
    if filename is None:
        plt.ion()
        plt.close(fig_num)

    num_imgs = len(images)

    if num_imgs <= 0:
        print("Nothing to show!\n")
        return

    if max_cols is None:
        max_cols = 5
    fig_size = [num_imgs % max_cols, num_imgs // max_cols + 1]
    if num_imgs % max_cols == 0:
        fig_size[1] -= 1
    if num_imgs >= max_cols:
        fig_size[0] = max_cols

    margin_title = 0
    if title is not None:
        margin_title = 0.5
    fig = plt.figure(fig_num, figsize=(1.6 * fig_size[0], 1.2 * fig_size[1] + margin_title))
    fig.clear()

    for i, img in enumerate(images):
        ax = fig.add_subplot(fig_size[1], fig_size[0], i + 1)

        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        if labels is not None:
            if labels2 is None:
                subfig_title = "{} = {}".format(label_description, labels[i])
            else:
                subfig_title = "{} = {} , {} = {}".format(label_description, labels[i],
                                                          label2_description, labels2[i])
            ax.set_title(subfig_title)
            if no_axis:
                plt.yticks([])
                plt.xticks([])

    # fix overlaps numbers in axis
    plt.tight_layout()

    if title is not None:
        fig.suptitle(title, fontsize="xx-large")
        fig.subplots_adjust(top=0.9 - (margin_title * 0.42 / fig_size[1]))

    if filename is None:
        plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()

def plot_weights(w, fig_num=0, filename=None, title=None, cmap=None):
    """
    Show weights of a 3D or 4D kernel.
    Dim0, Dim1: (x, y),
    Dim2: depth,
    Dim3: #kernels

    If w has only 3D, it is assumed that Dim2 is the #kernels, and depth is 1 (B/W kernels).
    If depths is different to 3 or 4, depth is set to 1, and only the 1st component is used

    If filename is None, the figure will be shown, otherwise it will be saved with name filename
    """
    num_imgs = 1
    if w.ndim == 4:
        num_imgs = w.shape[3]
        num_colors = w.shape[2]
        if num_colors < 3:
            w = w[:, :, 0, :]
        elif num_colors > 4:
            print("Too many dimensions, ignoring all but the first one")
            w = w[:, :, 0, :]
    elif w.ndim == 3:
        num_imgs = w.shape[2]
    NUM_ROWS = math.floor(num_imgs ** 0.5)
    NUM_COLS = math.ceil(num_imgs ** 0.5)
    if NUM_ROWS * NUM_COLS < num_imgs:
        NUM_ROWS += 1
    if filename is None:
        plt.ion()
    fig = plt.figure(fig_num)
    if title is not None:
        fig.suptitle(title)
    for i in range(num_imgs):
        subfig = fig.add_subplot(NUM_ROWS, NUM_COLS, i + 1)
        subfig.imshow(w[:, :, i], cmap=cmap)
        subfig.axis('off')
    if filename is None:
        plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()

def plot_history(history, fig_num=0, filename=None):
    """
    Plots loss and accuracy in history
    If filename is None, the figure will be shown, otherwise it will be saved with name filename
    """
    # Plot epoch history for accuracy and loss
    if filename is None:
        plt.ion()
    fig = plt.figure(fig_num)
    subfig = fig.add_subplot(122)
    subfig.plot(history.history['acc'], label="training")
    if history.history['val_acc'] is not None:
        subfig.plot(history.history['val_acc'], label="validation")
    subfig.set_title('Model Accuracy')
    subfig.set_xlabel('Epoch')
    subfig.legend(loc='upper left')
    subfig = fig.add_subplot(121)
    subfig.plot(history.history['loss'], label="training")
    if history.history['val_loss'] is not None:
        subfig.plot(history.history['val_loss'], label="validation")
    subfig.set_title('Model Loss')
    subfig.set_xlabel('Epoch')
    subfig.legend(loc='upper left')
    if filename is None:
        plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()

def plot_confusion_matrix(true_values, predicted_values, labels, fig_num=0, filename=None,
                          title=None, cmap="plasma", max_scale_factor=100.0, ignore_diagonal=False,
                          color_by_row=False, plot_half=False):
    """
    Plots a confusion matrix from a list of true and predicted labels. The variable labels contains
    the labels names. ignore_diagonal is used to leave the diagonal white (no color), and plot half
    only shows the lower half of the matrix (adding the top and bottom part together).
    max_scale_factor is used to show contrast even for small values in comparison to the larger
    numbers in the diagonal, set it to 1 to get the right color scale
    """
    if filename is None:
        plt.ion()

    # create confusion matrix
    confusion_matrix = np.array([[0] * len(labels) for _ in labels])
    label_mapper = dict([(label, i) for i, label in enumerate(labels)])
    for predicted, expected in zip(true_values, predicted_values):
        if plot_half:
            min_val = label_mapper[predicted]
            max_val = label_mapper[expected]
            if min_val > max_val:
                min_val, max_val = max_val, min_val
            confusion_matrix[max_val][min_val] += 1
        else:
            confusion_matrix[label_mapper[predicted]][label_mapper[expected]] += 1

    # crete mask to show some cells white
    mask = np.ones(confusion_matrix.shape, dtype=bool)
    if plot_half:
        mask *= np.tri(*mask.shape, dtype=bool)
    if not color_by_row:
        if ignore_diagonal:
            np.fill_diagonal(mask, False)
        max_cell = confusion_matrix[mask].max()

    # Adapted from: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    # create attributes to select cell colors
    norm_conf = []
    for i, row in enumerate(confusion_matrix):
        tmp_arr = []
        if color_by_row:
            divisor = sum(row, 0)
            if ignore_diagonal:
                divisor -= float(row[i])
        else:
            divisor = max_cell
        for j, el in enumerate(row):
            if divisor == 0:  # avoid dividing by 0
                tmp_arr.append(1)
            else:
                tmp_arr.append(min(float(el) / float(divisor) * max_scale_factor, 1))
            if mask[i][j] == False:
                tmp_arr[-1] = np.nan
        norm_conf.append(tmp_arr)

    # draw figure
    fig = plt.figure(fig_num)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)  # Make squares
    ax.xaxis.tick_top()  # Like in a table, I want the labels top and left
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("True Values")
    cmap = cm.get_cmap(cmap)
    # result = ax.imshow(np.array(norm_conf), cmap=cmap, norm=mpl_colors.LogNorm(vmin=1e-4, vmax=1),
    #                    interpolation='nearest')
    result = ax.imshow(np.array(norm_conf), cmap=cmap, interpolation='nearest')
    fig.colorbar(result)
    # print numbers in the middle of the cells
    side = range(len(labels))
    for x in side:
        for y in side:
            if plot_half and x < y:
                continue
            ax.annotate(str(confusion_matrix[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center')
    # print labels in axis
    plt.xticks(side, labels)
    plt.yticks(side, labels)
    if title is not None:
        fig.subplots_adjust(top=0.82, bottom=0.05)
        ax.set_title(title.strip(), fontsize="xx-large", y=1.125)
    else:
        fig.subplots_adjust(bottom=0.05)

    if filename is None:
        plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()
    return confusion_matrix

def plot_3D_bar_graph(X, Y, Z, axis_labels=None, title=None, suptitle=None, filename=None,
                      bars_dist=0.1, fig_num=0, cmap="plasma", view_elev=50, view_azim=45,
                      orthogonal_projection=False, subplot_position=111, fig_clear=True,
                      global_colorbar=False, color_scale=None, figsize=(1, 1), invert_xaxis=False,
                      invert_yaxis=False, zlim=None, save_without_prompt=False):
    """
    Receives list of X, Y and Z and plots them. X and Y can be strings or numbers.
    For example:
        plot_3D_bar_graph(["0", "0", "1", "1"], [0, 1, 0, 1], [0, 1, 1, 2])
    will plot a 2 by 2 matrix of bars with different heights.
    Many parameters can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial point of view...
    :param axis_labels: the label in every axis (x, y, z)
    :param title: the title for the subfigure
    :param suptitle: the global title for the figure
    :param filename: if None, the figure will be plotted, else, the figure will be saved (the user will be prompted with a save console interface)
    :param bars_dist: distance between bars (every bar is a square of side 1-bars_dist in a side 1 grid
    :param fig_num: number of the figure used in matplotlib
    :param cmap: name of colormap used
    :param view_elev: param to determine point of view elevation
    :param view_azim: param to determine point of view rotation
    :param orthogonal_projection: if True, an orthogonal projection is used, else the default oblique is used
    :param subplot_position: indicates the size of the whole figure and the position of the current subfigure (i.e. 122 means figure with 2 subfigs, and we draw in the second one)
    :param fig_clear: whether to clear the whole figure before drawing or not
    :param global_colorbar: whether if the colorbar is global (shared by all subfigures) or local (one for every subfigure)
    :param color_scale: tuple that represents the min and max values used in the scale to draw the colormap. If None, the scale will be picked automatically
    :param figsize: initial size of the figure. By default it is a (1, 1) square, but can be set to (1,2), to change the shape.
    :param invert_xaxis: inverts the xaxis
    :param invert_yaxis: inverts the yaxis
    :param zlim: if not None, sets the scale used in the zaxis, else it is set automatically
    :param save_without_prompt: if True, it will save without showing figure (filename must not be None), else, it shows figure and then it saves it once we press ENTER or cancel with Q
    :return: returns a list of all elevs and azims for all subfigures if filename is not None
    """
    # get X and Y axis, and order them
    X_labels = np.unique(X)
    Y_labels = np.unique(Y)
    X_mapper = {}
    for i, x in enumerate(X_labels):
        if invert_xaxis:
            X_mapper[x] = len(X_labels) - i - 1
        else:
            X_mapper[x] = i
    Y_mapper = {}
    for i, y in enumerate(Y_labels):
        if invert_yaxis:
            Y_mapper[y] = len(Y_labels) - i - 1
        else:
            Y_mapper[y] = i

    # create params needed for plotting
    minZ = min(Z)
    maxZ = max(Z)
    X_list = np.array([X_mapper[x] + bars_dist / 2.0 for x in X])
    Y_list = np.array([Y_mapper[y] + bars_dist / 2.0 for y in Y])
    Z_list = np.array(Z)
    if zlim is None:
        Z_offset = minZ - (maxZ - minZ) * 0.1
    else:
        Z_offset = zlim[0]
    dX = np.array([1 - bars_dist] * len(Z_list))
    dY = np.array([1 - bars_dist] * len(Z_list))
    dZ = Z_list - Z_offset
    cmap = cm.get_cmap(cmap)
    if color_scale is not None:
        if color_scale[0] is not None:
            minZ = color_scale[0]
        if color_scale[1] is not None:
            maxZ = color_scale[1]
    colors_list = cmap((Z_list - minZ) / np.float_(maxZ - minZ))

    # create figure
    figsize = (6.4 * figsize[0], 4.8 * figsize[1])
    fig = plt.figure(fig_num, figsize=figsize)
    if fig_clear:
        fig.clear()
    ax = fig.add_subplot(subplot_position, projection='3d')
    if orthogonal_projection:
        def matplotlib_orthogonal_projection(zfront, zback):
            """ Allows to see 3D figures in matplotlib in orthogonal projection """
            a = (zfront + zback) / (zfront - zback)
            b = -2 * (zfront * zback) / (zfront - zback)
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, a, b], [0, 0, -0.0001, zback]])
        proj3d.persp_transformation = matplotlib_orthogonal_projection

    # change initial perspective to be seen from above
    if view_elev is not None or view_azim is not None:
        if view_azim is None:
            ax.view_init(elev=view_elev)
        elif view_elev is None:
            ax.view_init(elev=50, azim=45)
            ax.view_init(azim=view_azim)
        else:
            ax.view_init(elev=view_elev, azim=view_azim)

    # change labels axis
    ax.set_xticks(np.array([X_mapper[xlabel] + 0.5 for xlabel in X_labels]), minor=False)
    ax.set_xticklabels(X_labels)
    ax.set_yticks(np.array([Y_mapper[ylabel] + 0.5 for ylabel in Y_labels]), minor=False)
    ax.set_yticklabels(Y_labels)

    # add space required for title and global colorbar
    margin_title = 0.25
    if suptitle is not None:
        margin_title += len(suptitle.strip().split("\n"))
    if title is not None:
        margin_title += len(title.strip().split("\n"))
        if suptitle is not None:
            margin_title += 0.25
    margin_colorbar = 0.0
    if global_colorbar:
        margin_colorbar = 0.16
    fig.subplots_adjust(bottom=margin_colorbar, top=1 - 0.06 * margin_title)

    # draw colorbar
    if global_colorbar:
        ax_cbar = fig.add_axes([0.1, 0.07, 0.8, 0.05])
        colorbar.ColorbarBase(ax_cbar, orientation="horizontal", cmap=cmap,
                              norm=mpl_colors.Normalize(vmin=minZ, vmax=maxZ))
    else:
        mappable = cm.ScalarMappable(cmap=cmap, norm=mpl_colors.Normalize(vmin=minZ, vmax=maxZ))
        mappable.set_array([minZ, maxZ])
        fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.05)

    # draw bar graph
    ax.bar3d(X_list, Y_list, np.array([Z_offset] * len(Z_list)), dX, dY, dZ, color=colors_list,
             edgecolors='black', linewidths=0.5)

    # set z axis to zlim
    if zlim is not None:
        ax.set_zlim(zlim)

    # add labels and title
    if axis_labels is not None:
        if axis_labels[0] is not None:
            ax.set_xlabel(axis_labels[0])
        if axis_labels[1] is not None:
            ax.set_ylabel(axis_labels[1])
        if axis_labels[2] is not None:
            ax.set_zlabel(axis_labels[2])
    if title is not None:
        ax.set_title(title.strip(), fontsize="xx-large", y=1.085)
    if suptitle is not None:
        fig.suptitle(suptitle.strip(), fontsize="xx-large")

    # wait for user actions and save graph
    if filename is not None:
        if save_without_prompt:
            fig.savefig("{}.png".format(filename.strip()), bbox_inches="tight")
            print("Figure saved in {}.png\n".format(filename.strip()))
        else:
            plt.ion()
            plt.show()
            txt = input("Position the figure in the preferred perspective, and press ENTER to save it.\nPress the Q key + ENTER to skip saving the figure.\n")
            if len(txt) < 1 or txt[0].lower() != "q":
                fig.savefig("{}.png".format(filename.strip()), bbox_inches="tight")
                print("Figure saved in {}.png\n".format(filename.strip()))
            else:
                print()
            plt.ioff()

        # return point of view params for every ax so next figure can be drawn from same point of view
        pts_of_view = []
        for axe in fig.axes:
            try:
                pts_of_view.append((axe.elev, axe.azim))
            except AttributeError:
                pass
        return pts_of_view

    # if figure is no shown, we don't care about point of view, so return None
    return None

def plot_colormap(X, Y, Z, axis_labels=None, title=None, suptitle=None, filename=None, fig_num=0,
                  cmap="plasma", subplot_position=111, fig_clear=True, global_colorbar=False,
                  color_scale=None, figsize=(1, 1), invert_xaxis=False, invert_yaxis=False,
                  save_without_prompt=False):
    """
    Receives list of X, Y and Z and plots them. X and Y can be strings or numbers.
    It will plot a matrix of squares, each with a color representing the number of Z.
    Many parameters can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial view...
    :param axis_labels: the label in every axis (x, y, z)
    :param title: the title for the subfigure
    :param suptitle: the global title for the figure
    :param filename: if None, the figure will be plotted, else, the figure will be saved (the user will be prompted with a save console interface)
    :param fig_num: number of the figure used in matplotlib
    :param cmap: name of colormap used
    :param subplot_position: indicates the size of the whole figure and the position of the current subfigure (i.e. 122 means figure with 2 subfigs, and we draw in the second one)
    :param fig_clear: whether to clear the whole figure before drawing or not
    :param global_colorbar: whether if the colorbar is global (shared by all subfigures) or local (one for every subfigure)
    :param color_scale: tuple that represents the min and max values used in the scale to draw the colormap. If None, the scale will be picked automatically
    :param figsize: initial size of the figure. By default it is a (1, 1) square, but can be set to (1,2), to change the shape.
    :param invert_xaxis: inverts the xaxis
    :param invert_yaxis: inverts the yaxis
    :param save_without_prompt: if True, it will save without showing figure (filename must not be None), else, it shows figure and then it saves it once we press ENTER or cancel with Q
    """
    # get X and Y axis, and order them
    X_labels = np.unique(X)
    Y_labels = np.unique(Y)
    X_mapper = {}
    for i, x in enumerate(X_labels):
        if invert_xaxis:
            X_mapper[x] = len(X_labels) - i - 1
        else:
            X_mapper[x] = i
    Y_mapper = {}
    for i, y in enumerate(Y_labels):
        if invert_yaxis:
            Y_mapper[y] = len(Y_labels) - i - 1
        else:
            Y_mapper[y] = i

    # create params needed for plotting
    minZ = min(Z)
    maxZ = max(Z)
    Z_list = np.array([[0.0 for x in X_labels] for y in Y_labels])
    Z_mask = np.array([[1.0 for x in X_labels] for y in Y_labels])
    for x, y, z in zip(X, Y, Z):
        Z_mask[Y_mapper[y]][X_mapper[x]] = 0
        Z_list[Y_mapper[y]][X_mapper[x]] = z
    Z_list = np.ma.array(Z_list, mask=Z_mask)
    cmap = cm.get_cmap(cmap)
    if color_scale is not None:
        if color_scale[0] is not None:
            minZ = color_scale[0]
        if color_scale[1] is not None:
            maxZ = color_scale[1]

    # create figure
    figsize = (6.4 * figsize[0], 4.8 * figsize[1])
    fig = plt.figure(fig_num, figsize=figsize)
    if fig_clear:
        fig.clear()
    ax = fig.add_subplot(subplot_position)

    # change labels axis
    ax.set_xticks(np.array([X_mapper[xlabel] + 0.5 for xlabel in X_labels]), minor=False)
    ax.set_xticklabels(X_labels)
    ax.set_yticks(np.array([Y_mapper[ylabel] + 0.5 for ylabel in Y_labels]), minor=False)
    ax.set_yticklabels(Y_labels)

    # add space required for title and global colorbar
    margin_title = 0.33
    if suptitle is not None:
        margin_title += len(suptitle.strip().split("\n"))
    if title is not None:
        margin_title += len(title.strip().split("\n"))
        if suptitle is not None:
            margin_title += 0.33
    margin_colorbar = 0.01
    if global_colorbar:
        margin_colorbar = 0.22
    fig.subplots_adjust(bottom=margin_colorbar, top=1 - 0.06 * margin_title)

    # draw colorbar
    if global_colorbar:
        ax_cbar = fig.add_axes([0.1, 0.07, 0.8, 0.05])
        colorbar.ColorbarBase(ax_cbar, orientation="horizontal", cmap=cmap,
                              norm=mpl_colors.Normalize(vmin=minZ, vmax=maxZ))
    else:
        mappable = cm.ScalarMappable(cmap=cmap, norm=mpl_colors.Normalize(vmin=minZ, vmax=maxZ))
        mappable.set_array([minZ, maxZ])
        fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.17)

    # draw color map
    ax.pcolor(Z_list, edgecolors='black', linewidths=0.3, cmap=cmap, norm=mpl_colors.Normalize(vmin=minZ, vmax=maxZ))

    # add labels and title
    if axis_labels is not None:
        if axis_labels[0] is not None:
            ax.set_xlabel(axis_labels[0])
        if axis_labels[1] is not None:
            ax.set_ylabel(axis_labels[1])
    if title is not None:
        ax.set_title(title.strip(), fontsize="xx-large", y=1.03)
    if suptitle is not None:
        fig.suptitle(suptitle.strip(), fontsize="xx-large")

    # hide lines for labels
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Ensure heatmap cells are square
    ax.set_aspect('equal')

    # wait for user actions and save graph
    if filename is not None:
        if save_without_prompt:
            fig.savefig("{}.png".format(filename.strip()), bbox_inches="tight")
            print("Figure saved in {}.png\n".format(filename.strip()))
        else:
            plt.ion()
            plt.show()
            txt = input("Position the figure in the preferred perspective, and press ENTER to save it.\nPress the Q key + ENTER to skip saving the figure.\n")
            if len(txt) < 1 or txt[0].lower() != "q":
                fig.savefig("{}.png".format(filename.strip()), bbox_inches="tight")
                print("Figure saved in {}.png\n".format(filename.strip()))
            else:
                print()
            plt.ioff()

def plot_graph_grid(X, Y, Z, subaxis_labels=None, axis_labels=None, suptitle=None, filename=None,
                    fig_num=0, scaleX=None, scaleY=None, fig_clear=True, simplified_style=True,
                    legend_label=None, invert_xaxis=False, invert_yaxis=False):
    """
    Receives list of X, Y and Z (Z is a list of lists of points, one for each (X, Y)) and plots them.
    X and Y can be strings or numbers.
    Many parameters can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial view...
    """

    # get X and Y axis, and order them
    X_labels = np.unique(X)
    Y_labels = np.unique(Y)
    X_mapper = {}
    for i, x in enumerate(X_labels):
        if invert_xaxis:
            X_mapper[x] = len(X_labels) - i - 1
        else:
            X_mapper[x] = i
    Y_mapper = {}
    for i, y in enumerate(Y_labels):
        if invert_yaxis:
            Y_mapper[y] = len(Y_labels) - i - 1
        else:
            Y_mapper[y] = i

    # create params needed for plotting
    Z_list = [[[] for x in X_labels] for y in Y_labels]
    Z_mask = np.array([[1.0 for x in X_labels] for y in Y_labels])
    minXaxis = 0
    maxXaxis = None
    minYaxis = None
    maxYaxis = None
    if scaleX is not None:
        maxXaxis = scaleX
    if scaleY is not None:
        if scaleY[0] is not None:
            minYaxis = scaleY[0]
        if scaleY[1] is not None:
            maxYaxis = scaleY[1]
    for x, y, z in zip(X, Y, Z):
        Z_mask[Y_mapper[y]][X_mapper[x]] = 0
        Z_list[Y_mapper[y]][X_mapper[x]] = z
        try:
            minYaxis = min(minYaxis, min(z))
            maxYaxis = max(maxYaxis, max(z))
            maxXaxis = max(maxXaxis, len(z))
        except TypeError:
            minYaxis = min(z)
            maxYaxis = max(z)
            maxXaxis = len(z)

    Z_list = np.ma.array(Z_list, mask=Z_mask)

    # create figure
    NUM_ROWS = len(Y_labels)
    NUM_COLS = len(X_labels)
    figsize = (6.4 * NUM_COLS / 3.0, 4.8 * NUM_ROWS / 3.0)
    fig = plt.figure(fig_num, figsize=figsize)
    if fig_clear:
        fig.clear()

    # plot all graphs
    for x in X_labels:
        i = X_mapper[x]
        for y in Y_labels:
            j = Y_mapper[y]
            if Z_mask[j][i]:
                continue
            ax = fig.add_subplot(NUM_ROWS, NUM_COLS, j * NUM_COLS + i + 1)
            if legend_label is not None:
                ax.plot(Z_list[j][i], label=legend_label)
            else:
                ax.plot(Z_list[j][i])
            ax.set_xlim([minXaxis, maxXaxis])
            ax.set_ylim([minYaxis, maxYaxis])

            if simplified_style:
                xlabel = ""
                ylabel = ""
                if subaxis_labels is not None and subaxis_labels[0] is not None:
                    xlabel += "{}\n".format(subaxis_labels[0])
                if axis_labels is not None:
                    if axis_labels[0] is not None:
                        xlabel += "{}: {}".format(axis_labels[0], x)
                    if axis_labels[1] is not None:
                        ylabel += "{}: {}\n".format(axis_labels[1], y)
                if subaxis_labels is not None and subaxis_labels[1] is not None:
                    ylabel += "{}\n".format(subaxis_labels[1])

                if i > 0:
                    ax.tick_params(labelleft='off')
                elif len(ylabel.strip()) > 0:
                    ax.set_ylabel(ylabel.strip())
                if j < NUM_ROWS - 1:
                    ax.tick_params(labelbottom='off')
                elif len(xlabel.strip()) > 0:
                    ax.set_xlabel(xlabel.strip())

            else:
                if axis_labels is not None:
                    label_title = ""
                    if axis_labels[0] is not None:
                        label_title += "{}: {}".format(axis_labels[0], x)
                    if axis_labels[1] is not None:
                        if len(label_title) > 0:
                            label_title += ",  "
                        label_title += "{}: {}".format(axis_labels[1], y)
                    ax.set_title(label_title.strip())
                if subaxis_labels is not None:
                    if subaxis_labels[0] is not None:
                        ax.set_xlabel(subaxis_labels[0])
                    if subaxis_labels[1] is not None:
                        ax.set_ylabel(subaxis_labels[1])

            if legend_label is not None and i == (NUM_COLS - 1) // 2 and j == 0:
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=len(ax.lines))

    # fix overlaps numbers in axis
    plt.tight_layout()

    # increase margins if necessary and print title
    margin_title = 1
    if suptitle is not None:
        margin_title += len(suptitle.strip().split("\n"))
    if legend_label is not None:
        margin_title += 1
    fig.subplots_adjust(top=1 - 0.06 * margin_title)
    if suptitle is not None:
        fig.suptitle(suptitle.strip(), fontsize="xx-large")

    # wait for user actions and save graph
    if filename is not None:
        plt.ion()
        plt.show()
        txt = input("Position the figure in the preferred perspective, and press ENTER to save it.\nPress the Q key + ENTER to skip saving the figure.\n")
        if len(txt) < 1 or txt[0].lower() != "q":
            fig.savefig("{}.png".format(filename.strip()), bbox_inches="tight")
            print("Figure saved in {}.png\n".format(filename.strip()))
        else:
            print()
        plt.ioff()

    return minXaxis, (minYaxis, maxYaxis)  # return scaleX and scaleY (can be used as input to next call)


if __name__ == "__main__":
    import random
    X = "1st 2nd 3rd 4th".split() + "1st 3rd 4th".split() + "1st 2nd 3rd 4th".split()
    Y = ["a", "a", "a", "a"] + ["b", "b", "b"] + ["c", "c", "c", "c"]
    Z1 = [random.random() * 0.5 + 0.5 for _ in X]
    Z2 = [random.random() * 0.5 + 0.5 for _ in X]
    Z3 = [[i + random.random() * 2 - 1 for i in range(20)] for _ in X]
    Z4 = [[i + random.random() * 2 - 1.5 for i in range(20)] for _ in X]
    suptitle = None
    suptitle = "One Line Long Title"
    # suptitle = "Two Lines Long Title\nTwo Lines Long Title"
    # suptitle = "Three Lines Long Title\nThree Lines Long Title\nThree Lines Long Title"
    title = None
    # title = "One Line Long Title"
    # title = "Two Lines Long Title\nTwo Lines Long Title"
    # title = "Three Lines Long Title\nThree Lines Long Title\nThree Lines Long Title"

    plot_graph_grid(X, Y, Z3, axis_labels=("x", "y"), subaxis_labels=("Epochs", "Accuracy"),
                    suptitle=suptitle, fig_num=1, fig_clear=True, legend_label="Asdfg")
    plot_graph_grid(X, Y, Z4, axis_labels=("x", "y"), subaxis_labels=("Epochs", "Accuracy"),
                    suptitle=suptitle, fig_num=1, fig_clear=False, legend_label="Ghjklopqwerty",
                    filename="test3.png")

    plot_graph_grid(X, Y, Z3, axis_labels=("x", "y"), subaxis_labels=("Epochs", "Accuracy"),
                    suptitle=suptitle, fig_num=1, fig_clear=True, legend_label="Asdfg",
                    simplified_style=False)
    plot_graph_grid(X, Y, Z4, axis_labels=None, subaxis_labels=None, simplified_style=False,
                    suptitle=suptitle, fig_num=1, fig_clear=False, legend_label="Ghjklopqwerty")

    plot_colormap(X, Y, Z1, axis_labels=("x", "y"), title=title, subplot_position=121,
                  fig_clear=True, suptitle=suptitle, global_colorbar=False, figsize=(2, 1))
    plot_colormap(X, Y, Z2, axis_labels=("x", "y"), title=title, subplot_position=122,
                  fig_clear=False, suptitle=suptitle, filename="test0.png", global_colorbar=False,
                  color_scale=(0.5, 1.2), figsize=(2, 1))

    plot_colormap(X, Y, Z1, axis_labels=("x", "y"), title=title, subplot_position=121,
                  fig_clear=True, suptitle=suptitle, global_colorbar=True, figsize=(2, 1))
    plot_colormap(X, Y, Z2, axis_labels=("x", "y"), title=title, subplot_position=122,
                  fig_clear=False, suptitle=suptitle, filename="test0.png", global_colorbar=True,
                  color_scale=(0.5, 1.2), figsize=(2, 1))

    elev1 = None
    azim1 = None
    elev2 = None
    azim2 = None
    while True:
        plot_3D_bar_graph(X, Y, Z1, axis_labels=("x", "y", "z1"), title=title, subplot_position=121,
                          fig_clear=True, suptitle=suptitle, global_colorbar=False, figsize=(2, 1),
                          view_azim=azim1, view_elev=elev1)
        pt_view = plot_3D_bar_graph(X, Y, Z2, axis_labels=("x", "y", "z2"), title=title,
                                    subplot_position=122, fig_clear=False, suptitle=suptitle,
                                    filename="test1.png", global_colorbar=False,
                                    color_scale=(0.5, 1.2), figsize=(2, 1), view_azim=azim2,
                                    view_elev=elev2)
        elev1 = pt_view[0][0]
        azim1 = pt_view[0][1]
        elev2 = pt_view[1][0]
        azim2 = pt_view[1][1]

        plot_3D_bar_graph(X, Y, Z1, axis_labels=("x", "y", "z1"), title=title, subplot_position=121,
                          fig_clear=True, suptitle=suptitle, global_colorbar=True,
                          color_scale=(0.5, 1.2), figsize = (2, 1), view_azim=azim1, view_elev=elev1)
        pt_view = plot_3D_bar_graph(X, Y, Z2, axis_labels=("x", "y", "z2"), title=title,
                                    subplot_position=122, fig_clear=False, suptitle=suptitle,
                                    filename="test2.png", global_colorbar=True,
                                    color_scale=(0.5, 1.2), figsize=(2, 1), view_azim=azim2,
                                    view_elev=elev2)
        elev1 = pt_view[0][0]
        azim1 = pt_view[0][1]
        elev2 = pt_view[1][0]
        azim2 = pt_view[1][1]
