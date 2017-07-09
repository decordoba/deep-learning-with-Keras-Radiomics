import numpy as np
from mpl_toolkits.mplot3d import proj3d
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors as mpl_colors
import math


def plot_images(images, fig_num=0, labels=None, label_description="Label", labels2=None,
                label2_description="Label", show_errors_only=False, cmap="Greys"):
    """
    Show all images in images list, one at a time, waiting for an ENTER to show the next one
    If q + ENTER is pressed, the function is terminated
    """
    plt.ion()  # Allows plots to be non-blocking
    fig = plt.figure(fig_num)
    for i, img in enumerate(images):
        if cmap is None:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=cmap)
        if labels is not None:
            if labels2 is None:
                title = "{} = {}".format(label_description, labels[i])
            else:
                if show_errors_only and labels[i] == labels2[i]:
                    continue
                title = "{} = {} , {} = {}".format(label_description, labels[i],
                                                   label2_description, labels2[i])
            plt.title(title, fontsize="xx-large")
        plt.pause(0.001)
        s = input("Press ENTER to see the next image, or Q (q) to continue:  ")
        if len(s) > 0 and s[0].lower() == "q":
            break
    plt.close()  # Hide plotting window
    fig.clear()
    plt.ioff()  # Make plots blocking again

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
        subfig = fig.add_subplot(NUM_ROWS, NUM_COLS, + i + 1)
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

def plot_3D_bar_graph(X, Y, Z, axis_labels=None, title=None, suptitle=None, filename=None,
                      bars_dist=0.1, fig_num=0, cmap="plasma", view_elev=50, view_azim=45,
                      orthogonal_projection=False, subplot_position=111, fig_clear=True,
                      global_colorbar=False, scale=None, figsize=(1, 1)):
    """
    Receives list of X, Y and Z and plots them. X and Y can be strings or numbers.
    For example:
        plot_3D_bar_graph(["0", "0", "1", "1"], [0, 1, 0, 1], [0, 1, 1, 2])
    will plot a 2 by 2 matrix of bars with different heights.
    Many parmateres can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial view...
    """
    # get X and Y axis, and order them
    X_labels = np.unique(X)
    Y_labels = np.unique(Y)
    X_mapper = {}
    for i, x in enumerate(X_labels):
        X_mapper[x] = i
    Y_mapper = {}
    for i, y in enumerate(Y_labels):
        Y_mapper[y] = i

    # create params needed for plotting
    minZ = min(Z)
    maxZ = max(Z)
    X_list = np.array([X_mapper[x] + bars_dist / 2.0 for x in X])
    Y_list = np.array([Y_mapper[y] + bars_dist / 2.0 for y in Y])
    Z_list = np.array(Z)
    Z_offset = minZ - (maxZ - minZ) * 0.1
    dX = np.array([1 - bars_dist] * len(Z_list))
    dY = np.array([1 - bars_dist] * len(Z_list))
    dZ = Z_list - Z_offset
    if scale is not None:
        if scale[0] is not None:
            minZ = scale[0]
        if scale[1] is not None:
            maxZ = scale[1]
    cmap = cm.get_cmap(cmap)
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
        print("elev", view_elev, "azim", view_azim)

    # change labels axis
    ax.set_xticks(np.arange(X_labels.shape[0]) + 0.5)
    ax.set_xticklabels(X_labels)
    ax.set_yticks(np.arange(Y_labels.shape[0]) + 0.5)
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

def plot_colormap(X, Y, Z, axis_labels=None, title=None, filename=None, fig_num=0, cmap="plasma",
                  subplot_position=111, fig_clear=True):
    """
    Receives list of X, Y and Z and plots them. X and Y can be strings or numbers.
    For example:
        plot_3D_bar_graph(["0", "0", "1", "1"], [0, 1, 0, 1], [0, 1, 1, 2])
    will plot a 2 by 2 matrix of bars with different heights.
    Many parmateres can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial view...
    """
    # get X and Y axis, and order them
    X_labels = np.unique(X)
    Y_labels = np.unique(Y)
    X_mapper = {}
    for i, x in enumerate(X_labels):
        X_mapper[x] = i
    Y_mapper = {}
    for i, y in enumerate(Y_labels):
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

    # create figure
    fig = plt.figure(fig_num)
    if fig_clear:
        fig.clear()
    ax = fig.add_subplot(subplot_position)

    # change labels axis
    ax.set_xticks(np.arange(X_labels.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(X_labels)
    ax.set_yticks(np.arange(Y_labels.shape[0]) + 0.5, minor=False)
    ax.set_yticklabels(Y_labels)

    # draw colorbar
    fig.subplots_adjust(bottom=0.25)
    ax_cbar = fig.add_axes([0.1, 0.07, 0.8, 0.05])
    colorbar.ColorbarBase(ax_cbar, orientation="horizontal", cmap=cmap,
                          norm=mpl_colors.Normalize(vmin=minZ, vmax=maxZ))

    # draw color map
    ax.pcolor(Z_list, edgecolors='black', linewidths=0.3, cmap=cmap)

    # add labels and title
    if axis_labels is not None:
        if axis_labels[0] is not None:
            ax.set_xlabel(axis_labels[0])
        if axis_labels[1] is not None:
            ax.set_ylabel(axis_labels[1])
    if title is not None:
        ax.set_title(title.strip(), fontsize="xx-large", y=1.1)

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
        plt.ion()
        plt.show()
        txt = input("Position the figure in the preferred perspective, and press ENTER to save it.\nPress the Q key + ENTER to skip saving the figure.\n")
        if len(txt) < 1 or txt[0].lower() != "q":
            fig.savefig("{}.png".format(filename.strip()), bbox_inches="tight")
            print("Figure saved in {}.png\n".format(filename.strip()))
        else:
            print()
        plt.ioff()


if __name__ == "__main__":
    import random
    X = "1st 2nd 3rd 4th".split() + "1st 3rd 4th".split() + "1st 2nd 3rd 4th".split()
    Y = ["a", "a", "a", "a"] + ["b", "b", "b"] + ["c", "c", "c", "c"]
    Z1 = [random.random() * 0.5 + 0.5 for _ in X]
    Z2 = [random.random() * 0.5 + 0.5 for _ in X]
    # plot_colormap(X, Y, Z, axis_labels=("x", "y"), title="Hello")
    # suptitle = None
    suptitle = "One Line Long Title"
    # suptitle = "Two Lines Long Title\nTwo Lines Long Title"
    # suptitle = "Three Lines Long Title\nThree Lines Long Title\nThree Lines Long Title"
    # title = None
    title = "One Line Long Title"
    # title = "Two Lines Long Title\nTwo Lines Long Title"
    # title = "Three Lines Long Title\nThree Lines Long Title\nThree Lines Long Title"

    # plot_3D_bar_graph(X, Y, Z1, axis_labels=("x", "y", "z1"), title=title, subplot_position=121,
    #                   fig_clear=True, suptitle=suptitle, global_colorbar=False)
    # plot_3D_bar_graph(X, Y, Z2, axis_labels=("x", "y", "z2"), title=title, subplot_position=122,
    #                   fig_clear=False, suptitle=suptitle, filename="test1.png", global_colorbar=False)
    #
    # plot_3D_bar_graph(X, Y, Z1, axis_labels=("x", "y", "z1"), title=title, subplot_position=121,
    #                   fig_clear=True, suptitle=suptitle, global_colorbar=True)
    # plot_3D_bar_graph(X, Y, Z2, axis_labels=("x", "y", "z2"), title=title, subplot_position=122,
    #                   fig_clear=False, suptitle=suptitle, filename="test1.png", global_colorbar=True)

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
                                    scale=(0.5, 1.2), figsize=(2, 1), view_azim=azim2,
                                    view_elev=elev2)
        elev1 = pt_view[0][0]
        azim1 = pt_view[0][1]
        elev2 = pt_view[1][0]
        azim2 = pt_view[1][1]

        plot_3D_bar_graph(X, Y, Z1, axis_labels=("x", "y", "z1"), title=title, subplot_position=121,
                          fig_clear=True, suptitle=suptitle, global_colorbar=True, scale=(0.5, 1.2),
                          figsize = (2, 1), view_azim=azim1, view_elev=elev1)
        pt_view = plot_3D_bar_graph(X, Y, Z2, axis_labels=("x", "y", "z2"), title=title,
                                    subplot_position=122, fig_clear=False, suptitle=suptitle,
                                    filename="test2.png", global_colorbar=True, scale=(0.5, 1.2),
                                    figsize=(2, 1), view_azim=azim2, view_elev=elev2)
        elev1 = pt_view[0][0]
        azim1 = pt_view[0][1]
        elev2 = pt_view[1][0]
        azim2 = pt_view[1][1]
