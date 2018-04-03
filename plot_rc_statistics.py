#!/usr/bin/env python3.5
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import operator
import sys
sys.path.insert(0, 'create_datasets')
from save_datasets import save_plt_figures_to_pdf


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """Make a scatter plot of circles.

    Similar to plt.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_)
               for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def plot_statistics_for_r_c(statistics_file, plot=False, static_statistics=None,
                            comparison_metrics=None, skip_plots=False):
    """Docstring for plot_statistics."""
    # Open file with all realizations of R and C
    with open(statistics_file) as f:
        lines = f.readlines()
    statistics = []
    for i, line in enumerate(lines):
        if i == 0:
            names = line.strip().split(", ")
            if names[-3] != "c" or names[-2] != "r" or names[-1] != "n":
                print(names)
                raise Exception("Unexpected data format. Expected to see n, c and r in -1, -2 and "
                                "-3 positions in '{}'".format(statistics_file))
        else:
            statistics.append([float(x) for x in line.split(", ")])
    # Format and extract data
    print(names)
    statistics = np.array(statistics)
    c = statistics[:, -3]
    r = statistics[:, -2]
    statistics = statistics[:, :-3]
    # Find unique Rs and Cs in realizations
    ht = {}
    c_map = {}
    unique_c = list(np.unique(c))
    print("Unique C:", unique_c)
    for i, cc in enumerate(unique_c):
        c_map[cc] = i
    for i, (rr, cc) in enumerate(zip(r, c)):
        if rr not in ht:
            ht[rr] = [-1] * len(unique_c)
        ht[rr][c_map[cc]] = i
    unique_r = sorted(list(ht.keys()))
    print("Unique R:", unique_r)
    # Find combination most similar to real data
    if static_statistics is None:
        return
    if "real label 0" not in static_statistics or "real label 1" not in static_statistics:
        return
    differences = np.array(static_statistics["real label 0"]) - static_statistics["real label 1"]
    differences = differences[comparison_metrics]
    distances_ht = {}
    for i in range(statistics.shape[0]):
        for j in range(statistics.shape[0]):
            diff = statistics[i, comparison_metrics] - statistics[j, comparison_metrics]
            perc = np.abs(differences - diff) / differences
            dist = np.sqrt(np.sum(np.square(perc)))
            distances_ht[(i, j)] = dist
    sorted_dist = sorted(distances_ht.items(), key=operator.itemgetter(1))
    print(names)
    print("Comparison based on metrics:\n  {}".format(np.array(names)[comparison_metrics]))
    for i in range(9, -1, -1):
        print("Top {} distance:\n  Distance: {}".format(i + 1, sorted_dist[i][1]))
        print("  Label 0.  R: {}, C: {}".format(r[sorted_dist[i][0][0]], c[sorted_dist[i][0][0]]))
        print("  Label 1.  R: {}, C: {}".format(r[sorted_dist[i][0][1]], c[sorted_dist[i][0][1]]))
    static_statistics["optimal label 0"] = statistics[sorted_dist[i][0][0], :]
    static_statistics["optimal label 1"] = statistics[sorted_dist[i][0][1], :]
    # Plot all realizations R-C and best result
    if plot:
        plt.ion()
    if not skip_plots:
        for i in range(statistics.shape[1]):
            values = statistics[:, i]
            fig = plt.figure()
            plt.scatter(c, r, c=values)
            plt.colorbar()
            plt.title(names[i])
            plt.xlabel("C")
            plt.ylabel("R")
            fig = plt.figure()
            ax = plt.subplot(111)
            for k in unique_r:
                plt.plot(unique_c, statistics[ht[k], i], label="R: {}".format(k))
            if static_statistics is not None:
                static_names = sorted(list(static_statistics.keys()))
                for kk, k in enumerate(static_names):
                    st = [":", "--", "-.", "-"][kk % 4]
                    if i < len(static_statistics[k]):
                        plt.axhline(y=static_statistics[k][i], color="k", linestyle=st, label=k)
            # Reduce box height by 10%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.title(names[i])
            plt.xlabel("C")
            ylabel = names[i].replace("_", " ", 1)
            ylabel = ylabel + " intensity" if ylabel.endswith("mean") else ylabel
            ylabel = ylabel + " intensity" if ylabel.endswith("median") else ylabel
            ylabel = ylabel + " intensity" if ylabel.endswith("stddev") else ylabel
            ylabel = ylabel + " (px)" if ylabel.endswith("surface") else ylabel
            ylabel = ylabel + " (px)" if ylabel.endswith("volume") else ylabel
            plt.ylabel(ylabel)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if plot:
            input("{}. Press ENTER to continue".format(i))
        save_plt_figures_to_pdf("RC_plots.pdf", verbose=True)
    if plot:
        plt.ioff()


if __name__ == "__main__":
    """
    Statistics real data:
    +--------------------+-----------------+----------------+----------------+---------+--------+------------------+------------------+----------------+------------------+
    |                    |       mean      |     median     |     stddev     | surface | volume |  surf_vol_ratio  |  dissimilarity   |  correlation   |       asm        |
    +--------------------+-----------------+----------------+----------------+---------+--------+------------------+------------------+----------------+------------------+
    |  Label 0 Medians   |  14.7422319315  | 10.4442642452  |  19.123010876  |  321.0  | 623.0  |  0.540308747856  |  0.443541803638  | 0.910109991577 |  0.208770332886  |
    |  Label 1 Medians   |  15.1923547764  | 9.64817589985  | 18.1819843505  |  413.0  | 783.0  |  0.602209944751  |  0.460260519395  | 0.892966048314 |  0.241893132802  |
    | Labels Differences | -0.450122844961 | 0.796088345316 | 0.941026525509 |  -92.0  | -160.0 | -0.0619011968955 | -0.0167187157572 | 0.017143943263 | -0.0331227999155 |
    +--------------------+-----------------+----------------+----------------+---------+--------+------------------+------------------+----------------+------------------+
    +--------------------+---------------+---------------+----------------+----------------+----------------+------------------+-------------------+-----------------+------------------+
    |                    |      mean     |     median    |     stddev     |    surface     |     volume     |  surf_vol_ratio  |   dissimilarity   |   correlation   |       asm        |
    +--------------------+---------------+---------------+----------------+----------------+----------------+------------------+-------------------+-----------------+------------------+
    |   Label 0 Means    | 16.8036008686 |  11.759901739 | 19.7299917319  | 440.688888889  | 891.244444444  |  0.552568761085  |   0.449349508121  |  0.910054738859 |  0.22031286031   |
    |   Label 1 Means    | 15.3408895241 | 10.0117188637 | 19.3893984186  |     611.2      | 1036.33333333  |  0.615367393711  |   0.456820439769  |  0.892641063124 |  0.237164536596  |
    | Labels Differences |  1.4627113445 | 1.74818287526 | 0.340593313279 | -170.511111111 | -145.088888889 | -0.0627986326255 | -0.00747093164828 | 0.0174136757352 | -0.0168516762862 |
    +--------------------+---------------+---------------+----------------+----------------+----------------+------------------+-------------------+-----------------+------------------+
    +--------------------+---------------+---------------+-----------------+----------------+----------------+------------------+------------------+-------------------+-------------------+
    |                    |      mean     |     median    |      stddev     |    surface     |     volume     |  surf_vol_ratio  |  dissimilarity   |    correlation    |        asm        |
    +--------------------+---------------+---------------+-----------------+----------------+----------------+------------------+------------------+-------------------+-------------------+
    |  Label 0 Std Dev   | 7.37845725416 | 5.78504237684 |  5.87692363223  | 244.562086843  | 656.871276264  |  0.129301950581  |  0.115485617053  |  0.0252230828979  |  0.0946236123828  |
    |  Label 1 Std Dev   | 6.12458330477 | 4.48838945867 |  6.00040242802  | 446.426283575  | 751.267787735  |  0.12462061253   |  0.139292437778  |  0.0242256531482  |  0.0959558285797  |
    | Labels Differences |  1.2538739494 | 1.29665291817 | -0.123478795799 | -201.864196732 | -94.3965114718 | 0.00468133805142 | -0.0238068207243 | 0.000997429749706 | -0.00133221619692 |
    +--------------------+---------------+---------------+-----------------+----------------+----------------+------------------+------------------+-------------------+-------------------+

    Statistics Original Lumpy model: (r1: 1.5, c1: 500, r0: 2.5, c0: 150)
    +--------------------+---------------+--------+---------------+---------+--------+------------------+------------------+-----------------+-------------------+
    |                    |      mean     | median |     stddev    | surface | volume |  surf_vol_ratio  |  dissimilarity   |   correlation   |        asm        |
    +--------------------+---------------+--------+---------------+---------+--------+------------------+------------------+-----------------+-------------------+
    |  Label 0 Medians   | 33.5298671875 |  25.0  | 27.9289712981 |  1145.5 | 4280.5 |  0.266054690903  |  0.367429596757  |  0.974697509413 |   0.180107021879  |
    |  Label 1 Medians   |    33.22825   |  26.0  |  24.750600714 |  1298.0 | 3686.0 |  0.348953452046  |  0.417232358098  |  0.960320123324 |   0.183312189403  |
    | Labels Differences |  0.3016171875 |  -1.0  | 3.17837058415 |  -152.5 | 594.5  | -0.0828987611431 | -0.0498027613412 | 0.0143773860882 | -0.00320516752471 |
    +--------------------+---------------+--------+---------------+---------+--------+------------------+------------------+-----------------+-------------------+
    +--------------------+----------------+----------------+---------------+----------------+---------------+------------------+------------------+-----------------+-------------------+
    |                    |      mean      |     median     |     stddev    |    surface     |     volume    |  surf_vol_ratio  |  dissimilarity   |   correlation   |        asm        |
    +--------------------+----------------+----------------+---------------+----------------+---------------+------------------+------------------+-----------------+-------------------+
    |   Label 0 Means    | 33.7726217784  | 25.5077319588  |  28.138551585 | 1149.24226804  | 4327.92783505 |  0.266791467412  |  0.368839864463  |  0.974630250697 |   0.184604439662  |
    |   Label 1 Means    | 33.5801766006  | 26.5365853659  | 24.9198847014 |  1353.5804878  | 3801.54146341 |  0.355117467852  |  0.424858812666  |  0.959243120786 |   0.185896365519  |
    | Labels Differences | 0.192445177741 | -1.02885340709 | 3.21866688365 | -204.338219764 | 526.386371637 | -0.0883260004402 | -0.0560189482027 | 0.0153871299109 | -0.00129192585666 |
    +--------------------+----------------+----------------+---------------+----------------+---------------+------------------+------------------+-----------------+-------------------+
    +--------------------+-----------------+-----------------+----------------+----------------+---------------+------------------+------------------+-------------------+-------------------+
    |                    |       mean      |      median     |     stddev     |    surface     |     volume    |  surf_vol_ratio  |  dissimilarity   |    correlation    |        asm        |
    +--------------------+-----------------+-----------------+----------------+----------------+---------------+------------------+------------------+-------------------+-------------------+
    |  Label 0 Std Dev   |  3.56566947871  |  3.61571484289  | 2.48693787274  | 181.702736923  | 764.311768409 | 0.0136541701933  | 0.0290712896293  |  0.0015917405576  |  0.0376980413245  |
    |  Label 1 Std Dev   |   4.0065203964  |  4.25238848795  | 2.17351617341  | 304.266442303  | 729.664489881 | 0.0308175809048  | 0.0532344943783  |  0.00564634201602 |  0.0400667235716  |
    | Labels Differences | -0.440850917693 | -0.636673645067 | 0.313421699336 | -122.563705381 | 34.6472785281 | -0.0171634107115 | -0.0241632047491 | -0.00405460145842 | -0.00236868224707 |
    +--------------------+-----------------+-----------------+----------------+----------------+---------------+------------------+------------------+-------------------+-------------------+
    """
    r_median0 = [14.7422319315, 10.4442642452, 19.123010876, 321.0, 623.0, 0.540308747856, 0.443541803638, 0.910109991577, 0.208770332886]
    r_mean0 = [16.8036008686, 11.759901739, 19.7299917319, 440.688888889, 891.244444444, 0.552568761085, 0.449349508121, 0.910054738859, 0.22031286031]
    r_std0 = [7.37845725416, 5.78504237684, 5.87692363223, 244.562086843, 656.871276264, 0.129301950581, 0.115485617053, 0.0252230828979, 0.0946236123828]
    r_median1 = [15.1923547764, 9.64817589985, 18.1819843505, 413.0, 783.0, 0.602209944751, 0.460260519395, 0.892966048314, 0.241893132802]
    r_mean1 = [15.3408895241, 10.0117188637, 19.3893984186, 611.2, 1036.33333333, 0.615367393711, 0.456820439769, 0.892641063124, 0.237164536596]
    r_std1 = [6.12458330477, 4.48838945867, 6.00040242802, 446.426283575, 751.267787735, 0.12462061253, 0.139292437778, 0.0242256531482, 0.0959558285797]
    l_median0 = [33.529867187500003, 25.0, 27.92897129811309, 1145.5, 4280.5, 0.26605469090312761, 0.36742959675651976, 0.97469750941250233, 0.18010702187870076]
    l_mean0 = [33.772621778350526, 25.507731958762886, 28.138551585045118, 1149.2422680412371, 4327.927835051546, 0.26679146741223225, 0.36883986446337785, 0.97463025069676656, 0.18460443966222509]
    l_std0 = [3.5656694787107308, 3.6157148428853061, 2.4869378727446954, 181.7027369226046, 764.31176840887565, 0.013654170193297819, 0.02907128962926004, 0.0015917405575988663, 0.03769804132451459]
    l_median1 = [33.228250000000003, 26.0, 24.750600713965479, 1298.0, 3686.0, 0.34895345204623557, 0.41723235809774267, 0.96032012332428518, 0.18331218940341465]
    l_mean1 = [33.580176600609761, 26.536585365853657, 24.919884701394967, 1353.580487804878, 3801.5414634146341, 0.35511746785240444, 0.42485881266603581, 0.95924312078585727, 0.1858963655188865]
    l_std1 = [4.006520396403995, 4.2523884879522429, 2.1735161734088924, 304.26644230338184, 729.66448988078105, 0.030817580904812177, 0.053234494378323852, 0.0056463420160174343, 0.040066723571589462]
    # static_statistics = {"real label 0": r_mean0 + r_median0 + r_std0,
    #                      "real label 1": r_mean1 + r_median1 + r_std1,
    #                      "lumpy label 0": l_mean0 + l_median0 + l_std0,
    #                      "lumpy label 1": l_mean1 + l_median1 + l_std1}
    static_statistics = {"real label 0": r_mean0 + r_median0 + r_std0,
                         "real label 1": r_mean1 + r_median1 + r_std1}
    """
    27 metrics: 9 x 3 (9 are written below, 3 are mean, median and std in this order)
    'mean', 'median', 'stddev', 'surface', 'volume', 'surf_vol_ratio', 'dissimilarity', 'correlation', 'asm',
    """
    comp_metrics = [9, 11, 15]  # median mean, median stddev and median dissimilarity
    statistics_location = "create_datasets/artificial_images/statistics.csv"
    plot_statistics_for_r_c(statistics_location, static_statistics=static_statistics,
                            comparison_metrics=comp_metrics, skip_plots=False)



