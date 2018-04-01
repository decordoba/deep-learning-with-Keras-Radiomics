#!/usr/bin/env python3.5
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


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


def plot_statistics_for_r_c(statistics_file):
    """Docstring for plot_statistics."""
    with open(statistics_file) as f:
        lines = f.readlines()
    statistics = []
    for i, line in enumerate(lines):
        if i == 0:
            names = line.split(", ")
            if names[-2] != "c" or names[-3] != "r" or names[-1] != "n":
                raise Exception("Unexpected data format. Expected to see n, c and r in -1, -2 and "
                                "-3 positions in '{}'".format(statistics_file))
        else:
            statistics.append([float(x) for x in line.split(", ")])
    print(names)
    statistics = np.array(statistics)
    c = statistics[:, -2]
    r = statistics[:, -3]
    statistics = statistics[:, :-3]
    for i in range(statistics.shape[1]):
        values = statistics[:, i]
        minv = np.min(values)
        maxv = np.max(values)
        plt.figure()
        circles(c, r, (values - minv) / (maxv - minv))
        for j in range(statistics.shape[0]):
            plt.text(c[j], r[j], statistics[i, j], ha='center', va='center')
        input("{}. Press ENTER to continue".format(i))
