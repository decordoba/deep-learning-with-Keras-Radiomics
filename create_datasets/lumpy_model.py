#!/usr/bin/env python3.5
import argparse
import numpy as np
from matplotlib import pyplot as plt


def lumpy_backround(dim=(64, 64), nbar=200, dc=10, lump_function="GaussLmp", pars=(1, 10),
                    discretize_lumps_positions=False, rng=None):
    """Generate 2D or 3D (or 1D) matrix with lumpy model.

    : param dim: Output image dimensions. Can be a 3D tuple, 2D tuple, 1D tuple or an int
                 (int will be converted to a cubic image of dimensions (int, int, int))
    : param nbar: Mean number of lumps in the image
    : param dc: DC offset of output image
    : param lump_function: Either 'GaussLmp' or 'CircLmp', for Gaussian or Circular lumps
    : param pars: (magnitude, stddev) for 'GaussLmp'
                  (magnitude, radius) for 'CircLmp'
    : param discretize_lumps_positions: If True, all positions are ints, else, they can be floats
    : return: (image, n, lumps_pos)
               image: Generated image with lumps
               N: Number of lumps
               lumps_pos: Position of every lump in image
    """
    # Assume square image if dim is an integer
    if isinstance(dim, int):
        dim = (dim, dim, dim)

    # Initialize values that will be returned
    image = dc * np.ones(dim)
    n = np.random.poisson(nbar)
    lumps_pos = []

    for i in range(n):
        # Random position of lump, uniform throughout image
        if discretize_lumps_positions:
            pos = [int(np.random.rand() * d) for d in dim]
        else:
            pos = [np.random.rand() * d for d in dim]
        pos = tuple(pos)
        lumps_pos.append(pos)

        # Set up a grid of points
        coords = np.meshgrid(*[np.array(range(dim[i])) - pos[i] for i in range(len(dim))])

        # Generate a lump centered at pos
        squares_sum = np.sum(c ** 2 for c in coords)
        if lump_function == "GaussLmp":
            lump = pars[0] * np.exp(-0.5 * squares_sum / (pars[1] ** 2))
        elif lump_function == "CircLmp":
            lump = pars[0] * (squares_sum <= (pars[1] ** 2))
        else:
            raise Exception("Unknown lump function '{}'".format(lump_function))

        # Add lump to the image
        image = image + lump

    # Rescale image to range rng
    if rng is not None:
        # If range is int, assume rng comes from 0
        if isinstance(rng, int):
            rng = (0, rng)
        min_v = image.min()
        max_v = image.max()
        if min_v == max_v:  # Avoid dividing by zero
            image = rng[0] * np.ones(dim)
        else:
            image = (image - min_v) / (max_v - min_v) * (rng[1] - rng[0]) + rng[0]

    return image, n, lumps_pos


def create_lumps_pos_matrix(lumps_pos, dim=(64, 64), discrete_lumps_positions=False):
    """Create matrix with 1s in all lumps_pos and 0s elsewhere.

    :param dim: Output image dimensions. Can be a 3D tuple, 2D tuple, 1D tuple or an int
                (int will be converted to a cubic image of dimensions (int, int, int))
    :param lumps_pos: Position of every lump in image
    :param discrete_lumps_positions: If True, all positions will be discretized (floored), else,
                                     they can be floats
    :return: matrix with lumps positions
    """
    # Assume square image if dim is an integer
    if isinstance(dim, int):
        dim = (dim, dim, dim)

    # Put a 1 in the matrix in all the lump positions.
    # If the position is not discrete, split this 1 among the discrete positions in image
    image = np.zeros(dim)
    for pos in lumps_pos:
        if discrete_lumps_positions:
            image[tuple([int(p) for p in pos])] += 1
        else:
            x = pos[0]
            xl_pos = int(x)
            xh_pos = int(x) + 1
            xl = x - xl_pos
            xh = xh_pos - x
            if len(dim) > 1:
                y = pos[1]
                yl_pos = int(y)
                yh_pos = int(y) + 1
                yl = y - yl_pos
                yh = yh_pos - y
                z = pos[2]
            if len(dim) > 2:
                zl_pos = int(z)
                zh_pos = int(z) + 1
                zl = z - zl_pos
                zh = zh_pos - z
            if len(dim) == 1:
                image[xl_pos] += xh
                if xh_pos < dim[0]:
                    image[xh_pos] += xl
            elif len(dim) == 2:
                image[xl_pos, yl_pos] += xh * yh
                if xh_pos < dim[0]:
                    image[xh_pos, yl_pos] += xl * yh
                if yh_pos < dim[1]:
                    image[xl_pos, yh_pos] += xh * yl
                if xh_pos < dim[0] and yh_pos < dim[1]:
                    image[xh_pos, yh_pos] += xl * yl
            elif len(dim) == 3:
                image[xl_pos, yl_pos, zl_pos] += xh * yh * zh
                if xh_pos < dim[0]:
                    image[xh_pos, yl_pos, zl_pos] += xl * yh * zh
                if yh_pos < dim[1]:
                    image[xl_pos, yh_pos, zl_pos] += xh * yl * zh
                if zh_pos < dim[2]:
                    image[xl_pos, yl_pos, zh_pos] += xh * yh * zl
                if xh_pos < dim[0] and yh_pos < dim[1]:
                    image[xh_pos, yh_pos, zl_pos] += xl * yl * zh
                if yh_pos < dim[1] and zh_pos < dim[2]:
                    image[xl_pos, yh_pos, zh_pos] += xh * yl * zl
                if zh_pos < dim[2] and xh_pos < dim[0]:
                    image[xh_pos, yl_pos, zh_pos] += xl * yh * zl
                if xh_pos < dim[0] and yh_pos < dim[1] and zh_pos < dim[2]:
                    image[xh_pos, yh_pos, zh_pos] += xl * yl * zl

    return image


def plot_slices(volume, title=None, fig_num=0, filename=None, show=True):
    """Plot all slices of volume in one figure."""
    # Plot epoch history for accuracy and loss
    if filename is None and show:
        plt.ion()
    try:
        num_curves = volume.shape[2]
    except IndexError:
        num_curves = 1
    h = int(np.floor(np.sqrt(num_curves)))
    w = int(np.ceil(np.sqrt(num_curves)))
    if w * h < num_curves:
        h += 1
    fig = plt.figure(fig_num, figsize=(1.5 * w, 1.2 * h))
    vmin = np.min(volume)
    vmax = np.max(volume)
    cmap = plt.cm.gray
    plt.clf()
    for i in range(num_curves):
        subfig = fig.add_subplot(h, w, i + 1)
        try:
            subfig.pcolormesh(volume[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap)
        except IndexError:
            subfig.pcolormesh(volume[:, :], vmin=vmin, vmax=vmax, cmap=cmap)
        subfig.axis('equal')
        subfig.axis('off')
    if title is not None:
        # fig.suptitle(title)
        fig.canvas.set_window_title("Figure {} - {}".format(fig_num, title))
    if filename is None:
        if show:
            plt.show()
            plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()


def parse_arguments(d, n, dc, p1, p2, r):
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Create lumpy image based on lumpy model")
    parser.add_argument('-d', '--dim', default=d, type=int, help="default: {}".format(d))
    parser.add_argument('-n', '--nbar', default=n, type=int, help="default: {}".format(n))
    parser.add_argument('-dc', '--dc', default=dc, type=int, help="default: {}".format(dc))
    parser.add_argument('-p1', '--pars1', default=p1, type=int, help="default: {}".format(p1))
    parser.add_argument('-p2', '--pars2', default=p2, type=int, help="default: {}".format(p2))
    parser.add_argument('-r', '--range', default=r, type=int, help="default: {}".format(r))
    parser.add_argument('--discrete', default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Default arguments
    DIM = 40  # 5
    NBAR = 100  # 2
    DC = 0
    LUMP_FUNCTION = "GaussLmp"
    PARS = (1, 4)
    DISCRETE_LUMPS = False
    RANGE_VALUES = (0, 255)

    # Get arguments reveived from command line
    args = parse_arguments(DIM, NBAR, DC, PARS[0], PARS[1], RANGE_VALUES[1])
    DIM = args.dim
    NBAR = args.nbar
    DC = args.dc
    PARS = (args.pars1, args.pars2)
    DISCRETE_LUMPS = args.discrete
    RANGE_VALUES = (0, args.range)

    np.random.seed(123)  # for reproducibility

    image, n, lumps_pos = lumpy_backround(dim=DIM, nbar=NBAR, dc=DC, lump_function=LUMP_FUNCTION,
                                          pars=PARS, discretize_lumps_positions=DISCRETE_LUMPS,
                                          rng=RANGE_VALUES)

    image_pos = create_lumps_pos_matrix(dim=DIM, lumps_pos=lumps_pos)

    print("Image:\n{}".format(image))
    print("Position matrix:\n{}".format(image_pos))
    print("Number of lumps:", n)
    print("Lumps position:\n{}".format(np.array(lumps_pos)))

    plot_slices(image, title="Lumpy image")
    plot_slices(image_pos, fig_num=1, title="Lumpy centers")
    input("Press ENTER to close all figures and exit.")
