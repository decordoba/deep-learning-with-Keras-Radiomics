#!/usr/bin/env python3.5
import argparse
import os
import PIL
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter


def lumpy_backround(dim=(64, 64), nbar=200, dc=10, lump_function="GaussLmp", pars=(1, 10),
                    discretize_lumps_positions=False, rng=None, exact=False, rnd_type=1):
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
    # Assume cubic image if dim is an integer
    if isinstance(dim, int):
        dim = (dim, dim, dim)
    dim = np.array(dim)

    # Initialize values that will be returned
    image = dc * np.ones(dim)
    n = np.random.poisson(nbar) if not exact else nbar

    # Create some useful variables
    n_dim = len(dim)
    scales = [np.arange(dim[j]) for j in range(n_dim)]

    # Random position of lump, uniform throughout image
    if rnd_type == 0:
        lumps_pos = np.random.rand(n, n_dim) * dim
    elif rnd_type == 1:
        lumps_pos = np.zeros((n, 0))
        for i, d in enumerate(dim):
            std_dev = d * (0.1 + 0.1 * np.random.rand())
            coords = np.random.normal(loc=int(d / 2), scale=std_dev, size=n)
            lumps_pos = np.column_stack([lumps_pos, coords])
    if discretize_lumps_positions:
        lumps_pos = lumps_pos.astype(int)
    for i in range(n):
        # Set up a grid of points
        coords = np.meshgrid(*[scales[j] - lumps_pos[i, j] for j in range(n_dim)])

        # Generate a lump centered at pos = lumps_pos[i, :]
        squares_sum = np.sum(c ** 2 for c in coords)
        pars0 = np.random.poisson(pars[0]) if not exact else pars[0]
        pars1 = np.random.poisson(pars[1]) if not exact else pars[1]
        pars0 = pars[0] if pars0 == 0 else pars0
        pars1 = pars[1] if pars1 == 0 else pars1
        if lump_function == "GaussLmp":
            lump = pars0 * np.exp(-0.5 * squares_sum / (pars1 ** 2))
        elif lump_function == "CircLmp":
            lump = pars0 * (squares_sum <= (pars1 ** 2))
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


def add_background(image, sigma=5, attenuation=1):
    """Add gaussian background to image."""
    background = gaussian_filter(np.random.normal(size=image.shape), sigma=sigma)
    rng = [image.min(), image.max()]
    background = background * (rng[1] - rng[0]) * attenuation
    return background + image, background


def generate_mask(image, threshold=0.5):
    """Return mask of 0s and 1s: 0s when pixel value <= threshold, 1s when pixel > threshold."""
    min_val = image.min()
    max_val = image.max()
    return (((image - min_val) / (max_val - min_val)) > threshold).astype(int)


def rescale_image(image, rng, convert_to_int=False):
    """Rescale image to go from rng[0] to rng[1]."""
    if isinstance(rng, int):
        rng = (0, rng)
    rng = np.array(rng)
    min_v = image.min()
    max_v = image.max()
    if min_v == max_v:  # Avoid dividing by zero
        image = rng[0] * np.ones(image.shape)
    else:
        if convert_to_int:
            rng[1] += 0.999
        image = (image - min_v) / (max_v - min_v) * (rng[1] - rng[0]) + rng[0]
    if convert_to_int:
        image = image.astype(int)
    return image


def get_lumpy_image(DIM, NBAR, DC, LUMP_FUNCTION, PARS, DISCRETE_LUMPS, RANGE_VALUES, SIGMA,
                    MASK_THRESHOLD):
    """Create lumpy image and add a noisy background to it."""
    image, n, lumps_pos = lumpy_backround(dim=DIM, nbar=NBAR, dc=DC, lump_function=LUMP_FUNCTION,
                                          pars=PARS, discretize_lumps_positions=DISCRETE_LUMPS,
                                          exact=True)

    noisy_image, background = add_background(image, sigma=SIGMA, attenuation=1)
    final_image = rescale_image(noisy_image, rng=RANGE_VALUES, convert_to_int=True)
    return final_image, image, background, lumps_pos


def create_lumps_pos_matrix(lumps_pos, dim=(64, 64), discrete_lumps_positions=False):
    """Create matrix with 1s in all lumps_pos and 0s elsewhere.

    :param dim: Output image dimensions. Can be a 3D tuple, 2D tuple, 1D tuple or an int
                (int will be converted to a cubic image of dimensions (int, int, int))
    :param lumps_pos: Position of every lump in image
    :param discrete_lumps_positions: If True, all positions will be discretized (floored), else,
                                     they can be floats
    :return: matrix with lumps positions
    """
    # Assume cubic image if dim is an integer
    if isinstance(dim, int):
        dim = (dim, dim, dim)

    # Create some useful variables
    n_dim = len(dim)

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
            if n_dim > 1:
                y = pos[1]
                yl_pos = int(y)
                yh_pos = int(y) + 1
                yl = y - yl_pos
                yh = yh_pos - y
                z = pos[2]
            if n_dim > 2:
                zl_pos = int(z)
                zh_pos = int(z) + 1
                zl = z - zl_pos
                zh = zh_pos - z
            if n_dim == 1:
                image[xl_pos] += xh
                if xh_pos < dim[0]:
                    image[xh_pos] += xl
            elif n_dim == 2:
                image[xl_pos, yl_pos] += xh * yh
                if xh_pos < dim[0]:
                    image[xh_pos, yl_pos] += xl * yh
                if yh_pos < dim[1]:
                    image[xl_pos, yh_pos] += xh * yl
                if xh_pos < dim[0] and yh_pos < dim[1]:
                    image[xh_pos, yh_pos] += xl * yl
            elif n_dim == 3:
                if xl_pos >= dim[0] or yl_pos >= dim[1] or zl_pos >= dim[2]:
                    continue
                if xl_pos < 0 or yl_pos < 0 or zl_pos < 0:
                    continue
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


def plot_slices(volume, title=None, fig_num=0, filename=None, show=True, max_slices=None,
                mask=None):
    """Plot all slices of volume in one figure."""
    # Plot epoch history for accuracy and loss
    if filename is None and show:
        plt.ion()
    try:
        num_slices = volume.shape[2]
    except IndexError:
        num_slices = 1
    if max_slices is not None and num_slices > max_slices:
        num_slices = max_slices
    h = int(np.floor(np.sqrt(num_slices)))
    w = int(np.ceil(np.sqrt(num_slices)))
    if w * h < num_slices:
        h += 1
    fig = plt.figure(fig_num, figsize=(1.5 * w, 1.2 * h))
    vmin = np.min(volume)
    vmax = np.max(volume)
    cmap = plt.cm.gray
    if mask is not None and mask.shape == volume.shape:
        masked_volume = np.ma.masked_array(volume, mask)
        cmap.set_bad('r', 1)
    plt.clf()
    for i in range(num_slices):
        subfig = fig.add_subplot(h, w, i + 1)
        if mask is None:
            try:
                subfig.pcolormesh(volume[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap)
            except IndexError:
                subfig.pcolormesh(volume[:, :], vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            try:
                subfig.pcolormesh(masked_volume[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap,
                                  rasterized=True, linewidth=0)
            except IndexError:
                subfig.pcolormesh(masked_volume[:, :], vmin=vmin, vmax=vmax, cmap=cmap,
                                  rasterized=True, linewidth=0)
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


def parse_arguments(d, n, dc, p1, p2, r, t, v):
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Create lumpy image based on lumpy model")
    parser.add_argument('-d', '--dim', default=d, type=int, help="default: {}".format(d),
                        metavar='N')
    parser.add_argument('-dx', '--dim_x', default=None, type=int, help="default: {}".format("dim"),
                        metavar='N')
    parser.add_argument('-dy', '--dim_y', default=None, type=int, help="default: {}".format("dim"),
                        metavar='N')
    parser.add_argument('-dz', '--dim_z', default=None, type=int, help="default: {}".format("dim"),
                        metavar='N')
    parser.add_argument('-n', '--nbar', default=n, type=int, help="default: {}".format(n),
                        metavar='N')
    parser.add_argument('-dc', '--dc_offset', default=dc, type=int, help="default: {}".format(dc),
                        metavar='N')
    parser.add_argument('-p1', '--pars1', default=p1, type=int, help="default: {}".format(p1),
                        metavar='N')
    parser.add_argument('-p2', '--pars2', default=p2, type=int, help="default: {}".format(p2),
                        metavar='N')
    parser.add_argument('-r', '--range', default=r, type=int, help="default: {}".format(r),
                        metavar='N')
    parser.add_argument('-t', '--threshold', default=t, type=float, help="default: {}".format(t),
                        metavar='N')
    parser.add_argument('-r0', default=None, type=float, help="default: {}".format(None),
                        metavar='N')
    parser.add_argument('-r1', default=None, type=float, help="default: {}".format(None),
                        metavar='N')
    parser.add_argument('-c0', default=None, type=int, help="default: {}".format(None),
                        metavar='N')
    parser.add_argument('-c1', default=None, type=int, help="default: {}".format(None),
                        metavar='N')
    parser.add_argument('--discrete', default=False, action="store_true")
    parser.add_argument('--circle_lumps', default=False, action="store_true")
    parser.add_argument('--random', default=False, action="store_true",
                        help="don't use seed for random numbers generator")
    parser.add_argument('--label0', default=False, action="store_true")
    parser.add_argument('--label1', default=False, action="store_true")
    parser.add_argument('--version', default=v, type=int, help="default: {}".format(v))
    parser.add_argument('--save_image', default=False, action="store_true")
    return parser.parse_args()


def get_params_label_0(version=0, discrete_positions=False, c=None, r=None):
    """Return parameters for label 0."""
    if version < 0:  # for version < 0, both label0 and label1 return the same parameters
        return get_params_label_0(0, discrete_positions=discrete_positions)
    if version == 0:
        DIM = 40
        NBAR = 150
        DC = 0
        LUMP_FUNCTION = "GaussLmp"
        PARS = (1, 2.5)
        DISCRETE_LUMPS = discrete_positions
        RANGE_VALUES = (0, 255)
        SIGMA = 4  # Should be the same for label 0 and label 1
        MASK_THRESHOLD = 0.3
    elif version == 1:
        DIM = 40
        NBAR = 150
        DC = 0
        LUMP_FUNCTION = "GaussLmp"
        PARS = (1, 2.5)
        DISCRETE_LUMPS = discrete_positions
        RANGE_VALUES = (0, 255)
        SIGMA = 4  # Should be the same for label 0 and label 1
        MASK_THRESHOLD = 0.3
    elif version == 2:
        DIM = 40
        NBAR = 250
        DC = 0
        LUMP_FUNCTION = "GaussLmp"
        PARS = (1, 2.25)
        DISCRETE_LUMPS = discrete_positions
        RANGE_VALUES = (0, 255)
        SIGMA = 4  # Should be the same for label 0 and label 1
        MASK_THRESHOLD = 0.3
    if c is not None:
        NBAR = c
    if r is not None:
        PARS = (PARS[0], r)
    return DIM, NBAR, DC, LUMP_FUNCTION, PARS, DISCRETE_LUMPS, RANGE_VALUES, SIGMA, MASK_THRESHOLD


def get_params_label_1(version=0, discrete_positions=False, c=None, r=None):
    """Return parameters for label 1."""
    if version < 0:  # for version < 0, both label0 and label1 return the same parameters
        return get_params_label_0(0, discrete_positions=discrete_positions)
    if version == 0:
        DIM = 40
        NBAR = 500
        DC = 0
        LUMP_FUNCTION = "GaussLmp"
        PARS = (1, 1.5)
        DISCRETE_LUMPS = discrete_positions
        RANGE_VALUES = (0, 255)
        SIGMA = 4  # Should be the same for label 0 and label 1
        MASK_THRESHOLD = 0.3
    elif version == 1:
        DIM = 40
        NBAR = 350
        DC = 0
        LUMP_FUNCTION = "GaussLmp"
        PARS = (1, 2)
        DISCRETE_LUMPS = discrete_positions
        RANGE_VALUES = (0, 255)
        SIGMA = 4  # Should be the same for label 0 and label 1
        MASK_THRESHOLD = 0.3
    elif version == 2:
        DIM = 40
        NBAR = 350
        DC = 0
        LUMP_FUNCTION = "GaussLmp"
        PARS = (1, 2)
        DISCRETE_LUMPS = discrete_positions
        RANGE_VALUES = (0, 255)
        SIGMA = 4  # Should be the same for label 0 and label 1
        MASK_THRESHOLD = 0.3
    if c is not None:
        NBAR = c
    if r is not None:
        PARS = (PARS[0], r)
    return DIM, NBAR, DC, LUMP_FUNCTION, PARS, DISCRETE_LUMPS, RANGE_VALUES, SIGMA, MASK_THRESHOLD


def main():
    """Run whole code."""
    # Default arguments
    DIM = 100  # 40  # 5
    NBAR = 200  # 100  # 2
    DC = 0
    LUMP_FUNCTION = "GaussLmp"
    PARS = (1, 5)  # (1, 4)
    DISCRETE_LUMPS = False
    RANGE_VALUES = (0, 255)
    SIGMA = 5
    MASK_THRESHOLD = 0.3
    version = 1  # This makes us use different parameters for label 0 and label 1. Original=0
    title = "Lumpy image"  # Title for plot
    title2 = "Lumpy image with mask"

    # Get arguments reveived from command line
    args = parse_arguments(DIM, NBAR, DC, PARS[0], PARS[1], RANGE_VALUES[1], MASK_THRESHOLD,
                           version)
    if args.save_image:
        try:
            os.mkdir("images")
        except FileExistsError:
            pass  # File exists
    if not args.random:
        np.random.seed(123)  # for reproducibility

    # Consider other arguments or not depending on value of args.label0 and args.label1
    if not args.label0 and not args.label1:
        # Read arguments passed by user
        DIM = (args.dim if args.dim_x is None else args.dim_x,
               args.dim if args.dim_y is None else args.dim_y,
               args.dim if args.dim_z is None else args.dim_z)
        NBAR = args.nbar
        DC = args.dc_offset
        LUMP_FUNCTION = "CircLmp" if args.circle_lumps else LUMP_FUNCTION
        PARS = (args.pars1, args.pars2)
        DISCRETE_LUMPS = args.discrete
        RANGE_VALUES = (0, args.range)
        MASK_THRESHOLD = args.threshold
        params = (DIM, NBAR, DC, LUMP_FUNCTION, PARS, DISCRETE_LUMPS, RANGE_VALUES, SIGMA,
                  MASK_THRESHOLD)
    elif args.label0 and args.label1:
        # Show one example of label 0 and one example of label 1
        if not args.random:
            np.random.seed(123)  # for reproducibility
        # Create lumpy image for label 1 and plot it
        params = get_params_label_1(args.version, r=args.r1, c=args.c1)
        image, lumps, background, lumps_pos = get_lumpy_image(*params)
        if args.save_image:
            for i in range(image.shape[2]):
                img = PIL.Image.fromarray(image[:, :, i].astype(np.uint8))
                img.save("images/label{}_img{:02d}.png".format(1, i))
        mask = generate_mask(image, params[-1])
        plot_slices(image, fig_num=2, title="Lumpy image (label 1)", max_slices=100)
        plot_slices(image, fig_num=3, title="Lumpy image with mask (label 1)", max_slices=100,
                    mask=mask)
        # Set params for label 0
        params = get_params_label_0(args.version, r=args.r0, c=args.c0)
        # Change title of next plot
        title = "Lumpy image (label 0)"
        title2 = "Lumpy image with mask (label 0)"
    elif args.label0:
        # Set params for label 0
        params = get_params_label_0(args.version, r=args.r0, c=args.c0)
    else:
        # Set params for label 1
        params = get_params_label_1(args.version, r=args.r1, c=args.c1)

    # Create lumpy image
    image, lumps, background, lumps_pos = get_lumpy_image(*params)
    if args.save_image:
        label = "?"
        label = "1" if args.label1 else label
        label = "0" if args.label0 else label
        for i in range(image.shape[2]):
            img = PIL.Image.fromarray(image[:, :, i].astype(np.uint8))
            img.save("images/label{}_img{:02d}.png".format(label, i))
    mask = generate_mask(image, params[-1])

    # Plot results
    plot_slices(image, fig_num=0, title=title, max_slices=100)
    plot_slices(image, fig_num=1, title=title2, max_slices=100, mask=mask)
    input("Press ENTER to close all figures and exit.")

    # Profiling (trying to understand what is slower in my function)!
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # with PyCallGraph(output=GraphvizOutput()):
    #     image, n, lumps_pos = lumpy_backround(dim=DIM, nbar=NBAR, dc=DC,
    #                                           lump_function=LUMP_FUNCTION, pars=PARS,
    #                                           discretize_lumps_positions=DISCRETE_LUMPS,
    #                                           rng=RANGE_VALUES)


if __name__ == "__main__":
    main()
