#!/usr/bin/env python3.5
import argparse
import PIL
import numpy as np
from lumpy_model import get_lumpy_image, generate_mask, plot_slices


def parse_arguments(d, n, dc, p1, p2, r, t):
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Create lumpy image based on lumpy model")
    parser.add_argument('-d', '--dim', default=d, type=int, help="default: {}".format(d),
                        metavar='N')
    parser.add_argument('-dx', '--dim_x', default=None, type=int, help="default: {}".format("dim"),
                        metavar='N')
    parser.add_argument('-dy', '--dim_y', default=None, type=int, help="default: {}".format("dim"),
                        metavar='N')
    parser.add_argument('-n', '--nbar', default=n, type=int, help="default: {}".format(n),
                        metavar='N')
    parser.add_argument('-dc', '--dc_offset', default=dc, type=int, help="default: {}".format(dc),
                        metavar='N')
    parser.add_argument('-p1', '--pars1', default=p1, type=float, help="default: {}".format(p1),
                        metavar='N')
    parser.add_argument('-p2', '--pars2', default=p2, type=float, help="default: {}".format(p2),
                        metavar='N')
    parser.add_argument('-r', '--range', default=r, type=int, help="default: {}".format(r),
                        metavar='N')
    parser.add_argument('-t', '--threshold', default=t, type=float, help="default: {}".format(t),
                        metavar='N')
    parser.add_argument('--discrete', default=False, action="store_true")
    parser.add_argument('--circle_lumps', default=False, action="store_true")
    parser.add_argument('--random', default=False, action="store_true",
                        help="don't use seed for random numbers generator")
    return parser.parse_args()


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
    SIGMA = 0  # For background noise added
    MASK_THRESHOLD = 0.3
    title = "Lumpy image"  # Title for plot
    title2 = "Lumpy image with mask"  # Title for plot

    # Get arguments reveived from command line
    args = parse_arguments(DIM, NBAR, DC, PARS[0], PARS[1], RANGE_VALUES[1], MASK_THRESHOLD)
    if not args.random:
        np.random.seed(123)  # for reproducibility

    # Read arguments passed by user
    DIM = (args.dim if args.dim_x is None else args.dim_x,
           args.dim if args.dim_y is None else args.dim_y)
    NBAR = args.nbar
    DC = args.dc_offset
    LUMP_FUNCTION = "CircLmp" if args.circle_lumps else LUMP_FUNCTION
    PARS = (args.pars1, args.pars2)
    DISCRETE_LUMPS = args.discrete
    RANGE_VALUES = (0, args.range)
    MASK_THRESHOLD = args.threshold
    params = (DIM, NBAR, DC, LUMP_FUNCTION, PARS, DISCRETE_LUMPS, RANGE_VALUES, SIGMA,
              MASK_THRESHOLD)

    # Create lumpy image
    image, lumps, background, lumps_pos = get_lumpy_image(*params, add_noise=False,
                                                          gaussian_probability=False)
    mask = generate_mask(image, params[-1])

    # Plot results
    plot_slices(image, fig_num=0, title=title, max_slices=100)
    plot_slices(image, fig_num=1, title=title2, max_slices=100, mask=mask)
    input("Press ENTER to close all figures and exit.")


if __name__ == "__main__":
    main()
