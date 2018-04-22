#!/usr/bin/env python3.5
import argparse
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import imageio
from calculate_dataset_statistics import read_dataset


def plot_volume_in_3D(volume, threshold=0.5, mask=None, fig_num=0, normalize_volume=True,
                      save_location=None, cmap="autumn", relative_colors=True, split_volume=False,
                      offset=None, show=True, axis_labels_off=True, white_background=False,
                      white_background_shrink=False, frame=None):
    """Plot 3D matrix as 3D pixels, show only pixels in mask or with threshold.

    Mask should be a boolean mask, where hidden pixels should have value 0 or False.
    Threshold is used if mask doesn't exist to create a mask: 1s for values >= threshold in volume.
    Normalize_volume makes values in volume go from 0 to 1 (only important if we use threshold).
    Save_location is the name of the gif saved with the volume spinning.
    Relative_colors makes the smallest number (threshold) as 0 for colouring.
    Split_volume splits volume in median to see what is inside.
    Axis_labels_off hides the numbers in the axis.
    White_background removes the grid background behind the tumor.
    White_background_shrink removes the white background, making the gif smaller.
    Frame is a tuple (x, y) of how much image to remove: (0.5, 0.5) cut width and hight to half.
    """
    if show:
        plt.ion()
    volume = np.array(volume)
    minv = volume.min()
    maxv = volume.max()
    # Normalize volume if required
    if normalize_volume:
        volume = (volume - minv) / (maxv - minv)
        minv = 0
        maxv = 1
    # If mask is None, we use threshold to paint or not every pixel
    if mask is None:
        mask = (volume >= threshold)
    else:
        mask = np.array(mask).astype(bool)
    # Create colors mask
    cmap = plt.cm.get_cmap(cmap)
    # Create normalizer that will convert values volume to 0-1 numbers
    if relative_colors:
        minv = (volume + (1 - mask) * maxv).min()
        if offset is None:
            norm = matplotlib.colors.Normalize(vmin=minv)
        else:
            norm = matplotlib.colors.Normalize(vmin=(minv - (maxv - minv) * offset))
    else:
        if offset is None:
            norm = matplotlib.colors.Normalize()
        else:
            norm = matplotlib.colors.Normalize(vmin=(minv - (maxv - minv) * offset))
    # Split volume so we can see inside
    if split_volume:
        median = int(volume.shape[0] / 2)
        separation = int(volume.shape[0] / 4)  # Separation: 25 % side
        v1 = volume[:median, :, :]
        m1 = mask[:median, :, :]
        v2 = volume[median:, :, :]
        m2 = mask[median:, :, :]
        vsep = np.ones((separation, volume.shape[1], volume.shape[2])) * minv
        csep = np.zeros((separation, mask.shape[1], mask.shape[2])) * minv
        volume = np.concatenate((np.concatenate((v1, vsep), axis=0), v2), axis=0)
        mask = np.concatenate((np.concatenate((m1, csep), axis=0), m2), axis=0)
    # Plot in 3D
    fig = plt.figure(fig_num)
    plt.clf()
    ax = fig.gca(projection='3d')
    ax.voxels(mask, facecolors=cmap(norm(volume)), edgecolor='k')
    # Correct axis if necessary
    if axis_labels_off:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    if white_background:
        ax.axis('off')
    if show:
        plt.show()
    if white_background and white_background_shrink:
        min_left_idx = None
        max_right_idx = None
        min_top_idx = None
        max_bottom_idx = None
    # Save figure rotating as gif
    if save_location is not None:
        images = []
        filename = "temporary_image.png"
        for view_angle in range(0, 360, 10):
            ax.view_init(elev=10., azim=view_angle)
            plt.savefig(filename)
            image = imageio.imread(filename)
            if white_background and white_background_shrink:
                left_idx = 0
                right_idx = image.shape[0] - 1
                top_idx = 0
                bottom_idx = image.shape[1] - 1
                tmp = np.unique(image[left_idx, :, :])
                while len(tmp) == 1 and tmp[0] == 255:
                    left_idx += 1
                    tmp = np.unique(image[left_idx, :, :])
                tmp = np.unique(image[right_idx, :, :])
                while len(tmp) == 1 and tmp[0] == 255:
                    right_idx -= 1
                    tmp = np.unique(image[right_idx, :, :])
                tmp = np.unique(image[:, top_idx, :])
                while len(tmp) == 1 and tmp[0] == 255:
                    top_idx += 1
                    tmp = np.unique(image[:, top_idx, :])
                tmp = np.unique(image[:, bottom_idx, :])
                while len(tmp) == 1 and tmp[0] == 255:
                    bottom_idx -= 1
                    tmp = np.unique(image[:, bottom_idx, :])
                try:
                    min_left_idx = np.min((min_left_idx, left_idx))
                    max_right_idx = np.max((max_right_idx, right_idx))
                    min_top_idx = np.min((min_top_idx, top_idx))
                    max_bottom_idx = np.max((max_bottom_idx, bottom_idx))
                except TypeError:
                    min_left_idx = left_idx
                    max_right_idx = right_idx
                    min_top_idx = top_idx
                    max_bottom_idx = bottom_idx
            images.append(image)
        if white_background and white_background_shrink:
            margin = 5
            min_left_idx = max(0, min_left_idx - margin)
            min_top_idx = max(0, min_top_idx - margin)
            max_right_idx += margin + 1
            max_bottom_idx += margin + 1
            for i, image in enumerate(images):
                images[i] = image[min_left_idx:max_right_idx, min_top_idx:max_bottom_idx]
        elif frame is not None:
            old_shape = np.array(images[0].shape[0:2])
            new_shape = np.round(old_shape * frame).astype(int)
            left_shape = ((old_shape - new_shape) / 2).astype(int)
            right_shape = left_shape + new_shape
            for i, image in enumerate(images):
                images[i] = image[left_shape[0]:right_shape[0], left_shape[1]:right_shape[1]]
        gif_filename = "{}.gif".format(save_location.split(".")[0])
        imageio.mimsave(gif_filename, images)
        print("Saved gif '{}'".format(gif_filename))
    # Close figures when done
    if show:
        plt.ioff()
        input("Press ENTER to continue...")
        plt.close("all")

    if save_location is not None:
        return images


def parse_arguments():
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Save all patients in dataset as 3D gifs.")
    parser.add_argument('-s', '--size', default=None, type=int, metavar="N",
                        help="number of patients to save as gifs (default: all)")
    parser.add_argument('-sp', '--skip_patients', default=0, type=int, metavar="N",
                        help="skip this number of patients when (default: 0)")
    parser.add_argument('-d', '--dataset', default="organized", type=str, metavar="PATH",
                        help="location of the dataset inside the ./data folder "
                        "(default: organized)")
    parser.add_argument('-sv', '--split_volume', default=False, action="store_true",
                        help="split volume to see the inside (recommended)")
    parser.add_argument('-wb', '--white_background', default=False, action="store_true",
                        help="remove background grid behind volume")
    parser.add_argument('-al', '--axis_labels', default=False, action="store_true",
                        help="show numbers in axis")
    parser.add_argument('-p', '--plot', default=False, action="store_true",
                        help="plot figures, don't save them")
    parser.add_argument('-saio', '--save_all_in_one', default=False, action="store_true",
                        help="create gif with all gifs for every label")
    parser.add_argument('-sisg', '--save_in_same_gif', default=False, action="store_true",
                        help="save both labels in the same gif (when -saio enabled)")
    parser.add_argument('-usd', '--use_saved_data', default=False, action="store_true",
                        help="get gifs from old 'volume_gifs.pkl' file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not args.use_saved_data:
        images, labels, masks, patients = read_dataset(args.dataset, num_patients_per_label=None)

        folder = "gifs"
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        gifs = [[], []]
        smallest_image_shape = None
        for i, (x, y, m, p) in enumerate(zip(images, labels, masks, patients)):
            # Skip patients if required
            if i < args.skip_patients:
                continue
            if args.size is not None and i >= args.skip_patients + args.size:
                break
            # Format label
            try:
                y = y[1]
            except TypeError:
                pass
            y = int(y)
            # Save or plot 3D volume
            print("{}/{}. Patient: {}, Label: {}".format(i + 1, len(patients), p, y))
            gif_name = "{}/patient_{}_label_{}".format(folder, p, y)
            gif_name = None if args.plot else gif_name  # Don't save if args.plot
            frame = None if args.axis_labels is True else (0.7, 0.7)
            frame = frame if args.white_background is False else (0.5, 0.5)
            gif = plot_volume_in_3D(x, mask=m, split_volume=args.split_volume, show=args.plot,
                                    white_background=args.white_background, frame=frame,
                                    axis_labels_off=not args.axis_labels, save_location=gif_name)
            # Calculate metrics that will be used if args.save_all_in_one is enabled
            gifs[y].append(gif)
            if gifs[y][-1] is not None:
                image_shape = gifs[y][-1][0].shape[0:2]
                try:
                    smallest_image_shape = np.min((image_shape, smallest_image_shape), axis=0)
                except TypeError:
                    smallest_image_shape = image_shape

        with open('volume_gifs.pkl', 'wb') as f:
            pickle.dump([gifs, smallest_image_shape], f)
    else:
        with open('volume_gifs.pkl', 'rb') as f:
            gifs, smallest_image_shape = pickle.load(f)

    # Create gif with all gifs with the same label
    if not args.plot and args.save_all_in_one and smallest_image_shape is not None:
        # Save label0 and label1 in different gifs
        if not args.save_in_same_gif:
            for i in range(len(gifs)):
                if len(gifs[i]) == 45:
                    w = 9
                    h = 5
                elif len(gifs[i]) == 15:
                    w = 3
                    h = 5
                else:
                    w = int(np.floor(np.sqrt(len(gifs[i]))))
                    h = int(np.ceil(np.sqrt(len(gifs[i]))))
                    w = w + 1 if w * h < len(gifs[i]) else w
                big_image_shape = (len(gifs[i][0]), h * smallest_image_shape[0],
                                   w * smallest_image_shape[1], 4)
                images = np.zeros(big_image_shape)
                for j, gif in enumerate(gifs[i]):
                    gif = np.array(gif)
                    x0, y0 = (j % h) * smallest_image_shape[0], (j // h) * smallest_image_shape[1]
                    x1, y1 = x0 + smallest_image_shape[0], y0 + smallest_image_shape[1]
                    xg0 = int((gif.shape[1] - smallest_image_shape[0]) / 2)
                    yg0 = int((gif.shape[2] - smallest_image_shape[1]) / 2)
                    xg1, yg1 = xg0 + smallest_image_shape[0], yg0 + smallest_image_shape[1]
                    images[:, x0:x1, y0:y1, :] = gif[:, xg0:xg1, yg0:yg1, :]
                images = images.astype("uint8")
                gif_filename = "label{}.gif".format(i)
                imageio.mimsave(gif_filename, images)
                print("Saved gif '{}'".format(gif_filename))
        # Save label0 and label1 in same gif
        else:
            num_gifs = 0
            for i in range(len(gifs)):
                num_gifs += len(gifs[i])
            if num_gifs == 60:
                w = 10
                h = 6
            elif num_gifs == 77:
                w = 11
                h = 7
            else:
                w = int(np.floor(np.sqrt(num_gifs)))
                h = int(np.ceil(np.sqrt(num_gifs)))
                w = w + 1 if w * h < num_gifs else w
            big_image_shape = (len(gifs[0][0]), h * smallest_image_shape[0],
                               w * smallest_image_shape[1], 4)
            images = np.zeros(big_image_shape)
            c = 0
            for i in range(len(gifs)):
                for j, gif in enumerate(gifs[i]):
                    gif = np.array(gif)
                    x0 = ((j + c) % h) * smallest_image_shape[0]
                    y0 = ((j + c) // h) * smallest_image_shape[1]
                    x1, y1 = x0 + smallest_image_shape[0], y0 + smallest_image_shape[1]
                    xg0 = int((gif.shape[1] - smallest_image_shape[0]) / 2)
                    yg0 = int((gif.shape[2] - smallest_image_shape[1]) / 2)
                    xg1, yg1 = xg0 + smallest_image_shape[0], yg0 + smallest_image_shape[1]
                    images[:, x0:x1, y0:y1, :] = gif[:, xg0:xg1, yg0:yg1, :]
                c += len(gifs[i])
            images = images.astype("uint8")
            gif_filename = "all_labels.gif".format(i)
            imageio.mimsave(gif_filename, images)
            print("Saved gif '{}'".format(gif_filename))

    # Example of use (no need to open):
    # d = 8
    # example_image = np.arange(d * d * d).reshape(d, d, d)
    # example_mask = np.zeros((d, d, d))
    # example_mask[1:d - 1, 1:d - 1, 1:d - 1] = 1
    # plot_volume_in_3D(example_image, mask=example_mask, split_volume=True, relative_colors=True,
    #                   cmap="autumn", show=False, save_location="example_box1")
    # plot_volume_in_3D(example_image, mask=example_mask, split_volume=True, relative_colors=True,
    #                   cmap="autumn", show=False, save_location="example_box2",
    #                   white_background=True)
    # plot_volume_in_3D(example_image, mask=example_mask, split_volume=True, relative_colors=True,
    #                   cmap="autumn", show=False, save_location="example_box3",
    #                   axis_labels_off=False)
