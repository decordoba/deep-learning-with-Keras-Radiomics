#!/usr/bin/env python3.5
import argparse
import numpy as np
from skimage import transform
sys.path.insert(0, '..')
from calculate_dataset_statistics import read_dataset


def get_3_medians(volume, mask):
    """Get 3 median slices of volume and mask and return as a 2D image with 3 channels.

    If volume and mask are not perfect cubes, there will be an error.
    """
    w = volume.shape[0]
    m = int(w / 2)
    volume_medians = np.zeros((3, w, w))
    volume_medians[0, :, :] = volume[m, :, :]
    volume_medians[1, :, :] = volume[:, m, :]
    volume_medians[2, :, :] = volume[:, :, m]
    w = mask.shape[0]
    m = int(w / 2)
    mask_medians = np.zeros((w, w, 3), dtype=int)
    mask_medians[:, :, 0] = mask[m, :, :]
    mask_medians[:, :, 1] = mask[:, m, :]
    mask_medians[:, :, 2] = mask[:, :, m]
    return volume_medians, mask_medians


def rotate_randomly(volume, mask):
    """Rotate volume and mask randomly."""
    rotations = np.random.random(3) * 360
    rotated_volume = volume
    rotated_mask = mask
    for i, theta in enumerate(rotations):
        rotated_volume = transform.rotate(rotated_volume, theta)
        rotated_mask = transform.rotate(rotated_mask, theta)
        if i == 0:
            rotated_volume = np.rot90(rotated_volume, axes=(1, 2))
            rotated_mask = np.rot90(rotated_mask, axes=(1, 2))
        elif i == 1:
            rotated_volume = np.rot90(rotated_volume, axes=(0, 2))
            rotated_mask = np.rot90(rotated_mask, axes=(0, 2))
    return rotated_volume, rotated_mask


def main(dataset, samples_per_image, size=None, plot_slices=False, plot=False):
    """Sample dataset using 3 medians."""
    # Load dataset
    volumes, labels, masks, patients = read_dataset(dataset, size, plot_slices, plot)

    samples_x = []
    samples_y = []
    samples_m = []
    samples_p = []
    for i, (x, y, m, p) in enumerate(zip(volumes, labels, masks, patients)):
        for j in range(samples_per_image):
            if j == 0:
                x_pt, m_pt = get_3_medians(x, m)
            else:
                x_pt, m_pt = rotate_randomly(x, m)
                x_pt, m_pt = get_3_medians(x_pt, m_pt)
            samples_x.append(x_pt)
            samples_m.append(m_pt)
            samples_y.append(y)
            samples_p.append(p)


def parse_arguments():
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Calculate several statistics from dataset.")
    parser.add_argument('-p', '--plot', default=False, action="store_true",
                        help="show figures before saving them")
    parser.add_argument('-ps', '--plot_slices', default=False, action="store_true",
                        help="show slices of volume in dataset")
    parser.add_argument('-s', '--size', default=None, type=int,
                        help="max number of patients per label (default: all)")
    parser.add_argument('-d', '--dataset', default="organized", type=str,
                        help="location of the dataset (default: organized)")
    parser.add_argument('-ns', '--num_samples', default=1, type=int,
                        help="number of median samples per patient (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.dataset, args.num_samples, args.size, args.plot_slices, args.plot)
