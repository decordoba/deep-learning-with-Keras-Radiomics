#!/usr/bin/env python3.5
import argparse
import numpy as np
from skimage import transform
from scipy import ndimage
from matplotlib import pyplot as plt
# import sys
# sys.path.insert(0, '..')
# from calculate_dataset_statistics import read_dataset


def plot_slices_volume(vol, vmin=0, vmax=1):
    """Docstring for plot_slices_volume."""
    w, h = int(np.ceil(np.sqrt(vol.shape[2]))), int(np.floor(np.sqrt(vol.shape[2])))
    h = h + 1 if w * h < vol.shape[2] else h
    fig = plt.figure()
    for i in range(vol.shape[2]):
        ax = fig.add_subplot(w, h, i + 1)
        ax.set_aspect('equal')
        plt.imshow(vol[:, :, i], interpolation='nearest', cmap="gray", vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])


def get_3_medians(volume, mask):
    """Get 3 median slices of volume and mask and return as a 2D image with 3 channels.

    If volume and mask are not perfect cubes, or they have not the same dimensions, there
    will be an error.
    """
    w = volume.shape[0]
    m = int(w / 2)
    volume_medians = np.zeros((w, w, 3))
    volume_medians[:, :, 0] = volume[m, :, :]
    volume_medians[:, :, 1] = volume[:, m, :]
    volume_medians[:, :, 2] = volume[:, :, m]
    mask_medians = np.zeros((w, w, 3), dtype=int)
    mask_medians[:, :, 0] = mask[m, :, :]
    mask_medians[:, :, 1] = mask[:, m, :]
    mask_medians[:, :, 2] = mask[:, :, m]
    return volume_medians, mask_medians


def convert_volumes_to_medians(volumes, masks):
    """Return dataset with 3 medians of all volumes and masks."""
    shape = np.array((len(volumes), ) + volumes[0].shape[:2] + (3, ))
    new_volumes = np.zeros(shape)
    new_masks = np.zeros(shape)
    for i, (v, m) in enumerate(zip(volumes, masks)):
        new_volumes[i, :, :, :], new_masks[i, :, :, :] = get_3_medians(v, m)
    return new_volumes, new_masks


def rotate_randomly(volume, mask, rotation=None):
    """Rotate volume and mask randomly.

    Warning! Won't work well if volume or mask have types int.
    """
    if rotation is None:
        rotation = np.random.random(3) * 360
    rotated_volume = volume
    rotated_mask = mask
    for i, theta in enumerate(rotation):
        rotated_volume = transform.rotate(rotated_volume, theta)
        rotated_mask = transform.rotate(rotated_mask, theta)
        if i == 0:
            rotated_volume = np.rot90(rotated_volume, axes=(1, 2))
            rotated_mask = np.rot90(rotated_mask, axes=(1, 2))
        elif i == 1:
            rotated_volume = np.rot90(rotated_volume, axes=(0, 2))
            rotated_mask = np.rot90(rotated_mask, axes=(0, 2))
    return rotated_volume, rotated_mask, rotation


def translate_randomly(volume, mask, translation=None, max_distance=5):
    """Translate (move) volume and mask randomly.

    Warning! Won't work well if volume or mask have types int.
    """
    if translation is None:
        dist = np.square(np.random.random() * max_distance)
        translation = np.zeros(3)
        pt1 = np.random.random() * dist
        pt2 = np.random.random() * dist
        if pt1 > pt2:
            pt1, pt2 = pt2, pt1
        translation[0] = pt1
        translation[1] = pt2 - pt1
        translation[2] = dist - pt1
        translation = np.sqrt(translation) * ((np.random.randint(0, 2, 3) * 2) - 1)
    translated_volume = volume
    translated_mask = mask
    for i, dist in enumerate(translation):
        minv = int(np.floor(dist))
        maxv = minv + 1
        maxf = dist - minv
        minf = 1 - maxf
        shift_min = np.zeros(3)
        shift_min[i] = minv
        shift_max = np.zeros(3)
        shift_max[i] = maxv
        translated_volume = (ndimage.interpolation.shift(translated_volume, shift_min) * minf +
                             ndimage.interpolation.shift(translated_volume, shift_max) * maxf)
        translated_mask = (ndimage.interpolation.shift(translated_mask, shift_min) * minf +
                           ndimage.interpolation.shift(translated_mask, shift_max) * maxf)
    return translated_volume, translated_mask, translation


def scale_volume(volume, mask, scales=(1, 1.2, 1.4, 1.6)):
    """Scale volume and mask, and cut it to have same shape as original.

    Volume and mask must have same shape.
    Warning! Won't work well if volume or mask have types int.
    """
    if type(scales) == float or type(scales) == int:
        v, m = scale_volume(volume, mask, scales=(scales, ))
        return v[0], m[0]
    scaled_volumes = []
    scaled_masks = []
    for scale in scales:
        scaled_volume = ndimage.zoom(volume, scale)
        scaled_mask = ndimage.zoom(mask, scale)
        ml = ((np.array(scaled_volume.shape) - volume.shape) / 2).astype(int)
        if scale >= 1:
            mr = ml + volume.shape
            scaled_volumes.append(scaled_volume[ml[0]:mr[0], ml[1]:mr[1], ml[2]:mr[2]])
            scaled_masks.append(scaled_mask[ml[0]:mr[0], ml[1]:mr[1], ml[2]:mr[2]])
        else:
            ml = -ml
            mr = ml + scaled_volume.shape
            scaled_volumes.append(np.zeros(volume.shape))
            scaled_volumes[-1][ml[0]:mr[0], ml[1]:mr[1], ml[2]:mr[2]] = scaled_volume
            scaled_masks.append(np.zeros(mask.shape))
            scaled_masks[-1][ml[0]:mr[0], ml[1]:mr[1], ml[2]:mr[2]] = scaled_mask
    return scaled_volumes, scaled_masks


def augment_dataset(volumes, labels, masks, patients, scale_samples=(1, 1.2, 1.4, 1.6),
                    num_rotate_samples=5, num_translate_samples=5, max_distance=4):
    """Augment dataset scaling, translating and rotating."""
    samples_x = []
    samples_y = []
    samples_m = []
    samples_p = []
    if type(scale_samples) == int:
        scale_samples = [1 + 0.2 * i for i in range(scale_samples)]
    for i, (x, y, m, p) in enumerate(zip(volumes, labels, masks, patients)):
        # Make sure we use floats instead of ints (transformations don't work for ints)
        if not np.issubdtype(x[0, 0, 0], np.floating):
            x = x.astype(float)
        if not np.issubdtype(m[0, 0, 0], np.floating):
            m = m.astype(float)
        # Create scaled volumes
        if scale_samples is None or len(scale_samples) == 0:
            augmented_x1, augmented_m1 = x, m
        else:
            augmented_x1, augmented_m1 = scale_volume(x, m, scales=scale_samples)
        # Create rotations of scaled volumes
        rotations = [(0, 0, 0)]  # First rotation will always be original image
        if num_rotate_samples is None or num_rotate_samples < 1:
            augmented_x2, augmented_m2 = augmented_x1, augmented_m1
        else:
            augmented_x2, augmented_m2 = [], []
            for j in range(num_rotate_samples):
                for aug_x, aug_m in zip(augmented_x1, augmented_m1):
                    try:
                        tmp_x, tmp_m, _ = rotate_randomly(aug_x, aug_m, rotation=rotations[j])
                    except IndexError:
                        tmp_x, tmp_m, rot = rotate_randomly(aug_x, aug_m)
                        rotations.append(rot)
                    augmented_x2.append(tmp_x)
                    augmented_m2.append(tmp_m)
        # Create translations of scaled and rotated volumes
        translations = [(0, 0, 0)]  # First translation will always be original
        if num_translate_samples is None or num_translate_samples < 1:
            augmented_x3, augmented_m3 = augmented_x2, augmented_m2
        else:
            augmented_x3, augmented_m3 = [], []
            for j in range(num_translate_samples):
                for aug_x, aug_m in zip(augmented_x2, augmented_m2):
                    try:
                        tmp_x, tmp_m, _ = translate_randomly(aug_x, aug_m,
                                                             translation=translations[j])
                    except IndexError:
                        tmp_x, tmp_m, tra = translate_randomly(aug_x, aug_m,
                                                               max_distance=max_distance)
                        translations.append(tra)
                    augmented_x3.append(tmp_x)
                    augmented_m3.append(tmp_m)
        # Covert mask to 0s and 1s again, limit number of decimals saved
        for j, (tmp_x, tmp_m) in enumerate(zip(augmented_x3, augmented_m3)):
            augmented_x3[j] = np.around(tmp_x, decimals=6)
            augmented_m3[j] = (tmp_m >= 0.5).astype(int)
        samples_x += augmented_x3
        samples_m += augmented_m3
        samples_y += [y] * len(augmented_x3)
        samples_p += [p] * len(augmented_x3)
    return samples_x, samples_y, samples_m, samples_p


def bootstrap_augment_dataset(volumes, labels, masks, patients, num_samples, max_distance=None,
                              max_scale_difference=0.5, balance_labels=False):
    """Augment dataset scaling, translating and rotating, bootstrapping data.

    Expects volumes and masks to be perfect cubes.
    max_scale_difference = how much bigger or smaller the volume can get (0.2 means 20%)
    max_distance=None means the max distance is half the radius of the tumor
    """
    # Get approximation of median radius of tumor and radius of every tumor
    middle = np.array(volumes[0].shape) / 2
    radius_mins = []
    radius_maxs = []
    radius = []
    for i, m in enumerate(masks):
        ones_pos = np.nonzero(m)
        radius_maxs.append(np.max(ones_pos, axis=1) - middle)
        radius_mins.append(middle - np.min(ones_pos, axis=1))
        radius.append(np.mean((radius_maxs[-1] + radius_mins[-1]) / 2))
    max_radius = np.percentile(radius_mins + radius_maxs, 66.6)
    min_radius = np.percentile(radius_mins + radius_maxs, 33.3)
    max_radius_diff = ((1 + max_scale_difference) ** (1 / 3))
    min_radius_diff = ((1 - max_scale_difference) ** (1 / 3))
    # Create structures to balance dataset later if necessary
    if balance_labels:
        idx_ones = np.argwhere(np.array(labels) == 1)
        idx_zeros = np.argwhere(np.array(labels) == 0)
    # Bootstrap augmentation
    if max_distance is not None:
        current_max_distance = max_distance
    samples_x = []
    samples_y = []
    samples_m = []
    samples_p = []
    num_patients = len(patients)
    print_when = int(num_samples / 100)
    for i in range(num_samples):
        # Pick patient with replacement
        if not balance_labels:
            idx = np.random.randint(num_patients)
        else:
            # If balance_labels, make sure 50% of samples are label 0 and 50% are label 1
            if i % 2 == 0:
                idx = np.random.randint(len(idx_zeros))
                idx = idx_zeros[idx][0]
            else:
                idx = np.random.randint(len(idx_ones))
                idx = idx_ones[idx][0]
        # Save patient and label
        samples_p.append(patients[idx])
        samples_y.append(labels[idx])
        # Make sure we use floats instead of ints (transformations don't work for ints)
        x = volumes[idx]
        m = masks[idx]
        if not np.issubdtype(x[0, 0, 0], np.floating):
            x = x.astype(float)
        if not np.issubdtype(m[0, 0, 0], np.floating):
            m = m.astype(float)
        # Get random scale (possible scales depend on the size of the tumor in comprison to others)
        if radius[idx] < min_radius:
            min_scale = 1
            max_scale = max_radius_diff  # median_radius / radius[idx]
        elif radius[idx] > max_radius:
            min_scale = min_radius_diff  # median_radius / radius[idx]
            max_scale = 1
        else:
            min_scale = min_radius_diff  # min_radius / radius[idx]
            max_scale = max_radius_diff  # max_radius / radius[idx]
        scale = np.random.random() * (max_scale - min_scale) + min_scale
        # Scale images
        scaled_x, scaled_m = scale_volume(x, m, scales=scale)
        scaled_x[scaled_x < 0] = 0  # When scaling, some values can go slightly below 0
        # Randomly rotate scaled image
        rotated_x, rotated_m, rot = rotate_randomly(scaled_x, scaled_m)
        # Randomly translate scaled image
        if max_distance is None:
            current_max_distance = radius[idx] * scale / 2  # only get slices close to middle
        translated_x, translated_m, tra = translate_randomly(rotated_x, rotated_m,
                                                             max_distance=current_max_distance)
        # Covert mask to 0s and 1s again, and limit the number of decimals saved
        translated_x = np.around(translated_x, decimals=6)
        translated_m = (translated_m >= 0.5).astype(int)
        # Save new image and mask
        samples_x.append(translated_x)
        samples_m.append(translated_m)
        if (i + 1) % print_when == 0:
            print("{}%. {}/{} samples".format(int(np.round((i + 1) * 100 / num_samples)), i + 1,
                                              num_samples))
    return samples_x, samples_y, samples_m, samples_p


def scale_dataset(volumes, masks, scale=0.5, verbose=False):
    """Scale volumes and masks by a factor."""
    # Get approximation of median radius of tumor and radius of every tumor
    new_volumes = []
    new_masks = []
    num_images = len(volumes)
    for i, (x, m) in enumerate(zip(volumes, masks)):
        # Transform dataset to floats if necessary
        if not np.issubdtype(x[0, 0, 0], np.floating):
            x = x.astype(float)
        if not np.issubdtype(m[0, 0, 0], np.floating):
            m = m.astype(float)
        # Scale image
        scaled_x, scaled_m = scale_volume(x, m, scales=scale)
        scaled_x[scaled_x < 0] = 0  # When scaling, some values can go slightly below 0
        # Covert mask to 0s and 1s again, limit the number of decimals saved, and put in lists
        new_volumes.append(np.around(scaled_x, decimals=6))
        new_masks.append((scaled_m >= 0.5).astype(int))
        if verbose and i % 5 == 4:
            print("{}/{} images and masks scaled".format(i + 1, num_images))
    return new_volumes, new_masks


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
    print("This main does nothing, sorry!")

    # args = parse_arguments()
    #
    # Load dataset
    # volumes, labels, masks, patients = read_dataset(args.dataset, args.size, args.plot_slices,
    #                                                 args.plot)
    # params = augment_dataset(volumes, labels, masks, patients, args.num_samples)
    # volumes, labels, masks, patients, rotations, translations, scales = params
    # volumes, masks = convert_volumes_to_medians(volumes, masks)
