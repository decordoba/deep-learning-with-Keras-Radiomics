#!/usr/bin/env python3.5
import argparse
import numpy as np
from skimage import transform
from scipy import ndimage
# import sys
# sys.path.insert(0, '..')
# from calculate_dataset_statistics import read_dataset


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
    """Rotate volume and mask randomly."""
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
    """Translate (move) volume and mask randomly."""
    if translation is None:
        dist = np.square(np.random.random() * max_distance)
        translation = np.zeros(3)
        translation[0] = np.random.random() * dist
        translation[1] = np.random.random() * (dist - translation[0])
        translation[2] = dist - translation[0] - translation[1]
        translation = np.sqrt(translation) * ((np.random.randint(0, 2, 3) * 2) - 1)
    translated_volume = volume
    translated_mask = mask
    for i, dist in enumerate(translation):
        minv = int(dist)
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


def boostrap_augment_dataset(volumes, labels, masks, patients, num_samples, max_distance=4):
    """Augment dataset scaling, translating and rotating, bootstrapping data."""
    samples_x = []
    samples_y = []
    samples_m = []
    samples_p = []
    num_patients = len(patients)
    for i in range(num_samples):
        # Pick patient with replacement
        idx = np.random.randint(num_patients)
        # Save patient and label
        samples_p.append(patients[idx])
        samples_y.append(labels[idx])
        # Get random scale and scale image
        scale = np.random.random() * 6 - 3
        scaled_x, scaled_m = scale_volume(volumes[idx], masks[idx], scales=scale)
        # Randomly rotate scaled image
        rotated_x, rotated_m, rot = rotate_randomly(scaled_x, scaled_m)
        # Randomly translate scaled image
        translated_x, translated_m, tra = translate_randomly(rotated_x, rotated_m,
                                                             max_distance=max_distance)
        # Covert mask to 0s and 1s again, and limit the number of decimals saved
        translated_x = np.around(translated_x, decimals=6)
        translated_m = (translated_m >= 0.5).astype(int)
        # Save new image and mask
        samples_x.append(translated_x)
        samples_m.append(translated_m)
    return samples_x, samples_y, samples_m, samples_p


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
    # Load dataset
    # volumes, labels, masks, patients = read_dataset(args.dataset, args.size, args.plot_slices,
    #                                                 args.plot)
    # params = augment_dataset(volumes, labels, masks, patients, args.num_samples)
    # volumes, labels, masks, patients, rotations, translations, scales = params
    # volumes, masks = convert_volumes_to_medians(volumes, masks)
