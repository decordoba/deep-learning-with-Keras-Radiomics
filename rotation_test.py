import numpy as np
import matplotlib.pylab as plt
from skimage import transform
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage


def rotate_randomly(volume, mask, rotations=None):
    """Rotate volume and mask randomly."""
    if rotations is None:
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
    return rotated_volume, rotated_mask, rotations


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

    Scales must be greater or equal to 1. Volume and mask must have same shape.
    """
    scaled_volumes = []
    scaled_masks = []
    if type(scales) == float:
        scales = (scales, )
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


def plot_volume_in_3d(vol):
    """Docstring for plot_volume_in_3d."""
    tmp = np.zeros(vol.shape)
    for x in range(1, vol.shape[0] - 1):
        for y in range(1, vol.shape[1] - 1):
            for z in range(1, vol.shape[2] - 1):
                if vol[x, y, z] <= 0.5:
                    continue
                if vol[x - 1, y, z] <= 0.5:
                    tmp[x, y, z] = 1
                    continue
                if vol[x + 1, y, z] <= 0.5:
                    tmp[x, y, z] = 1
                    continue
                if vol[x, y + 1, z] <= 0.5:
                    tmp[x, y, z] = 1
                    continue
                if vol[x, y - 1, z] <= 0.5:
                    tmp[x, y, z] = 1
                    continue
                if vol[x, y, z + 1] <= 0.5:
                    tmp[x, y, z] = 1
                    continue
                if vol[x, y, z - 1] <= 0.5:
                    tmp[x, y, z] = 1
                    continue
    fig = plt.figure()
    ax = Axes3D(fig)
    pos = np.nonzero(tmp)
    ax.scatter(pos[0], pos[1], pos[2], s=150, c="r", marker="o")


plt.close("all")
plt.ion()

# Create shape that can be easily recognized
a = np.zeros((21, 21, 21))
a[5:16, 5:16, 5:16] = 1  # this makes a cube
a[8:13, 1:5, 8:10] = 1  # this adds prism
a[3:5, 11:16, 5:16] = 1  # this adds prism
a[8:13, 12:16, 5:13] = 0  # this adds hole in cube

# Even simpler shape
# a = np.zeros((9, 9, 9))
# a[3:6, 3:6, 3:6] = 1
# a[1:3, 4:5, 4:5] = 1

# Plot slices volume
plot_slices_volume(a)

# Plot volume in 3D
plot_volume_in_3d(a)

test = "r"

if test == "r":
    ad, ae, rotations = rotate_randomly(a, a)
    print("Rotations:", rotations)
elif test == "t":
    ad, ae, translations = translate_randomly(a, a, translation=None, max_distance=5)
    print("Translations:", translations)
elif test == "s":
    scales = [0.8, 1, 1.2]
    ad, ae = scale_volume(a, a, scales=[0.8, 1, 1.2])
    print("Scales:", scales)

if test == "r" or test == "t":
    for i, (tmp_x, tmp_m) in enumerate(zip(ad, ae)):
        ad[i] = np.around(tmp_x, decimals=6)
        ae[i] = (tmp_m >= 0.5).astype(int)

if test == "r" or test == "t":
    # Plot slices volume
    plot_slices_volume(ad)
    # Plot volume in 3D
    plot_volume_in_3d(ad)
elif test == "s":
    # Plot slices volume
    plot_slices_volume(ad[0])
    plot_slices_volume(ad[1])
    plot_slices_volume(ad[2])
    # Plot volume in 3D
    plot_volume_in_3d(ad[0])
    plot_volume_in_3d(ad[1])
    plot_volume_in_3d(ad[2])

input("Press ENTER to exit")

# aa = np.rot90(a, axes=(0, 1))
# ab = np.rot90(aa, axes=(1, 2))
# ac = np.rot90(ab, axes=(0, 2))
#
# fig = plt.figure()
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i+1)
#     ax.set_aspect('equal')
#     plt.imshow(aa[:, :, i], interpolation='nearest', cmap="gray")
#
#
# fig = plt.figure()
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i+1)
#     ax.set_aspect('equal')
#     plt.imshow(ab[:, :, i], interpolation='nearest', cmap="gray")
#
# fig = plt.figure()
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i+1)
#     ax.set_aspect('equal')
#     plt.imshow(ac[:, :, i], interpolation='nearest', cmap="gray")
#
# # rotate 45 degrees
# c = transform.rotate(a, 45, order=1)
# d = transform.rotate(a, 45, order=2)
# e = transform.rotate(a, 45, order=3)
# f = transform.rotate(a, 45, order=4)
#
# plt.close("all")
# plt.ion()
# fig = plt.figure()
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i+1)
#     ax.set_aspect('equal')
#     plt.imshow(a[:, :, i], interpolation='nearest', cmap="gray")
#
# fig = plt.figure()
# for i in range(9):
#     ax = fig.add_subplot(3, 3, i+1)
#     ax.set_aspect('equal')
#     plt.imshow(c[:, :, i], interpolation='nearest', cmap="gray")
#
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1,1)
# # ax.set_aspect('equal')
# # plt.imshow(d, interpolation='nearest', cmap="gray")
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1,1)
# # ax.set_aspect('equal')
# # plt.imshow(e, interpolation='nearest', cmap="gray")
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1,1)
# # ax.set_aspect('equal')
# # plt.imshow(f, interpolation='nearest', cmap="gray")
# #
# # # rotate 45 degrees
# # c = imrotate(a, 45, "nearest")
# # d = imrotate(a, 45, "bilinear")
# # e = imrotate(a, 45, "cubic")
# # f = imrotate(a, 45, "bicubic")
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1,1)
# # ax.set_aspect('equal')
# # plt.imshow(d, interpolation='nearest', cmap="gray")
