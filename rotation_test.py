import numpy as np
import matplotlib.pylab as plt
from skimage import transform
from mpl_toolkits.mplot3d import Axes3D


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


def plot_slices_volume(vol):
    """Docstring for plot_slices_volume."""
    w, h = int(np.ceil(np.sqrt(vol.shape[2]))), int(np.floor(np.sqrt(vol.shape[2])))
    h = h + 1 if w * h < vol.shape[2] else h
    fig = plt.figure()
    for i in range(vol.shape[2]):
        ax = fig.add_subplot(w, h, i + 1)
        ax.set_aspect('equal')
        plt.imshow(vol[:, :, i], interpolation='nearest', cmap="gray")
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

# Plot slices volume
plot_slices_volume(a)

# Plot volume in 3D
plot_volume_in_3d(a)

ad, ae, rotations = rotate_randomly(a, a)
print("Rotations:", rotations)

# Plot slices volume
plot_slices_volume(ad)

# Plot volume in 3D
plot_volume_in_3d(ad)

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
