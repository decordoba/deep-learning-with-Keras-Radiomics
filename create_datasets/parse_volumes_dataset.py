#!/usr/bin/env python3.5
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import ndimage


"""
This code should be run after create_tumor_dataset.py
and before save_datasets.py
It creates four files, where the chosen contour has already been cut:
dataset.pkl
dataset_images.pkl
dataset_labels.pkl
dataset_patients.pkl
"""


def plot_pet_slice(pet_image, center=None, box=None, mask=None, mask_offset=None, label=None):
    """
    mask and pent image should be same size,
    box is 2 coordinates, we will cut mask and pet_image there
    """
    big_mask = None
    if mask is not None:
        if mask_offset is not None:
            big_mask = np.zeros(pet_image.shape)
            w, h, d = mask.shape
            for x in range(w):
                for y in range(h):
                    for z in range(d):
                        big_mask[mask_offset[0] + x, mask_offset[1] + y, mask_offset[2] + z] = mask[x, y, z]
            mask = big_mask
        if mask.shape == pet_image.shape:
            masked_pet_image = np.ma.masked_array(pet_image, mask)
        else:
            mask = None
    if box is not None:
        boxed_pet_image = pet_image[box[0][0]:box[1][0] + 1,
                                    box[0][1]:box[1][1] + 1,
                                    box[0][2]:box[1][2] + 1]
        if mask is not None:
            masked_pet_image = masked_pet_image[box[0][0]:box[1][0] + 1,
                                                box[0][1]:box[1][1] + 1,
                                                box[0][2]:box[1][2] + 1]
    else:
        boxed_pet_image = pet_image
    # normalize values
    vmin = np.min(boxed_pet_image)
    vmax = np.max(boxed_pet_image)
    cmap = plt.cm.gray
    cmap.set_bad('r', 1)
    i = 0
    while i < boxed_pet_image.shape[2]:
        if center is not None and i != center[2]:
            i += 1
            continue
        # show images
        fig = plt.figure(0)
        if label is not None:
            fig.canvas.set_window_title(label)
        plt.clf()
        plt.pcolormesh(boxed_pet_image[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap)
        print("Slice: {}/{}".format(i + 1, boxed_pet_image.shape[2]))
        if mask is not None:
            input("Press ENTER to reveal contour. ")
            plt.figure(0)
            plt.pcolormesh(masked_pet_image[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap,
                           rasterized=True, linewidth=0)
        c = input("ENTER=continue, R=repeat, N=see all, Q=quit. ")
        if c.startswith("r"):
            i -= 1
        elif c.startswith("n"):
            i = -1
            center = None
        elif c.startswith("q"):
            break
        i += 1
    return big_mask


def cut_image_given_corners(pet_image, box_center, box_size, corners):
    # cut pet_image acording to box, cented in box_center, with size box_size.
    # if any of the corners is outside the created box, increase size of box to meet those limits.
    if box_size[0] is not None:
        x0 = box_center[0] - int(box_size[0] / 2)
        x1 = x0 + box_size[0]
        x0 = min(x0, corners[0][0])
        x1 = max(x1, corners[1][0])
    else:
        x0 = corners[0][0]
        x1 = corners[1][0]
    if box_size[1] is not None:
        y0 = box_center[1] - int(box_size[1] / 2)
        y1 = y0 + box_size[1]
        y0 = min(y0, corners[0][1])
        y1 = max(y1, corners[1][1])
    else:
        y0 = corners[0][1]
        y1 = corners[1][1]
    if box_size[2] is not None:
        z0 = box_center[2] - int(box_size[2] / 2)
        z1 = z0 + box_size[2]
        z0 = min(z0, corners[0][2])
        z1 = max(z1, corners[1][2])
    else:
        z0 = corners[0][2]
        z1 = corners[1][2]
    return pet_image[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1], ((x0, y0, z0), (x1, y1, z1))


def cut_image(pet_image, box_center, box_size, min_box_size):
    # cut pet_image acording to box, cented in box_center, with size box_size.
    # if any value in box_size is None, use values in min_box_size
    # it is assumed that min_box is smaller or equal to box
    if box_size[0] is not None:
        x0 = box_center[0] - int(box_size[0] / 2)
        x1 = x0 + box_size[0]
    else:
        x0 = box_center[0] - int(min_box_size[0] / 2)
        x1 = x0 + min_box_size[0]
    if box_size[1] is not None:
        y0 = box_center[1] - int(box_size[1] / 2)
        y1 = y0 + box_size[1]
    else:
        y0 = box_center[1] - int(min_box_size[1] / 2)
        y1 = y0 + min_box_size[1]
    if box_size[2] is not None:
        z0 = box_center[2] - int(box_size[2] / 2)
        z1 = z0 + box_size[2]
    else:
        z0 = box_center[2] - int(min_box_size[2] / 2)
        z1 = z0 + min_box_size[2]
    return pet_image[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1], ((x0, y0, z0), (x1, y1, z1))


def find_centroid(image, discretize=False):
    # finds centroid of 2D or 3D image
    if len(image.shape) == 2:
        w, h = image.shape
        cumulative = 0
        centroid = [0, 0]
        for x in range(w):
            for y in range(h):
                centroid[0] += image[x, y] * x
                centroid[1] += image[x, y] * y
                cumulative += image[x, y]
        centroid = centroid[0] / cumulative, centroid[1] / cumulative
        if discretize:
            centroid = tuple([np.round(c) for c in centroid])
        return centroid
    elif len(image.shape) == 3:
        w, h, d = image.shape
        cumulative = 0
        centroid = [0, 0, 0]
        for x in range(w):
            for y in range(h):
                for z in range(d):
                    centroid[0] += image[x, y, z] * x
                    centroid[1] += image[x, y, z] * y
                    centroid[2] += image[x, y, z] * z
                    cumulative += image[x, y, z]
        centroid = centroid[0] / cumulative, centroid[1] / cumulative, centroid[2] / cumulative
        if discretize:
            centroid = list([int(np.round(c)) for c in centroid])
        return centroid
    return None


def get_distance_center_corners(centre, box):
    c = np.array(centre)
    d1 = c - box[0]
    d2 = box[1] - c
    d = []
    dist = 0
    for i in range(len(c)):
        d.append(max(d1[i], d2[i]))
        dist += (d[-1] ** 2)
    return np.sqrt(dist)


def atenuate_image_from_mask(image, mask):
    return image * mask


def atenuate_image_from_soft_mask2(image, mask, width=13):
    dot = np.zeros((width, width, width))
    dot[width // 2, width // 2, width // 2] = 1
    kernel_3d = ndimage.filters.gaussian_filter(dot, sigma=7)
    kernel_3d /= kernel_3d[width // 2, width // 2, width // 2]
    new_mask = ndimage.convolve(mask, kernel_3d, mode='constant', cval=0.0)
    return new_mask * image


def atenuate_image_from_soft_mask(image, mask):
    kernel_3d = [
                  [[.125, .25, .125],
                   [ .25,  .5,  .25],
                   [.125, .25, .125]],
                  [[ .25,  .5,  .25],
                   [  .5,   1,   .5],
                   [ .25,  .5,  .25]],
                  [[.125, .25, .125],
                   [ .25,  .5,  .25],
                   [.125, .25, .125]]
                ]
    new_mask = ndimage.convolve(mask, kernel_3d, mode='constant', cval=0.0)
    return new_mask * image


def atenuate_image_radially(image, centroid, distance, att_constant=100):
    # there will be no attenuation at a distance smaller or equal to distance
    c = np.array(centroid)
    new_image = np.copy(image)
    w, h, d = image.shape
    for x in range(w):
        for y in range(h):
            for z in range(d):
                dist = np.linalg.norm(c - [x, y, z])
                if dist > distance:
                    new_image[x, y, z] = max(0, new_image[x, y, z] - np.abs(dist - distance) * att_constant)
    return new_image


def check_contour_location(patient, folder, label):
    """
    This function holds the exceptional patients: those that have a weird number of structures
    folders and have to be specified manually
    """
    patient_dictionary = {
        "11101495": "FoR_002/Series_002_PT_001/structures_002",
        "11101607": "FoR_004/Series_001_PT_001/structures_002",
        "11101768": "FoR_004/Series_002_PT_001/structures_001",
        "11102431": "FoR_003/Series_001_PT_001/structures_002",
        "11102496": "FoR_002/Series_002_PT_001/structures_001",
        "11110878": "FoR_002/Series_002_PT_001/structures_002",
        "11111174": ("FoR_002/Series_002_PT_001/structures_001", "MTV.cervix PET"),
        "11111206": ("FoR_004/Series_001_PT_001/structures_001", "MTV.CERVIX pet"),
        "11111225": ("FoR_002/Series_002_PT_001/structures_001", "MTV.CERVIX - PET"),
        "11111228": ("FoR_003/Series_002_PT_001/structures_001", "MTV.CERVIX PET"),
        "11111307": "FoR_003/Series_001_PT_001/structures_002",
        "11111774": "FoR_008/Series_001_PT_001/structures_002",
        "11112386": ("FoR_003/Series_001_PT_001/structures_001", "p MTV Cervix"),
        "11120248": "FoR_003/Series_001_PT_001/structures_002",
        "11120585": "FoR_003/Series_001_PT_001/structures_002",
        "11112629": "FoR_003/Series_001_PT_001/structures_001",
        "20090808": "FoR_002/Series_001_PT_001/structures_002",
        "20090855": "FoR_007/Series_001_PT_001/structures_002",
        "20091159": ("FoR_002/Series_002_PT_001/structures_001", "MTV cervix"),
        "20093043": "FoR_008/Series_001_PT_001/structures_002"
    }
    if patient not in patient_dictionary:
        return True
    data = patient_dictionary[patient]
    if isinstance(data, tuple):
        print(label, data[1])
        if folder.endswith(data[0]) and label == data[1]:
            return True
    else:
        if folder.endswith(data):
            return True
    return False


if __name__ == "__main__":
    with open('volumes.pkl', 'rb') as f:
        volumes = pickle.load(f)

    # find box where we can fit all tumors
    max_box = [0, 0, 0]
    for patient in volumes:
        for mtv_volume in volumes[patient][1:]:
            if mtv_volume != ():
                box = mtv_volume[2]
                box_size = np.array(box[1]) - np.array(box[0])
                for i in range(3):
                    max_box[i] = max(max_box[i], box_size[i])
    print("Max box size: {}".format(max_box))
    print(" ")

    centroid_center = True  # False

    max_box = [0, 0, 0]
    # create dataset with 3D cuts of 40x40xN
    box_size = [39, 39, None]
    dataset = {}
    plt.ion()
    patients = sorted(volumes.keys())
    for i, patient in enumerate(patients):
        # skip patients with errors in their images
        if patient == "11101955":
            print(i, "/", len(patients), "Patient {} was skipped".format(patient))
            continue
        # get 3D image of patient (first element in volumes[patient])
        pet_image = volumes[patient][0]
        image_shape = pet_image.shape
        print(i, "/", len(patients), "Patient {}, dimensions ({}, {}, {})".format(patient, *image_shape))
        number_images = 0
        # go thorough all contours for this patient in volumes
        for mtv_volume in volumes[patient][1:]:
            if mtv_volume == ():  # skip empty contours
                continue
            mask, label, box, folder = mtv_volume
            # this function will return false (and skip it) if we get a contour that has to be ignored. 
            # WARNING: this function is custom for this dataset, it will not work for other data
            if check_contour_location(patient, folder, label) == False:
                continue
            number_images += 1
            print("Label:  {}".format(label))
            print("Folder: {}".format(folder))
            if not centroid_center:
                # center box using the middle point of the contour box
                min_box_size = np.array(box[1]) - np.array(box[0])
                centroid = (np.array(box[1]) + np.array(box[0])) / 2
                centroid = [int(b) for b in centroid]
                print("Tumor box:     ", box)
                print("Tumor box size:", list(np.array(box[1]) - np.array(box[0]) + 1))
                box_image, new_box = cut_image(pet_image, centroid, box_size, min_box_size)
                print("New box:       ", new_box)
                print("New box size:  ", box_image.shape)
            else:
                # center box using the centroid of the contour
                centroid = np.array(box[0]) + np.array(find_centroid(mask, discretize=True))
                print("Centroid:      ", centroid)
                print("Tumor box:     ", box)
                print("Tumor box size:", list(np.array(box[1]) - np.array(box[0]) + 1))
                box_image, new_box = cut_image_given_corners(pet_image, centroid, box_size, box)
                print("New box:       ", new_box)
                print("New box size:  ", box_image.shape)
            # Calculate max box size
            tumor_box_size = np.array(box[1]) - np.array(box[0])
            for j in range(3):
                max_box[j] = max(max_box[j], tumor_box_size[j])
            new_centroid = np.array(centroid) - new_box[0]  # used only for plotting
            # big_mask = plot_pet_slice(box_image, center=new_centroid, box=None,
            #                           mask=mask, mask_offset=np.array(box[0]) - new_box[0],
            #                           label="{}, {}, {}".format(patient, folder.split("/")[-1],
            #                                                     label))
            # print("Radial attenuation")
            # dist = get_distance_center_corners(centroid, box)
            # image1 = atenuate_image_radially(box_image, new_centroid, dist)
            # plot_pet_slice(image1, center=np.array(centroid) - new_box[0], box=None,
            #                 mask=mask, mask_offset=np.array(box[0]) - new_box[0],
            #                 label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
            # print("Clean cut")
            # image2 = atenuate_image_from_mask(box_image, big_mask)
            # plot_pet_slice(image2, center=np.array(centroid) - new_box[0], box=None,
            #                 mask=mask, mask_offset=np.array(box[0]) - new_box[0],
            #                 label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
            # print("1 pixel quadratic mask")
            # image3 = atenuate_image_from_soft_mask(box_image, big_mask)
            # plot_pet_slice(image3, center=np.array(centroid) - new_box[0], box=None,
            #                 mask=mask, mask_offset=np.array(box[0]) - new_box[0],
            #                 label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
            # print("Gaussian mask")
            # image4 = atenuate_image_from_soft_mask2(box_image, big_mask)
            # plot_pet_slice(image4, center=np.array(centroid) - new_box[0], box=None,
            #                 mask=mask, mask_offset=np.array(box[0]) - new_box[0],
            #                 label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
        # skip all patients that have more or less than one contour
        if number_images == 1:
            dataset[patient] = box_image
        print(" ")

    print("Max tumor box size: {}".format(np.array(max_box) + 1))
    print(" ")

    # Read labels file 'patient.txt' and remove patients that do not have a label
    labels = {}
    with open('patient.txt', 'r') as f:
        line = f.readline()
        while line:
            patient, label = line.split()
            labels[patient] = int(label)
            line = f.readline()
    print(labels)
    ignored_patients = []
    dataset_patients = []
    dataset_labels = []
    dataset_images = []
    patients = sorted(dataset.keys())
    i = 1
    for patient in patients:
        if patient in labels:
            dataset_images.append(dataset[patient])
            dataset_labels.append(labels[patient])
            dataset_patients.append(patient)
            print("{} Patient {} has label {}".format(i, dataset_patients[-1], dataset_labels[-1]))
            print(dataset[patient].shape)
            i += 1
        else:
            ignored_patients.append(patient)

    print("{} patients ignored".format(len(ignored_patients)))
    
    # Save data: dataset has all data together, and 
    # dataset_images, dataset_labels and dataset_patients hold in the same order
    # the 3D images, the label (0 or 1) and the patient id
    print("Saving data, this may take a few minutes")
    # Save the volumes
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print("Data saved in 'dataset.pkl'.")

    with open('dataset_images.pkl', 'wb') as f:
        pickle.dump(dataset_images, f)
    print("Data saved in 'dataset_images.pkl'.")

    with open('dataset_labels.pkl', 'wb') as f:
        pickle.dump(dataset_labels, f)
    print("Data saved in 'dataset_labels.pkl'.")

    with open('dataset_patients.pkl', 'wb') as f:
        pickle.dump(dataset_patients, f)
    print("Data saved in 'dataset_patients.pkl'.")