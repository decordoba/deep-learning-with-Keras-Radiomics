#!/usr/bin/env python3.5
import argparse
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
It requires a volumes.pkl file
"""


def get_big_mask(pet_image, mask=None, mask_offset=None):
    """
    Generate big mask from contour mask and offset
    Basically, we return the mask but resize it with the size of pet_image (which will be bigger)
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
    return big_mask


def plot_pet_slice(pet_image, center=None, box=None, mask=None, mask_offset=None, label=None,
                   figure=0):
    """
    mask and pet_image should be same size
    box is 2 coordinates, we will cut mask and pet_image there
    if center is None, all slices in the image are shown, else only the center'th slice is shown
    mask_offset is used if the mask and the pet_image are not aligned
    label is window title (I regret this name choice)
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
    center = (0, 0, center) if isinstance(center, int) else center
    while i < boxed_pet_image.shape[2]:
        if center is not None and i != center[2]:
            i += 1
            continue
        # show images
        fig = plt.figure(figure)
        if label is not None:
            fig.canvas.set_window_title(label + " - slice: {}/{}".format(i + 1,
                                                                         boxed_pet_image.shape[2]))
        else:
            fig.canvas.set_window_title("slice: {}/{}".format(i + 1, boxed_pet_image.shape[2]))
        plt.clf()
        plt.pcolormesh(boxed_pet_image[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap)
        print("Slice: {}/{}".format(i + 1, boxed_pet_image.shape[2]))
        if mask is not None:
            input("Press ENTER to reveal contour. ")
            plt.figure(figure)
            plt.pcolormesh(masked_pet_image[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap,
                           rasterized=True, linewidth=0)
        c = input("ENTER=continue, R=repeat, N=see all, P=previous, Q=quit. ")
        if c.startswith("r"):
            i -= 1
        elif c.startswith("p"):
            i -= 2
            if i < -1:
                i = -1
            center = None
        elif c.startswith("n"):
            i = -1
            center = None
        elif c.startswith("q"):
            break
        i += 1
    return big_mask


def cut_image_given_corners(pet_image, box_center, box_size, corners, offset_if_None=0):
    # cut pet_image acording to box, cented in box_center, with size box_size.
    # if any of the corners is outside the created box, increase size of box to meet those limits.
    if box_size[0] is not None:
        x0 = box_center[0] - int(box_size[0] / 2)
        x1 = x0 + box_size[0]
        x0 = min(x0, corners[0][0])
        x1 = max(x1, corners[1][0])
    else:
        x0 = corners[0][0] - offset_if_None
        x1 = corners[1][0] + offset_if_None
    if box_size[1] is not None:
        y0 = box_center[1] - int(box_size[1] / 2)
        y1 = y0 + box_size[1]
        y0 = min(y0, corners[0][1])
        y1 = max(y1, corners[1][1])
    else:
        y0 = corners[0][1] - offset_if_None
        y1 = corners[1][1] + offset_if_None
    if box_size[2] is not None:
        z0 = box_center[2] - int(box_size[2] / 2)
        z1 = z0 + box_size[2]
        z0 = min(z0, corners[0][2])
        z1 = max(z1, corners[1][2])
    else:
        z0 = corners[0][2] - offset_if_None
        z1 = corners[1][2] + offset_if_None
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


def expand_mask(mask):
    # receives a mask and adds 1s to its border (to any position that is at Manhattan distance
    # of offset or less of a 1)
    # This code is a bit embarassing, but it was easier to do it like this than to use recursion
    new_mask = np.zeros(mask.shape)
    for x in range(new_mask.shape[0]):
        for y in range(new_mask.shape[1]):
            for z in range(new_mask.shape[2]):
                if mask[x, y, z] == 1:
                    pos = (x, y, z)
                    new_mask[pos] = 1
                    off = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                    for ox, oy, oz in off:
                        pos1 = (pos[0] + ox, pos[1] + oy, pos[2] + oz)
                        try:
                            new_mask[pos1] = 1
                            for oox, ooy, ooz in off:
                                pos2 = (pos1[0] + oox, pos1[1] + ooy, pos1[2] + ooz)
                                try:
                                    new_mask[pos2] = 1
                                    for ooox, oooy, oooz in off:
                                        pos3 = (pos2[0] + ooox, pos2[1] + oooy, pos2[2] + oooz)
                                        try:
                                            new_mask[pos3] = 1
                                        except IndexError:
                                            pass
                                except IndexError:
                                    pass
                        except IndexError:
                            pass
    return new_mask


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
                   [.025, .05, .025],
                   [.125, .25, .125]],
                  [[.025, .05, .025],
                   [.005,   1, .005],
                   [.025, .05, .025]],
                  [[.125, .25, .125],
                   [.025, .05, .025],
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
        if folder.endswith(data[0]) and label == data[1]:
            return True
    else:
        if folder.endswith(data):
            return True
    return False


def parse_arguments():
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="This code uses the file 'volumes.pkl' created by"
                                     " 'create_tumor_dataset.py' to create five files, where the "
                                     "chosen contour has already been cut: 'dataset.pkl', "
                                     "'dataset_images.pkl', 'dataset_labels.pkl', "
                                     "'dataset_patients.pkl' and 'dataset_masks.pkl'. What "
                                     "the code does it to find the centroid of the contour "
                                     "and create a box of 40x40xNumSlicesInTheTumor representing "
                                     "the tumor of the patient. After that it checks the file "
                                     "'patients.txt' and saves only the patients that have a label"
                                     " assigned to them in such file. It saves images,"
                                     "labels, patients and masks separately.")
    parser.add_argument('-p', '--plot', default=False, action="store_true",
                        help="show figures before saving them")
    parser.add_argument('-f', '--format', default=0, type=int, choices=[0, 1, 2],
                        help="select how to save the volumes: 0=the original volume is unchanged"
                        " (just put into smaller box), 1=the volume is cut exactly by contour, "
                        "2=the volume is cut by contour but adding margin of 3 pixels "
                        "(default: 0)")
    parser.add_argument('-m', '--margin', default=0, type=int,
                        help="number of adjacent slices on top and bottom of volume that don't "
                        "have any contour pixel (default: 0)")
    parser.add_argument('--patients', default=None, type=str,
                        help="enter the list of patients that you want to see and save, separated"
                        "with spaces, and surroud them with ' or \" (i.e. 11111874 or "
                        "'02092013 11110482')")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    path = ""  # "../data/sources/"
    with open(path + 'volumes.pkl', 'rb') as f:
        volumes = pickle.load(f)

    if args.patients is not None:
        tmp_volumes = {}
        for patient in args.patients.split():
            tmp_volumes[patient] = volumes[patient]
        volumes = tmp_volumes

    # 0: original volume is unchanged (just put into smaller box)
    # 1: volume is cut exactly by contour
    # 2: volume is cut by contour but adding margin of 3 pixels
    dataset_format = args.format
    file_suffix = ""
    if dataset_format > 0:
        file_suffix = str(dataset_format)

    # find box where we can fit all tumors
    max_box = [0, 0, 0]
    for patient in volumes:
        if patient == "11101955":  # Skip patients with errors in their images
            continue
        for mtv_volume in volumes[patient][1:]:
            if mtv_volume != ():
                mask, label, box, folder = mtv_volume
                if check_contour_location(patient, folder, label) is False:
                    continue
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
    patient_masks = {}
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
        print("{}/{} Patient {}, dimensions ({}, {}, {})".format(i + 1, len(patients), patient,
                                                                 *image_shape))
        number_images = 0
        # go thorough all contours for this patient in volumes
        for mtv_volume in volumes[patient][1:]:
            if mtv_volume == ():  # skip empty contours
                continue
            mask, label, box, folder = mtv_volume
            # this will return false (and skip it) if we get a contour that has to be ignored.
            # WARNING: this function is custom for this dataset, it will not work for other data
            if check_contour_location(patient, folder, label) is False:
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
                box_image, new_box = cut_image_given_corners(pet_image, centroid, box_size, box,
                                                             offset_if_None=args.margin)
                print("New box:       ", new_box)
                print("New box size:  ", box_image.shape)
            # Calculate max box size
            tumor_box_size = np.array(box[1]) - np.array(box[0])
            for j in range(3):
                max_box[j] = max(max_box[j], tumor_box_size[j])
            big_mask = get_big_mask(box_image, mask, mask_offset=np.array(box[0]) - new_box[0])
            image2 = atenuate_image_from_mask(box_image, big_mask)
            expanded_big_mask = expand_mask(big_mask)
            image5 = atenuate_image_from_mask(box_image, expanded_big_mask)
            if args.plot:
                new_centroid = np.array(centroid) - new_box[0]  # used only for plotting
                plot_pet_slice(box_image, center=new_centroid, box=None,
                               mask=mask, mask_offset=np.array(box[0]) - new_box[0],
                               label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
                # print("Radial attenuation")
                # dist = get_distance_center_corners(centroid, box)
                # image1 = atenuate_image_radially(box_image, new_centroid, dist)
                # plot_pet_slice(image1, center=np.array(centroid) - new_box[0], box=None,
                #                 mask=mask, mask_offset=np.array(box[0]) - new_box[0],
                #                 label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
                print("Clean cut")
                plot_pet_slice(image2, center=np.array(centroid) - new_box[0], box=None,
                               mask=mask, mask_offset=np.array(box[0]) - new_box[0],
                               label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
                print("Clean cut with expanded mask")
                plot_pet_slice(image5, center=np.array(centroid) - new_box[0], box=None,
                               mask=mask, mask_offset=np.array(box[0]) - new_box[0],
                               label="{}, {}, {}".format(patient, folder.split("/")[-1], label))
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
            patient_masks[patient] = big_mask
            if dataset_format == 1:
                dataset[patient] = image2
            elif dataset_format == 2:
                dataset[patient] = image5
            else:
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
    ignored_patients = []
    dataset_patients = []
    dataset_labels = []
    dataset_images = []
    dataset_masks = []
    patients = sorted(dataset.keys())
    i = 1
    for patient in patients:
        if patient in labels:
            dataset_images.append(dataset[patient])
            dataset_labels.append(labels[patient])
            dataset_patients.append(patient)
            dataset_masks.append(patient_masks[patient])
            print("{} Patient {} has label {}".format(i, dataset_patients[-1], dataset_labels[-1]))
            print(dataset[patient].shape)
            i += 1
        else:
            ignored_patients.append(patient)

    print("{} patients ignored".format(len(ignored_patients)))

    print(" ")
    answer = ""
    while len(answer) <= 0 or answer[0].strip().lower() != "y":
        print("Are you sure you want to save? This may overwrite some files.")
        answer = input("Type 'y' to save data or Ctrl-C to abort.\n>> ")

    # Save data
    # dataset_images, dataset_labels and dataset_patients hold in the same order
    # the 3D images, the label (0 or 1) and the patient id
    print("Saving data, this may take a few minutes")
    # Save the volumes
    with open('dataset{}.pkl'.format(file_suffix), 'wb') as f:
        pickle.dump(dataset, f)
    print("Data saved in 'dataset{}.pkl'.".format(file_suffix))

    with open('dataset{}_images.pkl'.format(file_suffix), 'wb') as f:
        pickle.dump(dataset_images, f)
    print("Data saved in 'dataset{}_images.pkl'.".format(file_suffix))

    with open('dataset{}_labels.pkl'.format(file_suffix), 'wb') as f:
        pickle.dump(dataset_labels, f)
    print("Data saved in 'dataset{}_labels.pkl'.".format(file_suffix))

    with open('dataset{}_patients.pkl'.format(file_suffix), 'wb') as f:
        pickle.dump(dataset_patients, f)
    print("Data saved in 'dataset{}_patients.pkl'.".format(file_suffix))

    with open('dataset{}_masks.pkl'.format(file_suffix), 'wb') as f:
        pickle.dump(dataset_masks, f)
    print("Data saved in 'dataset{}_masks.pkl'.".format(file_suffix))
