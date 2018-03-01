
import dicom
import nibabel as nib
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt


"""
The goal of this code is to loop through all the patients and show their PET images
(plots will only appear if line ~247, with function 'plot_pet_volume' is not commented)
and their respective MTV shapes. See main (line ~253) to see how this happens. It should
have plenty of comments. Finally the code will save a file volumes.pkl with the full volumes
and MTV shapes. This .pkl file will be read by 'parse_volumes_dataset.py' to generate the
final numpy dataset.
"""


def plot_pet_volume(pet_image, pixel_shape, pixel_spacing, mask=None):
    """
    The transparent option makes all zeros transparent, and all ones red (expects image with only
    1s and 0s)
    """
    # create axis for plotting
    pixel_shape = pet_image.shape
    x = np.arange(0.0, (pixel_shape[1] + 1) * pixel_spacing[0], pixel_spacing[0])
    y = np.arange(0.0, (pixel_shape[0] + 1) * pixel_spacing[1], pixel_spacing[1])
    # z = np.arange(0.0, (pixel_shape[2] + 1) * pixel_spacing[2], pixel_spacing[2])
    if mask is not None:
        masked_pet_image = np.ma.masked_array(pet_image, mask)
    # normalize values
    vmin = np.min(pet_image)
    vmax = np.max(pet_image)
    cmap = plt.cm.gray
    cmap.set_bad('r', 1)
    i = 0
    while i < pet_image.shape[2]:
        # show images
        plt.figure(0)
        plt.clf()
        plt.pcolormesh(x, y, pet_image[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.xlabel('y')
        plt.ylabel('x')
        if mask is not None:
            input("Press ENTER to see reveal contour. ")
            plt.figure(0)
            plt.pcolormesh(x, y, masked_pet_image[:, :, i], vmin=vmin, vmax=vmax, cmap=cmap,
                           rasterized=True, linewidth=0)
        c = input("ENTER=continue, Q=quit, M=median, R=repeat, N=start over. ")
        if c.startswith("q"):
            break
        elif c.startswith("m"):
            i = int(pet_image.shape[2] / 2) - 1
        elif c.startswith("r"):
            i -= 1
        elif c.startswith("n"):
            i = -1
        i += 1


def plot_pet_image(pet_image, yz_slice_pos, xz_slice_pos, xy_slice_pos, pixel_shape,
                   pixel_spacing, mask=None):
    """
    The transparent option makes all zeros transparent, and all ones red (expects image with only
    1s and 0s)
    """
    # create axis for plotting
    x = np.arange(0.0, (pixel_shape[0] + 1) * pixel_spacing[0], pixel_spacing[0])
    y = np.arange(0.0, (pixel_shape[1] + 1) * pixel_spacing[1], pixel_spacing[1])
    z = np.arange(0.0, (pixel_shape[2] + 1) * pixel_spacing[2], pixel_spacing[2])
    if mask is not None:
        pet_image = np.ma.masked_array(pet_image, mask)
    # create slices that will be shown
    yz_slice = pet_image[yz_slice_pos, :, :]
    xz_slice = pet_image[:, xz_slice_pos, :]
    xy_slice = pet_image[:, :, xy_slice_pos]
    vmin = min(np.min(yz_slice), np.min(xz_slice), np.min(xy_slice))
    vmax = max(np.max(yz_slice), np.max(xz_slice), np.max(xy_slice))
    yz_slice = np.rot90(yz_slice)
    xz_slice = np.fliplr(np.rot90(xz_slice))
    # normalize values
    vmin = min(np.min(yz_slice), np.min(xz_slice), np.min(xy_slice))
    vmax = max(np.max(yz_slice), np.max(xz_slice), np.max(xy_slice))
    cmap = plt.cm.gray
    cmap.set_bad('r', 1)
    # show images
    plt.figure(0)
    plt.clf()
    plt.subplot(221)
    plt.pcolormesh(y, z, yz_slice, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.ylabel('z')
    plt.subplot(222)
    plt.pcolormesh(x, z, xz_slice, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel('x')
    plt.subplot(223)
    plt.pcolormesh(x, y, xy_slice, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.subplot(224)
    plt.axis([0, 5, 0, 4.5])
    plt.axis('off')
    plt.text(1, 3, "x: {:.4f}".format(yz_slice_pos * pixel_spacing[0]), fontsize=15)
    plt.text(1, 2, "y: {:.4f}".format(xz_slice_pos * pixel_spacing[1]), fontsize=15)
    plt.text(1, 1, "z: {:.4f}".format(xy_slice_pos * pixel_spacing[2]), fontsize=15)
    return vmin, vmax


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
            centroid = tuple([np.round(c) for c in centroid])
        return centroid
    return None


def get_pet_location(patient, options):
    """
    This function holds the exceptional patients: those that have a weird number of PT folders and
    have to be specified manually
    """
    patient_dictionary = {
        "11111774": "FoR_008/Series_001_PT_001",
        "11102077": "FoR_005/Series_002_PT_001",
        "20100039": "FoR_005/Series_004_PT_001",
        "20100052": "FoR_005/Series_004_PT_001",
        "20090735": "FoR_002/Series_001_PT_001",
        "11112002": "FoR_006/Series_002_PT_001",
        "11110941": "FoR_002/Series_001_PT_001",
        "20092802": "FoR_006/Series_001_PT_001"
    }
    if patient not in patient_dictionary:
        return None
    for op in options:
        if op.endswith(patient_dictionary[patient]):
            return op
    print("Problem found in the dictionary, ignoring patient")
    return None


def get_volumes(patient, pet_folder, struct_folders, number, volumes):
    """
    volumes is where the function writes the volumes found
    it is a dictionary, where keys are the names of the patients, and each value is a list
    where the first element is always the original 3D PET image, and the following are the
    contours of the volumes. Every contour is a dict with 4 fields: a mask (3D map of 1s and 0s),
    the contour label, a range (the 2 3D position of the opposite corners of the tumor box)
    and the folder where the contour was found.
    """
    print("Patient {:02d}: {}".format(number, patient))
    # get all dicom image's paths
    dicom_images = [pet_folder+"/"+f for f in os.listdir(pet_folder) if f.lower().endswith(".dcm")]
    dicom_images.sort()
    # get information from dicom header
    dicom_info = dicom.read_file(dicom_images[0])
    pixel_shape = (int(dicom_info.Rows), int(dicom_info.Columns), int(dicom_info.NumberOfSlices))
    pixel_spacing = (float(dicom_info.PixelSpacing[0]), float(dicom_info.PixelSpacing[1]),
                     float(dicom_info.SliceThickness))
    print(pixel_spacing)
    # create 3D array for pet image
    pet_image = np.zeros(pixel_shape, dtype=dicom_info.pixel_array.dtype)
    for i, dicom_img in enumerate(dicom_images):
        ds = dicom.read_file(dicom_img)
        pet_image[:, :, i] = ds.pixel_array
    # create contours structure
    mtv_variables = []
    for struct_folder in struct_folders:
        # extract contours labels and index from lvol.txt
        lvoltxt_file = struct_folder + "/lvol.txt"
        with open(lvoltxt_file) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if ("mtv" in line.lower() and ("cervix" in line.lower() or "tumor" in line.lower()) and
                    "nodal" not in line.lower() and "nodes" not in line.lower() and
                    "ring" not in line.lower() and "opt" not in line.lower()):
                struct = line.strip().split("|")
                mtv_variables.append((int(struct[0]), struct[-1], struct_folder))
    # return nothing if no mtv contours were found
    if len(mtv_variables) == 0:
        return [], volumes
    # add contours to original image and plot it
    prev_folder = None
    patient_volumes = [pet_image]
    for mtv_idx, mtv_label, mtv_folder in mtv_variables:
        # read and transform data from nii file
        if prev_folder != mtv_folder:
            # only read mtv_folder if it has changed
            nii_obj = nib.load(mtv_folder + "/lvol.nii")
            nii_data = nii_obj.get_data()
            volume = np.zeros(nii_data.shape[:3], dtype=int)
            for i in range(nii_data.shape[-1]):
                volume += nii_data[:, :, :, 0, i] << (8 * i)
            volume = np.swapaxes(volume, 0, 1)
            volume = np.flip(volume, 2)
            print("Structures folder: {}".format(mtv_folder.split("/")[-1]))
        print(mtv_idx, "--", mtv_label.split("/")[-1])
        prev_folder = mtv_folder
        # create 3D matrix with 1s where ROI is and 0s everwhere else
        try:
            tumor_volume = (np.bitwise_and(volume, 2 ** mtv_idx) > 0) * 1
        except TypeError:
            print("Error while reading volume for index: {}, label: {}!".format(mtv_idx,
                                                                                mtv_label))
            patient_volumes.append(())
            continue
        # find bounding box for volume
        mask_range = [[pixel_shape[0], pixel_shape[1], pixel_shape[2]], [-1, -1, -1]]
        tumor_exists = False
        for xx in range(pixel_shape[0]):
            for yy in range(pixel_shape[1]):
                for zz in range(pixel_shape[2]):
                    if tumor_volume[xx, yy, zz]:
                        tumor_exists = True
                        mask_range[0][0] = min(mask_range[0][0], xx)
                        mask_range[0][1] = min(mask_range[0][1], yy)
                        mask_range[0][2] = min(mask_range[0][2], zz)
                        mask_range[1][0] = max(mask_range[1][0], xx)
                        mask_range[1][1] = max(mask_range[1][1], yy)
                        mask_range[1][2] = max(mask_range[1][2], zz)
        # continue if the mask is all 0s
        if not tumor_exists:
            print("Volume not found for index: {}, label: {}!".format(mtv_idx, mtv_label))
            patient_volumes.append(())
            continue
        # Get ROI
        current_volume = pet_image[mask_range[0][0]:mask_range[1][0]+1,
                                   mask_range[0][1]:mask_range[1][1]+1,
                                   mask_range[0][2]:mask_range[1][2]+1]
        current_mask = tumor_volume[mask_range[0][0]:mask_range[1][0]+1,
                                    mask_range[0][1]:mask_range[1][1]+1,
                                    mask_range[0][2]:mask_range[1][2]+1]
        # Add volumes to patient_volumes
        patient_volumes.append((current_mask, mtv_label, mask_range, mtv_folder))
        # Plot volumes
        plot_pet_volume(current_volume, pixel_shape, pixel_spacing, mask=current_mask)
    volumes[patient] = patient_volumes
    return mtv_variables, volumes


if __name__ == "__main__":
    # path for all patients
    root_path = "/home/dani/Documents/disease-detection/Cervical Radiomic Images"

    # get all patients in dataset
    patient_folders = sorted(next(os.walk(root_path))[1])

    # create structure to ignore patients that have an unexpected folder structure
    ignored_patients = {p: False for p in patient_folders}
    num_ignored_patients = 0

    # loop to get PET folders (contain dicom images) and all structure folders (contain nii files)
    pet_folders = {}
    num_pet_folders = 0
    struct_folders = {}
    num_struct_folders = 0
    for patient in patient_folders:
        pet_scans_per_patient = 0
        path = "{}/{}".format(root_path, patient)
        FoR_folders = [f for f in next(os.walk(path))[1] if f.startswith("FoR_")]
        for folder in FoR_folders:
            FoR_path = "{}/{}".format(path, folder)
            PT_folders = [FoR_path + "/" + f for f in next(os.walk(FoR_path))[1] if f.find("PT") > -1]
            num_pet_folders += len(PT_folders)
            pet_scans_per_patient += len(PT_folders)
            if patient not in pet_folders:
                pet_folders[patient] = []
            pet_folders[patient] += PT_folders
        if pet_scans_per_patient != 1:
            location = get_pet_location(patient, pet_folders[patient])
            if location is not None:
                pet_folders[patient] = [location]
                pet_scans_per_patient = 1
        if pet_scans_per_patient != 1:
            num_ignored_patients += 1
            if pet_scans_per_patient == 0:
                print("Patient {} has {} PET images.\nThis patient will be ignored!\n"
                      "".format(patient, pet_scans_per_patient))
                ignored_patients[patient] = "Too few PET images: {}".format(pet_scans_per_patient)
            else:
                print("Patient {} has {} PET images in: \n{}\nThis patient will be ignored!\n"
                      "".format(patient, pet_scans_per_patient, "\n".join(pet_folders[patient])))
                ignored_patients[patient] = "Too many PET images: {}".format(pet_scans_per_patient)
        else:
            path = pet_folders[patient][0]
            s_folders = [path + "/" + f for f in next(os.walk(path))[1] if f.startswith("struct")]
            num_struct_folders += len(s_folders)
            struct_folders[patient] = s_folders

    print("{} patient folders found.".format(len(patient_folders)))
    print("{} PET folders found.".format(num_pet_folders))
    print("{} structures folders found.".format(num_struct_folders))
    print("{} patients ignored.".format(num_ignored_patients))

    # Get all volumes and save them
    plt.ion()
    contour_names = set()
    i = 0
    volumes = {}
    for patient in patient_folders:
        if ignored_patients[patient]:  # skip ignored patients
            continue
        i += 1
        # This function does all the volumes extraction, and also plots the tumors
        mtv_variables, volumes = get_volumes(patient, pet_folders[patient][0],
                                             struct_folders[patient], i, volumes)
        # Track all the names found
        for mtv_idx, mtv_label, mtv_folder in mtv_variables:
            contour_names.add(mtv_label)
        # If no contour is detected, add patient to ignored set
        if len(mtv_variables) == 0 or len(volumes[patient]) <= 1:
            ignored_patients[patient] = "No valid MTV contour found"
            num_ignored_patients += 1
            print("Patient", patient, "has no MTV contour. \nThis patient will be ignored!\n")
            if patient in volumes:
                volumes.pop(patient)
    plt.ioff()

    # Print some statistics and data from the extraction
    print("UNIQUE LABELS:")
    for c in contour_names:
        print(c)
    print(" ")
    print("DATASET STRUCTURE:")
    patients = sorted(volumes.keys())
    for i, patient in enumerate(patients):
        print("Patient {}: {}".format(i, patient))
        contents = [volumes[patient][0]]
        prev_folder = None
        for info in volumes[patient][1:]:
            if len(info) == 0:
                continue
            contents.append(info)
            current_mask, mtv_label, mask_range, mtv_folder = info
            if prev_folder != mtv_folder:
                print("Folder: {}".format(mtv_folder.split("/")[-1]))
            prev_folder = mtv_folder
            print(mtv_label, "  ", mask_range)
        print(" ")
        volumes[patient] = contents
    print("IGNORED PATIENTS:")
    i = 0
    for patient in ignored_patients:
        if ignored_patients[patient] is False:
            continue
        print("Patient {}: {}".format(i, patient))
        print("Reason: {}".format(ignored_patients[patient]))
        i += 1

    # Save the volumes
    print("Saving data, this may take a few minutes")
    with open('volumes.pkl', 'wb') as f:
        pickle.dump(volumes, f)
    print("Data saved in 'volumes.pkl'.")
