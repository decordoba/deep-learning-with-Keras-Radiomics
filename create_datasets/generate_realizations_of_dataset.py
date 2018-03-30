#!/usr/bin/env python3.5
import numpy as np
import pickle
from datetime import datetime
from lumpy_model import get_params_label_0, generate_mask, get_lumpy_image
from matplotlib import pyplot as plt


def get_current_time(time=True, date=False, microseconds=False):
    """Return string with current time and date well formatted."""
    now = datetime.now()
    s = ""
    if date:
        s += "{} ".format(now.date())
    if time:
        s += "{:02d}:{:02d}:{:02d}".format(now.hour, now.minute, now.second)
    if microseconds:
        if time:
            s += "."
        s += "{:06d}".format(now.microsecond)
    return s.strip()


def remove_healthy_top_and_bottom_slices(image, mask, margin):
    """Remove top and bottom slices that have all 0s masks, minus margin slices."""
    ones_pos = np.nonzero(mask)
    min_z = max(0, np.min(ones_pos[2]) - margin)
    max_z = min(image.shape[2], np.max(ones_pos[2]) + margin + 1)
    return image[:, :, min_z:max_z], mask[:, :, min_z:max_z]


def generate_data(c, r, dataset_name="lumpy_dataset", show_images=False, pause_images=False,
                  discrete_centers=False, lumps_version=0, num_samples=100,
                  number_first_patient=0, cut_edges_margin=None):
    """Generate num_samples lumpy images for label 0 and 1, save them, and possibly plot them."""
    print("Samples generated for each label: " + str(num_samples))

    # Save or show data
    percent = 5
    split_distance = num_samples * percent // 100
    split_distance = 1 if split_distance < 1 else split_distance
    params0 = get_params_label_0(version=lumps_version, discrete_positions=discrete_centers, c=c,
                                 r=r)
    volumes = []
    labels = []
    patients = []
    masks = []
    patient_counter = number_first_patient
    print("{}. 0% loaded (0/{} samples)".format(get_current_time(), num_samples))
    for i in range(num_samples):
        # Save lumpy images for label 0 and 1
        image0, lumps, background, pos_lumps0 = get_lumpy_image(*params0)
        mask0 = generate_mask(image0, params0[-1])
        if cut_edges_margin is not None:
            image0, mask0 = remove_healthy_top_and_bottom_slices(image0, mask0, cut_edges_margin)
        volumes.append(image0)
        masks.append(mask0)
        labels.append(0)
        patients.append("{:08d}".format(patient_counter))
        patient_counter += 1

        # Create and show plots
        if show_images:
            num0 = image0.shape[2]
            middle0 = int(num0 / 2)
            fig = plt.figure(0)
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(image0[:, :, middle0])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("Label 0 - Slice {}/{}".format(middle0, num0))
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(masks[-2][:, :, middle0])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("Label 0 - Mask Slice {}/{}".format(middle0, num0))
            # If pause images is not set, we will see the images briefly one after another
            if pause_images:
                s = input("Press ENTER to see the next image, or Q (q) to disable pause:  ")
                if len(s) > 0 and s[0].lower() == "q":
                    pause_images = False

        if (i + 1) % split_distance == 0:
            print("{}. {}% loaded ({}/{} samples)".format(get_current_time(),
                                                          (i + 1) * 100 // num_samples,
                                                          i + 1, num_samples))

    if show_images:
        plt.ioff()

    print(" ")
    print("Saving data, this may take a few minutes")
    # Save the volumes
    with open('{}_images.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(volumes, f)
    print("Data saved in '{}_images.pkl'.".format(dataset_name))

    with open('{}_labels.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(labels, f)
    print("Data saved in '{}_labels.pkl'.".format(dataset_name))

    with open('{}_patients.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(patients, f)
    print("Data saved in '{}_patients.pkl'.".format(dataset_name))

    with open('{}_masks.pkl'.format(dataset_name), 'wb') as f:
        pickle.dump(masks, f)
    print("Data saved in '{}_masks.pkl'.".format(dataset_name))


if __name__ == "__main__":
    args = parse_arguments()
    c_min = 50
    c_max = 1000
    c_step = 50
    r_min = 1
    r_max = 3
    r_step = 0.25
    n = 256
    num_comb = int((c_max - c_min) / c_step) * int((r_max - r_min) / r_step)
    print("Number of combinations: {}".format(num_comb))
    i = 0
    for c in range(c_min, c_max, c_step):
        for r in range(r_min, r_max, r_step):
            i += 1
            print("{}/{}. Centers: {}, Radius(stddev): {}".format(i, num_comb, c, r))
            name = "lumpy_model_c{}_r{}_n{}".format(c, r)
            generate_data(c=c, r=r, dataset_name="lumpy_dataset", num_samples=100,
                          show_images=False, pause_images=False,
                          discrete_centers=False, lumps_version=0,
                          number_first_patient=0, cut_edges_margin=None)
