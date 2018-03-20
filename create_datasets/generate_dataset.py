#!/usr/bin/env python3.5
import argparse
import numpy as np
import pickle
from datetime import datetime
from lumpy_model import get_params_label_0, get_params_label_1, generate_mask
from lumpy_model import get_lumpy_image, create_lumps_pos_matrix
from matplotlib import pyplot as plt


def get_current_time(time=True, date=False):
    """Return string with current time and date well formatted."""
    now = datetime.now()
    s = ""
    if date:
        s += "{} ".format(now.date())
    if time:
        s += "{:02d}:{:02d}:{:02d}".format(now.hour, now.minute, now.second)
    return s.strip()


def generate_data(save_lumps_pos=False, show_images=False, pause_images=False,
                  discrete_centers=False, dataset_name="lumpy_dataset",
                  num_samples=100):
    """Generate num_samples lumpy images for label 0 and 1, save them, and possibly plot them."""
    print("Samples generated for each label: " + str(num_samples))

    if show_images:
        plt.ion()

    # Save or show data
    percent = 1
    split_distance = num_samples * percent // 100
    split_distance = 1 if split_distance < 1 else split_distance
    params0 = get_params_label_0(discrete_centers)
    middle0 = int(params0[0] / 2) if isinstance(params0[0], int) else int(params0[0][2] / 2)
    params1 = get_params_label_1(discrete_centers)
    middle1 = int(params1[0] / 2) if isinstance(params1[0], int) else int(params1[0][2] / 2)
    volumes = []
    labels = []
    patients = []
    masks = []
    patient_counter = 0
    for i in range(num_samples):
        # Save lumpy images for label 0 and 1
        image0, lumps, background, pos_lumps0 = get_lumpy_image(*params0)
        volumes.append(image0)
        labels.append(0)
        patients.append("{:08d}".format(patient_counter))
        patient_counter += 1
        masks.append(generate_mask(image0, params0[-1]))
        image1, lumps, background, pos_lumps1 = get_lumpy_image(*params1)
        volumes.append(image1)
        labels.append(1)
        patients.append("{:08d}".format(patient_counter))
        patient_counter += 1
        masks.append(generate_mask(image1, params1[-1]))

        # Only create matrix with lumps centers if we are going to save it
        if save_lumps_pos:
            # Save all matrices with lumps centers for label 0 and 1
            pos_matrix0 = create_lumps_pos_matrix(lumps_pos=pos_lumps0, dim=params0[0])
            pos_matrix1 = create_lumps_pos_matrix(lumps_pos=pos_lumps1, dim=params1[0])
            try:
                centers = np.concatenate((centers, [pos_matrix0]))
            except NameError:
                centers = np.array([pos_matrix0])
            centers = np.concatenate((centers, [pos_matrix1]))

        # Create and show plots
        if show_images:
            if save_lumps_pos:
                fig = plt.figure(0)
                ax = fig.add_subplot(2, 3, 1)
                ax.imshow(pos_matrix0[:, :, middle0].T)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Centers Middle Slice")
                ax = fig.add_subplot(2, 3, 2)
                ax.imshow(image0[:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Middle Slice")
                ax = fig.add_subplot(2, 3, 3)
                ax.imshow(masks[-2][:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Mask Middle Slice")
                ax = fig.add_subplot(2, 3, 4)
                ax.imshow(pos_matrix1[:, :, middle1].T)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Centers Middle Slice")
                ax = fig.add_subplot(2, 3, 5)
                ax.imshow(image1[:, :, middle1])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Middle Slice")
                ax = fig.add_subplot(2, 3, 6)
                ax.imshow(masks[-1][:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Mask Middle Slice")
            else:
                fig = plt.figure(0)
                ax = fig.add_subplot(2, 2, 1)
                ax.imshow(image0[:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Middle Slice")
                ax = fig.add_subplot(2, 2, 2)
                ax.imshow(masks[-2][:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Mask Middle Slice")
                ax = fig.add_subplot(2, 2, 3)
                ax.imshow(image1[:, :, middle1])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Middle Slice")
                ax = fig.add_subplot(2, 2, 4)
                ax.imshow(masks[-1][:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Mask Middle Slice")
            plt.pause(0.00001)
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
    print("Dataset shape:    {}".format(np.array(volumes).shape))
    print("Dataset range:    {} - {}".format(np.array(volumes).min(), np.array(volumes).max()))
    print("Dataset median:   {}".format(np.median(volumes)))
    print("Dataset mean:     {}".format(np.mean(volumes)))
    print("Dataset std dev:  {}".format(np.std(volumes)))
    print("Labels shape:     {}".format(np.array(labels).shape))
    print("Labels available: {}".format(np.array(labels).shape))
    print("Patients range:   {} - {}".format(patients[0], patients[-1]))
    print("Masks shape:      {}".format(np.array(masks).shape))
    print("Masks range:      {} - {}".format(np.array(masks).min(), np.array(masks).max()))
    print("Masks median:     {}".format(np.median(masks)))
    print("Masks mean:       {}".format(np.mean(masks)))
    print("Masks std dev:    {}".format(np.std(masks)))
    print(" ")

    print("Saving data, this may take a few minutes")
    # Save the volumes
    file_suffix = ""
    with open('{}{}_images.pkl'.format(dataset_name, file_suffix), 'wb') as f:
        pickle.dump(volumes, f)
    print("Data saved in '{}{}_images.pkl'.".format(dataset_name, file_suffix))

    with open('{}{}_labels.pkl'.format(dataset_name, file_suffix), 'wb') as f:
        pickle.dump(labels, f)
    print("Data saved in '{}{}_labels.pkl'.".format(dataset_name, file_suffix))

    with open('{}{}_patients.pkl'.format(dataset_name, file_suffix), 'wb') as f:
        pickle.dump(patients, f)
    print("Data saved in '{}{}_patients.pkl'.".format(dataset_name, file_suffix))

    with open('{}{}_masks.pkl'.format(dataset_name, file_suffix), 'wb') as f:
        pickle.dump(masks, f)
    print("Data saved in '{}{}_masks.pkl'.".format(dataset_name, file_suffix))

    if save_lumps_pos:
        np.save(dataset_name + "_centers", centers)
        print("Lumps centers saved in '{}.npy'.".format(dataset_name + "_centers"))


def parse_arguments():
    """Parse arguments in code."""
    parser = argparse.ArgumentParser(description="Create dataset made of lumpy images")
    parser.add_argument('-c', '--centers', default=False, action="store_true",
                        help="store lumps centers")
    parser.add_argument('-p', '--plot', default=False, action="store_true", help="plot images")
    parser.add_argument('-w', '--wait', default=False, action="store_true",
                        help="pause when plotting")
    parser.add_argument('-d', '--discrete', default=False, action="store_true",
                        help="use only integer positions")
    parser.add_argument('-n', '--name', default="lumpy_dataset", type=str, help="dataset name")
    parser.add_argument('-s', '--size', default=100, type=int,
                        help="number of samples generated per label (default: 100)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    generate_data(save_lumps_pos=args.centers,
                  show_images=args.plot,
                  pause_images=args.wait,
                  discrete_centers=args.discrete,
                  dataset_name=args.name,
                  num_samples=args.size)

    """
    HOW TO USE:
        save_lumps_pos: True, a lumps_position matrix is saved, False, only the generated lumpy
                        image is saved
        show_images: True, shows images of lumpy image (and lumps_position matrix), False, doesn't
        pause_images: True, pauses in every image, you have to type ENTER to proceed, False doesn't
        discrete_centers: only use discrete values (ints) for the centers of the lumps, False
                          the center of the lump is shared by the pixels depending on distance to
                          every adjacent pixel
        dataset_name: string with name of dataset
        num_samples: number of samples saved for every label (if num_samples is 100, 100 samples
                     will be saved with label 0 and 100 will be saved with label 1: 200 in total)
    """
