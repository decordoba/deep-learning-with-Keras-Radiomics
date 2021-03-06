#!/usr/bin/env python3.5
import argparse
import numpy as np
import pickle
from datetime import datetime
from lumpy_model import get_params_label_0, get_params_label_1, generate_mask
from lumpy_model import get_lumpy_image, create_lumps_pos_matrix
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


def generate_data(save_lumps_pos=False, show_images=False, pause_images=False,
                  discrete_centers=False, dataset_name="lumpy_dataset", lumps_version=0,
                  num_samples=100, number_first_patient=0, cut_edges_margin=None):
    """Generate num_samples lumpy images for label 0 and 1, save them, and possibly plot them."""
    print("Samples generated for each label: " + str(num_samples))

    if lumps_version != 0:
        dataset_name += "_v{}".format(lumps_version)

    if show_images:
        plt.ion()

    # Save or show data
    percent = 1
    split_distance = num_samples * percent // 100
    split_distance = 1 if split_distance < 1 else split_distance
    params0 = get_params_label_0(version=lumps_version, discrete_positions=discrete_centers)
    params1 = get_params_label_1(version=lumps_version, discrete_positions=discrete_centers)
    volumes = []
    labels = []
    patients = []
    masks = []
    centers = []
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
        image1, lumps, background, pos_lumps1 = get_lumpy_image(*params1)
        mask1 = generate_mask(image1, params0[-1])
        if cut_edges_margin is not None:
            image1, mask1 = remove_healthy_top_and_bottom_slices(image1, mask1, cut_edges_margin)
        volumes.append(image1)
        masks.append(mask1)
        labels.append(1)
        patients.append("{:08d}".format(patient_counter))
        patient_counter += 1

        # Only create matrix with lumps centers if we are going to save it
        if save_lumps_pos:
            # Save all matrices with lumps centers for label 0 and 1
            pos_matrix0 = create_lumps_pos_matrix(lumps_pos=pos_lumps0, dim=params0[0])
            pos_matrix1 = create_lumps_pos_matrix(lumps_pos=pos_lumps1, dim=params1[0])
            centers.append(pos_matrix0)
            centers.append(pos_matrix1)

        # Create and show plots
        if show_images:
            num0 = image0.shape[2]
            num1 = image1.shape[2]
            middle0 = int(num0 / 2)
            middle1 = int(num1 / 2)
            if save_lumps_pos:
                fig = plt.figure(0)
                ax = fig.add_subplot(2, 3, 1)
                ax.imshow(pos_matrix0[:, :, middle0].T)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Centers Slice {}/{}".format(middle0, num0))
                ax = fig.add_subplot(2, 3, 2)
                ax.imshow(image0[:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Slice {}/{}".format(middle0, num0))
                ax = fig.add_subplot(2, 3, 3)
                ax.imshow(masks[-2][:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Mask Slice {}/{}".format(middle0, num0))
                ax = fig.add_subplot(2, 3, 4)
                ax.imshow(pos_matrix1[:, :, middle1].T)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Centers Slice {}/{}".format(middle1, num1))
                ax = fig.add_subplot(2, 3, 5)
                ax.imshow(image1[:, :, middle1])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Slice {}/{}".format(middle1, num1))
                ax = fig.add_subplot(2, 3, 6)
                ax.imshow(masks[-1][:, :, middle1])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Mask Slice {}/{}".format(middle1, num1))
            else:
                fig = plt.figure(0)
                ax = fig.add_subplot(2, 2, 1)
                ax.imshow(image0[:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Slice {}/{}".format(middle0, num0))
                ax = fig.add_subplot(2, 2, 2)
                ax.imshow(masks[-2][:, :, middle0])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 0 - Mask Slice {}/{}".format(middle0, num0))
                ax = fig.add_subplot(2, 2, 3)
                ax.imshow(image1[:, :, middle1])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Slice {}/{}".format(middle1, num1))
                ax = fig.add_subplot(2, 2, 4)
                ax.imshow(masks[-1][:, :, middle1])
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Label 1 - Mask Slice {}/{}".format(middle1, num1))
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
        elif num_samples > 8192 and i % 64 == 63:
            print("{}. {}% loaded ({}/{} samples)".format(get_current_time(),
                                                          (i + 1) * 100 // num_samples,
                                                          i + 1, num_samples))

    if show_images:
        plt.ioff()

    print(" ")
    print("Saving data, this may take a few minutes")
    # Save the volumes
    margin_suffix = "" if cut_edges_margin is None else "_m{}".format(cut_edges_margin)
    file_suffix = "{}_{}-{}".format(margin_suffix, number_first_patient,
                                    number_first_patient + num_samples)
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
        centers = np.array(centers)
        np.save(dataset_name + "_centers", centers)
        print("Lumps centers saved in '{}.npy'.".format(dataset_name + "_centers"))

    if cut_edges_margin is None:
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
    parser.add_argument('-e', '--edges', default=False, action="store_true",
                        help="remove top and bottom slices with mask of 0s")
    parser.add_argument('-n', '--name', default="lumpy_dataset", type=str,
                        help="dataset name (default: 'lumpy_dataset')")
    parser.add_argument('-s', '--size', default=100, type=int,
                        help="number of samples generated per label (default: 100)")
    parser.add_argument('-f', '--first', default=0, type=int, metavar='NUM',
                        help="number of first patient (default: 0)")
    parser.add_argument('-v', '--version', default=1, type=int, metavar='N',
                        help="version of lumps params for labels 0 & 1 (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    cut_edges_margin = None
    if args.edges:
        cut_edges_margin = 2  # leave 2 slices with 0s mask on top and bottom
    generate_data(save_lumps_pos=args.centers,
                  show_images=args.plot,
                  pause_images=args.wait,
                  discrete_centers=args.discrete,
                  dataset_name=args.name,
                  num_samples=args.size,
                  number_first_patient=args.first,
                  cut_edges_margin=cut_edges_margin,
                  lumps_version=args.version)

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
        number_first_patient: the number of the first patient, i.e. 1234 will become '00001234' for
                              the first patient, '00001235' for the second, etc.
        cut_edges_margin: if None, all slices are preserved, else all top and bottom slices
                          except the cut_edges_margin closer to the volume are removed. The slices
                          removed will have all zeros in their mask.
    """
