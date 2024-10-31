import numpy as np
import os
import scipy.io
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def interpolate_data(original_data_dict, target_timestamps):
    """
    Interpolate the original data to match the target timestamps.
    """

    # goes throught the elements, not the timesteps, of the actual data (i.e. goes through each of 4 elems of orientation)
    for key in original_data_dict.keys():
        interpolated_positions = np.zeros((len(target_timestamps), original_data_dict[key]['positions'].shape[1]))
        interpolated_orientations = np.zeros((len(target_timestamps), original_data_dict[key]['orientations'].shape[1]))
        interpolated_data = {key: {} for key in original_data_dict.keys()}

        for i in range(original_data_dict[key]['positions'].shape[1]):
            interp_func = interp1d(original_data_dict[key]['timestamps'], original_data_dict[key]['positions'][:, i], kind='linear', fill_value="extrapolate")
            interpolated_positions[:, i] = interp_func(target_timestamps)

        for i in range(original_data_dict[key]['orientations'].shape[1]):
            interp_func = interp1d(original_data_dict[key]['timestamps'], original_data_dict[key]['orientations'][:, i], kind='linear', fill_value="extrapolate")
            interpolated_orientations[:, i] = interp_func(target_timestamps)

        interpolated_data[key]['positions'] = np.array(interpolated_positions)
        interpolated_data[key]['orientations'] = np.array(interpolated_orientations)
        interpolated_data[key]['timestamps'] = target_timestamps

    return interpolated_data

def save_data(file_path, interpolated_data):
    """
    Save the interpolated data to a CSV file.
    """
    scipy.io.savemat(file_path, interpolated_data)

def main(data_location):

    # Get paths to data
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # follower_log_file_path = os.path.join(script_directory, data_location + "/follower/data.log")
    vive_data_file = os.path.join(script_directory, data_location + "vive/parsed_vive_data.mat")
    #TODO take only from one ifeel file?
    ifeel_data_file = os.path.join(script_directory, data_location + "leader/parsed_ifeel_data.mat")
    # leader_mat_file_path = os.path.join(script_directory, data_location + "/leader/parsed_ifeel_data.mat")
    output_file = os.path.join(script_directory, data_location + "interpolated_vive_data.mat")

    # Start all the data at the same time so that it lines up
    start_ind_dict = {"forward_backward": [3100, 2049, 48415], "left_right": [10275, 4539, 2411]}
    if "forward_backward" in vive_data_file:
        start_indices = start_ind_dict["forward_backward"] #iFeel leader, iFeel follower, Vive
    elif "left_right" in vive_data_file:
        start_indices = start_ind_dict["left_right"]
    else:
        KeyError("Start indices not defined for this data.")

    # Load the original data
    # original_vive_data = h5py.File(vive_data_file, 'r')
    original_vive_data = scipy.io.loadmat(vive_data_file)
    cut_data = {key: {} for key in original_vive_data.keys() if key not in ['__header__', '__version__', '__globals__']}
    for key in original_vive_data.keys():
        if key not in ['__header__', '__version__', '__globals__']:
            # Extract the array from the nested structure
            positions = original_vive_data[key]['positions'][0][0]
            orientations = original_vive_data[key]['orientations'][0][0]
            timestamps = original_vive_data[key]['timestamps'][0][0][0]

            # Cut the data at the start time
            positions = np.squeeze(positions[start_indices[2]:, :])
            orientations = orientations[start_indices[2]:]
            timestamps = timestamps[start_indices[2]:] - timestamps[start_indices[2]]

            cut_data[key]['positions'] = positions
            cut_data[key]['orientations'] = orientations
            cut_data[key]['timestamps'] = timestamps

    # Load the target timestamps
    ifeel_data = scipy.io.loadmat(ifeel_data_file)
    target_timestamps = ifeel_data['timestamps'][0]
    target_timestamps = target_timestamps[start_indices[0]:] - target_timestamps[start_indices[0]]

    print(cut_data['openxr_head2']['positions'])

    # Interpolate the data
    interpolated_vive_data = interpolate_data(cut_data, target_timestamps)

    # Plot the original and interpolated position data for some frames just to check
    plt.figure(1)
    plt.plot(cut_data['vive_tracker_waist_pose']['timestamps'], cut_data['vive_tracker_waist_pose']['positions'][:, 1], label='Original follower waist z')
    plt.plot(interpolated_vive_data['vive_tracker_waist_pose']['timestamps'], interpolated_vive_data['vive_tracker_waist_pose']['positions'][:, 1], linestyle="--", label='Interp follower waist z')
    plt.legend()
    plt.show()

    # Save the interpolated data
    save_data(output_file, interpolated_vive_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interpolate data to match target timestamps.")
    parser.add_argument("--data_location", help="Dataset folder to extract features from.",
                    type=str, default="../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/forward_backward/")
    args = parser.parse_args()
    data_location = args.data_location

    main(data_location)