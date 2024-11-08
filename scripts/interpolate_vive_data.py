import numpy as np
import os
import scipy.io
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp, Rotation
from pi_learning_for_collaborative_carrying.data_processing import utils

def interpolate_data(original_data_dict, target_timestamps):
    """
    Interpolate the original data to match the target timestamps.
    """
    interpolated_data = {}

    # goes throught the elements, not the timesteps, of the actual data (i.e. goes through each of 4 elems of orientation)
    for key in original_data_dict.keys():
        positions = original_data_dict[key]['positions']
        orientations = original_data_dict[key]['orientations']
        timestamps = original_data_dict[key]['timestamps']

        f = interp1d(timestamps, positions, axis=0, fill_value="extrapolate")
        interpolated_positions = f(target_timestamps)

        # Remove duplicate samples
        unique_indices = np.unique(timestamps, return_index=True)[1]
        timestamps = timestamps[unique_indices]
        orientations = orientations[unique_indices]

        # Create rotation matrix for interpolation
        tmp = Rotation.from_quat(orientations) # assume xyzw raw data form

        # Handle extrapolation for orientations
        if np.max(target_timestamps) > np.max(timestamps):
            max_timestamp = np.max(timestamps)
            extrapolation_indices = target_timestamps > max_timestamp
            interpolation_indices = ~extrapolation_indices

            slerp = Slerp(timestamps, tmp)
            interpolated_orientations = np.zeros((len(target_timestamps), 4))

            # Interpolate within the original timestamp range
            interpolated_orientations[interpolation_indices] = slerp(target_timestamps[interpolation_indices]).as_quat()

            # Extrapolate beyond the original timestamp range
            last_orientation = orientations[-1]
            interpolated_orientations[extrapolation_indices] = last_orientation
        else:
            slerp = Slerp(timestamps, tmp)
            interpolated_orientations = slerp(target_timestamps).as_quat()

        interpolated_data[key] = {
                'positions': interpolated_positions,
                'orientations': interpolated_orientations,
                'timestamps': target_timestamps
            }

    return interpolated_data

def save_data(file_path, interpolated_data):
    """
    Save the interpolated data to a CSV file.
    """
    scipy.io.savemat(file_path, interpolated_data)

def main(data_location):

    # Get paths to data
    script_directory = os.path.dirname(os.path.abspath(__file__))
    vive_data_file = os.path.join(script_directory, data_location + "vive/parsed_vive_data.mat")
    # Taking the timesteps from the leader iFeel data
    ifeel_data_file = os.path.join(script_directory, data_location + "leader/parsed_ifeel_data.mat")
    output_file = os.path.join(script_directory, data_location + "/vive/interpolated_vive_data.mat")

    # Start all the data at the same time so that it lines up
    start_ind_dict = {"forward_backward": [3100, 2049, 48415], "left_right": [10275, 4539, 2411]}
    if "forward_backward" in vive_data_file:
        start_indices = start_ind_dict["forward_backward"] #iFeel leader, iFeel follower, Vive
    elif "left_right" in vive_data_file:
        start_indices = start_ind_dict["left_right"]
    else:
        KeyError("Start indices not defined for this data.")

    # Load the original data
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

    # Interpolate the data
    interpolated_vive_data = interpolate_data(cut_data, target_timestamps)

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