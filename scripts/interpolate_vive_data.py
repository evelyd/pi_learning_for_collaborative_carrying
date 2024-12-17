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
        if key not in ['__header__', '__version__', '__globals__']:
            positions = original_data_dict[key]['positions'][0][0]
            orientations = original_data_dict[key]['orientations'][0][0]
            timestamps = original_data_dict[key]['timestamps'][0][0][0]

            f = interp1d(timestamps, positions, axis=0, fill_value="extrapolate")
            interpolated_positions = f(target_timestamps)

            # Remove duplicate samples
            unique_indices = np.unique(timestamps, return_index=True)[1]
            timestamps = timestamps[unique_indices]
            orientations = orientations[unique_indices]

            # Create rotation matrix for interpolation
            tmp = Rotation.from_quat(orientations) # assume xyzw raw data form

            # Handle extrapolation for orientations
            if (np.max(target_timestamps) > np.max(timestamps)) or (np.min(target_timestamps) < np.min(timestamps)):
                max_timestamp = np.max(timestamps)
                end_extrapolation_indices = target_timestamps > max_timestamp
                end_interpolation_indices = ~end_extrapolation_indices

                min_timestamp = np.min(timestamps)
                beginning_extrapolation_indices = target_timestamps < min_timestamp
                beginning_interpolation_indices = ~beginning_extrapolation_indices

                interpolation_indices = np.logical_and(beginning_interpolation_indices, end_interpolation_indices)

                slerp = Slerp(timestamps, tmp)
                interpolated_orientations = np.zeros((len(target_timestamps), 4))

                # Extrapolate before the original timestamp range
                first_orientation = orientations[0]
                interpolated_orientations[beginning_extrapolation_indices] = first_orientation

                # Interpolate within the original timestamp range
                interpolated_orientations[interpolation_indices] = slerp(target_timestamps[interpolation_indices]).as_quat()

                # Extrapolate beyond the original timestamp range
                last_orientation = orientations[-1]
                interpolated_orientations[end_extrapolation_indices] = last_orientation
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
    ifeel_leader_data_file = os.path.join(script_directory, data_location + "leader/parsed_ifeel_data.mat")
    ifeel_follower_data_file = os.path.join(script_directory, data_location + "follower/parsed_ifeel_data.mat")
    leader_output_file = os.path.join(script_directory, data_location + "vive/interpolated_leader_vive_data.mat")
    follower_output_file = os.path.join(script_directory, data_location + "vive/interpolated_follower_vive_data.mat")

    # Load the original data
    original_vive_data = scipy.io.loadmat(vive_data_file)

    # Load the target timestamps
    leader_ifeel_data = scipy.io.loadmat(ifeel_leader_data_file)
    leader_target_timestamps = leader_ifeel_data['timestamps'][0]
    follower_ifeel_data = scipy.io.loadmat(ifeel_follower_data_file)
    follower_target_timestamps = follower_ifeel_data['timestamps'][0]

    # Interpolate the data
    interpolated_leader_vive_data = interpolate_data(original_vive_data, leader_target_timestamps)
    interpolated_follower_vive_data = interpolate_data(original_vive_data, follower_target_timestamps)

    # Plot the interpolated data for vive_tracker_left_elbow_pose and vive_tracker_left_elbow_pose2 on the same plots
    key1 = 'vive_tracker_left_elbow_pose'
    key2 = 'vive_tracker_left_elbow_pose2'

    if key1 in interpolated_leader_vive_data and key2 in interpolated_leader_vive_data:
        plt.figure(figsize=(12, 6))

        # Plot positions
        plt.subplot(2, 1, 1)
        plt.plot(interpolated_leader_vive_data[key1]['timestamps'], interpolated_leader_vive_data[key1]['positions'], label=f'{key1} Positions')
        plt.plot(interpolated_leader_vive_data[key2]['timestamps'], interpolated_leader_vive_data[key2]['positions'], label=f'{key2} Positions', linestyle='--')
        plt.title('Interpolated Positions')
        plt.xlabel('Time (s)')
        plt.xlim([np.min(leader_target_timestamps), np.max(leader_target_timestamps)])
        plt.ylabel('Position (m)')
        plt.legend()

        # Plot orientations
        plt.subplot(2, 1, 2)
        plt.plot(interpolated_leader_vive_data[key1]['timestamps'], Rotation.from_quat(interpolated_leader_vive_data[key1]['orientations']).as_euler('xyz'), label=f'{key1} Orientations')
        plt.plot(interpolated_leader_vive_data[key2]['timestamps'], Rotation.from_quat(interpolated_leader_vive_data[key2]['orientations']).as_euler('xyz'), label=f'{key2} Orientations', linestyle='--')
        plt.title('Interpolated Orientations')
        plt.xlabel('Time (s)')
        plt.xlim([np.min(leader_target_timestamps), np.max(leader_target_timestamps)])
        plt.ylabel('Orientation (quaternion)')
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        if key1 not in interpolated_leader_vive_data:
            print(f"Key '{key1}' not found in interpolated data.")
        if key2 not in interpolated_leader_vive_data:
            print(f"Key '{key2}' not found in interpolated data.")

    # Save the interpolated data
    save_data(leader_output_file, interpolated_leader_vive_data)
    save_data(follower_output_file, interpolated_follower_vive_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interpolate data to match target timestamps.")
    parser.add_argument("--data_location", help="Dataset folder to extract features from.",
                    type=str, default="../datasets/collaborative_payload_carrying/ifeel_and_vive/dec12_2024/1_fb_straight/")
    args = parser.parse_args()
    data_location = args.data_location

    main(data_location)