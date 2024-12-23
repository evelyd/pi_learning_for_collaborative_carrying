import re
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import argparse

# Function to parse poses with balanced parentheses
def parse_poses(poses):
    pose_list = []
    stack = []
    current_pose = []
    for i, char in enumerate(poses):
        if char == '(':
            stack.append(char)
            if len(stack) == 1:
                current_pose = []
        elif char == ')':
            if stack:
                stack.pop()
                if len(stack) == 0:
                    current_pose.append(char)
                    pose_list.append(''.join(current_pose))
        if len(stack) > 0:
            current_pose.append(char)
        # Check if we are at the final character
        if i == len(poses) - 1 and stack:
            pose_list.append(''.join(current_pose))
    # Remove '(' and ')' from each pose in the pose_list
    pose_list = [pose.strip('()') for pose in pose_list]

    return pose_list

# Function to parse the log file
def parse_and_organize_log_file(log_file_path):
    organized_data = {}
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r'-1 (\d+\.\d+) \[ok\] \(\((.*)\)\)', line)
            if match:
                timestamp = float(match.group(1))
                poses = match.group(2)
                pose_matches = parse_poses(poses)
                for pose in pose_matches:
                    parts = pose.split()
                    origin = parts[0]
                    name = parts[1]
                    pose_timestamp = float(parts[2])
                    pos = list(map(float, parts[4:7]))
                    xyzw = list(map(float, parts[7:11]))
                    if name not in organized_data:
                        organized_data[name] = {
                            'timestamps': [],
                            'positions': [],
                            'orientations': []
                        }
                    organized_data[name]['timestamps'].append(pose_timestamp)
                    organized_data[name]['positions'].append(pos)
                    organized_data[name]['orientations'].append(xyzw)

    # Convert lists to numpy arrays
    for name in organized_data:
        if '2' in name:
            base_pose_name = name.replace('2', '')
            if base_pose_name in organized_data:
                organized_data[name]['timestamps'] = organized_data[base_pose_name]['timestamps']
        else:
            organized_data[name]['timestamps'] = np.array(organized_data[name]['timestamps'])
        organized_data[name]['positions'] = np.array(organized_data[name]['positions'])
        organized_data[name]['orientations'] = np.array(organized_data[name]['orientations'])

    return organized_data

def save_organized_data_to_mat(organized_data, mat_file_path):
    scipy.io.savemat(mat_file_path, organized_data)

# Function to plot positions
def plot_positions(data, pose_names, ax, colors=['r', 'b']):
    for pose_name, color in zip(pose_names, colors):
        timestamps = data[pose_name]['timestamps']
        positions = np.array(data[pose_name]['positions'])

        ax.plot(timestamps, positions[:, 0], c=color, label=f'{pose_name} X')
        ax.plot(timestamps, positions[:, 1], c=color, linestyle='--', label=f'{pose_name} Y')
        ax.plot(timestamps, positions[:, 2], c=color, linestyle=':', label=f'{pose_name} Z')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    ax.legend()

# Main script
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", help="Dataset folder to extract features from.",
                    type=str, default="../datasets/collaborative_payload_carrying/ifeel_and_vive/dec10_2024/1_fb_straight/")
args = parser.parse_args()
data_location = args.data_location

# Get path to retargeted data
script_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_directory, data_location + "vive/data.log")
mat_file_path = os.path.join(script_directory, data_location + "vive/parsed_vive_data.mat")

# Organize the data by field
organized_data = parse_and_organize_log_file(log_file_path)

fig = plt.figure()

plot_positions(organized_data, ['vive_tracker_waist_pose', 'vive_tracker_waist_pose2'], fig.add_subplot(321))
plot_positions(organized_data, ['vive_tracker_left_elbow_pose', 'vive_tracker_left_elbow_pose2'], fig.add_subplot(322))
plot_positions(organized_data, ['vive_tracker_right_elbow_pose', 'vive_tracker_right_elbow_pose2'], fig.add_subplot(323))
plot_positions(organized_data, ['vive_tracker_left_foot_pose', 'vive_tracker_left_foot_pose2'], fig.add_subplot(324))
plot_positions(organized_data, ['vive_tracker_right_foot_pose', 'vive_tracker_right_foot_pose2'], fig.add_subplot(325))

plt.show()

# Save the parsed data to a .mat file
save_organized_data_to_mat(organized_data, mat_file_path)