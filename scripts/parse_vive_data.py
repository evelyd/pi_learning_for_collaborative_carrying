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
                    zyx = list(map(float, parts[4:7]))
                    xyzw = list(map(float, parts[7:11]))
                    if name not in organized_data:
                        organized_data[name] = {
                            'timestamps': [],
                            'positions': [],
                            'orientations': []
                        }
                    organized_data[name]['timestamps'].append(pose_timestamp)
                    organized_data[name]['positions'].append(zyx)
                    organized_data[name]['orientations'].append(xyzw)

    # Convert lists to numpy arrays
    for name in organized_data:
        organized_data[name]['timestamps'] = np.array(organized_data[name]['timestamps'])
        organized_data[name]['positions'] = np.array(organized_data[name]['positions'])
        organized_data[name]['orientations'] = np.array(organized_data[name]['orientations'])

    return organized_data

def save_organized_data_to_mat(organized_data, mat_file_path):
    scipy.io.savemat(mat_file_path, organized_data)

# Function to plot positions
def plot_positions(data, pose_names, ax, colors = ['r', 'b']):

    for pose_name, color in zip(pose_names, colors):
        filtered_positions = np.array(data[pose_name]['positions'])
        ax.scatter(-filtered_positions[:, 2], -filtered_positions[:, 0], filtered_positions[:, 1], c=color, label=pose_name)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

# Function to print all unique pose names
def print_pose_names(data):
    pose_names = set(d['name'] for d in data)
    print("Pose names:", pose_names)

# Main script
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", help="Dataset folder to extract features from.",
                    type=str, default="../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/forward_backward/vive/")
args = parser.parse_args()
data_location = args.data_location

# Get path to retargeted data
script_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_directory, data_location + "data.log")
mat_file_path = os.path.join(script_directory, data_location + "parsed_vive_data.mat")

# Organize the data by field
organized_data = parse_and_organize_log_file(log_file_path)

# Save the parsed data to a .mat file
save_organized_data_to_mat(organized_data, mat_file_path)

# Create a plot of all the 4 different positions of headset 1 on a 3d plot
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(15, 15))
pose_names = ['vive_tracker_waist_pose', 'openxr_head', 'vive_tracker_left_elbow_pose', 'vive_tracker_right_elbow_pose']
colors = ['r', 'b', 'g', 'y']
plot_positions(organized_data, pose_names, ax, colors=colors)

# Create a plot of all the 4 different positions of headset 2 on a 3d plot
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(15, 15))
pose_names = ['vive_tracker_waist_pose2', 'openxr_head2', 'vive_tracker_left_elbow_pose2', 'vive_tracker_right_elbow_pose2']
colors = ['r', 'b', 'g', 'y']
plot_positions(organized_data, pose_names, ax, colors=colors)

plt.show()