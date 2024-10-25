import re
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.transform import Rotation

# Function to parse the log file
def parse_log_file(log_file_path):
    data = []
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
                    #TODO is data actually in zyx form? check
                    zyx = list(map(float, parts[4:7]))
                    xyzw = list(map(float, parts[7:11]))
                    data.append({
                        'timestamp': timestamp,
                        'origin': origin,
                        'name': name,
                        'pose_timestamp': pose_timestamp,
                        'position': zyx,
                        'orientation': xyzw
                    })
    return data

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

# Function to organize data by field
def organize_data_by_field(data):
    organized_data = {}
    for entry in data:
        name = entry['name']
        if name not in organized_data:
            organized_data[name] = {
                'timestamps': [],
                'positions': [],
                'orientations': []
            }
        organized_data[name]['timestamps'].append(entry['pose_timestamp'])
        organized_data[name]['positions'].append(entry['position'])
        organized_data[name]['orientations'].append(entry['orientation'])

    # Convert lists to numpy arrays
    for name in organized_data:
        organized_data[name]['timestamps'] = np.array(organized_data[name]['timestamps'])
        organized_data[name]['positions'] = np.array(organized_data[name]['positions'])
        organized_data[name]['orientations'] = np.array(organized_data[name]['orientations'])

    return organized_data

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

# Function to save data to a .mat file
def save_to_mat(data, mat_file_path):

    mat_data = {
        'timestamps': np.array([d['timestamp'] for d in data]),
        'origins': np.array([d['origin'] for d in data], dtype=object),
        'names': np.array([d['name'] for d in data], dtype=object),
        'pose_timestamps': np.array([d['pose_timestamp'] for d in data]),
        'positions': np.array([d['position'] for d in data]),
        'orientations': np.array([d['orientation'] for d in data])
    }
    scipy.io.savemat(mat_file_path, mat_data)

def save_organized_data_to_mat(organized_data, mat_file_path):
    scipy.io.savemat(mat_file_path, organized_data)

# Function to plot positions
def plot_positions(data, pose_names, ax, colors = ['r', 'b']):
    # colors = ['r', 'b']  # Colors for each pose

    for pose_name, color in zip(pose_names, colors):
        # filtered_positions = [d['position'] for d in data if d['name'] == pose_name]
        filtered_positions = np.array(data[pose_name]['positions'])
        # filtered_positions = filtered_positions - filtered_positions[0]  # Set the first position as the origin
        ax.scatter(filtered_positions[:, 0], filtered_positions[:, 2], filtered_positions[:, 1], c=color, label=pose_name)

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
                    type=str, default="../datasets/collaborative_payload_carrying/ifeel_and_vive/")
args = parser.parse_args()
data_location = args.data_location

# Get path to retargeted data
set_number = "set1"
script_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_directory, data_location + "data_" + set_number + ".log")
mat_file_path = os.path.join(script_directory, data_location + "parsed_vive_data_" + set_number + ".mat")

# Parse the log file
# data = parse_log_file(log_file_path)

# Organize the data by field
# organized_data = organize_data_by_field(data)
organized_data = parse_and_organize_log_file(log_file_path)

# Save the parsed data to a .mat file
# save_to_mat(data, mat_file_path)
# Save the organized data to a .mat file
save_organized_data_to_mat(organized_data, mat_file_path)

# Print all unique pose names
# print_pose_names(data)

# Create a 2x2 grid of 3D subplots
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(15, 15))
# fig = plt.figure()

# Plot the positions in each subplot
pose_names = ['vive_tracker_right_elbow_pose1', 'vive_tracker_right_elbow_pose2']
plot_positions(organized_data, pose_names, axs[0, 0])

pose_names = ['vive_tracker_left_elbow_pose1', 'vive_tracker_left_elbow_pose2']
plot_positions(organized_data, pose_names, axs[0, 1])

pose_names = ['vive_tracker_left_elbow_pose1', 'vive_tracker_right_elbow_pose1']
plot_positions(organized_data, pose_names, axs[1, 0])

pose_names = ['vive_tracker_left_elbow_pose2', 'vive_tracker_right_elbow_pose2']
plot_positions(organized_data, pose_names, axs[1, 1])

plt.tight_layout()

# Create a plot of right and left elbow positions over time
fig = plt.figure()
plt.plot(organized_data['vive_tracker_right_elbow_pose2']['timestamps'], organized_data['vive_tracker_right_elbow_pose2']['positions'], organized_data['vive_tracker_left_elbow_pose2']['timestamps'], organized_data['vive_tracker_left_elbow_pose2']['positions'])
plt.legend(['right x', 'right z', 'right y', 'left x', 'left z', 'left y'])

# Create a plot of all the 3 different positions of headset 2 on a 3d plot
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(15, 15))
pose_names = ['openxr_headset2', 'vive_tracker_left_elbow_pose2', 'vive_tracker_right_elbow_pose2']
colors = ['r', 'b', 'g']
plot_positions(organized_data, pose_names, ax, colors=colors)

plt.show()