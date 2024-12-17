import re
import os
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Function to parse the log file and organize data by node
def parse_and_organize_log_file(log_file_path):
    organized_data = {}
    timestamps = []
    with open(log_file_path, 'r') as file:
        for line in file:
            if (('dec10_2024' in log_file_path) or ('dec12_2024' in log_file_path)) and ('leader' in log_file_path):
                match = re.search(r'0 (\d+\.\d+) "iFeelSuit::" \(\((.*)\)\)', line)
            else:
                match = re.search(r'0 (\d+\.\d+) (\d+\.\d+) "iFeelSuit::" \(\((.*)\)\)', line)
            if match:
                if (('dec10_2024' in log_file_path) or ('dec12_2024' in log_file_path)) and ('leader' in log_file_path):
                    pose_timestamp = float(match.group(1))
                    poses = match.group(2)
                else:
                    pose_timestamp = float(match.group(2))
                    poses = match.group(3)
                pose_matches = parse_poses(poses)
                for pose in pose_matches:
                    if any(data_type in pose for data_type in ['orient', 'gyro', 'ft6D']):
                        if any(foot_part in pose for foot_part in ['Back', 'Front']):
                            continue
                        parts = pose.split()
                        node_name = parts[0]
                        node_number = re.search(r'Node#(\d+)', node_name).group(1)
                        data_type = node_name.split('::')[1]
                        if data_type == 'ft6D':
                            data_length = 6
                        elif data_type == 'orient':
                            data_length = 4
                        else:
                            data_length = 3
                        start_index = parts.index('1') + 1
                        values = [float(part.strip('()')) for part in parts[start_index:start_index+data_length]]
                        node_key = f'node{node_number}'
                        if node_key not in organized_data:
                            organized_data[node_key] = {
                                'orient': [],
                                'ft6D': [],
                                'gyro': [],
                            }

                        organized_data[node_key][data_type].append(values)
                timestamps.append(pose_timestamp)
    # Convert lists to numpy arrays
    for node_key in organized_data:
        for data_type in ['orient', 'ft6D', 'gyro']:
            organized_data[node_key][data_type] = np.array(organized_data[node_key][data_type])
    organized_data['timestamps'] = np.array(timestamps)

    return organized_data

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

    fully_split_poses = []
    for pose in pose_list:
        if any(data_type in pose for data_type in ['orient', 'gyro', 'ft6D', 'fbAcc', 'mag', 'vLink']):
            partially_split_poses = []
            stack = []
            current_pose = []
            for i, char in enumerate(pose):
                if char == '(':
                    stack.append(char)
                    if len(stack) == 1:
                        current_pose = []
                elif char == ')':
                    if stack:
                        stack.pop()
                        if len(stack) == 0:
                            current_pose.append(char)
                            partially_split_poses.append(''.join(current_pose))
                if len(stack) > 0:
                    current_pose.append(char)
                # Check if we are at the final character
                if i == len(pose) - 1 and stack:
                    partially_split_poses.append(''.join(current_pose))
            fully_split_poses.extend(partially_split_poses)
        else:
            fully_split_poses.append(pose)

    return fully_split_poses

# Function to save organized data to a .mat file
def save_organized_data_to_mat(organized_data, mat_file_path):
    scipy.io.savemat(mat_file_path, organized_data)

# Main script
parser = argparse.ArgumentParser()
parser.add_argument("--data_location", help="Dataset folder to extract features from.",
                    type=str, default="../datasets/collaborative_payload_carrying/ifeel_and_vive/dec12_2024/1_fb_straight/")
args = parser.parse_args()
data_location = args.data_location

# Get path to retargeted data
script_directory = os.path.dirname(os.path.abspath(__file__))
follower_log_file_path = os.path.join(script_directory, data_location + "follower/data.log")
follower_mat_file_path = os.path.join(script_directory, data_location + "follower/parsed_ifeel_data.mat")
leader_log_file_path = os.path.join(script_directory, data_location + "leader/data.log")
leader_mat_file_path = os.path.join(script_directory, data_location + "leader/parsed_ifeel_data.mat")

# Parse and organize the log file
follower_organized_data = parse_and_organize_log_file(follower_log_file_path)
leader_organized_data = parse_and_organize_log_file(leader_log_file_path)

# Switch node3 to node5 for follower for oct25_2024 data
if "oct25_2024" in follower_log_file_path:
    follower_organized_data['node5'] = follower_organized_data.pop('node3')
# Save the organized data to a .mat file
save_organized_data_to_mat(follower_organized_data, follower_mat_file_path)
save_organized_data_to_mat(leader_organized_data, leader_mat_file_path)

# Plot node5 data for both follower and leader
def plot_node5_data(follower_data, leader_data):
    # fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plt.figure()
    plt.title('Node5 Data for Follower and Leader')

    data_type = 'orient'

    from scipy.spatial.transform import Rotation

    if 'node5' in follower_data and 'node5' in leader_data:
        follower_node5_data = follower_data['node5'][data_type]
        follower_node5_rpy = Rotation.from_quat(follower_node5_data, scalar_first=True).as_euler('xyz', degrees=True)
        leader_node5_data = leader_data['node5'][data_type]
        leader_node5_rpy = Rotation.from_quat(leader_node5_data, scalar_first=True).as_euler('xyz', degrees=True)
        follower_timestamps = follower_data['timestamps']
        leader_timestamps = leader_data['timestamps']

        plt.plot(follower_timestamps, follower_node5_rpy, label=f'Follower Node5 {data_type}')
        plt.plot(leader_timestamps, leader_node5_rpy, label=f'Leader Node5 {data_type}')
        plt.title(f'Node5 {data_type}')
        plt.xlabel('Time (s)')
        plt.ylabel(data_type)
        plt.legend()

    plt.tight_layout()
    plt.show()

plot_node5_data(follower_organized_data, leader_organized_data)

print("Data has been parsed and saved to", follower_mat_file_path, leader_mat_file_path)