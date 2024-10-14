# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import json
import argparse
from pi_learning_for_collaborative_carrying.data_processing import utils
from pi_learning_for_collaborative_carrying.data_processing import feature_extractor
import jaxsim.api as js
import resolve_robotics_uri_py
import h5py
import numpy as np
# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

# Our custom dataset is divided in two datasets: D2 and D3
parser.add_argument("--data_location", help="Dataset folder to extract features from.",
                    type=str, default="../datasets/collaborative_payload_carrying/leader_backward/")
# Plot configuration
parser.add_argument("--plot_global_velocities", help="Visualize the raw and smoothed global velocities.",action="store_true")
parser.add_argument("--plot_human_features", help="Visualize the transformed and smoothed human features.",action="store_true")
parser.add_argument("--plot_robot_v_human", help="Visualize the robot and human base positions.",action="store_true")
parser.add_argument("--plot_global", help="Visualize the computed global features.",action="store_true")
parser.add_argument("--plot_local", help="Visualization the computed local features.",action="store_true")
# Store configuration
parser.add_argument("--save", help="Store the network input and output vectors in json format.",action="store_true")

args = parser.parse_args()

data_location = args.data_location
plot_global_velocities = args.plot_global_velocities
plot_human_features = args.plot_human_features
plot_robot_v_human = args.plot_robot_v_human
plot_global = args.plot_global
plot_local = args.plot_local
store_as_json = args.save

# Get path to retargeted data
script_directory = os.path.dirname(os.path.abspath(__file__))
retargeted_mocap_path = os.path.join(script_directory, data_location + "retargeted_motion_follower.txt")

# Load the retargeted mocap data
timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=retargeted_mocap_path)

# ===============
# MODEL INSERTION
# ===============

# Retrieve the robot urdf model
urdf_path = str(resolve_robotics_uri_py.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf"))

# Init jaxsim model for visualization and joint names/positions
js_model = js.model.JaxSimModel.build_from_model_description(
    model_description=urdf_path, is_urdf=True
)

# Get the joint name list
joint_names = [str(joint_name) for joint_name in js_model.joint_names()]

# Define the joints of interest for the feature computation and their associated indexes in the robot joints  list
controlled_joints = ['l_hip_pitch', 'l_hip_roll', 'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',  # left leg
                     'r_hip_pitch', 'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll',  # right leg
                     'torso_pitch', 'torso_roll', 'torso_yaw',  # torso
                     'neck_pitch', 'neck_roll', 'neck_yaw', # neck
                     'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', # left arm
                     'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow'] # right arm
controlled_joints_indexes = [joint_names.index(elem) for elem in controlled_joints]

# ========================================
# HUMAN DATA EXTRACTION AND TIME ALIGNMENT
# ========================================

# Align robot and human data in time
human_mocap_filename = os.path.join(script_directory, data_location + "leader.mat")
robot_mocap_filename = os.path.join(script_directory, data_location + "follower.mat")

# Define the start time, cutting the t-pose off for time alignment
start_time_dict = {"leader_backward/follower": 5.0, "leader_forward/follower": 8.0, "leader_backward/leader": 16.25, "leader_forward/leader": 33.0}

# Extract the relevant part of the file name to determine the start time
human_file_key = None
robot_file_key = None
for key in start_time_dict.keys():
    if key in human_mocap_filename:
        human_file_key = key
    elif key in robot_mocap_filename:
        robot_file_key = key

human_start_time = start_time_dict[human_file_key]
robot_start_time = start_time_dict[robot_file_key]

# Read in the mat file of the data as a h5py file, which acts the same as a python dict
human_mocap_data = h5py.File(human_mocap_filename, 'r')['robot_logger_device']

# Get the timestamps starting at calibration time
human_zeroed_timestamps = np.squeeze(human_mocap_data['human_state']['base_position']['timestamps'][:] - human_mocap_data['human_state']['base_position']['timestamps'][0])
robot_zeroed_timestamps = np.array(timestamps) - timestamps[0]

human_start_time_index = np.argmin(np.abs(human_zeroed_timestamps - human_start_time))
robot_start_time_index = np.argmin(np.abs(robot_zeroed_timestamps - robot_start_time))

# Get the human state positions and velocities
human_data = {"timestamps": human_zeroed_timestamps[human_start_time_index:] - human_zeroed_timestamps[human_start_time_index],
              "base_positions": np.squeeze(human_mocap_data['human_state']['base_position']['data'][human_start_time_index:]),
              "base_orientations": np.squeeze(human_mocap_data['human_state']['base_orientation']['data'][human_start_time_index:]),
              "base_linear_velocities": np.squeeze(human_mocap_data['human_state']['base_linear_velocity']['data'][human_start_time_index:]),
              "base_angular_velocities": np.squeeze(human_mocap_data['human_state']['base_angular_velocity']['data'][human_start_time_index:])}

robot_timestamps = robot_zeroed_timestamps[robot_start_time_index:]
ik_solutions = ik_solutions[robot_start_time_index:]

# Ensure the time series are the same length
if len(ik_solutions) > len(human_data["base_positions"]):
    # If the robot data is longer, cut it to match the human data
    ik_solutions = ik_solutions[:len(human_data["base_positions"])]
elif len(human_data["base_positions"]) > len(ik_solutions):
    # If the human data is longer, cut it to match the robot data
    human_data["timestamps"] = human_data["timestamps"][:len(ik_solutions)]
    human_data["base_positions"] = human_data["base_positions"][:len(ik_solutions)]
    human_data["base_orientations"] = human_data["base_orientations"][:len(ik_solutions)]
    human_data["base_linear_velocities"] = human_data["base_linear_velocities"][:len(ik_solutions)]
    human_data["base_angular_velocities"] = human_data["base_angular_velocities"][:len(ik_solutions)]

# ===================
# FEATURE EXTRACTION
# ===================

# Instantiate the feature extractor
extractor = feature_extractor.FeatureExtractor.build(ik_solutions=ik_solutions,
                                                       human_data=human_data,
                                                       controlled_joints_indexes=controlled_joints_indexes,
                                                       plot_global_vels=plot_global_velocities,
                                                       plot_human_features=plot_human_features,
                                                       plot_robot_v_human=plot_robot_v_human)
# Extract the features
extractor.compute_features()

# ===========================================
# NETWORK INPUT AND OUTPUT VECTORS GENERATION
# ===========================================

# Generate the network input vector X
X = extractor.compute_X()

if store_as_json:

    # Define the path to store the input X associated to the selected subsection of the dataset
    input_path = "extracted_features_X.txt"
    input_path = os.path.join(script_directory, input_path)

    input("Press Enter to store the computed X")

    # Store the retrieved input X in a JSON file
    with open(input_path, 'w') as outfile:
        json.dump(X, outfile)

    # Debug
    print("Input features have been saved in", input_path)

# Generate the network output vector Y
Y = extractor.compute_Y()

if store_as_json:

    # Define the path to store the output Y associated to the selected subsection of the dataset
    output_path = "extracted_features_Y.txt"
    output_path = os.path.join(script_directory, output_path)

    input("Press Enter to store the computed Y")

    # Store the retrieved output Y in a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(Y, outfile)

    # Debug
    print("Output features have been saved in", output_path)

# =======================================================
# VISUALIZE THE RETARGETED MOTION AND THE GLOBAL FEATURES
# =======================================================

if plot_global:

    input("Press Enter to start the visualization of the GLOBAL features")
    utils.visualize_global_features(global_window_features=extractor.get_global_window_features(),
                                    global_frame_features=extractor.get_global_frame_features(),
                                    ik_solutions=ik_solutions,
                                    js_model=js_model,
                                    controlled_joints=controlled_joints)

# =======================================================
# VISUALIZE THE RETARGETED MOTION AND THE LOCAL FEATURES
# =======================================================

if plot_local:

    input("Press Enter to start the visualization of the LOCAL features")
    utils.visualize_local_features(local_window_features=extractor.get_local_window_features(),
                                   global_frame_features=extractor.get_global_frame_features(),
                                   ik_solutions=ik_solutions,
                                   js_model=js_model,
                                   controlled_joints=controlled_joints)
