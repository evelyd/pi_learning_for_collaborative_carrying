# Author: Evelyn D'Elia
import pi_learning_for_collaborative_carrying.trajectory_generation.DualVisualizer as vis
import pi_learning_for_collaborative_carrying.trajectory_generation.utils as utils
import idyntree.bindings as idyn
import numpy as np
import manifpy as manif
import bipedal_locomotion_framework as blf
import resolve_robotics_uri_py
import argparse
import os
import json
from scipy.spatial.transform import Rotation
import torch
from typing import List, Dict


import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif'})

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="Directory of the trained model to be simulated.", type=str, default="../datasets/trained_models/training_test_collab_no_pi_i_h_hb_no_bending_start_origin_subsampled_20241202-164937/")

args = parser.parse_args()
model_dir = args.model_dir

# Get the configuration files
script_directory = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_directory, "../config/config_mann.toml")
params_network = blf.parameters_handler.TomlParametersHandler()
params_network.set_from_file(str(config_path))

# Get the path of the robot model
robot_model_path = str(resolve_robotics_uri_py.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf"))
ml = idyn.ModelLoader()
ml.loadReducedModelFromFile(robot_model_path, params_network.get_parameter_vector_string("joints_list"))

# Get the path to the human model
human_urdf_path = str(resolve_robotics_uri_py.resolve_robotics_uri("package://human-gazebo/humanSubjectWithMesh.urdf")) #TODO fixed by putting the urdf and meshes inside the human-gazebo package in the conda env/share folder
human_ml = idyn.ModelLoader()
human_ml.loadReducedModelFromFile(human_urdf_path, params_network.get_parameter_vector_string("human_joints_list"))

# Get the human joint positions from the retargeted data (subsampled by 2x)
human_data_path = os.path.join(script_directory, "../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/left_right/retargeted_motion_leader.txt")
robot_data_path = os.path.join(script_directory, "../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/left_right/retargeted_motion_follower.txt")
human_data = utils.get_human_base_pose_from_retargeted_data(human_data_path, robot_data_path)

# Extract the relevant part of the file name to determine the start time
start_end_dict = {"forward_backward": [600, 2670], "left_right": [590, 4090]} # Cut off bending over
file_key = None
for key in start_end_dict.keys():
    if key in human_data_path:
        file_key = key

start_ind = int(start_end_dict[file_key][0]/2)
end_ind = int(start_end_dict[file_key][1]/2)

# Get the human control inputs
human_base_poses = [np.array(data["base_pose"]) for data in human_data[start_ind:end_ind]]
human_base_positions = [pose[:3,3] for pose in human_base_poses]
human_base_orientations = [pose[:3,:3].reshape(9) for pose in human_base_poses]
human_base_linear_velocities = [np.array(data["base_linear_velocity"]) for data in human_data[start_ind:end_ind]]
human_base_angular_velocities = [np.array(data["base_angular_velocity"]) for data in human_data[start_ind:end_ind]]
human_joint_positions = [np.array(data["joint_positions"]) for data in human_data[start_ind:end_ind]]

# prepare visualizer
viz = vis.DualVisualizer(ml1=ml, ml2=human_ml, model1_name="robot", model2_name="human")
viz.load_model()

# Set the initial input values from the data
initial_input_path = script_directory + "/../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/" + "left_right" + "/extracted_features_X.txt"
initial_output_path = script_directory + "/../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/" + "left_right" + "/extracted_features_Y.txt"

with open(initial_input_path, 'r') as file:
    input_stuff = json.load(file)
# Get the initial input vector
input_vector = np.array(input_stuff[0])

# Get the data inputs
feature_human_base_positions = np.array([entry[136:139] for entry in input_stuff])

# Replace the human inputs with those of the data
start_at = 100
input_vector[136:139] = human_base_positions[start_at]
input_vector[139:148] = human_base_orientations[start_at]
input_vector[148:151] = human_base_linear_velocities[start_at]
input_vector[151:154] = human_base_angular_velocities[start_at]

# Compute component-wise input mean and standard deviation
datapath = os.path.join(model_dir, "normalization/")
Xmean, Xstd = utils.load_input_mean_and_std(datapath)
Ymean, Ystd = utils.load_output_mean_and_std(datapath)

# Define the initial past robot base velocities
current_past_base_linear_velocities = [[0.0, 0.0, 0.0] for _ in range(51)]
current_past_base_angular_velocities = [[0.0, 0.0, 0.0] for _ in range(51)] #TODO should really get this from a standing still portion of dataset

# Prepare the model for querying
model_path = model_dir + "/models/model_149.pth"
learned_model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
learned_model.eval()

predicted_base_positions = []
predicted_base_orientations = []

length_of_time = 800
for i in range(length_of_time):

    # Compute current robot base pose
    current_robot_base_position = input_vector[124:127]
    current_robot_base_orientation = input_vector[127:136]
    current_robot_base_pose = utils.get_base_pose(current_robot_base_position, current_robot_base_orientation)

    # Get the human base pose from the input vector
    human_base_position = input_vector[136:139]
    human_base_orientation = input_vector[139:148]
    human_base_pose = utils.get_base_pose(human_base_position, human_base_orientation)

    # Normalize input vector
    input_vector = (input_vector - Xmean) / Xstd

    # Make input vector into a n x 1 tensor
    input_vector = torch.tensor(input_vector).unsqueeze(0)

    # Get network output from input
    current_output = learned_model.inference(torch.tensor(input_vector)).cpu().numpy()

    # Get rid of extra dimension
    current_output = current_output.squeeze()

    # Denormalize output
    denormalized_current_output = current_output * Ystd + Ymean

    # Get the parts of the output
    output_dict = utils.parse_output(denormalized_current_output)

    # Get the robot pose and joint state from the output
    robot_base_pose = utils.get_base_pose(output_dict["robot_base_position"], output_dict["robot_base_orientation"])

    predicted_base_positions.append(output_dict["robot_base_position"])
    predicted_base_orientations.append(output_dict["robot_base_orientation"])

    # Get the human joint state from the input vector
    #TODO ith or i+1 th?
    human_joint_state = human_joint_positions[i]

    # Update visualization
    viz.update_models(output_dict["robot_joint_state"], human_joint_state, robot_base_pose, human_base_pose)

    if i == 0:
        input("Press a key to start the trajectory generation")

    # Form input vector for next iteration
    new_input_vector = []

    # Compute the new last 50 linear velocities
    current_past_base_linear_velocities = utils.form_next_past_velocity_window(current_past_base_linear_velocities, output_dict["current_linear_velocity"], current_robot_base_pose[:3,:3], robot_base_pose[:3,:3])

    # Get the past window from the previous velocities
    subsampled_past_linear_velocities = [item for sublist in [current_past_base_linear_velocities[j] for j in [0, 10, 20, 30, 40, 50]] for item in sublist]
    new_input_vector.extend(subsampled_past_linear_velocities)
    new_input_vector.extend(output_dict["future_linear_velocities"])

    # Compute the new last 50 angular velocities
    current_past_base_angular_velocities = utils.form_next_past_velocity_window(current_past_base_angular_velocities, output_dict["current_angular_velocity"], current_robot_base_pose[:3,:3], robot_base_pose[:3,:3])

    # Get the past window from the previous velocities
    subsampled_past_angular_velocities = [item for sublist in [current_past_base_angular_velocities[j] for j in [0, 10, 20, 30, 40, 50]] for item in sublist]
    new_input_vector.extend(subsampled_past_angular_velocities)
    new_input_vector.extend(output_dict["future_angular_velocities"])

    # Add the robot joint information and base pose information
    new_input_vector.extend(output_dict["robot_joint_state"])
    new_input_vector.extend(output_dict["robot_joint_velocity"])
    new_input_vector.extend(output_dict["robot_base_position"])
    new_input_vector.extend(output_dict["robot_base_orientation"])

    # Compute the next human base pose and velocity (from the "user input" data)
    new_human_base_position = human_base_positions[start_at + i + 1]
    new_human_base_orientation = human_base_orientations[start_at + i + 1]
    new_human_base_linear_velocity = human_base_linear_velocities[start_at + i + 1]
    new_human_base_angular_velocity = human_base_angular_velocities[start_at + i + 1]
    new_input_vector.extend(new_human_base_position)
    new_input_vector.extend(new_human_base_orientation)
    new_input_vector.extend(new_human_base_linear_velocity)
    new_input_vector.extend(new_human_base_angular_velocity)

    input_vector = np.array(new_input_vector)

# Plot the predicted vs data robot base positions
with open(initial_output_path, 'r') as file:
    output_stuff = json.load(file)

# Get the robot base positions from the output feature data
feature_robot_base_positions = np.array([entry[94:97] for entry in output_stuff])
feature_robot_base_orientations = np.array([entry[97:106] for entry in output_stuff])

predicted_base_positions = np.array(predicted_base_positions)
predicted_base_orientations = np.array(predicted_base_orientations)

# Compute the MSE between the predicted and actual robot base orientations
mse_position = np.mean((predicted_base_positions - feature_robot_base_positions[:len(predicted_base_positions)])**2)
mse_orientation = np.mean((predicted_base_orientations - feature_robot_base_orientations[:len(predicted_base_orientations)])**2)

print(f'MSE for positions: {mse_position}')
print(f'MSE for orientations: {mse_orientation}')

plt.figure()
plt.plot(range(len(predicted_base_positions)), predicted_base_positions[:, 0], label='Predicted X Position')
plt.plot(range(len(predicted_base_positions)), predicted_base_positions[:, 1], label='Predicted Y Position')
plt.plot(range(len(predicted_base_positions)), predicted_base_positions[:, 2], label='Predicted Z Position')
plt.plot(range(len(feature_robot_base_positions)), feature_robot_base_positions[:, 0], label='Actual X Position', linestyle='--')
plt.plot(range(len(feature_robot_base_positions)), feature_robot_base_positions[:, 1], label='Actual Y Position', linestyle='--')
plt.plot(range(len(feature_robot_base_positions)), feature_robot_base_positions[:, 2], label='Actual Z Position', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Predicted vs Actual Robot Base Positions Over Time')
plt.legend()

# Convert the rotation matrices to Euler angles
predicted_robot_base_angles = np.array([Rotation.from_matrix(predicted_base_orientations[i].reshape(3, 3)).as_euler('xyz') for i in range(len(predicted_base_orientations))])
feature_robot_base_angles = np.array([Rotation.from_matrix(feature_robot_base_orientations[i].reshape(3, 3)).as_euler('xyz') for i in range(len(feature_robot_base_orientations))])

plt.figure()
plt.plot(range(len(predicted_robot_base_angles)), predicted_robot_base_angles[:, 0], label='Predicted Roll')
plt.plot(range(len(predicted_robot_base_angles)), predicted_robot_base_angles[:, 1], label='Predicted Pitch')
plt.plot(range(len(predicted_robot_base_angles)), predicted_robot_base_angles[:, 2], label='Predicted Yaw')
plt.plot(range(len(feature_robot_base_angles)), feature_robot_base_angles[:, 0], label='Actual Roll', linestyle='--')
plt.plot(range(len(feature_robot_base_angles)), feature_robot_base_angles[:, 1], label='Actual Pitch', linestyle='--')
plt.plot(range(len(feature_robot_base_angles)), feature_robot_base_angles[:, 2], label='Actual Yaw', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Orientation (radians)')
plt.title('Predicted vs Actual Robot Base Orientations Over Time')
plt.legend()
plt.show()