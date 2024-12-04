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

def load_input_mean_and_std(datapath: str) -> (Dict, Dict):
    """Compute component-wise input mean and standard deviation."""

    # Full-input mean and std
    Xmean = read_from_file(datapath + 'X_mean.txt')
    Xstd = read_from_file(datapath + 'X_std.txt')

    # Remove zeroes from Xstd
    for i in range(Xstd.size):
        if Xstd[i] == 0:
            Xstd[i] = 1

    return Xmean, Xstd

def load_output_mean_and_std(datapath: str) -> (List, List):
    """Compute output mean and standard deviation."""

    # Full-output mean and std
    Ymean = read_from_file(datapath + 'Y_mean.txt')
    Ystd = read_from_file(datapath + 'Y_std.txt')

    # Remove zeroes from Ystd
    for i in range(Ystd.size):
        if Ystd[i] == 0:
            Ystd[i] = 1

    return Ymean, Ystd

def read_from_file(filename: str) -> np.array:
    """Read data as json from file."""

    with open(filename, 'r') as openfile:
        data = json.load(openfile)

    return np.array(data)

def form_next_past_velocity_window(current_past_trajectory_base_velocities: List, current_base_velocity: List, current_world_R_base: np.array, new_world_R_base: np.array) -> List:
    """Form the next velocity window from the current past trajectory velocities, for either linear or angular velocities."""

    # Update the full window storing the past base velocities
    new_past_trajectory_base_velocities = []
    for k in range(len(current_past_trajectory_base_velocities) - 1):
        # Element in the reference frame defined by the previous base position + orientation
        base_elem = current_past_trajectory_base_velocities[k + 1]
        # Express element in world frame
        world_elem = current_world_R_base.dot(base_elem)
        # Express element in the frame defined by the new base position + orientation
        new_base_elem = np.linalg.inv(new_world_R_base).dot(world_elem)
        # Store updated element
        new_past_trajectory_base_velocities.append(new_base_elem)

    # Add as last element the current (local) base velocity (from the output)
    new_past_trajectory_base_velocities.append(current_base_velocity)

    return new_past_trajectory_base_velocities

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
human_data_path = os.path.join(script_directory, "../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/forward_backward/retargeted_motion_leader.txt")
robot_data_path = os.path.join(script_directory, "../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/forward_backward/retargeted_motion_follower.txt")
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
human_base_orientations = [Rotation.from_matrix(pose[:3,:3]).as_euler('xyz') for pose in human_base_poses]
human_base_linear_velocities = [np.array(data["base_linear_velocity"]) for data in human_data[start_ind:end_ind]]
human_base_angular_velocities = [np.array(data["base_angular_velocity"]) for data in human_data[start_ind:end_ind]]
human_joint_positions = [np.array(data["joint_positions"]) for data in human_data[start_ind:end_ind]]

# prepare visualizer
viz = vis.DualVisualizer(ml1=ml, ml2=human_ml, model1_name="robot", model2_name="human")
viz.load_model()

# Set the initial input values from the data
initial_input_path = script_directory + "/../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/" + "forward_backward" + "/extracted_features_X.txt"
initial_output_path = script_directory + "/../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/" + "forward_backward" + "/extracted_features_Y.txt"

with open(initial_input_path, 'r') as file:
    input_stuff = json.load(file)
# Get the initial input vector
input_vector = np.array(input_stuff[0])

# Compute component-wise input mean and standard deviation
datapath = os.path.join(model_dir, "normalization/")
Xmean, Xstd = load_input_mean_and_std(datapath)
Ymean, Ystd = load_output_mean_and_std(datapath)

# Define the initial past robot base velocities
current_past_base_linear_velocities = [[0.0, 0.0, 0.0] for _ in range(51)]
current_past_base_angular_velocities = [[0.0, 0.0, 0.0] for _ in range(51)] #TODO should really get this from a standing still portion of dataset

# Prepare the model for querying
model_path = model_dir + "/models/model_149.pth"
learned_model = torch.load(model_path, weights_only=False)
learned_model.eval()

length_of_time = 500
for i in range(length_of_time):

    # Get the human base pose from the input vector
    human_base_position = input_vector[130:133]
    human_base_orientation = input_vector[133:136]
    human_base_pose = np.vstack((np.hstack((Rotation.from_euler('xyz', human_base_orientation).as_matrix(), human_base_position.reshape(3,1))), np.array([0, 0, 0, 1])))

    # Compute current robot base pose
    current_robot_base_position = input_vector[124:127]
    current_robot_base_orientation = input_vector[127:130]
    current_robot_base_pose = np.vstack((np.hstack((Rotation.from_euler('xyz', current_robot_base_orientation).as_matrix(), current_robot_base_position.reshape(3,1))), np.array([0, 0, 0, 1])))

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

    # Get the robot pose and joint state from the output
    robot_joint_state = denormalized_current_output[42:68]
    robot_base_position = denormalized_current_output[94:97]
    robot_base_orientation = denormalized_current_output[97:]
    robot_base_pose = np.vstack((np.hstack((Rotation.from_euler('xyz', robot_base_orientation).as_matrix(), robot_base_position.reshape(3,1))), np.array([0, 0, 0, 1])))

    # Get the human joint state from the input vector
    #TODO ith or i+1 th?
    human_joint_state = human_joint_positions[i]

    # Update visualization
    viz.update_models(robot_joint_state, human_joint_state, robot_base_pose, human_base_pose)

    if i == 0:
        input("Press a key to start the trajectory generation")

    # Form input vector for next iteration
    new_input_vector = []

    # Compute the next velocity windows
    current_linear_velocity = denormalized_current_output[0:3]

    # Compute the new last 50 velocities
    current_past_base_linear_velocities = form_next_past_velocity_window(current_past_base_linear_velocities, current_linear_velocity, current_robot_base_pose[:3,:3], robot_base_pose[:3,:3])

    # Get the past window from the previous velocities
    subsampled_past_linear_velocities = [item for sublist in [current_past_base_linear_velocities[j] for j in [0, 10, 20, 30, 40, 50]] for item in sublist]
    future_linear_velocities = denormalized_current_output[3:21]
    new_input_vector.extend(subsampled_past_linear_velocities)
    new_input_vector.extend(future_linear_velocities)

    current_angular_velocity = denormalized_current_output[21:24]

    # Compute the new last 50 velocities
    current_past_base_angular_velocities = form_next_past_velocity_window(current_past_base_angular_velocities, current_angular_velocity, current_robot_base_pose[:3,:3], robot_base_pose[:3,:3])

    # Get the past window from the previous velocities
    subsampled_past_angular_velocities = [item for sublist in [current_past_base_angular_velocities[j] for j in [0, 10, 20, 30, 40, 50]] for item in sublist]
    new_input_vector.extend(subsampled_past_angular_velocities)

    future_angular_velocities = denormalized_current_output[24:42]
    new_input_vector.extend(future_angular_velocities)

    # Compute the next robot joint state
    new_input_vector.extend(robot_joint_state)

    # Compute the next robot joint velocity vector
    robot_joint_velocity = denormalized_current_output[68:94]
    new_input_vector.extend(robot_joint_velocity)

    # Compute next base position and orientation
    new_input_vector.extend(robot_base_position)
    new_input_vector.extend(robot_base_orientation)

    # Compute the next human base pose and velocity (from the "user input" data)
    new_human_base_position = human_base_positions[i+1]
    new_human_base_orientation = human_base_orientations[i+1]
    new_human_base_linear_velocity = human_base_linear_velocities[i+1]
    new_human_base_angular_velocity = human_base_angular_velocities[i+1]
    new_input_vector.extend(new_human_base_position)
    new_input_vector.extend(new_human_base_orientation)
    new_input_vector.extend(new_human_base_linear_velocity)
    new_input_vector.extend(new_human_base_angular_velocity)

    input_vector = np.array(new_input_vector)

