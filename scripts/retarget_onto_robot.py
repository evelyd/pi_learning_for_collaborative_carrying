# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
import numpy as np
from pi_learning_for_collaborative_carrying.data_processing import utils
from pi_learning_for_collaborative_carrying.data_processing import motion_data
from pi_learning_for_collaborative_carrying.data_processing import data_converter
from pi_learning_for_collaborative_carrying.data_processing import ifeel_data_retargeter
import pathlib
import idyntree.swig as idyn
import bipedal_locomotion_framework.bindings as blf
import jaxsim.api as js
from scipy.spatial.transform import Rotation
import resolve_robotics_uri_py
import biomechanical_analysis_framework as baf
import h5py
import manifpy as manif

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--filename", help="Mocap file to be retargeted. Relative path from script folder.",
                    type=str, default="../datasets/collaborative_payload_carrying/leader_backward/cheng1.mat")
parser.add_argument("--mirroring", help="Mirror the mocap data.", action="store_true")
parser.add_argument("--KFWBGR", help="Kinematically feasible Whole-Body Geometric Retargeting.", action="store_true")
parser.add_argument("--save", help="Store the retargeted motion in json format.", action="store_true")
parser.add_argument("--deactivate_horizontal_feet", help="Deactivate horizontal feet enforcing.", action="store_true")
parser.add_argument("--deactivate_straight_head", help="Deactivate straight head enforcing.", action="store_true")
parser.add_argument("--deactivate_wider_legs", help="Deactivate wider legs enforcing.", action="store_true")
parser.add_argument("--deactivate_visualization", help="Do not visualize the retargeted motion.", action="store_true")
parser.add_argument("--plot_ik_solutions", help="Show plots of the target task values and the IK solutions.", action="store_true")

args = parser.parse_args()

mocap_filename = args.filename
mirroring = args.mirroring
kinematically_feasible_base_retargeting = args.KFWBGR
store_as_json = args.save
horizontal_feet = not args.deactivate_horizontal_feet
straight_head = not args.deactivate_straight_head
wider_legs = not args.deactivate_wider_legs
visualize_retargeted_motion = not args.deactivate_visualization
plot_ik_solutions = args.plot_ik_solutions

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

# Load the idyn model
model_loader = idyn.ModelLoader()
assert model_loader.loadReducedModelFromFile(str(urdf_path), joint_names)

# create KinDynComputationsDescriptor
kindyn = idyn.KinDynComputations()
assert kindyn.loadRobotModel(model_loader.model())

# ===========================
# INVERSE KINEMATICS SETTINGS
# ===========================

# Set the parameters from toml file
qp_ik_params = blf.parameters_handler.TomlParametersHandler()
toml = pathlib.Path("../src/pi_learning_for_collaborative_carrying/data_processing/qpik.toml").expanduser()
assert toml.is_file()
ok = qp_ik_params.set_from_file(str(toml))

assert ok

# =====================
# IFEEL DATA CONVERSION
# =====================

# Basically want to run the same IK as ifeel uses, but on raw data -> robot instead of to human

# Original mocap data
script_directory = os.path.dirname(os.path.abspath(__file__))
mocap_filename = os.path.join(script_directory, mocap_filename)

# Define the relevant data for retargeting purposes
# for cheng 1 dataset: use time 17.19 for calibration with arms down, or 12.59 for calibration in t-pose
# for cheng 2 dataset: use time  for calibration with arms down, or 25.74 for calibration in t-pose
start_time = 12.59
metadata = motion_data.MocapMetadata.build(start_time=start_time)
metadata.add_timestamp()

# Add the tasks to which to assign the target orientation or force data
for task_name in qp_ik_params.get_parameter_vector_string("tasks"):
    task_type = qp_ik_params.get_group(task_name).get_parameter_string("type")
    # Check for node number, if there is none, use 0
    node_number = qp_ik_params.get_group(task_name).get_parameter_int("node_number") if ("node_number" in str(qp_ik_params.get_group(task_name))) else 0

    # Add a new task to the metadata
    metadata.add_task(task_name, task_type, node_number)

# Instantiate the data converter
converter = data_converter.DataConverter.build(mocap_filename=mocap_filename,
                                                          mocap_metadata=metadata)
# Convert the mocap data
motiondata = converter.convert()

# =======================
# HUMAN IK INITIALIZATION
# =======================

humanIK = baf.ik.HumanIK()

# TODO here need to set T pose to kindyn
kindyn.setJointPos(qp_ik_params.get_parameter_vector_float("calibration_joint_positions"))

humanIK.initialize(qp_ik_params, kindyn)
humanIK.setDt(0.01)

mocap_data = h5py.File(mocap_filename, 'r')
mocap_data_cleaned = mocap_data['robot_logger_device']

# Cut the data at the start time
# Get the index of the timestamp closest to the start time
start_time = 12.59
zeroed_timestamps = np.squeeze(mocap_data_cleaned['shoe1']['FT']['timestamps'][:] - mocap_data_cleaned['shoe1']['FT']['timestamps'][0])
start_time_index = np.argmin(np.abs(zeroed_timestamps - start_time))

# Assign the data for the SO3 and Gravity tasks into a struct
# Create the list of nodes in order of the toml file
orientation_nodes = [3, 6, 7, 8, 5, 4, 11, 12, 9, 10]
floor_contact_nodes = [1, 2]

node_struct = {}
for node in orientation_nodes + floor_contact_nodes:
    # Define time series of rotations for this node
    I_R_IMU = [manif.SO3(quaternion=utils.normalize_quaternion(utils.to_xyzw(quat))) for quat in np.squeeze(mocap_data_cleaned['node' + str(node)]['orientation']['data'][start_time_index:])]
    # Define time series of angular velocities for this node
    I_omega_IMU = [manif.SO3Tangent(omega) for omega in np.squeeze(mocap_data_cleaned['node' + str(node)]['angVel']['data'][start_time_index:])]
    # Assign these values to the node struct
    nodeData = baf.ik.nodeData()
    nodeData.I_R_IMU = I_R_IMU[0]
    nodeData.I_omega_IMU = I_omega_IMU[0]
    node_struct[node] = nodeData

# ===========
# RETARGETING
# ===========

# Define robot-specific feet frames
feet_frames, feet_links = utils.define_feet_frames_and_links(robot="ergoCubV1")

# Define robot-specific feet vertices positions in the foot frame
local_foot_vertices_pos = utils.define_foot_vertices(robot="ergoCubV1")

# Define robot-specific quaternions from the robot base frame to the target base frame
robot_to_target_base_quat = utils.define_robot_to_target_base_quat(robot="ergoCubV1")

initial_base_height = utils.define_initial_base_height(robot="ergoCubV1")

# Instantiate the retargeter
if kinematically_feasible_base_retargeting:
    retargeter = ifeel_data_retargeter.KFWBGR.build(motiondata=motiondata,
                                                     metadata=metadata,
                                                     param_handler=qp_ik_params,
                                                     joint_names=joint_names,
                                                     mirroring=mirroring,
                                                     horizontal_feet=horizontal_feet,
                                                     straight_head=straight_head,
                                                     wider_legs=wider_legs,
                                                     robot_to_target_base_quat=robot_to_target_base_quat,
                                                     kindyn=kindyn,
                                                     local_foot_vertices_pos=local_foot_vertices_pos,
                                                     feet_frames=feet_frames,
                                                     initial_base_height=initial_base_height)
else:
    retargeter = ifeel_data_retargeter.WBGR.build(motiondata=motiondata,
                                                   metadata=metadata,
                                                   humanIK=humanIK,
                                                   node_struct=node_struct,
                                                   param_handler=qp_ik_params,
                                                   joint_names=joint_names,
                                                   kindyn=kindyn,
                                                   mirroring=mirroring,
                                                   horizontal_feet=horizontal_feet,
                                                   straight_head=straight_head,
                                                   wider_legs=wider_legs,
                                                   robot_to_target_base_quat=robot_to_target_base_quat,
                                                   initial_base_height=initial_base_height)

# Retrieve ik solutions
if kinematically_feasible_base_retargeting:
    timestamps, ik_solutions = retargeter.KF_retarget(plot_ik_solutions=plot_ik_solutions)
else:
    timestamps, ik_solutions = retargeter.retarget(plot_ik_solutions=plot_ik_solutions)

# =============
# STORE AS JSON
# =============

if store_as_json:

    outfile_name = os.path.join(script_directory, "retargeted_motion.txt")

    input("Press Enter to store the retargeted mocap into a json file")
    utils.store_retargeted_mocap_as_json(timestamps=timestamps, ik_solutions=ik_solutions, outfile_name=outfile_name)
    print("\nThe retargeted mocap data have been saved in", outfile_name, "\n")

# ===============================
# VISUALIZE THE RETARGETED MOTION
# ===============================

if visualize_retargeted_motion:

    input("Press Enter to start the visualization of the retargeted motion")
    utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions,
                                      js_model=js_model, controlled_joints=joint_names)

input("Close")
