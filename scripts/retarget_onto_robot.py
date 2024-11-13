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
import resolve_robotics_uri_py
import biomechanical_analysis_framework as baf
from scipy.spatial.transform import Rotation

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--data_location", help="Mocap file to be retargeted. Relative path from script folder.",
                    type=str, default="../datasets/collaborative_payload_carrying/ifeel_and_vive/oct25_2024/forward_backward")
parser.add_argument("--leader", help="Retarget the leader data.", action="store_true")
parser.add_argument("--save", help="Store the retargeted motion in json format.", action="store_true")
parser.add_argument("--deactivate_visualization", help="Do not visualize the retargeted motion.", action="store_true")

args = parser.parse_args()

data_location = args.data_location
retarget_leader = args.leader
store_as_json = args.save
visualize_retargeted_motion = not args.deactivate_visualization

# ===============
# MODEL INSERTION
# ===============

# Retrieve the robot urdf model
if retarget_leader:
    script_directory = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_directory, "../human_model/humanSubjectWithMesh.urdf")
else:
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
if retarget_leader:
    toml = pathlib.Path("../src/pi_learning_for_collaborative_carrying/data_processing/qpik_leader.toml").expanduser()
else:
    toml = pathlib.Path("../src/pi_learning_for_collaborative_carrying/data_processing/qpik_follower.toml").expanduser()
assert toml.is_file()
ok = qp_ik_params.set_from_file(str(toml))

assert ok

# =====================
# IFEEL DATA CONVERSION
# =====================

# Basically want to run the same IK as ifeel uses, but on raw data -> robot instead of to human

# Original mocap data
script_directory = os.path.dirname(os.path.abspath(__file__))
if retarget_leader:
    mocap_filename = os.path.join(script_directory, data_location + "/leader/parsed_ifeel_data.mat")
else:
    mocap_filename = os.path.join(script_directory, data_location + "/follower/parsed_ifeel_data.mat")
vive_path = os.path.join(script_directory, data_location + "/vive/interpolated_vive_data.mat")

start_end_ind_dict = {"forward_backward/follower": [2049, 5173], "left_right/follower": [4539, 9193], "forward_backward/leader": [3100, 6224], "left_right/leader": [10275, 14929]}
# Extract the relevant part of the file name to determine the start time
file_key = None
for key in start_end_ind_dict.keys():
    if key in mocap_filename:
        file_key = key
        break

if file_key is None:
    raise ValueError("The file name does not correspond to a defined start time.")

start_ind = start_end_ind_dict[file_key][0]
end_ind = start_end_ind_dict[file_key][1]
metadata = motion_data.MocapMetadata.build(start_ind=start_ind, end_ind=end_ind)
metadata.add_timestamp()

# Add the tasks to which to assign the target orientation or force data
for task_name in qp_ik_params.get_parameter_vector_string("tasks"):
    task_type = qp_ik_params.get_group(task_name).get_parameter_string("type")
    # Check for node number, if there is none, use 0
    node_number = qp_ik_params.get_group(task_name).get_parameter_int("node_number") if ("node_number" in str(qp_ik_params.get_group(task_name))) else 0

    if ("target_frame_name" in str(qp_ik_params.get_group(task_name))):
        frame_name = qp_ik_params.get_group(task_name).get_parameter_string("target_frame_name")
    elif ("frame_name" in str(qp_ik_params.get_group(task_name))):
        frame_name = qp_ik_params.get_group(task_name).get_parameter_string("frame_name")
    else:
        frame_name = ""

    # Add a new task to the metadata
    metadata.add_task(task_name, task_type, node_number, frame_name)

# Instantiate the data converter
converter = data_converter.DataConverter.build(mocap_filename=mocap_filename, vive_filename=vive_path,
                                                          mocap_metadata=metadata, retarget_leader=retarget_leader)
# Convert the mocap data
motiondata = converter.convert()

# =======================
# HUMAN IK INITIALIZATION
# =======================

humanIK = baf.ik.HumanIK()

# Set the robot to calibration pose
if not retarget_leader:
    kindyn.setJointPos(qp_ik_params.get_parameter_vector_float("calibration_joint_positions"))

# Define the initial base height
if retarget_leader:
    initial_base_height = utils.define_initial_base_height(robot="humanSubjectWithMesh")
else:
    initial_base_height = utils.define_initial_base_height(robot="ergoCubV1")

# Set the robot to start at the pose of the first base measurement
initial_transform = idyn.Transform(motiondata.initial_base_pose)
kindyn.setWorldBaseTransform(initial_transform)

humanIK.initialize(qp_ik_params, kindyn)
humanIK.setDt(0.01)

# ===========
# RETARGETING
# ===========

# Instantiate the retargeter
retargeter = ifeel_data_retargeter.WBGR.build(motiondata=motiondata,
                                                metadata=metadata,
                                                humanIK=humanIK,
                                                joint_names=joint_names,
                                                kindyn=kindyn,
                                                initial_base_height=initial_base_height,
                                                retarget_leader=retarget_leader)

# Retrieve ik solutions
timestamps, ik_solutions = retargeter.retarget()

# =============
# STORE AS JSON
# =============

if store_as_json:

    if retarget_leader:
        participant = "leader"
    else:
        participant = "follower"

    outfile_name = os.path.join(script_directory, data_location + "/retargeted_motion_" + participant + ".txt")

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
