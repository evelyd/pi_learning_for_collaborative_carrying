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

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--filename", help="Mocap file to be retargeted. Relative path from script folder.",
                    type=str, default="../datasets/collaborative_payload_carrying/leader_backward/cheng1.mat")
parser.add_argument("--mirroring", help="Mirror the mocap data.", action="store_true")
parser.add_argument("--save", help="Store the retargeted motion in json format.", action="store_true")
parser.add_argument("--deactivate_visualization", help="Do not visualize the retargeted motion.", action="store_true")

args = parser.parse_args()

mocap_filename = args.filename
mirroring = args.mirroring
store_as_json = args.save
visualize_retargeted_motion = not args.deactivate_visualization

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
start_time_dict = {"cheng1": 12.59, "cheng2": 25.44, "evelyn1": 11.73, "evelyn2": 25.79}

# Extract the relevant part of the file name to determine the start time
file_key = None
for key in start_time_dict.keys():
    if key in mocap_filename:
        file_key = key
        break

if file_key is None:
    raise ValueError("The file name does not correspond to a defined start time.")

start_time = start_time_dict[file_key]
metadata = motion_data.MocapMetadata.build(start_time=start_time)
metadata.add_timestamp()
metadata.add_calibration()

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

# Set the robot to calibration pose
kindyn.setJointPos(qp_ik_params.get_parameter_vector_float("calibration_joint_positions"))

humanIK.initialize(qp_ik_params, kindyn)
humanIK.setDt(0.01)

# ===========
# RETARGETING
# ===========

# Define the initial base height
initial_base_height = utils.define_initial_base_height(robot="ergoCubV1")

# Instantiate the retargeter
retargeter = ifeel_data_retargeter.WBGR.build(motiondata=motiondata,
                                                metadata=metadata,
                                                humanIK=humanIK,
                                                joint_names=joint_names,
                                                kindyn=kindyn,
                                                mirroring=mirroring,
                                                initial_base_height=initial_base_height)

# Retrieve ik solutions
timestamps, ik_solutions = retargeter.retarget()

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
