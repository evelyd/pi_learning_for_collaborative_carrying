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
urdf_path = pathlib.Path("../src/pi_learning_for_collaborative_carrying/model/ergoCubGazeboV1_xsens/ergoCubGazeboV1_xsens.urdf")

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
qp_ik_params.set_parameter_string(name="robot_velocity_variable_name", value="robotVelocity")

# =========================================
# Set the extra joint limit task parameters

# Get the model joint limits
lim_joints = qp_ik_params.get_group("JOINT_LIMITS_TASK").get_parameter_vector_string("joints_list")
k_limits = qp_ik_params.get_group("JOINT_LIMITS_TASK").get_parameter_float("k_limits")
upper_bounds = qp_ik_params.get_group("JOINT_LIMITS_TASK").get_parameter_vector_float("upper_bounds")
lower_bounds = qp_ik_params.get_group("JOINT_LIMITS_TASK").get_parameter_vector_float("lower_bounds")

# Get joint limits from model urdf
lower = np.array([kindyn.model().getJoint(i).getMinPosLimit(i) for i in range(len(joint_names))])
upper = np.array([kindyn.model().getJoint(i).getMaxPosLimit(i) for i in range(len(joint_names))])
# Update with the specified joints
for joint in lim_joints:
    lower[joint_names.index(joint)] = lower_bounds[lim_joints.index(joint)]
    upper[joint_names.index(joint)] = upper_bounds[lim_joints.index(joint)]

# Assign the vector values to the task
qp_ik_params.get_group("JOINT_LIMITS_TASK").set_parameter_vector_float(name="lower_limits", value=lower)
qp_ik_params.get_group("JOINT_LIMITS_TASK").set_parameter_vector_float(name="upper_limits", value=upper)
qp_ik_params.get_group("JOINT_LIMITS_TASK").set_parameter_vector_float(name="klim", value=np.array([k_limits] * len(joint_names)))

# ==================================================
# Set the extra joint regularization task parameters

qp_ik_params.get_group("JOINT_REG_TASK").set_parameter_vector_float(name="kp", value=np.array([0.0] * len(joint_names)))
weight_float = qp_ik_params.get_group("JOINT_REG_TASK").get_parameter_float(name="weight")
qp_ik_params.get_group("JOINT_REG_TASK").set_parameter_vector_float(name="weight", value=np.array([weight_float] * len(joint_names)))

# ==================================================
# Set the extra joint velocity limit task parameters

upper_limit = qp_ik_params.get_group("JOINT_VEL_LIMITS_TASK").get_parameter_float("upper_limit")
lower_limit = qp_ik_params.get_group("JOINT_VEL_LIMITS_TASK").get_parameter_float("lower_limit")
qp_ik_params.get_group("JOINT_VEL_LIMITS_TASK").set_parameter_vector_float(name="upper_limits", value=np.array([upper_limit] * len(joint_names)))
qp_ik_params.get_group("JOINT_VEL_LIMITS_TASK").set_parameter_vector_float(name="lower_limits", value=np.array([lower_limit] * len(joint_names)))

assert ok

# Build the QPIK object
(variables_handler, tasks, ik_solver) = blf.ik.QPInverseKinematics.build(
    param_handler=qp_ik_params, kin_dyn=kindyn
)

ik_solver: blf.ik.QPInverseKinematics

# =====================
# IFEEL DATA CONVERSION
# =====================

# Basically want to run the same IK as ifeel uses, but on raw data -> robot instead of to human

# Original mocap data
script_directory = os.path.dirname(os.path.abspath(__file__))
mocap_filename = os.path.join(script_directory, mocap_filename)

# Define the relevant data for retargeting purposes
metadata = motion_data.MocapMetadata.build(start_time=14.38)
metadata.add_timestamp()

# Add the tasks to which to assign the target orientation or force data
for task_name in qp_ik_params.get_parameter_vector_string("tasks"):
    # Get the values from the parameter handler
    if "target_frame_name" in str(qp_ik_params.get_group(task_name)):
        frame = qp_ik_params.get_group(task_name).get_parameter_string("target_frame_name")
    elif "frame_name" in str(qp_ik_params.get_group(task_name)):
        frame = qp_ik_params.get_group(task_name).get_parameter_string("frame_name")
    else:
        frame = ""

    task_type = qp_ik_params.get_group(task_name).get_parameter_string("type")
    # Check for node number, if there is none, use 0
    node_number = qp_ik_params.get_group(task_name).get_parameter_int("node_number") if ("node_number" in str(qp_ik_params.get_group(task_name))) else 0
    # Check for a rotation matrix, if there is none, use identity
    if "rotation_matrix" in str(qp_ik_params.get_group(task_name)):
        rotation_matrix = qp_ik_params.get_group(task_name).get_parameter_vector_float("rotation_matrix")
        IMU_R_link = Rotation.from_matrix(np.reshape(rotation_matrix, (3, 3)))
    else:
        IMU_R_link = Rotation.identity()

    if "vertical_force_threshold" in str(qp_ik_params.get_group(task_name)):
        force_threshold = qp_ik_params.get_group(task_name).get_parameter_float("vertical_force_threshold")
    else:
        force_threshold = 0.0

    if "weight" in str(qp_ik_params.get_group(task_name)):
        weight = qp_ik_params.get_group(task_name).get_parameter_vector_float("weight")
    else:
        weight = [0.0, 0.0, 0.0]

    # Add a new task to the metadata
    metadata.add_task(task_name, task_type, frame, node_number, IMU_R_link, force_threshold, weight)

# Instantiate the data converter
converter = data_converter.DataConverter.build(mocap_filename=mocap_filename,
                                                          mocap_metadata=metadata)
# Convert the mocap data
motiondata = converter.convert()

# ===========
# RETARGETING
# ===========

# Define robot-specific feet frames
feet_frames, feet_links = utils.define_feet_frames_and_links(robot="ergoCubV1")

# Define robot-specific feet vertices positions in the foot frame
local_foot_vertices_pos = utils.define_foot_vertices(robot="ergoCubV1")

# Define robot-specific quaternions from the robot base frame to the target base frame
robot_to_target_base_quat = utils.define_robot_to_target_base_quat(robot="ergoCubV1")

# Instantiate the retargeter
if kinematically_feasible_base_retargeting:
    retargeter = ifeel_data_retargeter.KFWBGR.build(motiondata=motiondata,
                                                     metadata=metadata,
                                                     ik_solver=ik_solver,
                                                     param_handler=qp_ik_params,
                                                     joint_names=joint_names,
                                                     mirroring=mirroring,
                                                     horizontal_feet=horizontal_feet,
                                                     straight_head=straight_head,
                                                     wider_legs=wider_legs,
                                                     robot_to_target_base_quat=robot_to_target_base_quat,
                                                     kindyn=kindyn,
                                                     local_foot_vertices_pos=local_foot_vertices_pos,
                                                     feet_frames=feet_frames)
else:
    retargeter = ifeel_data_retargeter.WBGR.build(motiondata=motiondata,
                                                   metadata=metadata,
                                                   ik_solver=ik_solver,
                                                   param_handler=qp_ik_params,
                                                   joint_names=joint_names,
                                                   kindyn=kindyn,
                                                   mirroring=mirroring,
                                                   horizontal_feet=horizontal_feet,
                                                   straight_head=straight_head,
                                                   wider_legs=wider_legs,
                                                   robot_to_target_base_quat=robot_to_target_base_quat)

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
