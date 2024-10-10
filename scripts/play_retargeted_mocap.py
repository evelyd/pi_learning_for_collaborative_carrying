# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import os
import argparse
from pi_learning_for_collaborative_carrying.data_processing import utils
import jaxsim.api as js
import resolve_robotics_uri_py

# ==================
# USER CONFIGURATION
# ==================

parser = argparse.ArgumentParser()

parser.add_argument("--filename", help="Retargeted file to play. Relative path from script folder.",
                    type=str, default="retargeted_motion.txt")

args = parser.parse_args()

filename = args.filename

# ====================
# LOAD RETARGETED DATA
# ====================

# Retrieve script directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Load the retargeted mocap data
timestamps, ik_solutions = utils.load_retargeted_mocap_from_json(input_file_name=filename)

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

# ===============================
# VISUALIZE THE RETARGETED MOTION
# ===============================

input("Press Enter to start the visualization of the retargeted motion")
utils.visualize_retargeted_motion(timestamps=timestamps, ik_solutions=ik_solutions, js_model=js_model,
                                  controlled_joints=controlled_joints, controlled_joints_indexes=controlled_joints_indexes)
