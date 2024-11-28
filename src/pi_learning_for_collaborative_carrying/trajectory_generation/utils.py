# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import math
import time
import json
import numpy as np
from typing import List, Dict
from scipy.spatial.transform import Rotation
import dataclasses
import idyntree.swig as idyn

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# ===========================
# TRAJECTORY GENERATION UTILS
# ===========================

def get_human_base_pose_from_retargeted_data(human_file_path: str, robot_file_path: str) -> List[Dict[str, float]]:
    """
    Reads data from a file and returns it as a list of dictionaries.

    :param file_path: Path to the file containing the data.
    :return: List of dictionaries with the data.
    """
    human_data = []
    with open(human_file_path, 'r') as file:
        content = file.read()
        content = json.loads(content)

        for data_point in content:

            base_position = data_point.get("base_position", [])
            base_quaternion = data_point.get("base_quaternion", [])
            base_linear_velocity = data_point.get("base_linear_velocity", [])
            base_angular_velocity = data_point.get("base_angular_velocity", [])
            joint_positions = data_point.get("joint_positions", [])
            human_data.append({
                "base_position": base_position,
                "base_quaternion": base_quaternion,
                "base_linear_velocity": base_linear_velocity,
                "base_angular_velocity": base_angular_velocity,
                "joint_positions": joint_positions
                })

    robot_data = []
    with open(robot_file_path, 'r') as file:
        content = file.read()
        content = json.loads(content)

        for data_point in content:

            base_position = data_point.get("base_position", [])
            base_quaternion = data_point.get("base_quaternion", [])
            robot_data.append({
                "base_position": base_position,
                "base_quaternion": base_quaternion
                })

    # For each timestep, rotate the world human base pose into the robot base frame
    RB_human_data_HB = []

    #Create the homogeneous robot transformation matrix
    robot_base_position = robot_data[0]["base_position"]
    robot_base_quaternion = robot_data[0]["base_quaternion"]
    robot_base_rotation_matrix = Rotation.from_quat(robot_base_quaternion, scalar_first=True).as_matrix()
    I_H_RB_initial = np.eye(4)
    I_H_RB_initial[:3, :3] = robot_base_rotation_matrix
    I_H_RB_initial[:3, 3] = np.array(robot_base_position)
    for human_data in human_data:
        human_base_position = human_data["base_position"]
        human_base_quaternion = human_data["base_quaternion"]
        human_base_linear_velocity = human_data["base_linear_velocity"]
        human_base_angular_velocity = human_data["base_angular_velocity"]
        human_joint_positions = human_data["joint_positions"]

        # Convert the quaternion to a rotation matrix, retargeted data uses wxyz quaternion form
        human_base_rotation_matrix = Rotation.from_quat(human_base_quaternion, scalar_first=True).as_matrix()

        # Create the homogeneous human transformation matrix
        Idata_H_HB = np.eye(4)
        Idata_H_HB[:3, :3] = human_base_rotation_matrix
        Idata_H_HB[:3, 3] = np.array(human_base_position)

        # For each timestep, rotate the world human base pose into the initial robot base frame (but don't translate off the ground), so basically only rotate/translate in the xy plane
        I_R_RB_initial_xy_angles = Rotation.from_matrix(I_H_RB_initial[:3, :3]).as_euler('xyz')
        I_R_RB_initial_z = Rotation.from_euler('z', I_R_RB_initial_xy_angles[2]).as_matrix()
        I_H_RB_initial_xy = np.eye(4)
        I_H_RB_initial_xy[:3, :3] = I_R_RB_initial_z # set rotation to be only the z rotation
        I_H_RB_initial_xy[:3, 3] = np.array([I_H_RB_initial[0,3], I_H_RB_initial[1,3], 0])
        Iinference_H_HB = np.linalg.inv(I_H_RB_initial_xy) @ Idata_H_HB

        # Rotate the velocities into the robot base frame
        RB_human_base_linear_velocity = np.linalg.inv(robot_base_rotation_matrix).dot(human_base_linear_velocity)
        RB_human_base_angular_velocity = np.linalg.inv(robot_base_rotation_matrix).dot(human_base_angular_velocity)

        # Store the data in the new format
        RB_human_data_HB.append({"base_pose": Iinference_H_HB,
                                 "base_linear_velocity": RB_human_base_linear_velocity,
                                 "base_angular_velocity": RB_human_base_angular_velocity,
                                 "joint_positions": human_joint_positions})

    return RB_human_data_HB