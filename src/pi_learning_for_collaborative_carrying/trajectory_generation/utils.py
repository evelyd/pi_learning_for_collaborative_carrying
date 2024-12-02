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

def moving_average(vect, n) -> List:
        moving_averaged_vect = []
        for idx in range(0, len(vect)):
            if idx < n//2: # When there are less than N/2 frames before the current frame, average over the available frames
                moving_averaged_vect.append(np.mean(vect[:idx + n//2], axis=0))
            elif idx >= len(vect) - n//2: # When there are less than N/2 frames after the current frame, average over the available frames
                moving_averaged_vect.append(np.mean(vect[idx - n//2:], axis=0))
            else: # Average over N frames
                moving_averaged_vect.append(np.mean(vect[idx - n//2:idx + n//2], axis=0))

        return moving_averaged_vect

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

        for idx, data_point in enumerate(content):
            if idx % 2 == 0:
                continue

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

        for idx, data_point in enumerate(content):
            if idx % 2 == 0:
                continue

            base_position = data_point.get("base_position", [])
            base_quaternion = data_point.get("base_quaternion", [])
            robot_data.append({
                "base_position": base_position,
                "base_quaternion": base_quaternion
                })

    I_human_data_HB = []

    #Create the homogeneous robot transformation matrix
    robot_base_position = robot_data[0]["base_position"]
    robot_base_quaternion = robot_data[0]["base_quaternion"]
    robot_base_rotation_matrix = Rotation.from_quat(robot_base_quaternion, scalar_first=True).as_matrix()

    # Retargeted data always has to be transformed to start at the origin
    I_H_RB_initial = np.eye(4)
    I_H_RB_initial[:3, :3] = robot_base_rotation_matrix
    I_H_RB_initial[:3, 3] = np.array(robot_base_position)

    # For each timestep, rotate the world human base pose into the initial robot base frame (but don't translate off the ground), so basically only rotate/translate in the xy plane
    I_R_RB_initial_xy_angles = Rotation.from_matrix(I_H_RB_initial[:3, :3]).as_euler('xyz')
    I_R_RB_initial_z = Rotation.from_euler('z', I_R_RB_initial_xy_angles[2]).as_matrix()
    I_H_RB_initial_xy = np.eye(4)
    I_H_RB_initial_xy[:3, :3] = I_R_RB_initial_z # set rotation to be only the z rotation
    I_H_RB_initial_xy[:3, 3] = np.array([I_H_RB_initial[0,3], I_H_RB_initial[1,3], 0])

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

        # If the data used doesn't start at the origin it has to be transformed to start at the origin
        I_H_HB = np.linalg.inv(I_H_RB_initial_xy) @ Idata_H_HB

        # Store the data in the new format
        I_human_data_HB.append({"base_pose": I_H_HB,
                                 "base_linear_velocity": human_base_linear_velocity,
                                 "base_angular_velocity": human_base_angular_velocity,
                                 "joint_positions": human_joint_positions})

    # Smooth out the human base velocities the same as in feature extraction
    human_base_linear_velocities = [data["base_linear_velocity"] for data in I_human_data_HB]
    human_base_angular_velocities = [data["base_angular_velocity"] for data in I_human_data_HB]

    N = 9
    human_base_linear_velocities_smoothed = moving_average(human_base_linear_velocities, N)
    human_base_angular_velocities_smoothed = moving_average(human_base_angular_velocities, N)

    # Create new dict with human velocities in robot base frame
    RB_human_data_HB = []
    for i, data in zip(range(len(human_base_linear_velocities_smoothed)), I_human_data_HB):

        # Rotate the smoothed velocities into the robot base frame
        RB_human_base_linear_velocity = np.linalg.inv(robot_base_rotation_matrix).dot(human_base_linear_velocities_smoothed[i])
        RB_human_base_angular_velocity = np.linalg.inv(robot_base_rotation_matrix).dot(human_base_angular_velocities_smoothed[i])

        RB_human_data_HB.append({"base_pose": data["base_pose"],
                                 "base_linear_velocity": RB_human_base_linear_velocity,
                                 "base_angular_velocity": RB_human_base_angular_velocity,
                                 "joint_positions": data["joint_positions"]})

    # Extract human base positions over time
    human_positions = [data["base_pose"][:3, 3] for data in RB_human_data_HB]

    # Convert to numpy array for easier manipulation
    human_positions = np.array(human_positions)

    # Plot the human base positions over time
    plt.figure()
    plt.plot(human_positions[:, 0], human_positions[:, 1], label='Human Base Position')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Human Base Position Over Time')
    plt.legend()
    plt.grid(True)

    # Extract human base linear velocities over time
    human_linear_velocities = [data["base_linear_velocity"] for data in RB_human_data_HB]

    # Convert to numpy array for easier manipulation
    human_linear_velocities = np.array(human_linear_velocities)

    # Plot the human base linear velocities over time
    plt.figure()
    plt.plot(human_linear_velocities[:, 0], label='Linear Velocity X')
    plt.plot(human_linear_velocities[:, 1], label='Linear Velocity Y')
    plt.plot(human_linear_velocities[:, 2], label='Linear Velocity Z')
    plt.xlabel('Time Step')
    plt.ylabel('Linear Velocity')
    plt.title('Human Base Linear Velocity Over Time')
    plt.legend()
    plt.grid(True)

    # Extract human base angular velocities over time
    human_angular_velocities = [data["base_angular_velocity"] for data in RB_human_data_HB]

    # Convert to numpy array for easier manipulation
    human_angular_velocities = np.array(human_angular_velocities)

    # Plot the human base angular velocities over time
    plt.figure()
    plt.plot(human_angular_velocities[:, 0], label='Angular Velocity X')
    plt.plot(human_angular_velocities[:, 1], label='Angular Velocity Y')
    plt.plot(human_angular_velocities[:, 2], label='Angular Velocity Z')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Velocity')
    plt.title('Human Base Angular Velocity Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    return RB_human_data_HB