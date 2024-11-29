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

import mujoco
import mujoco.viewer
import jaxsim.api as js
from jaxsim.mujoco import UrdfToMjcf

import pi_learning_for_collaborative_carrying.trajectory_generation.DualVisualizer as vis

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# =================
# RETARGETING UTILS
# =================

@dataclasses.dataclass
class IKSolution:

    joint_configuration: np.ndarray
    joint_velocities: np.ndarray
    base_position: np.array = dataclasses.field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    base_quaternion: np.array = dataclasses.field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    base_linear_velocity: np.array = dataclasses.field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    base_angular_velocity: np.array = dataclasses.field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

def define_initial_base_height(robot: str) -> List:
    """Define the robot-specific quaternions from the robot base frame to the target base frame."""

    if robot == "ergoCubV1":
        initial_base_height = 0.7754
    elif robot == "humanSubjectWithMesh":
        initial_base_height = 0.917163
    else:
        raise Exception("Initial base height only defined for ergoCubV1.")

    return initial_base_height

def to_xyzw(wxyz: List) -> List:
    """Auxiliary function to convert quaternions from wxyz to xyzw format."""

    return wxyz[[1, 2, 3, 0]]

def to_wxyz(xyzw: List) -> List:
    """Auxiliary function to convert quaternions from xyzw to wxyz format."""

    return xyzw[[3, 0, 1, 2]]

def normalize_quaternion(quat: List) -> List:
    """Auxiliary function to normalize quaternions."""

    norm = np.linalg.norm(quat)
    normalized_quat = [quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm]

    return normalized_quat

def store_retargeted_mocap_as_json(timestamps: List, ik_solutions: List, outfile_name: str) -> None:
    """Auxiliary function to store the retargeted motion."""

    ik_solutions_json = []

    for i in range(1, len(ik_solutions)):

        ik_solution = ik_solutions[i]

        ik_solution_json = {"joint_positions": ik_solution.joint_configuration.tolist(),
                            "joint_velocities": ik_solution.joint_velocities.tolist(),
                            "base_position": ik_solution.base_position,
                            "base_quaternion": ik_solution.base_quaternion.tolist(),
                            "base_linear_velocity": ik_solution.base_linear_velocity.tolist(),
                            "base_angular_velocity": ik_solution.base_angular_velocity.tolist(),
                            "timestamp": timestamps[i]}

        ik_solutions_json.append(ik_solution_json)

    with open(outfile_name, "w") as outfile:
        json.dump(ik_solutions_json, outfile)

def load_retargeted_mocap_from_json(input_file_name: str, initial_frame: int = 0, final_frame: int = -1) -> (List, List):
    """Auxiliary function to load the retargeted mocap data."""

    # Load ik solutions
    with open(input_file_name, 'r') as openfile:
        ik_solutions = json.load(openfile)

    # If a final frame has been passed, extract relevant ik solutions
    if initial_frame != -1:
        ik_solutions = ik_solutions[initial_frame:final_frame]

    # Extract timestamps
    timestamps = [ik_solution["timestamp"] for ik_solution in ik_solutions]

    return timestamps, ik_solutions

# =========================
# FEATURE EXTRACTION UTILS
# =========================

def transform_from_pos_quat(position: List, quaternion: List) -> np.ndarray:

    H = np.eye(4)
    H[:3,3] = position
    H[:3,:3] = Rotation.from_quat(to_xyzw(quaternion)).as_matrix()

    return H

def reset_robot_configuration(kindyn: idyn.KinDynComputations, joint_positions: List, base_position: List, base_quaternion: List) -> None:
        """Reset the robot configuration."""

        world_H_base = idyn.Transform(transform_from_pos_quat(base_position, base_quaternion))
        s = idyn.VectorDynSize.FromPython(joint_positions)
        ds = idyn.VectorDynSize.FromPython(np.zeros(len(joint_positions)))
        base_twist = idyn.Twist.FromPython(np.zeros(6))
        g = idyn.Vector3.FromPython(np.array([0, 0, -9.806]))

        kindyn.setRobotState(world_H_base, s, base_twist, ds, g)

def idyn_transform_to_np(H_idyn: idyn.Transform) -> np.ndarray:

    transform_as_string = H_idyn.toString()
    idyn_style_transform_as_np = np.fromstring(transform_as_string, dtype=float, sep=' ').reshape(4,3)
    H_np = np.eye(4)
    H_np[:3,:3] = idyn_style_transform_as_np[:3,:]
    H_np[:3,-1] = idyn_style_transform_as_np[-1,:]

    return H_np

# ===================
# VISUALIZATION UTILS
# ===================

def visualize_retargeted_motion(timestamps: List,
                                ik_solutions: List,
                                js_model: js.model.JaxSimModel,
                                controlled_joints: List,
                                controlled_joints_indexes: List = []) -> None:
    """Auxiliary function to visualize retargeted motion."""

    timestamp_prev = -1

    mjcf_string, assets = UrdfToMjcf.convert(urdf=js_model.built_from)

    # Create the mujoco objects
    env = mujoco.MjModel.from_xml_string(mjcf_string, assets)
    data = mujoco.MjData(env)

    # Launch a passive viewer
    handle = mujoco.viewer.launch_passive(
            env, data, show_left_ui=False, show_right_ui=False
        )

    # Define the vantage point and convert the env to a mujoco string
    camera = {
    "trackbodyid": mujoco.mj_name2id(env, mujoco.mjtObj.mjOBJ_BODY, 'root_link'),
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
    }

    for key, value in camera.items():
            if isinstance(value, np.ndarray):
                getattr(handle.cam, key)[:] = value
            else:
                setattr(handle.cam, key, value)

    with handle as viewer:
        for i in range(0, len(ik_solutions)-1):

            ik_solution = ik_solutions[i]

            # Retrieve the base pose and the joint positions, based on the type of ik_solution
            if type(ik_solution) == IKSolution:
                joint_positions = ik_solution.joint_configuration
                base_position = ik_solution.base_position
                base_quaternion = ik_solution.base_quaternion

            elif type(ik_solution) == dict:
                joint_positions = np.asarray(ik_solution["joint_positions"])
                base_position = np.asarray(ik_solution["base_position"])
                base_quaternion = np.asarray(ik_solution["base_quaternion"])

                # Reorder the joint positions to match the controlled joints
                joint_positions = np.array([joint_positions[index] for index in controlled_joints_indexes])

            # Set the base position
            data.qpos[:3] = np.array(base_position)

            # Set the base orientation (expects wxyz form)
            data.qpos[3:7] = np.array(base_quaternion)

            # Set joint positions
            for joint_name in controlled_joints:
                data.joint(joint_name).qpos = joint_positions[controlled_joints.index(joint_name)]

            # Visualize the retargeted motion at the time rate of the collected data
            timestamp = timestamps[i]
            if timestamp_prev == -1:
                dt = 1 / 100
            else:
                dt = timestamp - timestamp_prev
            timestamp_prev = timestamp

            # Update the camera to follow the link
            with viewer.lock():
                viewer.cam.lookat[:] = data.qpos[:3]

            mujoco.mj_step(env, data)
            viewer.sync()
            time.sleep(dt)

            if i == 0:
                input("Start visualization")

    print("Visualization ended")
    time.sleep(1)

def visualize_global_features(global_window_features,
                              global_frame_features,
                              ik_solutions: List,
                              js_model: js.model.JaxSimModel,
                              controlled_joints: List,
                              plot_facing_directions: bool = True,
                              plot_base_velocities: bool = True) -> None:
    """Visualize the retargeted frames along with the associated global features."""

    window_length_frames = global_window_features.window_length_frames
    window_step = global_window_features.window_step
    window_indexes = global_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    # Define the vantage point and convert the env to a mujoco string
    camera = {
    "trackbodyid": mujoco.mj_name2id(env, mujoco.mjtObj.mjOBJ_BODY, 'root_link'),
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
    }

    for key, value in camera.items():
            if isinstance(value, np.ndarray):
                getattr(handle.cam, key)[:] = value
            else:
                setattr(handle.cam, key, value)

    mjcf_string, assets = UrdfToMjcf.convert(urdf=js_model.built_from, cameras=camera)

    # Create the mujoco objects
    env = mujoco.MjModel.from_xml_string(mjcf_string, assets)
    data = mujoco.MjData(env)

    # Launch a passive viewer
    handle = mujoco.viewer.launch_passive(
            env, data, show_left_ui=False, show_right_ui=False
        )

    with handle as viewer:
        for i in range(initial_frame, final_frame):

            if not viewer.is_running():
                break
            # Debug
            print(i - initial_frame, "/", final_frame - initial_frame)

            # Retrieve the base pose and the joint positions
            joint_positions = global_frame_features.s[i]
            base_position = global_frame_features.base_positions[i]
            base_quaternion = global_frame_features.base_quaternions[i]

            # Set the base position
            data.qpos[:3] = np.array(base_position)

            # Set the base orientation (expects wxyz form)
            data.qpos[3:7] = np.array(base_quaternion)

            # Set joint positions
            for joint_name in controlled_joints:
                data.joint(joint_name).qpos = joint_positions[controlled_joints.index(joint_name)]

            # Retrieve global features
            base_positions = global_window_features.base_positions[i - window_length_frames]
            base_velocities = global_window_features.base_velocities[i - window_length_frames]
            base_quaternions = global_window_features.base_quaternions[i - window_length_frames]

            # Update the camera to follow the link
            with viewer.lock():
                viewer.cam.lookat[:] = data.qpos[:3]

            mujoco.mj_step(env, data)
            viewer.sync()
            time.sleep(1/50)

            if i == 1:
                input("Start visualization")

            # =================
            # FACING DIRECTIONS
            # =================

            if plot_facing_directions:

                # Figure 1 for the facing directions
                plt.figure(1)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot base directions
                    base_euler_angle = Rotation.from_quat(to_xyzw(base_quaternions[j])).as_euler('xyz')[2]
                    base_direction = np.array([np.cos(base_euler_angle), np.sin(base_euler_angle)])
                    base_direction = base_direction / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current base direction in blue
                        plt.plot([base_position[0], base_position[0] + 2 * base_direction[0]],
                                 [base_position[1], base_position[1] + 2 * base_direction[1]],
                                 'b', label="Current facing direction")
                    else:
                        # Other base directions in green
                        plt.plot([base_position[0], base_position[0] + base_direction[0]],
                                 [base_position[1], base_position[1] + base_direction[1]], 'g')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Facing directions (global view)")
                plt.legend()

            # ===============
            # BASE VELOCITIES
            # ===============

            if plot_base_velocities:

                # Figure 2 for the base velocities
                plt.figure(2)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot base velocities
                    base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current base velocity in magenta
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]],
                                  'm', label="Current base velocity")
                    else:
                        # Other base velocities in gray
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]], 'gray')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Base velocities (global view)")
                plt.legend()

            # Plot
            plt.show()
            plt.pause(0.0001)

def visualize_local_features(local_window_features,
                             global_frame_features,
                             ik_solutions: List,
                             js_model: js.model.JaxSimModel,
                             controlled_joints: List,
                             plot_facing_directions: bool = True,
                             plot_base_velocities: bool = True) -> None:
    """Visualize the retargeted frames along with the associated local features."""

    window_length_frames = local_window_features.window_length_frames
    window_step = local_window_features.window_step
    window_indexes = local_window_features.window_indexes
    initial_frame = window_length_frames
    final_frame = round(len(ik_solutions)/2) - window_length_frames - window_step - 1

    plt.ion()

    # Define the vantage point and convert the env to a mujoco string
    camera = {
    "trackbodyid": mujoco.mj_name2id(env, mujoco.mjtObj.mjOBJ_BODY, 'root_link'),
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
    }

    for key, value in camera.items():
            if isinstance(value, np.ndarray):
                getattr(handle.cam, key)[:] = value
            else:
                setattr(handle.cam, key, value)

    mjcf_string, assets = UrdfToMjcf.convert(urdf=js_model.built_from, cameras=camera)

    # Create the mujoco objects
    env = mujoco.MjModel.from_xml_string(mjcf_string, assets)
    data = mujoco.MjData(env)

    # Launch a passive viewer
    handle = mujoco.viewer.launch_passive(
            env, data, show_left_ui=False, show_right_ui=False
        )

    with handle as viewer:
        for i in range(initial_frame, final_frame):

            if not viewer.is_running():
                break
            # Debug
            print(i - initial_frame, "/", final_frame - initial_frame)

            # Retrieve the base pose and the joint positions
            joint_positions = global_frame_features.s[i]
            base_position = global_frame_features.base_positions[i]
            base_quaternion = global_frame_features.base_quaternions[i]

            # Set the base position
            data.qpos[:3] = np.array(base_position)

            # Set the base orientation (expects wxyz form)
            data.qpos[3:7] = np.array(base_quaternion)

            # Set joint positions
            for joint_name in controlled_joints:
                data.joint(joint_name).qpos = joint_positions[controlled_joints.index(joint_name)]

            # Retrieve global features
            base_positions = local_window_features.base_positions[i - window_length_frames]
            base_velocities = local_window_features.base_velocities[i - window_length_frames]
            base_quaternions = local_window_features.base_quaternions[i - window_length_frames]

            # Update the camera to follow the link
            with viewer.lock():
                viewer.cam.lookat[:] = data.qpos[:3]

            mujoco.mj_step(env, data)
            viewer.sync()
            time.sleep(1/50)

            if i == 1:
                input("Start visualization")

            # =================
            # FACING DIRECTIONS
            # =================

            if plot_facing_directions:

                # Figure 1 for the facing directions
                plt.figure(1)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot facing directions
                    base_euler_angle = Rotation.from_quat(to_xyzw(base_quaternions[j])).as_euler('xyz')[2]
                    base_direction = np.array([np.cos(base_euler_angle), np.sin(base_euler_angle)])
                    base_direction = base_direction / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current facing direction in blue
                        plt.plot([base_position[0], base_position[0] + 2 * base_direction[0]],
                                [base_position[1], base_position[1] + 2 * base_direction[1]], 'b',
                                label="Current facing direction")
                    else:
                        # Other facing directions in green
                        plt.plot([base_position[0], base_position[0] + base_direction[0]],
                                 [base_position[1], base_position[1] + base_direction[1]], 'g')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Facing directions (local view)")
                plt.legend()

            # ===============
            # BASE VELOCITIES
            # ===============

            if plot_base_velocities:

                # Figure 2 for the base velocities
                plt.figure(2)
                plt.clf()

                for j in range(len(base_positions)):

                    # Plot base positions
                    base_position = base_positions[j]
                    if window_indexes[j] == 0:
                        # Set the center of the plot
                        center = base_position

                        # Current base position in red
                        plt.scatter(base_position[0], base_position[1], c='r', label="Current base position")
                    else:
                        # Other base positions in black
                        plt.scatter(base_position[0], base_position[1], c='k')

                    # Plot base velocities
                    base_velocity = base_velocities[j] / 10  # scaled for visualization purposes
                    if window_indexes[j] == 0:
                        # Current base velocity in magenta
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]],
                                  'm', label="Current base velocity")
                    else:
                        # Other base velocities in gray
                        plt.plot([base_position[0], base_position[0] + base_velocity[0]],
                                 [base_position[1], base_position[1] + base_velocity[1]], 'gray')

                # Configuration
                plt.xlim([center[0]-1, center[0]+1])
                plt.ylim([center[1]-1, center[1]+1])
                plt.title("Base velocities (local view)")
                plt.legend()

            # Plot
            plt.show()
            plt.pause(0.0001)

def visualize_meshcat(global_frame_features,
                      robot_ml, human_ml):

    # prepare visualizer
    viz = vis.DualVisualizer(ml1=robot_ml, ml2=human_ml, model1_name="robot", model2_name="human")
    viz.load_model()

    input("Start visualization")

    for i in range(len(global_frame_features.base_positions)):

        # Retrieve the base pose and the joint positions
        base_position = global_frame_features.base_positions[i]
        base_quaternion = global_frame_features.base_quaternions[i]
        joint_positions = global_frame_features.s[i]

        human_base_position = global_frame_features.human_base_positions[i]
        human_base_quaternion = global_frame_features.human_base_quaternions[i]
        human_joint_positions = global_frame_features.human_s[i]

        I_H_RB = np.vstack((np.hstack((Rotation.from_quat(base_quaternion, scalar_first=True).as_matrix(), base_position.reshape(3, 1))), [0, 0, 0, 1]))

        I_H_HB = np.vstack((np.hstack((Rotation.from_quat(human_base_quaternion, scalar_first=True).as_matrix(), human_base_position.reshape(3, 1))), [0, 0, 0, 1]))

        # Update the model configuration
        viz.update_models(joint_positions, human_joint_positions, I_H_RB, I_H_HB)