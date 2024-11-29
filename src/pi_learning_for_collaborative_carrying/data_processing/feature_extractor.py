# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import List
from dataclasses import dataclass, field
from pi_learning_for_collaborative_carrying.data_processing import utils
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath

@dataclass
class GlobalFrameFeatures:
    """Class for the global features associated to each retargeted frame."""

    # Feature computation
    ik_solutions: List
    human_ik_solutions: List
    controlled_joints_indexes: List
    dt_mean: float

    # Feature storage
    base_positions: List = field(default_factory=list)
    base_linear_velocities: List = field(default_factory=list)
    base_angular_velocities: List = field(default_factory=list)
    s: List = field(default_factory=list)
    s_dot: List = field(default_factory=list)
    base_quaternions: List = field(default_factory=list)

    s_dot_raw: List = field(default_factory=list)
    base_linear_velocities_raw: List = field(default_factory=list)
    base_angular_velocities_raw: List = field(default_factory=list)

    # Human feature storage
    human_base_positions: List = field(default_factory=list)
    human_base_quaternions: List = field(default_factory=list)
    human_base_linear_velocities_raw: List = field(default_factory=list)
    human_base_angular_velocities_raw: List = field(default_factory=list)
    human_base_linear_velocities: List = field(default_factory=list)
    human_base_angular_velocities: List = field(default_factory=list)

    human_s: List = field(default_factory=list)

    plot_global_vels: bool = False
    plot_human_features: bool = False
    plot_robot_v_human: bool = False
    start_at_origin: bool = False

    @staticmethod
    def build(ik_solutions: List,
              leader_ik_solutions: dict,
              controlled_joints_indexes: List,
              dt_mean: float,
              plot_global_vels: bool = False,
              plot_human_features: bool = False,
              plot_robot_v_human: bool = False,
              start_at_origin: bool = False) -> "GlobalFrameFeatures":
        """Build an empty GlobalFrameFeatures."""

        return GlobalFrameFeatures(ik_solutions=ik_solutions,
                                   human_ik_solutions=leader_ik_solutions,
                                   controlled_joints_indexes=controlled_joints_indexes,
                                   dt_mean=dt_mean, plot_global_vels=plot_global_vels,
                                   plot_human_features=plot_human_features,
                                   plot_robot_v_human=plot_robot_v_human,
                                   start_at_origin=start_at_origin
                                   )

    def plot_raw_and_smoothed_vels(self) -> None:

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        axs[0].plot([np.linalg.norm(vel) for vel in self.base_linear_velocities_raw], label='Raw linear velocity')
        axs[0].plot([np.linalg.norm(vel) for vel in self.base_linear_velocities], label='Smoothed linear velocity')
        axs[0].set_title('Base linear velocity')
        axs[0].set_xlabel('Frame index')
        axs[0].set_ylabel('Norm of the velocity (m/s)')
        axs[0].legend()

        axs[1].plot([np.linalg.norm(vel) for vel in self.base_angular_velocities_raw], label='Raw angular velocity')
        axs[1].plot([np.linalg.norm(vel) for vel in self.base_angular_velocities], label='Smoothed angular velocity')
        axs[1].set_title('Base angular velocity')
        axs[1].set_xlabel('Frame index')
        axs[1].set_ylabel('Norm of the velocity (rad/s)')
        axs[1].legend()

        plt.show()

    def plot_human_data(self) -> None:

        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

        axs[0,0].plot(self.human_base_positions)
        axs[0,0].plot([arr[0] for arr in self.human_base_positions], [arr[1] for arr in self.human_base_positions])
        axs[0,0].plot([arr[0] for arr in self.base_positions], [arr[1] for arr in self.base_positions])
        axs[0,0].set_title('Human base position')
        axs[0,0].set_title('Human and robot global base position')
        axs[0,0].set_xlabel('Frame index')
        axs[0,0].set_ylabel('Position displacement (m)')
        axs[0,0].set_xlabel('X displacement (m)')
        axs[0,0].set_ylabel('Y displacement (m)')
        axs[0,0].legend(['human', 'robot'])

        axs[0,1].plot(Rotation.from_quat(self.human_base_quaternions, scalar_first=True).as_euler('xyz'))
        axs[0,1].set_title('Human base orientation')
        axs[0,1].set_xlabel('Frame index')
        axs[0,1].set_ylabel('Orientation displacement (rad)')
        axs[0,1].legend(['roll', 'pitch', 'yaw'])

        axs[1,0].plot(self.human_base_linear_velocities)
        axs[1,0].set_title('Human base linear velocity')
        axs[1,0].set_xlabel('Frame index')
        axs[1,0].set_ylabel('Velocity (m/s)')
        axs[1,0].legend(['x','y','z'])

        axs[1,1].plot(self.human_base_angular_velocities)
        axs[1,1].set_title('Human base angular velocity')
        axs[1,1].set_xlabel('Frame index')
        axs[1,1].set_ylabel('Angular velocity (rad/s)')
        axs[1,1].legend(['roll', 'pitch', 'yaw'])

        plt.show()

    def colorline(
        self, x, y, z=None, cmap='Blues', norm=plt.Normalize(0.0, 1.0),
            linewidth=3, alpha=1.0):
        """
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        """

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])

        z = np.asarray(z)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                linewidth=linewidth, alpha=alpha)

        ax = plt.gca()
        ax.add_collection(lc)

        return lc

    def plot_robot_and_human_positions(self) -> None:

        # Plot the displacements over time of the human and the robot
        fig = plt.figure()
        plt.plot(np.arange(0, len(self.base_positions) * self.dt_mean, self.dt_mean), self.base_positions)
        plt.plot(np.arange(0, len(self.human_base_positions) * self.dt_mean, self.dt_mean), self.human_base_positions)
        plt.title('Base positions over time')
        plt.xlabel('Time (s)')
        plt.ylabel('Position displacement (m)')
        plt.legend(['robot x','robot y','robot z', 'human x', 'human y', 'human z'])

        x_human = [arr[0] for arr in self.human_base_positions]
        y_human = [arr[1] for arr in self.human_base_positions]
        x_robot = [arr[0] for arr in self.base_positions]
        y_robot = [arr[1] for arr in self.base_positions]
        human_path = mpath.Path(np.column_stack([x_human, y_human]))
        robot_path = mpath.Path(np.column_stack([x_robot, y_robot]))

        human_verts = human_path.interpolated(steps=3).vertices
        x_human, y_human = human_verts[:, 0], human_verts[:, 1]
        robot_verts = robot_path.interpolated(steps=3).vertices
        x_robot, y_robot = robot_verts[:, 0], robot_verts[:, 1]

        human_z = np.linspace(0, 1, len(x_human))
        robot_z = np.linspace(0, 1, len(x_robot))

        # Plot the x v y displacements of both the human and the robot
        fig = plt.figure()
        plt.title('Ground base positions')
        plt.xlabel('X displacement (m)')
        plt.ylabel('Y displacement (m)')
        plt.legend(['human', 'robot'])
        self.colorline(x_human, y_human, z=human_z, cmap='Blues', linewidth=2)
        self.colorline(x_robot, y_robot, z=robot_z, cmap='Reds', linewidth=2)
        plt.xlim(np.concatenate((x_human, x_robot)).min(), np.concatenate((x_human, x_robot)).max())
        plt.ylim(np.concatenate((y_human, y_robot)).min(), np.concatenate((y_human, y_robot)).max())

        plt.savefig('../datasets/plots/robot_follower_ground_base_positions.png')

        fig = plt.figure()
        plt.title('Ground distance between robot and human over time')
        distance = np.array([np.linalg.norm(human_pos[:2] - robot_pos[:2]) for human_pos, robot_pos in zip(self.human_base_positions, self.base_positions)])
        input(self.human_base_positions[0][:2] - self.base_positions[0][:2])
        plt.plot(np.arange(0, len(distance) * self.dt_mean, self.dt_mean), distance)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (m)')
        plt.savefig('../datasets/plots/human_robot_distance.png')

        plt.show()

    def plot_robot_and_human_orientations(self) -> None:
        plt.figure()

        robot_orientations = [Rotation.from_quat(quat, scalar_first=True).as_euler('xyz') for quat in self.base_quaternions]
        human_orientations = [Rotation.from_quat(quat, scalar_first=True).as_euler('xyz') for quat in self.human_base_quaternions]

        plt.plot([orientation[0] for orientation in robot_orientations], label='Robot roll')
        plt.plot([orientation[1] for orientation in robot_orientations], label='Robot pitch')
        plt.plot([orientation[2] for orientation in robot_orientations], label='Robot yaw')
        plt.plot([orientation[0] for orientation in human_orientations], label='Human roll', linestyle='dashed')
        plt.plot([orientation[1] for orientation in human_orientations], label='Human pitch', linestyle='dashed')
        plt.plot([orientation[2] for orientation in human_orientations], label='Human yaw', linestyle='dashed')
        plt.title('Orientation angles over time')
        plt.xlabel('Frame index')
        plt.ylabel('Angle (rad)')
        plt.legend()

        plt.figure()
        robot_headings = [np.arctan2(orientation[1], orientation[0]) for orientation in robot_orientations]
        human_headings = [np.arctan2(orientation[1], orientation[0]) for orientation in human_orientations]

        plt.plot(robot_headings, label='Robot heading')
        plt.plot(human_headings, label='Human heading', linestyle='dashed')
        plt.title('Ground heading over time')
        plt.xlabel('Frame index')
        plt.ylabel('Heading (rad)')
        plt.legend()

        plt.show()

    def moving_average(self, vect, n) -> List:
        moving_averaged_vect = []
        for idx in range(0, len(vect)):
            if idx < n//2: # When there are less than N/2 frames before the current frame, average over the available frames
                moving_averaged_vect.append(np.mean(vect[:idx + n//2], axis=0))
            elif idx >= len(vect) - n//2: # When there are less than N/2 frames after the current frame, average over the available frames
                moving_averaged_vect.append(np.mean(vect[idx - n//2:], axis=0))
            else: # Average over N frames
                moving_averaged_vect.append(np.mean(vect[idx - n//2:idx + n//2], axis=0))

        return moving_averaged_vect

    def compute_global_frame_features(self) -> None:
        """Extract global features associated to each retargeted frame"""

        # Debug
        print("Computing global frame features")

        # Subsampling (discard one ik solution over two)
        for frame_idx in range(0, len(self.ik_solutions)):

            ik_solution = self.ik_solutions[frame_idx]
            human_ik_solution = self.human_ik_solutions[frame_idx]

            # Retrieve the base pose and the joint positions
            joint_positions = np.asarray(ik_solution["joint_positions"])
            joint_velocities = np.asarray(ik_solution["joint_velocities"])
            base_position = np.asarray(ik_solution["base_position"])
            base_quaternion = np.asarray(ik_solution["base_quaternion"])
            base_linear_velocity_raw = np.asarray(ik_solution["base_linear_velocity"])
            base_angular_velocity_raw = np.asarray(ik_solution["base_angular_velocity"])

            # Retrieve the human base pose
            human_base_position = np.asarray(human_ik_solution["base_position"])
            human_base_quaternion = np.asarray(human_ik_solution["base_quaternion"])
            human_base_linear_velocity_raw = np.asarray(human_ik_solution["base_linear_velocity"])
            human_base_angular_velocity_raw = np.asarray(human_ik_solution["base_angular_velocity"])

            # Include the human joints for visualization purposes only
            human_joint_positions = np.asarray(human_ik_solution["joint_positions"])

            # If the start at origin var is false, then transform both robot and human s.t. robot starts at xy origin (yaw 0)
            if self.start_at_origin:

                # Get the initial robot transform
                initial_solution = self.ik_solutions[0]
                initial_base_position = np.asarray(initial_solution["base_position"])
                initial_base_quaternion = np.asarray(initial_solution["base_quaternion"])
                initial_rotation = Rotation.from_quat(initial_base_quaternion, scalar_first=True)
                initial_R_yaw = Rotation.from_euler('z', initial_rotation.as_euler('xyz')[2])
                initial_translation = np.array([initial_base_position[0], initial_base_position[1], 0])

                # Transform the base position and orientation to the initial frame
                current_rotation = Rotation.from_quat(base_quaternion, scalar_first=True)
                current_translation = base_position

                transformed_translation = initial_R_yaw.inv().apply(current_translation - initial_translation)
                transformed_rotation = initial_R_yaw.inv() * current_rotation

                base_position = transformed_translation
                base_quaternion = transformed_rotation.as_quat(scalar_first=True)

                # Transform the human base position and orientation to the initial frame
                current_human_rotation = Rotation.from_quat(human_base_quaternion, scalar_first=True)
                current_human_translation = human_base_position

                transformed_human_translation = initial_R_yaw.inv().apply(current_human_translation - initial_translation)
                transformed_human_rotation = initial_R_yaw.inv() * current_human_rotation

                human_base_position = transformed_human_translation
                human_base_quaternion = transformed_human_rotation.as_quat(scalar_first=True)

            # Base position
            self.base_positions.append(base_position)

            # Base quaternion
            self.base_quaternions.append(base_quaternion)

            # Human base position
            self.human_base_positions.append(human_base_position)

            # Human base orientation
            self.human_base_quaternions.append(human_base_quaternion)

            # Joint angles
            joint_positions_controlled = np.array([joint_positions[index] for index in self.controlled_joints_indexes])
            self.s.append(joint_positions_controlled)

            # Joint velocities
            joint_velocities_controlled = np.array([joint_velocities[index] for index in self.controlled_joints_indexes])
            self.s_dot.append(joint_velocities_controlled)

            # Store the raw base velocities
            self.base_linear_velocities_raw.append(base_linear_velocity_raw)
            self.base_angular_velocities_raw.append(base_angular_velocity_raw)

            # Store the raw human base velocities
            self.human_base_linear_velocities_raw.append(human_base_linear_velocity_raw)
            self.human_base_angular_velocities_raw.append(human_base_angular_velocity_raw)

            self.human_s.append(human_joint_positions)

        # Smooth out the base velocities
        N = 9 # Filter window size, centered around current frame
        self.base_linear_velocities = self.moving_average(self.base_linear_velocities_raw, N)
        self.base_angular_velocities = self.moving_average(self.base_angular_velocities_raw, N)

        # Smooth out the human base velocities
        self.human_base_linear_velocities = self.moving_average(self.human_base_linear_velocities_raw, N)
        self.human_base_angular_velocities = self.moving_average(self.human_base_angular_velocities_raw, N)

        if self.plot_global_vels:
            self.plot_raw_and_smoothed_vels()
        if self.plot_human_features:
            self.plot_human_data()
        if self.plot_robot_v_human:
            self.plot_robot_and_human_positions()
            self.plot_robot_and_human_orientations()



@dataclass
class GlobalWindowFeatures:
    """Class for the global features associated to a window of retargeted frames."""

    # Feature computation
    window_length_frames: int
    window_step: int
    window_indexes: List

    # Feature storage
    base_positions: List = field(default_factory=list)
    base_linear_velocities: List = field(default_factory=list)
    base_angular_velocities: List = field(default_factory=list)
    base_quaternions: List = field(default_factory=list)

    # Human feature storage
    human_base_positions: List = field(default_factory=list)
    human_base_quaternions: List = field(default_factory=list)
    human_base_linear_velocities: List = field(default_factory=list)
    human_base_angular_velocities: List = field(default_factory=list)

    @staticmethod
    def build(window_length_frames: int,
              window_step: int,
              window_indexes: List) -> "GlobalWindowFeatures":
        """Build an empty GlobalWindowFeatures."""

        return GlobalWindowFeatures(window_length_frames=window_length_frames,
                                    window_step=window_step,
                                    window_indexes=window_indexes)

    def compute_global_window_features(self, global_frame_features: GlobalFrameFeatures) -> None:
        """Extract global features associated to a window of retargeted frames."""

        # Debug
        print("Computing global window features")

        initial_frame = self.window_length_frames
        final_frame = len(global_frame_features.base_positions) - self.window_length_frames - self.window_step - 1

        # For each window of retargeted frames
        for i in range(initial_frame, final_frame):

            # Initialize placeholders for the current window
            current_global_base_positions = []
            current_global_base_linear_velocities = []
            current_global_base_angular_velocities = []
            current_global_base_quaternions = []
            current_global_human_base_positions = []
            current_global_human_base_quaternions = []
            current_global_human_base_linear_velocities = []
            current_global_human_base_angular_velocities = []

            for window_index in self.window_indexes:

                # Store the base positions, facing directions and base velocities in the current window
                current_global_base_positions.append(global_frame_features.base_positions[i + window_index])
                current_global_base_linear_velocities.append(global_frame_features.base_linear_velocities[i + window_index])
                current_global_base_angular_velocities.append(global_frame_features.base_angular_velocities[i + window_index])
                current_global_base_quaternions.append(global_frame_features.base_quaternions[i + window_index])

                # Store the human base positions, facing directions and base velocities in the current window
                current_global_human_base_positions.append(global_frame_features.human_base_positions[i + window_index])
                current_global_human_base_quaternions.append(global_frame_features.human_base_quaternions[i + window_index])
                current_global_human_base_linear_velocities.append(global_frame_features.human_base_linear_velocities[i + window_index])
                current_global_human_base_angular_velocities.append(global_frame_features.human_base_angular_velocities[i + window_index])

            # Store global features for the current window
            self.base_positions.append(current_global_base_positions)
            self.base_linear_velocities.append(current_global_base_linear_velocities)
            self.base_angular_velocities.append(current_global_base_angular_velocities)
            self.base_quaternions.append(current_global_base_quaternions)

            # Store human features for the current window
            self.human_base_positions.append(current_global_human_base_positions)
            self.human_base_quaternions.append(current_global_human_base_quaternions)
            self.human_base_linear_velocities.append(current_global_human_base_linear_velocities)
            self.human_base_angular_velocities.append(current_global_human_base_angular_velocities)

@dataclass
class LocalFrameFeatures:
    """Class for the local features associated to each retargeted frame."""

    # Features storage
    human_base_positions: List = field(default_factory=list)
    human_base_quaternions: List = field(default_factory=list)
    human_base_linear_velocities: List = field(default_factory=list)
    human_base_angular_velocities: List = field(default_factory=list)

    plot_local_human_features: bool = False

    @staticmethod
    def build(plot_local_human_features: bool = False) -> "LocalFrameFeatures":
        """Build an empty LocalFrameFeatures."""

        return LocalFrameFeatures(plot_local_human_features=plot_local_human_features)

    def plot_local_human_data(self) -> None:

        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

        axs[0,0].plot([arr[0] for arr in self.human_base_positions], [arr[1] for arr in self.human_base_positions])
        axs[0,0].set_title('Human base position wrt robot base frame')
        axs[0,0].set_xlabel('X displacement (m)')
        axs[0,0].set_ylabel('Y displacement (m)')

        human_base_angles = [Rotation.from_quat(quat, scalar_first=True).as_euler('xyz') for quat in self.human_base_quaternions]

        axs[0,1].plot(human_base_angles)
        axs[0,1].set_title('Human base orientation wrt robot base frame')
        axs[0,1].set_xlabel('Frame index')
        axs[0,1].set_ylabel('Orientation displacement (rad)')
        axs[0,1].legend(['roll', 'pitch', 'yaw'])

        axs[1,0].plot(self.human_base_linear_velocities)
        axs[1,0].set_title('Human base linear velocity')
        axs[1,0].set_xlabel('Frame index')
        axs[1,0].set_ylabel('Velocity (m/s)')
        axs[1,0].legend(['x','y','z'])

        axs[1,1].plot(self.human_base_angular_velocities)
        axs[1,1].set_title('Human base angular velocity')
        axs[1,1].set_xlabel('Frame index')
        axs[1,1].set_ylabel('Angular velocity (rad/s)')
        axs[1,1].legend(['roll', 'pitch', 'yaw'])

        plt.show()

    def compute_local_frame_features(self, global_frame_features: GlobalFrameFeatures) -> None:
        """Extract local features associated to each retargeted frame"""

        # Debug
        print("Computing local frame features")

        for i in range(0, len(global_frame_features.base_positions)):

            # Retrieve the robot and human poses and the human velocities to compute the transform
            current_global_base_position = global_frame_features.base_positions[i]
            current_global_base_orientation = global_frame_features.base_quaternions[i]
            current_global_human_base_position = global_frame_features.human_base_positions[i]
            current_global_human_base_orientation = global_frame_features.human_base_quaternions[i]
            current_global_human_base_linear_velocity = global_frame_features.base_linear_velocities[i]
            current_global_human_base_angular_velocity = global_frame_features.base_angular_velocities[i]

            # Compute the transforms
            I_H_RB = np.vstack((np.hstack((Rotation.from_quat(np.array(current_global_base_orientation), scalar_first=True).as_matrix(), current_global_base_position.reshape(-1, 1))), np.array([0, 0, 0, 1])))
            I_H_HB = np.vstack((np.hstack((Rotation.from_quat(np.array(current_global_human_base_orientation), scalar_first=True).as_matrix(), current_global_human_base_position.reshape(-1, 1))), np.array([0, 0, 0, 1])))

            # Compute the pose of the human base in the robot base frame
            RB_H_HB = np.linalg.inv(I_H_RB) @ I_H_HB
            current_local_human_base_position = RB_H_HB[:3, 3]
            current_local_human_base_orientation = Rotation.from_matrix(RB_H_HB[:3, :3]).as_quat(scalar_first=True)

            # Compute base linear velocity in the robot base frame
            RB_R_I = I_H_RB[:3, :3].T
            current_local_base_linear_velocity = RB_R_I.dot(current_global_human_base_linear_velocity)
            current_local_human_base_angular_velocity = RB_R_I.dot(current_global_human_base_angular_velocity)

            # Store the values in the local reference frame
            self.human_base_positions.append(current_local_human_base_position)
            self.human_base_quaternions.append(current_local_human_base_orientation)
            self.human_base_linear_velocities.append(current_local_base_linear_velocity)
            self.human_base_angular_velocities.append(current_local_human_base_angular_velocity)

        if self.plot_local_human_features:
            self.plot_local_human_data()

@dataclass
class LocalWindowFeatures:
    """Class for the local features associated to a window of retargeted frames."""

    # Feature computation
    window_length_frames: int
    window_step: int
    window_indexes: List

    # Feature storage
    base_positions: List = field(default_factory=list)
    base_quaternions: List = field(default_factory=list)
    base_linear_velocities: List = field(default_factory=list)
    base_angular_velocities: List = field(default_factory=list)

    @staticmethod
    def build(window_length_frames: int,
              window_step: int,
              window_indexes: List) -> "LocalWindowFeatures":
        """Build an empty GlobalWindowFeatures."""

        return LocalWindowFeatures(window_length_frames=window_length_frames,
                                   window_step=window_step,
                                   window_indexes=window_indexes)

    def compute_local_window_features(self, global_window_features: GlobalWindowFeatures) -> None:
        """Extract local features associated to a window of retargeted frames."""

        # Debug
        print("Computing local window features")

        # For each window of retargeted frames
        for i in range(len(global_window_features.base_linear_velocities)):

            # Store the global features associated to the currently-considered window of retargeted frames
            current_global_base_positions = global_window_features.base_positions[i]
            current_global_base_linear_velocities = global_window_features.base_linear_velocities[i]
            current_global_base_angular_velocities = global_window_features.base_angular_velocities[i]
            current_global_base_quaternions = global_window_features.base_quaternions[i]

            # Placeholders for the local features associated to the currently-considered window of retargeted frames
            current_local_base_positions = []
            current_local_base_quaternions = []
            current_local_base_linear_velocities = []
            current_local_base_angular_velocities = []

            # Find the current reference frame with respect to which the local quantities will be expressed
            for j in range(len(current_global_base_linear_velocities)):

                # The current reference frame is defined by the central frame of the window. Skip the others
                if global_window_features.window_indexes[j] != 0:
                    continue

                # Get the current transform from world to robot base
                I_R_RB=Rotation.from_quat(np.array(current_global_base_quaternions[j]),
                                                 scalar_first=True).as_matrix()
                RB_R_I = np.linalg.inv(I_R_RB)
                base_ref_position = current_global_base_positions[j]

            for j in range(len(current_global_base_linear_velocities)):

                # Retrieve global features
                current_global_base_position = current_global_base_positions[j]
                current_global_base_quaternion = current_global_base_quaternions[j]
                current_global_base_linear_vel = current_global_base_linear_velocities[j]
                current_global_base_ang_vel = current_global_base_angular_velocities[j]

                # Express them locally
                current_local_base_position = RB_R_I.dot(current_global_base_position) - base_ref_position
                current_global_base_rotation = Rotation.from_quat(np.array(current_global_base_quaternion),
                                                                  scalar_first=True).as_matrix()
                current_local_base_rotation = RB_R_I.dot(current_global_base_rotation)
                current_local_base_quaternion = np.array(Rotation.as_quat(Rotation.from_matrix(current_local_base_rotation), scalar_first=True))
                current_local_base_linear_vel = RB_R_I.dot(current_global_base_linear_vel)
                current_local_base_ang_vel = RB_R_I.dot(current_global_base_ang_vel)

                # Fill the placeholders for the local features associated to the current window
                current_local_base_positions.append(current_local_base_position)
                current_local_base_quaternions.append(current_local_base_quaternion)
                current_local_base_linear_velocities.append(current_local_base_linear_vel)
                current_local_base_angular_velocities.append(current_local_base_ang_vel)

            # Store local features for the current window
            self.base_positions.append(current_local_base_positions)
            self.base_quaternions.append(current_local_base_quaternions)
            self.base_linear_velocities.append(current_local_base_linear_velocities)
            self.base_angular_velocities.append(current_local_base_angular_velocities)


@dataclass
class FeatureExtractor:
    """Class for the extracting features from retargeted mocap data."""

    global_frame_features: GlobalFrameFeatures
    global_window_features: GlobalWindowFeatures
    local_frame_features: LocalFrameFeatures
    local_window_features: LocalWindowFeatures

    plot_global_vels: bool = False
    plot_human_features: bool = False
    plot_robot_v_human: bool = False
    plot_local_human_features: bool = False
    start_at_origin: bool = False

    @staticmethod
    def build(ik_solutions: List,
              leader_ik_solutions: dict,
              controlled_joints_indexes: List,
              dt_mean: float = 1/100,
              window_length_s: float = 1,
              window_granularity_s: float = 0.2,
              plot_global_vels: bool = False,
              plot_human_features: bool = False,
              plot_robot_v_human: bool = False,
              plot_local_human_features: bool = False,
              start_at_origin: bool = False) -> "FeatureExtractor":
        """Build a FeatureExtractor."""

        # Define the lenght, expressed in frames, of the window of interest (default=50)
        window_length_frames = round(window_length_s / dt_mean)

        # Define the step, expressed in frames, between the relevant time instants in the window of interest (default=10)
        window_step = round(window_length_frames * window_granularity_s)

        # Define the indexes, expressed in frames, of the relevant time instants in the window of interest (default = [-50, -40, ... , 0, ..., 50, 60])
        window_indexes = list(range(-window_length_frames, window_length_frames + 2 * window_step, window_step))

        # Instantiate all the features
        gff = GlobalFrameFeatures.build(ik_solutions=ik_solutions,
                                        leader_ik_solutions=leader_ik_solutions,
                                        controlled_joints_indexes=controlled_joints_indexes,
                                        dt_mean=dt_mean, plot_global_vels=plot_global_vels,
                                        plot_human_features=plot_human_features,
                                        plot_robot_v_human=plot_robot_v_human,
                                        start_at_origin=start_at_origin)
        gwf = GlobalWindowFeatures.build(window_length_frames=window_length_frames,
                                         window_step=window_step,
                                         window_indexes=window_indexes)
        lff = LocalFrameFeatures.build(plot_local_human_features=plot_local_human_features)
        lwf = LocalWindowFeatures.build(window_length_frames=window_length_frames,
                                        window_step=window_step,
                                        window_indexes=window_indexes)

        return FeatureExtractor(global_frame_features=gff,
                                 global_window_features=gwf,
                                 local_frame_features=lff,
                                 local_window_features=lwf)

    def compute_features(self) -> None:
        """Compute all the features."""

        self.global_frame_features.compute_global_frame_features()
        self.global_window_features.compute_global_window_features(global_frame_features=self.global_frame_features)
        self.local_frame_features.compute_local_frame_features(global_frame_features=self.global_frame_features)
        self.local_window_features.compute_local_window_features(global_window_features=self.global_window_features)

    def compute_X(self) -> List:
        """Generate the network input vector X."""

        window_length_frames = self.global_window_features.window_length_frames
        window_step = self.global_window_features.window_step
        initial_frame = window_length_frames
        final_frame = len(self.global_frame_features.base_positions) - window_length_frames - window_step - 2

        # Initialize input vector
        X = []

        # For each window of retargeted frames
        for i in range(initial_frame, final_frame):

            print(i)

            # Initialize current input vector
            X_i = []

            # Add current local base velocities (36 components)
            current_local_base_linear_velocities = []
            for local_base_linear_velocity in self.local_window_features.base_linear_velocities[i - window_length_frames]:
                current_local_base_linear_velocities.extend(local_base_linear_velocity)
            X_i.extend(current_local_base_linear_velocities)

            # Add current local base angular velocities (36 components)
            current_local_base_angular_velocities = []
            for local_base_angular_velocity in self.local_window_features.base_angular_velocities[i - window_length_frames]:
                current_local_base_angular_velocities.extend(local_base_angular_velocity)
            X_i.extend(current_local_base_angular_velocities)

            # Add previous joint positions (26 components)
            prev_s = self.global_frame_features.s[i - 1]
            X_i.extend(prev_s)

            # Add previous joint velocities (26 components)
            prev_s_dot = self.global_frame_features.s_dot[i - 1]
            X_i.extend(prev_s_dot)

            # Add previous inertial frame base position (3 components)
            prev_base_position = self.global_frame_features.base_positions[i - 1]
            X_i.extend(prev_base_position)

            # Add previous inertial frame base euler angles (3 components)
            prev_base_euler_angles = Rotation.from_quat(
                self.global_frame_features.base_quaternions[i - 1], scalar_first=True).as_euler('xyz')
            X_i.extend(prev_base_euler_angles)

            # Add previous human base position expressed in inertial frame (3 components)
            prev_human_base_position = self.global_frame_features.human_base_positions[i - 1]
            X_i.extend(prev_human_base_position)

            # Add previous human base orientation expressed in inertial frame (3 components)
            prev_human_base_euler_angles = Rotation.from_quat(
                self.global_frame_features.human_base_quaternions[i - 1], scalar_first=True).as_euler('xyz')
            X_i.extend(prev_human_base_euler_angles)

            # Add previous human base linear velocity expressed in robot base frame (3 components)
            prev_human_base_linear_velocity = self.local_frame_features.human_base_linear_velocities[i - 1]
            X_i.extend(prev_human_base_linear_velocity)

            # Add previous human base angular velocity expressed in robot base frame (3 components)
            prev_human_base_angular_velocity = self.local_frame_features.human_base_angular_velocities[i - 1]
            X_i.extend(prev_human_base_angular_velocity)

            # Store current input vector (142 components)
            X.append(X_i)

        # Debug
        print("X size:", len(X), "x", len(X[0]))

        return X

    def compute_Y(self) -> List:
        """Generate the network output vector Y."""

        window_length_frames = self.global_window_features.window_length_frames
        window_step = self.global_window_features.window_step
        window_indexes = self.global_window_features.window_indexes
        initial_frame = window_length_frames
        final_frame = len(self.global_frame_features.base_positions) - window_length_frames - window_step - 2

        # Initialize output vector
        Y = []

        # For each window of retargeted frames
        for i in range(initial_frame, final_frame):

            # Initialize current input vector
            Y_i = []

            # Add future local base velocities (21 components)
            next_local_base_linear_velocities = []
            for j in range(len(self.local_window_features.base_linear_velocities[i - window_length_frames + 1])):
                if window_indexes[j] >= 0:
                    next_local_base_linear_velocities.extend(self.local_window_features.base_linear_velocities[i - window_length_frames + 1][j])
            Y_i.extend(next_local_base_linear_velocities)

            # Add future local base angular velocities (21 components)
            next_local_base_angular_velocities = []
            for j in range(len(self.local_window_features.base_angular_velocities[i - window_length_frames + 1])):
                if window_indexes[j] >= 0:
                    next_local_base_angular_velocities.extend(self.local_window_features.base_angular_velocities[i - window_length_frames + 1][j])
            Y_i.extend(next_local_base_angular_velocities)

            # Add current joint positions (26 components)
            current_s = self.global_frame_features.s[i]
            Y_i.extend(current_s)

            # Add current joint velocities (26 components)
            current_s_dot = self.global_frame_features.s_dot[i]
            Y_i.extend(current_s_dot)

            # Add current inertial frame base position (3 components)
            current_base_position = self.global_frame_features.base_positions[i]
            Y_i.extend(current_base_position)

            # Add current inertial frame base euler angles (3 components)
            current_base_euler_angles = Rotation.from_quat(
                self.global_frame_features.base_quaternions[i], scalar_first=True).as_euler('xyz')
            Y_i.extend(current_base_euler_angles)

            # Store current output vector (100 components)
            Y.append(Y_i)

        # Debug
        print("Y size:", len(Y), "x", len(Y[0]))

        return Y

    def get_global_window_features(self) -> GlobalWindowFeatures:
        """Get the global features associated to a window of retargeted frames."""

        return self.global_window_features

    def get_local_window_features(self) -> LocalWindowFeatures:
        """Get the local features associated to a window of retargeted frames."""

        return self.local_window_features

    def get_global_frame_features(self) -> LocalWindowFeatures:
        """Get the global features associated to the retargeted frames."""

        return self.global_frame_features
