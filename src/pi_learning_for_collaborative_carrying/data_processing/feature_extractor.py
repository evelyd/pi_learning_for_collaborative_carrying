# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import List
from dataclasses import dataclass, field
from pi_learning_for_collaborative_carrying.data_processing import utils
from scipy.spatial.transform import Rotation

@dataclass
class GlobalFrameFeatures:
    """Class for the global features associated to each retargeted frame."""

    # Feature computation
    ik_solutions: List
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

    @staticmethod
    def build(ik_solutions: List,
              controlled_joints_indexes: List,
              dt_mean: float
              ) -> "GlobalFrameFeatures":
        """Build an empty GlobalFrameFeatures."""

        return GlobalFrameFeatures(ik_solutions=ik_solutions,
                                   controlled_joints_indexes=controlled_joints_indexes,
                                   dt_mean=dt_mean
                                   )

    def compute_global_frame_features(self) -> None:
        """Extract global features associated to each retargeted frame"""

        # Debug
        print("Computing global frame features")

        # Subsampling (discard one ik solution over two)
        for frame_idx in range(0, len(self.ik_solutions)):

            ik_solution = self.ik_solutions[frame_idx]

            # Retrieve the base pose and the joint positions
            joint_positions = np.asarray(ik_solution["joint_positions"])
            joint_velocities = np.asarray(ik_solution["joint_velocities"])
            base_position = np.asarray(ik_solution["base_position"])
            base_quaternion = np.asarray(ik_solution["base_quaternion"])
            base_linear_velocity_raw = np.asarray(ik_solution["base_linear_velocity"])
            base_angular_velocity_raw = np.asarray(ik_solution["base_angular_velocity"])

            # Base position
            self.base_positions.append(base_position)

            # Base quaternion
            self.base_quaternions.append(base_quaternion)

            # Joint angles
            joint_positions_controlled = np.array([joint_positions[index] for index in self.controlled_joints_indexes])
            self.s.append(joint_positions_controlled)

            # Joint velocities
            joint_velocities_controlled = np.array([joint_velocities[index] for index in self.controlled_joints_indexes])
            self.s_dot.append(joint_velocities_controlled)

            # Store the raw base velocities
            self.base_linear_velocities_raw.append(base_linear_velocity_raw)
            self.base_angular_velocities_raw.append(base_angular_velocity_raw)

        # Smooth out the base velocities
        N = 9 # Filter window size, centered around current frame
        for idx in range(0, len(self.base_linear_velocities_raw)):
            if idx < N//2: # When there are less than N/2 frames before the current frame, average over the available frames
                self.base_linear_velocities.append(np.mean(self.base_linear_velocities_raw[:idx + N//2], axis=0))
                self.base_angular_velocities.append(np.mean(self.base_angular_velocities_raw[:idx + N//2], axis=0))
            elif idx >= len(self.base_linear_velocities_raw) - N//2: # When there are less than N/2 frames after the current frame, average over the available frames
                self.base_linear_velocities.append(np.mean(self.base_linear_velocities_raw[idx - N//2:], axis=0))
                self.base_angular_velocities.append(np.mean(self.base_angular_velocities_raw[idx - N//2:], axis=0))
            else: # Average over N frames
                self.base_linear_velocities.append(np.mean(self.base_linear_velocities_raw[idx - N//2:idx + N//2], axis=0))
                self.base_angular_velocities.append(np.mean(self.base_angular_velocities_raw[idx - N//2:idx + N//2], axis=0))



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

            for window_index in self.window_indexes:

                # Store the base positions, facing directions and base velocities in the current window
                current_global_base_positions.append(global_frame_features.base_positions[i + window_index])
                current_global_base_linear_velocities.append(global_frame_features.base_linear_velocities[i + window_index])
                current_global_base_angular_velocities.append(global_frame_features.base_angular_velocities[i + window_index])
                current_global_base_quaternions.append(global_frame_features.base_quaternions[i + window_index])

            # Store global features for the current window
            self.base_positions.append(current_global_base_positions)
            self.base_linear_velocities.append(current_global_base_linear_velocities)
            self.base_angular_velocities.append(current_global_base_angular_velocities)
            self.base_quaternions.append(current_global_base_quaternions)

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

                base_rotation=Rotation.from_quat(utils.to_xyzw(
                    np.array(current_global_base_quaternions[j]))).as_matrix()
                base_R_world = np.linalg.inv(base_rotation)
                base_ref_position = current_global_base_positions[j]

            for j in range(len(current_global_base_linear_velocities)):

                # Retrieve global features
                current_global_base_position = current_global_base_positions[j]
                current_global_base_quaternion = current_global_base_quaternions[j]
                current_global_base_linear_vel = current_global_base_linear_velocities[j]
                current_global_base_ang_vel = current_global_base_angular_velocities[j]

                # Express them locally
                current_local_base_position = base_R_world.dot(current_global_base_position) - base_ref_position
                current_global_base_rotation = Rotation.from_quat(utils.to_xyzw(np.array(current_global_base_quaternion))).as_matrix()
                current_local_base_rotation = base_R_world.dot(current_global_base_rotation)
                current_local_base_quaternion = np.array(utils.to_wxyz(Rotation.as_quat(
                    Rotation.from_matrix(current_local_base_rotation))))
                current_local_base_linear_vel = base_R_world.dot(current_global_base_linear_vel)
                current_local_base_ang_vel = base_R_world.dot(current_global_base_ang_vel)

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
    local_window_features: LocalWindowFeatures

    @staticmethod
    def build(ik_solutions: List,
            #   kindyn: kindyncomputations.KinDynComputations,
              controlled_joints_indexes: List,
              dt_mean: float = 1/50,
              window_length_s: float = 1,
              window_granularity_s: float = 0.2) -> "FeatureExtractor":
        """Build a FeatureExtractor."""

        # Define the lenght, expressed in frames, of the window of interest (default=50)
        window_length_frames = round(window_length_s / dt_mean)

        # Define the step, expressed in frames, between the relevant time instants in the window of interest (default=10)
        window_step = round(window_length_frames * window_granularity_s)

        # Define the indexes, expressed in frames, of the relevant time instants in the window of interest (default = [-50, -40, ... , 0, ..., 50, 60])
        window_indexes = list(range(-window_length_frames, window_length_frames + 2 * window_step, window_step))

        # Instantiate all the features
        gff = GlobalFrameFeatures.build(ik_solutions=ik_solutions,
                                        controlled_joints_indexes=controlled_joints_indexes,
                                        dt_mean=dt_mean#,
                                        # kindyn=kindyn
                                        )
        gwf = GlobalWindowFeatures.build(window_length_frames=window_length_frames,
                                         window_step=window_step,
                                         window_indexes=window_indexes)
        lwf = LocalWindowFeatures.build(window_length_frames=window_length_frames,
                                        window_step=window_step,
                                        window_indexes=window_indexes)

        return FeatureExtractor(global_frame_features=gff,
                                 global_window_features=gwf,
                                 local_window_features=lwf)

    def compute_features(self) -> None:
        """Compute all the features."""

        self.global_frame_features.compute_global_frame_features()
        self.global_window_features.compute_global_window_features(global_frame_features=self.global_frame_features)
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
            prev_s_dot = self.global_frame_features.s_dot[i - 2]
            X_i.extend(prev_s_dot)

            # Add previous inertial frame base position (3 components)
            prev_base_position = self.global_frame_features.base_positions[i - 1]
            X_i.extend(prev_base_position)

            # Add previous inertial frame base euler angles (3 components)
            base_euler_angles = Rotation.from_quat(utils.to_xyzw(self.global_frame_features.base_quaternions[i - 1])).as_euler('xyz')
            prev_base_euler_angles = base_euler_angles
            X_i.extend(prev_base_euler_angles)

            # Store current input vector (130 components)
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
            current_s_dot = self.global_frame_features.s_dot[i - 1]
            Y_i.extend(current_s_dot)

            # Add current inertial frame base position (3 components)
            current_base_position = self.global_frame_features.base_positions[i]
            Y_i.extend(current_base_position)

            # Add current inertial frame base euler angles (3 components)
            base_euler_angles = Rotation.from_quat(utils.to_xyzw(self.global_frame_features.base_quaternions[i])).as_euler('xyz')
            current_base_euler_angles = base_euler_angles
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
