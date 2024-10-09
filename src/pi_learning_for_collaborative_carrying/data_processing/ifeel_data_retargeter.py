# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from pi_learning_for_collaborative_carrying.data_processing import utils
from pi_learning_for_collaborative_carrying.data_processing import motion_data
import bipedal_locomotion_framework.bindings as blf
import idyntree.swig as idyn
from scipy.spatial.transform import Rotation
import manifpy as manif
import biomechanical_analysis_framework as baf

@dataclass
class IKTargets:
    """Class to manipulate the targets for the IK used in the retargeting pipeline."""

    timestamps: List[float]
    root_link: str
    base_pose_targets: Dict
    link_orientation_targets: Dict

    @staticmethod
    def build(motiondata: motion_data.MotionData,
              metadata: motion_data.MocapMetadata) -> "IKTargets":
        """Build an instance of IKTargets."""

        link_orientation_targets = {}
        base_pose_targets = {}

        for link in motiondata.Links:

            link_orientation_targets[link['name']] = np.array(link['orientations'])

            if link['name'] == metadata.root_link:
                base_pose_targets['positions'] = np.array(link['positions'])
                base_pose_targets['orientations'] = np.array(link['orientations'])

        return IKTargets(timestamps=motiondata.SampleDurations,
                         base_pose_targets=base_pose_targets,
                         link_orientation_targets=link_orientation_targets,
                         root_link=metadata.root_link)

    @staticmethod
    def mirror_quat_wrt_xz_world_plane(quat) -> np.array:
        """Mirror a quaternion w.r.t. the world X-Z plane."""

        R = Rotation.from_quat(utils.to_xyzw(np.asarray(quat)))
        RPY = R.as_euler('xyz')

        mirrored_R = Rotation.from_euler('xyz', [-RPY[0], RPY[1], -RPY[2]])
        mirrored_quat = utils.to_wxyz(mirrored_R.as_quat())

        return np.array(mirrored_quat)

    @staticmethod
    def mirror_pos_wrt_xz_world_plane(pos) -> np.array:
        """Mirror a position w.r.t. the world X-Z plane."""

        mirrored_pos = pos
        mirrored_pos[1] *= -1

        return np.array(mirrored_pos)

    def mirror_ik_targets(self) -> None:
        """Mirror the ik targets. The base poses are mirrored w.r.t. the world X-Z plane. The left and
        right link orientations for the limbs are switched and mirrored w.r.t. the model's sagittal plane.
        """

        # Define a mapping between the links in order to exchange left and right limbs
        link_to_link_mapping = {link:
                                    "Right" + link[4:] if "Left" in link
                                    else "Left" + link[5:] if "Right" in link
                                    else link
                                for link in self.link_orientation_targets}

        # ====================
        # BASE POSES MIRRORING
        # ====================

        # Replace original base positions with mirrored base positions
        base_positions = self.base_pose_targets['positions']
        mirrored_base_positions = [self.mirror_pos_wrt_xz_world_plane(np.asarray(base_pos)) for base_pos in base_positions]
        self.base_pose_targets['positions'] = mirrored_base_positions

        # Replace original base orientations with mirrored base orientations
        base_orientations = self.base_pose_targets['orientations']
        mirrored_base_orientations = [self.mirror_quat_wrt_xz_world_plane(np.asarray(base_quat)) for base_quat in base_orientations]
        self.base_pose_targets['orientations'] = mirrored_base_orientations

        # Store mirrored base (B) orientations w.r.t. world (W) and original world orientations w.r.t. base (used below)
        mirrored_W_Rs_B = [Rotation.from_quat(utils.to_xyzw(base_quat)) for base_quat in mirrored_base_orientations]
        original_B_Rs_W = [Rotation.from_matrix(np.linalg.inv(Rotation.from_quat(utils.to_xyzw(base_quat)).as_matrix()))
                           for base_quat in base_orientations]

        # ===========================
        # LINK ORIENTATIONS MIRRORING
        # ===========================

        original_orientations = self.link_orientation_targets.copy()

        for link in self.link_orientation_targets:

            # Skip the root link
            if link == self.root_link:
                continue

            # Match link with its mirror link according to the predefined mapping
            mirror_link = link_to_link_mapping[link]
            print("Assign to", link, "the references of", mirror_link)

            # Retrieve original mirror-link quaternions (in the world frame)
            W_mirror_link_quat = original_orientations[mirror_link]

            # Initialize mirrored mirror-link quaternions (in the world frame)
            W_mirror_link_mirrored_quaternions = []

            for i in range(len(W_mirror_link_quat)):

                # Compute mirror-link RPYs (in the original base frame)
                W_mirror_link_orientation = Rotation.from_quat(utils.to_xyzw(np.array(W_mirror_link_quat[i])))
                B_mirror_link_orientation = Rotation.from_matrix(original_B_Rs_W[i].as_matrix().dot(W_mirror_link_orientation.as_matrix()))
                B_mirror_link_RPY = B_mirror_link_orientation.as_euler('xyz')

                # Mirror mirror-link orientation w.r.t. the model's sagittal plane (i.e. revert roll and yaw signs)
                B_mirror_link_mirrored_orientation = \
                    Rotation.from_euler('xyz', [-B_mirror_link_RPY[0], B_mirror_link_RPY[1], -B_mirror_link_RPY[2]])

                # Express the mirrored mirror-link orientation in the world frame (using the mirrored base orientation)
                W_mirror_link_mirrored_orientation = Rotation.from_matrix(mirrored_W_Rs_B[i].as_matrix().dot(B_mirror_link_mirrored_orientation.as_matrix()))

                # Retrieve quaternions and add them to the mirrored mirror-link quaternions (in the world frame)
                W_mirror_link_mirrored_quaternion = utils.to_wxyz(W_mirror_link_mirrored_orientation.as_quat())
                W_mirror_link_mirrored_quaternions.append(W_mirror_link_mirrored_quaternion)

            # Assign to the link the mirrored mirror-link quaternions
            self.link_orientation_targets[link] = np.array(W_mirror_link_mirrored_quaternions)

@dataclass
class WBGR:
    """Class implementing the Whole-Body Geometric Retargeting (WBGR)."""

    motiondata: motion_data.MotionData
    metadata: motion_data.MocapMetadata
    joint_names: List
    kindyn: idyn.KinDynComputations
    calibration_matrices = {}
    IMU_link_rotations = {}
    initial_base_height: float
    humanIK: baf.ik.HumanIK

    @staticmethod
    def build(motiondata: motion_data.MotionData,
              metadata: motion_data.MocapMetadata,
              humanIK: baf.ik.HumanIK,
              joint_names: List,
              kindyn: idyn.KinDynComputations,
              mirroring: bool = False,
              initial_base_height: float = 0.0) -> "WBGR":
        """Build an instance of WBGR."""

        #TODO update this for using ifeel data
        # if mirroring:
        #     # Mirror the ik targets
        #     ik_targets.mirror_ik_targets()

        return WBGR(motiondata=motiondata, metadata=metadata, joint_names=joint_names, kindyn=kindyn, initial_base_height=initial_base_height, humanIK=humanIK)

    def retarget(self) -> (List, List):
        """Apply Whole-Body Geometric Retargeting (WBGR)."""

        timestamps = []
        ik_solutions = []

        # Get the height of the front foot frame off the ground
        foot_height = utils.idyn_transform_to_np(self.kindyn.getWorldTransform("r_foot_front"))[2,3]

        # ====================================================
        # Calibrate using humanIK
        # ====================================================

        self.humanIK.clearCalibrationMatrices()
        self.humanIK.calibrateWorldYaw(self.motiondata.CalibrationData)
        self.humanIK.calibrateAllWithWorld(self.motiondata.CalibrationData, "r_foot_front")

        # Keep track of the frames jumped due to IK failure
        jumped_frames = 0

        for i in range(len(self.motiondata.SampleDurations)):

            print(i, "/", len(self.motiondata.SampleDurations))

            timestamps.append(self.motiondata.SampleDurations[i])

            # ==============
            # UPDATE TARGETS
            # ==============

            # Update orientation and gravity tasks
            for task in self.motiondata.SO3Tasks + self.motiondata.GravityTasks:

                # Extract data for the update
                group_name = task['name']
                task_type = self.metadata.metadata[group_name]['type']
                I_quat_IMU = np.array(task['orientations'][i])
                node_number = self.metadata.metadata[group_name]['node_number']

                # Get the orientation data in manif SO3 format
                I_R_IMU_manif = manif.SO3(quaternion=utils.to_xyzw(I_quat_IMU))

                if task_type == 'SO3Task':
                    # Get the angular velocity data in manif SO3Tangent format
                    I_omega_IMU = np.array(task['angular_velocities'][i])
                    I_omega_IMU_manif = manif.SO3Tangent(I_omega_IMU)

                    assert self.humanIK.updateOrientationTask(node_number, I_R_IMU_manif, I_omega_IMU_manif)

                else: # for GravityTask
                    assert self.humanIK.updateGravityTask(node_number, I_R_IMU_manif)

            # Update FloorContactTasks
            for task in self.motiondata.FloorContactTasks:
                node_number = self.metadata.metadata[group_name]['node_number']
                group_name = task['name']
                force = task['forces'][i]
                node_number = self.metadata.metadata[group_name]['node_number']

                assert self.humanIK.updateFloorContactTask(node_number, force, foot_height)

            # Update JointLimitsTask
            assert self.humanIK.updateJointConstraintsTask()

            # # Update JointTrackingTask
            assert self.humanIK.updateJointRegularizationTask()

            # ========
            # SOLVE IK
            # ========

            try:
                # Step the solver
                self.humanIK.advance()

            except Exception as e:
                # Skip this ik solution and keep track of how many skipped
                print("Frame skipped due to Exception:", e)
                jumped_frames += 1

                continue

            # Get the solution values from human IK
            ok, new_joint_positions = self.humanIK.getJointPositions()
            ok, new_joint_velocities = self.humanIK.getJointVelocities()
            ok, new_base_position = self.humanIK.getBasePosition()
            ok, new_base_rotation = self.humanIK.getBaseOrientation()
            new_base_quaternion = utils.to_wxyz(Rotation.from_matrix(new_base_rotation).as_quat())

            # Update ik solution
            ik_solution = utils.IKSolution(base_position=[new_base_position[0], new_base_position[1], new_base_position[2] + self.initial_base_height],
                                     base_quaternion=new_base_quaternion,
                                     joint_configuration=new_joint_positions)

            # Store the ik solutions
            ik_solutions.append(ik_solution)

        return timestamps, ik_solutions