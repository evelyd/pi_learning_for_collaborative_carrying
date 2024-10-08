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

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 36

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

    def enforce_horizontal_feet(self) -> None:
        """Enforce zero roll and pitch target orientation for the feet, i.e. enforce feet parallel to the ground."""

        for link in ["RightFoot", "LeftFoot"]:

            print("Enforcing horizontal", link)

            updated_orientations = []

            for i in range(len(self.link_orientation_targets[link])):

                # Retrieve original target yaw
                original_quaternions = self.link_orientation_targets[link][i]
                original_rotation = Rotation.from_quat(utils.to_xyzw(original_quaternions))
                original_yaw = original_rotation.as_euler('xyz')[2]

                # Enforce zero pitch and roll
                updated_rotation = Rotation.from_euler('xyz', [0, 0, original_yaw])
                updated_quaternions = utils.to_wxyz(updated_rotation.as_quat())
                updated_orientations.append(updated_quaternions)

            self.link_orientation_targets[link] = np.array(updated_orientations)

    def enforce_straight_head(self) -> None:
        """Enforce torso roll and pitch target orientation for the head, while keeping the yaw unchanged."""

        print("Enforcing straight Head")

        updated_head_orientations = []

        for i in range(len(self.link_orientation_targets["Head"])):

            # Retrieve original head target yaw
            original_head_quaternions = self.link_orientation_targets["Head"][i]
            original_head_rotation = Rotation.from_quat(utils.to_xyzw(original_head_quaternions))
            original_head_yaw = original_head_rotation.as_euler('xyz')[2]

            # Retrieve torso target roll and pitch
            torso_quaternions = self.link_orientation_targets["T8"][i]
            torso_rotation = Rotation.from_quat(utils.to_xyzw(torso_quaternions))
            torso_euler_angles = torso_rotation.as_euler('xyz')
            torso_roll = torso_euler_angles[0]
            torso_pitch = torso_euler_angles[1]

            # Enforce torso roll and pitch target orientation for the head, while keeping the yaw unchanged
            updated_head_rotation = Rotation.from_euler('xyz', [torso_roll, torso_pitch, original_head_yaw])
            updated_head_quaternions = utils.to_wxyz(updated_head_rotation.as_quat())
            updated_head_orientations.append(updated_head_quaternions)

        self.link_orientation_targets["Head"] = np.array(updated_head_orientations)

    def enforce_wider_legs(self) -> None:
        """Enforce offsets to the upper leg target orientations so to avoid feet crossing."""

        print("Enforcing wider legs")

        updated_orientations_left = []
        updated_orientations_right = []
        roll_difference = []

        # Define threshold for the minimum difference between the left and the right rolls
        roll_difference_threshold = 0.1

        for i in range(len(self.link_orientation_targets["LeftUpperLeg"])):

            # Retrieve original left target rpy
            original_quaternions_left = self.link_orientation_targets["LeftUpperLeg"][i]
            original_rotation_left = Rotation.from_quat(utils.to_xyzw(original_quaternions_left))
            original_rpy_left = original_rotation_left.as_euler('xyz')

            # Retrieve original right target rpy
            original_quaternions_right = self.link_orientation_targets["RightUpperLeg"][i]
            original_rotation_right = Rotation.from_quat(utils.to_xyzw(original_quaternions_right))
            original_rpy_right = original_rotation_right.as_euler('xyz')

            # Update left and right rolls if needed (based on the difference between the left and right rolls)
            if original_rpy_left[0] - original_rpy_right[0] < roll_difference_threshold:
                delta = roll_difference_threshold - (original_rpy_left[0] - original_rpy_right[0])
                original_rpy_left += [delta/2.0, 0, 0]
                original_rpy_right -= [delta/2.0, 0, 0]

            # Retrieve update left target rpy
            updated_rotation_left = Rotation.from_euler('xyz', original_rpy_left)
            updated_quaternions_left = utils.to_wxyz(updated_rotation_left.as_quat())
            updated_orientations_left.append(updated_quaternions_left)

            # Retrieve updated left target rpy
            updated_rotation_right = Rotation.from_euler('xyz', original_rpy_right)
            updated_quaternions_right = utils.to_wxyz(updated_rotation_right.as_quat())
            updated_orientations_right.append(updated_quaternions_right)

        # Store updated targets
        self.link_orientation_targets["LeftUpperLeg"] = np.array(updated_orientations_left)
        self.link_orientation_targets["RightUpperLeg"] = np.array(updated_orientations_right)


@dataclass
class WBGR:
    """Class implementing the Whole-Body Geometric Retargeting (WBGR)."""

    param_handler: blf.parameters_handler.IParametersHandler
    motiondata: motion_data.MotionData
    metadata: motion_data.MocapMetadata
    robot_to_target_base_quat: List
    joint_names: List
    kindyn: idyn.KinDynComputations
    calibration_matrices = {}
    IMU_link_rotations = {}
    initial_base_height: float
    humanIK: baf.ik.HumanIK
    node_struct: dict

    @staticmethod
    def build(motiondata: motion_data.MotionData,
              metadata: motion_data.MocapMetadata,
              humanIK: baf.ik.HumanIK,
              node_struct: dict,
              param_handler: blf.parameters_handler.IParametersHandler,
              joint_names: List,
              kindyn: idyn.KinDynComputations,
              mirroring: bool = False,
              horizontal_feet: bool = False,
              straight_head: bool = False,
              wider_legs: bool = False,
              robot_to_target_base_quat: List = None,
              initial_base_height: float = 0.0) -> "WBGR":
        """Build an instance of WBGR."""

        #TODO update this for using ifeel data
        # if mirroring:
        #     # Mirror the ik targets
        #     ik_targets.mirror_ik_targets()

        # if wider_legs:
            # Enforce wider legs
            # ik_targets.enforce_wider_legs()

        return WBGR(param_handler=param_handler, motiondata=motiondata, metadata=metadata, joint_names=joint_names, kindyn=kindyn, robot_to_target_base_quat=robot_to_target_base_quat, initial_base_height=initial_base_height, humanIK=humanIK, node_struct=node_struct)

    def retarget(self, plot_ik_solutions: bool) -> (List, List):
        """Apply Whole-Body Geometric Retargeting (WBGR)."""

        timestamps = []
        ik_solutions = []

        # Get the height of the front foot frame off the ground
        foot_height = utils.idyn_transform_to_np(self.kindyn.getWorldTransform("r_foot_front"))[2,3]

        # ====================================================
        # Calibrate using humanIK
        # ====================================================

        self.humanIK.clearCalibrationMatrices()
        self.humanIK.calibrateWorldYaw(self.node_struct)
        self.humanIK.calibrateAllWithWorld(self.node_struct, "r_foot_front")

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


@dataclass
class KinematicComputations:
    """Class for the kinematic computations exploited by the Kinematically-Feasible Whole-Body
    Geometric Retargeting (KFWBGR).
    """

    kindyn: idyn.KinDynComputations
    local_foot_vertices_pos: List
    feet_frames: Dict

    @staticmethod
    def build(kindyn: idyn.KinDynComputations,
              local_foot_vertices_pos: List,
              feet_frames: Dict) -> "KinematicComputations":
        """Build an instance of KinematicComputations."""

        return KinematicComputations(kindyn=kindyn, local_foot_vertices_pos=local_foot_vertices_pos, feet_frames=feet_frames)

    def compute_W_vertices_pos(self) -> List:
        """Compute the feet vertices positions in the world (W) frame."""

        # Retrieve front-left (FL), front-right (FR), back-left (BL) and back-right (BR) vertices in the foot frame
        FL_vertex_pos = self.local_foot_vertices_pos[0]
        FR_vertex_pos = self.local_foot_vertices_pos[1]
        BL_vertex_pos = self.local_foot_vertices_pos[2]
        BR_vertex_pos = self.local_foot_vertices_pos[3]

        # Compute right foot (RF) transform w.r.t. the world (W) frame
        world_H_base = utils.idyn_transform_to_np(self.kindyn.getWorldBaseTransform())
        base_H_r_foot = utils.idyn_transform_to_np(self.kindyn.getRelativeTransform("root_link",
                                                                                    self.feet_frames["right_foot"]))
        W_H_RF = world_H_base.dot(base_H_r_foot)

        # Get the right-foot vertices positions in the world frame
        W_RFL_vertex_pos_hom = W_H_RF @ np.concatenate((FL_vertex_pos, [1]))
        W_RFR_vertex_pos_hom = W_H_RF @ np.concatenate((FR_vertex_pos, [1]))
        W_RBL_vertex_pos_hom = W_H_RF @ np.concatenate((BL_vertex_pos, [1]))
        W_RBR_vertex_pos_hom = W_H_RF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_RFL_vertex_pos = W_RFL_vertex_pos_hom[0:3]
        W_RFR_vertex_pos = W_RFR_vertex_pos_hom[0:3]
        W_RBL_vertex_pos = W_RBL_vertex_pos_hom[0:3]
        W_RBR_vertex_pos = W_RBR_vertex_pos_hom[0:3]

        # Compute left foot (LF) transform w.r.t. the world (W) frame
        world_H_base = utils.idyn_transform_to_np(self.kindyn.getWorldBaseTransform())
        base_H_l_foot = utils.idyn_transform_to_np(self.kindyn.getRelativeTransform("root_link",
                                                                                    self.feet_frames["left_foot"]))
        W_H_LF = world_H_base.dot(base_H_l_foot)

        # Get the left-foot vertices positions wrt the world frame
        W_LFL_vertex_pos_hom = W_H_LF @ np.concatenate((FL_vertex_pos, [1]))
        W_LFR_vertex_pos_hom = W_H_LF @ np.concatenate((FR_vertex_pos, [1]))
        W_LBL_vertex_pos_hom = W_H_LF @ np.concatenate((BL_vertex_pos, [1]))
        W_LBR_vertex_pos_hom = W_H_LF @ np.concatenate((BR_vertex_pos, [1]))

        # Convert homogeneous to cartesian coordinates
        W_LFL_vertex_pos = W_LFL_vertex_pos_hom[0:3]
        W_LFR_vertex_pos = W_LFR_vertex_pos_hom[0:3]
        W_LBL_vertex_pos = W_LBL_vertex_pos_hom[0:3]
        W_LBR_vertex_pos = W_LBR_vertex_pos_hom[0:3]

        # Store the positions of both right-foot and left-foot vertices in the world frame
        W_vertices_positions = [W_RFL_vertex_pos, W_RFR_vertex_pos, W_RBL_vertex_pos, W_RBR_vertex_pos,
                                W_LFL_vertex_pos, W_LFR_vertex_pos, W_LBL_vertex_pos, W_LBR_vertex_pos]

        return W_vertices_positions

    def compute_support_vertex_pos(self, support_foot: str, support_vertex: int) -> List:
        """Compute the support vertex position in the world (W) frame."""

        # Compute the transform of the support foot (SF) wrt the world (W) frame
        world_H_base = utils.idyn_transform_to_np(self.kindyn.getWorldBaseTransform())
        base_H_SF = utils.idyn_transform_to_np(self.kindyn.getRelativeTransform("root_link", support_foot))
        W_H_SF = world_H_base.dot(base_H_SF)

        # Get the support vertex position wrt the world frame
        F_support_vertex_pos = self.local_foot_vertices_pos[support_vertex]
        F_support_vertex_pos_hom = np.concatenate((F_support_vertex_pos, [1]))
        W_support_vertex_pos_hom = W_H_SF @ F_support_vertex_pos_hom
        W_support_vertex_pos = W_support_vertex_pos_hom[0:3]

        return W_support_vertex_pos

    def compute_base_position_by_leg_odometry(self, support_vertex_pos: List, support_foot: str,
                                              support_vertex_offset: List) -> List:
        """Compute kinematically-feasible base position using leg odometry."""

        # Get the base (B) position in the world (W) frame
        W_pos_B = utils.idyn_transform_to_np(self.kindyn.getWorldBaseTransform())[:3,-1]

        # Get the support vertex position in the world (W) frame
        W_support_vertex_pos = support_vertex_pos

        # Get the support vertex orientation in the world (W) frame, defined as the support foot (SF) orientation
        world_H_base = utils.idyn_transform_to_np(self.kindyn.getWorldBaseTransform())
        base_H_SF = utils.idyn_transform_to_np(self.kindyn.getRelativeTransform("root_link", support_foot))
        W_H_SF = world_H_base.dot(base_H_SF)
        W_support_vertex_quat = utils.to_wxyz(Rotation.from_matrix(W_H_SF[0:3, 0:3]).as_quat())

        # Compute the transform of the support vertex (SV) in the world (W) frame
        W_H_SV = utils.transform_from_pos_quat(position=np.asarray(W_support_vertex_pos),
                                               quaternion=np.asarray(W_support_vertex_quat))

        # Express the base (B) position in the support vertex (SV) reference frame
        SV_H_W = np.linalg.inv(W_H_SV)
        W_pos_B_hom = np.concatenate((W_pos_B, [1]))
        SV_pos_B_hom = SV_H_W @ W_pos_B_hom

        # Express the base (B) position in a reference frame oriented as the world but positioned in the support vertex (SV)
        mixed_H_SV = utils.transform_from_pos_quat(position=np.asarray([0, 0, 0]),
                                                   quaternion=np.asarray(W_support_vertex_quat))
        mixed_pos_B_hom = mixed_H_SV @ SV_pos_B_hom

        # Convert homogeneous to cartesian coordinates
        mixed_pos_B = mixed_pos_B_hom[0:3]

        # Compute the kinematically-feasible base position, i.e. the base position such that the support
        # vertex remains fixed while the robot configuration changes
        b_pos = mixed_pos_B + support_vertex_offset

        return b_pos


@dataclass
class KFWBGR(WBGR):
    """Class implementing the Kinematically-Feasible Whole-Body Geometric Retargeting (KFWBGR)."""

    kinematic_computations: KinematicComputations
    initial_base_height: float

    @staticmethod
    def build(motiondata: motion_data.MotionData,
              metadata: motion_data.MocapMetadata,
              joint_names: List,
              mirroring: bool = False,
              horizontal_feet: bool = False,
              straight_head: bool = False,
              wider_legs: bool = False,
              robot_to_target_base_quat: List = None,
              kindyn: idyn.KinDynComputations = None,
              local_foot_vertices_pos: List = None,
              feet_frames: Dict = None,
              initial_base_height: float = 0.0) -> "KFWBGR":
        """Build an instance of KFWBGR."""

        # Instantiate IKTargets
        ik_targets = IKTargets.build(motiondata=motiondata, metadata=metadata)

        if mirroring:
            # Mirror the ik targets
            ik_targets.mirror_ik_targets()

        if wider_legs:
            # Enforce wider legs
            ik_targets.enforce_wider_legs()

        kinematic_computations = KinematicComputations.build(
            kindyn=kindyn, local_foot_vertices_pos=local_foot_vertices_pos, feet_frames=feet_frames)

        return KFWBGR(ik_targets=ik_targets,
                      joint_names=joint_names,
                      kindyn=kindyn, robot_to_target_base_quat=robot_to_target_base_quat, initial_base_height=initial_base_height, kinematic_computations=kinematic_computations)

    def KF_retarget(self, plot_ik_solutions: bool) -> (List, List):
        """Apply Kinematically-Feasible Whole-Body Geometric Retargeting (KFWBGR)."""

        # WBGR
        timestamps, ik_solutions = self.retarget(plot_ik_solutions)

        # Compute kinematically-feasible linear base motion
        kinematically_feasible_base_position = self.compute_kinematically_feasible_base_motion(ik_solutions)

        # Override base position in the ik solutions
        for i in range(len(kinematically_feasible_base_position)):
            ik_solutions[i+1].base_position = kinematically_feasible_base_position[i]

        return timestamps, ik_solutions

    def compute_kinematically_feasible_base_motion(self, ik_solutions: List) -> List:
        """Compute a kinematically-feasible base linear motion."""

        # Initialize output list
        kinematically_feasible_base_positions = []

        # Set initial robot configuration
        initial_joint_configuration = ik_solutions[0].joint_configuration
        initial_base_position = ik_solutions[0].base_position
        initial_base_quaternion = ik_solutions[0].base_quaternion
        utils.reset_robot_configuration(self.kinematic_computations.kindyn,
                                        joint_positions=initial_joint_configuration,
                                        base_position=initial_base_position,
                                        base_quaternion=initial_base_quaternion)

        # Associate indexes to feet vertices names, from Right-foot Front Left (RFL) to Left-foot Back Right (LBR)
        vertex_indexes_to_names = {0: "RFL", 1: "RFR", 2: "RBL", 3: "RBR",
                                   4: "LFL", 5: "LFR", 6: "LBL", 7: "LBR"}

        # Define the initial support vertex index and the initial support foot
        support_vertex_prev = 0 # i.e. right-foot front-left vertex (RFL)
        support_foot = self.kinematic_computations.feet_frames["right_foot"]

        # Compute the initial support vertex position in the world frame and its ground projection
        support_vertex_pos = self.kinematic_computations.compute_support_vertex_pos(
            support_foot=support_foot, support_vertex=support_vertex_prev)
        support_vertex_offset = [support_vertex_pos[0], support_vertex_pos[1], 0]

        for ik_solution in ik_solutions[1:]:

            # ===========================================
            # UPDATE JOINT POSITIONS AND BASE ORIENTATION
            # ===========================================

            joint_positions = ik_solution.joint_configuration
            base_quaternion = ik_solution.base_quaternion
            previous_base_position = utils.idyn_transform_to_np(self.kinematic_computations.kindyn.getWorldBaseTransform())[:3,-1]
            utils.reset_robot_configuration(self.kinematic_computations.kindyn,
                                            joint_positions=joint_positions,
                                            base_position=previous_base_position,
                                            base_quaternion=base_quaternion)

            # ==============================
            # UPDATE SUPPORT VERTEX POSITION
            # ==============================

            # Retrieve the vertices positions in the world (W) frame
            W_vertices_positions = self.kinematic_computations.compute_W_vertices_pos()

            # Update support vertex position
            support_vertex_pos = W_vertices_positions[support_vertex_prev]

            # =======================
            # RECOMPUTE BASE POSITION
            # =======================

            # Compute kinematically-feasible base position by leg odometry
            kinematically_feasible_base_position = self.kinematic_computations.compute_base_position_by_leg_odometry(
                support_vertex_pos=support_vertex_pos,
                support_foot=support_foot,
                support_vertex_offset=support_vertex_offset)

            # Update the robot configuration with the kinematically-feasible base position
            utils.reset_robot_configuration(self.kinematic_computations.kindyn,
                                            joint_positions=joint_positions,
                                            base_position=kinematically_feasible_base_position,
                                            base_quaternion=base_quaternion)

            # ======================================
            # UPDATE SUPPORT VERTEX AND SUPPORT FOOT
            # ======================================

            # Retrieve the vertices positions in the world (W) frame
            W_vertices_positions = self.kinematic_computations.compute_W_vertices_pos()

            # Compute the current support vertex as the lowest among the feet vertices
            vertices_heights = [W_vertex[2] for W_vertex in W_vertices_positions]
            support_vertex = np.argmin(vertices_heights)

            if support_vertex == support_vertex_prev:

                # Update the support vertex position only
                support_vertex_pos = W_vertices_positions[support_vertex]

            else:

                # Update the support foot
                if vertex_indexes_to_names[support_vertex][0] == "R":
                    support_foot = self.kinematic_computations.feet_frames["right_foot"]
                else:
                    support_foot = self.kinematic_computations.feet_frames["left_foot"]

                # Debug
                print("Change of support vertex: from", vertex_indexes_to_names[support_vertex_prev],
                      "to", vertex_indexes_to_names[support_vertex])
                print("New support foot:", support_foot)

                # Update the support vertex position and its ground projection
                support_vertex_pos = W_vertices_positions[support_vertex]
                support_vertex_offset = [support_vertex_pos[0], support_vertex_pos[1], 0]

                support_vertex_prev = support_vertex

            # =======================
            # RECOMPUTE BASE POSITION
            # =======================

            # Compute kinematically-feasible base position by leg odometry
            kinematically_feasible_base_position = self.kinematic_computations.compute_base_position_by_leg_odometry(
                support_vertex_pos=support_vertex_pos,
                support_foot=support_foot,
                support_vertex_offset=support_vertex_offset)
            kinematically_feasible_base_positions.append(kinematically_feasible_base_position)

            # Update the robot configuration with the kinematically-feasible base position
            utils.reset_robot_configuration(self.kinematic_computations.kindyn,
                                            joint_positions=joint_positions,
                                            base_position=kinematically_feasible_base_position,
                                            base_quaternion=base_quaternion)

        return kinematically_feasible_base_positions
