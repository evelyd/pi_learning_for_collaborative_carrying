# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, asdict
from pi_learning_for_collaborative_carrying.data_processing import motion_data
import h5py
from pi_learning_for_collaborative_carrying.data_processing import utils
import numpy as np
import biomechanical_analysis_framework as baf
import manifpy as manif
import scipy.io
from scipy.spatial.transform import Rotation


@dataclass
class DataConverter:
    """Class for converting MoCap data collected using iFeel into the intermediate MoCap data format."""

    mocap_data: dict
    vive_data: dict
    mocap_metadata: motion_data.MocapMetadata
    retarget_leader: bool = False
    vive_filename: str = ""

    @staticmethod
    def build(mocap_filename: str, vive_filename: str, mocap_metadata: motion_data.MocapMetadata,
              retarget_leader: bool = False) -> "DataConverter":
        """Build a DataConverter."""

        # Read in the mat file of the data as a h5py file, which acts the same as a python dict
        mocap_data = scipy.io.loadmat(mocap_filename)

        # Read the vive data into a dict
        vive_data = scipy.io.loadmat(vive_filename)

        return DataConverter(mocap_data=mocap_data, vive_data=vive_data, mocap_metadata=mocap_metadata,
                             retarget_leader=retarget_leader, vive_filename=vive_filename)

    def clean_and_clip_data(self):
        """Clean and clip the data based on the start and end time."""
        mocap_data_cleaned = {key: {} for key in self.mocap_data.keys() if key not in ['__header__', '__version__', '__globals__']}

        # Get time stamps first
        key = 'timestamps'
        start_time = self.mocap_metadata.start_time
        time_diffs = np.abs(self.mocap_data[key][0] - start_time)
        ifeel_start_ind = np.argmin(time_diffs)

        end_time = self.mocap_metadata.end_time
        time_diffs_end = np.abs(self.mocap_data[key][0] - end_time)
        ifeel_end_ind = np.argmin(time_diffs_end)

        # Put the start time at 0
        timestamps = self.mocap_data[key][0][ifeel_start_ind:ifeel_end_ind] - self.mocap_data[key][0][ifeel_start_ind]

        mocap_data_cleaned[key] = timestamps
        for key in self.mocap_data.keys():
            if key not in ['__header__', '__version__', '__globals__', 'timestamps']:

                # Extract the array from the nested structure
                orientations = self.mocap_data[key]['orient'][0][0]
                forces = self.mocap_data[key]['ft6D'][0][0]
                ang_vels = self.mocap_data[key]['gyro'][0][0]

                # Cut the data at the start time
                orientations = orientations[ifeel_start_ind:ifeel_end_ind]
                forces = forces[ifeel_start_ind:ifeel_end_ind]
                ang_vels = ang_vels[ifeel_start_ind:ifeel_end_ind]

                mocap_data_cleaned[key]['orient'] = orientations
                mocap_data_cleaned[key]['ft6D'] = forces
                mocap_data_cleaned[key]['gyro'] = ang_vels

        start_time = self.mocap_metadata.start_time
        end_time = self.mocap_metadata.end_time

        clipped_vive_data = {}

        key = 'world_fixed'
        timestamps = self.vive_data[key]['timestamps'][0][0][0]
        start_ind = np.argmin(np.abs(timestamps - start_time))
        end_ind = np.argmin(np.abs(timestamps - end_time))

        for key in self.vive_data.keys():
            if key not in ['__header__', '__version__', '__globals__']:

                clipped_vive_data[key] = {}
                for sub_key in self.vive_data[key].dtype.names:
                    # Put the start time at 0
                    clipped_vive_data[key][sub_key] = self.vive_data[key][sub_key][0][0][start_ind:end_ind]

        return mocap_data_cleaned, clipped_vive_data

    def convert(self) -> motion_data.MotionData:
        """Convert the collected mocap data from the original to the intermediate format."""

        motiondata = motion_data.MotionData.build()

        mocap_data_cleaned, vive_data_cleaned = self.clean_and_clip_data()

        node_struct = {}

        if self.retarget_leader:
            task_name_dict = {'PELVIS_TASK': 'vive_tracker_waist_pose', 'LEFT_HAND_TASK': 'vive_tracker_left_elbow_pose', 'RIGHT_HAND_TASK': 'vive_tracker_right_elbow_pose', 'HEAD_TASK': 'openxr_head', 'LEFT_TOE_TASK': 'vive_tracker_left_foot_pose', 'RIGHT_TOE_TASK': 'vive_tracker_right_foot_pose'} #TODO this depends on who was the follower
        else:
            task_name_dict = {'PELVIS_TASK': 'vive_tracker_waist_pose2', 'LEFT_HAND_TASK': 'vive_tracker_left_elbow_pose2', 'RIGHT_HAND_TASK': 'vive_tracker_right_elbow_pose2', 'HEAD_TASK': 'openxr_head2', 'LEFT_TOE_TASK': 'vive_tracker_left_foot_pose2', 'RIGHT_TOE_TASK': 'vive_tracker_right_foot_pose2'}

        for key, item in self.mocap_metadata.metadata.items():

            item_type = item['type']

            # Retrieve and store timestamps for the entire dataset
            if item_type == "TimeStamp":
                motiondata.SampleDurations = mocap_data_cleaned['timestamps']

            # Store pose task data
            elif item_type == "SE3Task":

                # Rotate the values into the world frame
                I_H_openxr_origin = np.array([[0, 0, -1, 0],
                                        [-1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]])

                waist_tracker_H_root_link = np.array([[0, 1, 0, 0],
                                                 [0, 0, 1, -0.1],
                                                 [1, 0, 0, 0.1],
                                                 [0, 0, 0, 1]])

                # Apply small corrections due to imperfect sensor placement
                if self.retarget_leader and any(x in self.vive_filename for x in ['1', '2', '3', '4', '5']):
                    # Add a 10 degree pitch rotation to the waist tracker H root link
                    pitch_angle = np.deg2rad(10)
                    pitch_rotation_matrix = Rotation.from_euler('x', pitch_angle).as_matrix()
                    waist_tracker_H_root_link[:3, :3] = waist_tracker_H_root_link[:3, :3] @ pitch_rotation_matrix
                elif not self.retarget_leader and any(x in self.vive_filename for x in ['1', '2', '3', '4', '5']):
                    # Add a 5 degree pitch rotation to the waist tracker H root link
                    pitch_angle = np.deg2rad(5)
                    pitch_rotation_matrix = Rotation.from_euler('x', pitch_angle).as_matrix()
                    waist_tracker_H_root_link[:3, :3] = waist_tracker_H_root_link[:3, :3] @ pitch_rotation_matrix

                # Get the base position and rotate into world frame
                positions = vive_data_cleaned[task_name_dict[key]]['positions']

                # Rotate orientations into the world frame, correctly this time
                quaternions = vive_data_cleaned[task_name_dict[key]]['orientations']

                if key == 'PELVIS_TASK':
                    # Define the transformation matrix of raw data
                    openxr_origin_H_waist_trackers = [np.vstack((np.hstack((Rotation.from_quat(quat).as_matrix(), pos.reshape(3, 1))), [0, 0, 0, 1])) for pos, quat in zip(positions, quaternions)]

                    # Apply the transformation to the data
                    I_H_root_links = [I_H_openxr_origin @ openxr_origin_H_waist_tracker @ waist_tracker_H_root_link for openxr_origin_H_waist_tracker in openxr_origin_H_waist_trackers]

                    # Extract the positions and orientations
                    positions = [pose[:3, 3] for pose in I_H_root_links]
                    quaternions = [Rotation.from_matrix(pose[:3, :3]).as_quat() for pose in I_H_root_links]
                else:
                    # Define the transformation matrix of raw data
                    openxr_origin_H_sensors = [np.vstack((np.hstack((Rotation.from_quat(quat).as_matrix(), pos.reshape(3, 1))), [0, 0, 0, 1])) for pos, quat in zip(positions, quaternions)]

                    I_H_sensors = [I_H_openxr_origin @ openxr_origin_H_sensor for openxr_origin_H_sensor in openxr_origin_H_sensors]

                    # Extract the positions and orientations
                    positions = [pose[:3, 3] for pose in I_H_sensors]
                    quaternions = [Rotation.from_matrix(pose[:3, :3]).as_quat() for pose in I_H_sensors]

                # Normalize the quaternions
                quaternions = [utils.normalize_quaternion(quat) for quat in quaternions]

                task = motion_data.SE3Task(name=key, positions=positions, orientations=quaternions)
                motiondata.SE3Tasks.append(asdict(task))

                # Save the first base pose
                if key == 'PELVIS_TASK':
                    initial_base_pose = I_H_root_links[0]
                    np.put(initial_base_pose, 11, 0.0)
                    motiondata.initial_base_pose = initial_base_pose

            # Store orientation task data
            elif item_type == "SO3Task":
                # Assumes wxyz format for raw data
                quaternions = [utils.normalize_quaternion(utils.to_xyzw(quat)) for quat in mocap_data_cleaned['node' + str(item['node_number'])]['orient']]

                angular_velocities = mocap_data_cleaned['node' + str(item['node_number'])]['gyro']

                task = motion_data.SO3Task(name=key, orientations=quaternions, angular_velocities=angular_velocities)
                motiondata.SO3Tasks.append(asdict(task))

                # Update node struct for calibration
                I_R_IMU_calib = manif.SO3(quaternion=np.array(quaternions[0]))
                I_omega_IMU_calib = manif.SO3Tangent(angular_velocities[0])

                nodeData = baf.ik.nodeData()
                nodeData.I_R_IMU = I_R_IMU_calib
                nodeData.I_omega_IMU = I_omega_IMU_calib
                node_struct[item['node_number']] = nodeData

            # Store gravity task data
            elif item_type == "GravityTask":

                # Assumes wxyz format for raw data
                quaternions = [utils.normalize_quaternion(utils.to_xyzw(quat)) for quat in mocap_data_cleaned['node' + str(item['node_number'])]['orient']]

                task = motion_data.GravityTask(name=key, orientations=quaternions)
                motiondata.GravityTasks.append(asdict(task))

                # Update node struct for calibration
                I_R_IMU_calib = manif.SO3(quaternion=np.array(quaternions[0]))
                I_omega_IMU_calib = manif.SO3Tangent(mocap_data_cleaned['node' + str(item['node_number'])]['gyro'][0])

                nodeData = baf.ik.nodeData()
                nodeData.I_R_IMU = I_R_IMU_calib
                nodeData.I_omega_IMU = I_omega_IMU_calib
                node_struct[item['node_number']] = nodeData

            # Store position task data
            elif item_type == "FloorContactTask":
                # Rotate the values into the world frame
                I_H_openxr_origin = np.array([[0, 0, -1, 0],
                                        [-1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]])

                waist_tracker_H_root_link = np.array([[0, 1, 0, 0],
                                                 [0, 0, 1, -0.1],
                                                 [1, 0, 0, 0.1],
                                                 [0, 0, 0, 1]])

                # Get the base position and rotate into world frame
                positions = vive_data_cleaned[task_name_dict[key]]['positions']

                # Rotate orientations into the world frame, correctly this time
                quaternions = vive_data_cleaned[task_name_dict[key]]['orientations']

                # Define the transformation matrix of raw data
                openxr_origin_H_sensors = [np.vstack((np.hstack((Rotation.from_quat(quat).as_matrix(), pos.reshape(3, 1))), [0, 0, 0, 1])) for pos, quat in zip(positions, quaternions)]

                I_H_sensors = [I_H_openxr_origin @ openxr_origin_H_sensor for openxr_origin_H_sensor in openxr_origin_H_sensors]

                # Extract the positions and orientations
                positions = [pose[:3, 3] for pose in I_H_sensors]
                quaternions = [Rotation.from_matrix(pose[:3, :3]).as_quat() for pose in I_H_sensors]

                # Normalize the quaternions
                quaternions = [utils.normalize_quaternion(quat) for quat in quaternions]

                # Take the vertical force from the FT sensor
                forces = np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['ft6D'])[:,2]

                task = motion_data.FloorContactTask(name=key, positions=positions, orientations=quaternions, forces=forces)
                motiondata.FloorContactTasks.append(asdict(task))

        # Update the calibration data once all tasks have been added
        motiondata.CalibrationData = node_struct

        return motiondata
