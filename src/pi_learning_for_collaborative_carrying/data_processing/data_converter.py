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

    @staticmethod
    def build(mocap_filename: str, vive_filename: str, mocap_metadata: motion_data.MocapMetadata,
              retarget_leader: bool = False) -> "DataConverter":
        """Build a DataConverter."""

        # Read in the mat file of the data as a h5py file, which acts the same as a python dict
        mocap_data = scipy.io.loadmat(mocap_filename)

        # Read the vive data into a dict
        vive_data = scipy.io.loadmat(vive_filename)

        return DataConverter(mocap_data=mocap_data, vive_data=vive_data, mocap_metadata=mocap_metadata,
                             retarget_leader=retarget_leader)

    def clean_mocap_data(self):
        """Clean the mocap data from irrelevant information."""
        mocap_data_cleaned = {key: {} for key in self.mocap_data.keys() if key not in ['__header__', '__version__', '__globals__']}
        for key in self.mocap_data.keys():
            if key not in ['__header__', '__version__', '__globals__']:

                if key == 'timestamps':
                    timestamps = self.mocap_data[key][0][self.mocap_metadata.start_ind:self.mocap_metadata.end_ind] - self.mocap_data[key][0][self.mocap_metadata.start_ind]
                    mocap_data_cleaned[key] = timestamps
                else:
                    # Extract the array from the nested structure
                    orientations = self.mocap_data[key]['orient'][0][0]
                    forces = self.mocap_data[key]['ft6D'][0][0]
                    ang_vels = self.mocap_data[key]['gyro'][0][0]

                    # Cut the data at the start time
                    orientations = orientations[self.mocap_metadata.start_ind:self.mocap_metadata.end_ind]
                    forces = forces[self.mocap_metadata.start_ind:self.mocap_metadata.end_ind]
                    ang_vels = ang_vels[self.mocap_metadata.start_ind:self.mocap_metadata.end_ind]

                    mocap_data_cleaned[key]['orient'] = orientations
                    mocap_data_cleaned[key]['ft6D'] = forces
                    mocap_data_cleaned[key]['gyro'] = ang_vels

        return mocap_data_cleaned

    def convert(self) -> motion_data.MotionData:
        """Convert the collected mocap data from the original to the intermediate format."""

        motiondata = motion_data.MotionData.build()

        mocap_data_cleaned = self.clean_mocap_data()

        node_struct = {}

        if self.retarget_leader:
            task_name_dict = {'PELVIS_TASK': 'vive_tracker_waist_pose2', 'LEFT_HAND_TASK': 'vive_tracker_right_elbow_pose2', 'RIGHT_HAND_TASK': 'vive_tracker_left_elbow_pose2', 'HEAD_TASK': 'openxr_head2'}
        else:
            task_name_dict = {'PELVIS_TASK': 'vive_tracker_waist_pose', 'LEFT_HAND_TASK': 'vive_tracker_right_elbow_pose', 'RIGHT_HAND_TASK': 'vive_tracker_left_elbow_pose', 'HEAD_TASK': 'openxr_head'} #TODO this depends on who was the follower

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

                # Get the base position and rotate into world frame
                positions = self.vive_data[task_name_dict[key]]['positions'][0][0]

                # Rotate orientations into the world frame, correctly this time
                quaternions = self.vive_data[task_name_dict[key]]['orientations'][0][0]

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

                    # Scale the positions as in teleoperation: https://github.com/robotology/human-dynamics-estimation/blob/82b84f4cb28bc37cdff6cb3250d145b5d29d68cf/conf/xml/RobotStateProvider_ergoCub_openxr_ifeel.xml
                    # Only scale the follower, not the leader
                    if not self.retarget_leader:

                        # Transform into base frame
                        # start with openxr_origin_H_sensors, to get those in world frame do I_H_openxr_origin @ openxr_origin_H_sensors, then put in base frame with root_link_H_I @ I_H_openxr_origin @ openxr_origin_H_sensors = root_link_H_sensors
                        root_link_H_sensors = [I_H_root_link.T @ I_H_openxr_origin @ openxr_origin_H_sensor for openxr_origin_H_sensor, I_H_root_link in zip(openxr_origin_H_sensors, I_H_root_links)]

                        if key == 'HEAD_TASK':
                            scaling_factor = [0.7, 0.7, 0.6]
                        else:
                            # Scale only x and y for hands such that the height is the same as the human for carrying
                            scaling_factor = [0.7, 0.7, 1.0]
                        root_link_p_sensors_scaled = [[scaling_factor[0] * root_link_H_sensor[0, 3],
                                                       scaling_factor[1] * root_link_H_sensor[1, 3],
                                                       scaling_factor[2] * root_link_H_sensor[2, 3]]
                                                       for root_link_H_sensor in root_link_H_sensors]

                        # Put the new poses back into the world frame
                        root_link_H_sensor_scaleds = [np.vstack((np.hstack((root_link_H_sensor[:3, :3], np.array(root_link_p_sensor).reshape(3, 1))), [0, 0, 0, 1])) for root_link_H_sensor, root_link_p_sensor in zip(root_link_H_sensors, root_link_p_sensors_scaled)]
                        I_H_sensors = [I_H_root_link @ root_link_H_sensor_scaled for root_link_H_sensor_scaled, I_H_root_link in zip(root_link_H_sensor_scaleds, I_H_root_links)]

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

                # Take the vertical force from the FT sensor
                forces = np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['ft6D'])[:,2]

                task = motion_data.FloorContactTask(name=key, forces=forces)
                motiondata.FloorContactTasks.append(asdict(task))

        # Update the calibration data once all tasks have been added
        motiondata.CalibrationData = node_struct

        return motiondata
