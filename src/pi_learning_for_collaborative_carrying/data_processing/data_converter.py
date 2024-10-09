# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, asdict
from pi_learning_for_collaborative_carrying.data_processing import motion_data
import h5py
from pi_learning_for_collaborative_carrying.data_processing import utils
import numpy as np
import biomechanical_analysis_framework as baf
import manifpy as manif

@dataclass
class DataConverter:
    """Class for converting MoCap data collected using iFeel into the intermediate MoCap data format."""

    mocap_data: dict
    mocap_metadata: motion_data.MocapMetadata

    @staticmethod
    def build(mocap_filename: str, mocap_metadata: motion_data.MocapMetadata) -> "DataConverter":
        """Build a DataConverter."""

        # Read in the mat file of the data as a h5py file, which acts the same as a python dict
        mocap_data = h5py.File(mocap_filename, 'r')

        return DataConverter(mocap_data=mocap_data, mocap_metadata=mocap_metadata)

    def convert(self) -> motion_data.MotionData:
        """Convert the collected mocap data from the original to the intermediate format."""

        # Clean mocap frames from irrelevant information
        mocap_data_cleaned = self.mocap_data['robot_logger_device']

        motiondata = motion_data.MotionData.build()

        node_struct = {}

        for key, item in self.mocap_metadata.metadata.items():

            item_type = item['type']

            # Get the index of the timestamp closest to the start time
            zeroed_timestamps = np.squeeze(mocap_data_cleaned['shoe1']['FT']['timestamps'][:] - mocap_data_cleaned['shoe1']['FT']['timestamps'][0])

            start_time_index = np.argmin(np.abs(zeroed_timestamps - self.mocap_metadata.start_time))

            # Retrieve and store timestamps for the entire dataset
            if item_type == "TimeStamp":
                timestamps = zeroed_timestamps[start_time_index:] - zeroed_timestamps[start_time_index]
                motiondata.SampleDurations = timestamps

            # Store orientation task data
            elif item_type == "SO3Task":
                # Assumes wxyz format
                quaternions = [utils.normalize_quaternion(quat) for quat in np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['orientation']['data'][start_time_index:])]
                angular_velocities = np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['angVel']['data'][start_time_index:])

                task = motion_data.SO3Task(name=key, orientations=quaternions, angular_velocities=angular_velocities)
                motiondata.SO3Tasks.append(asdict(task))

                # Update node struct for calibration
                I_R_IMU_calib = manif.SO3(quaternion=utils.normalize_quaternion(utils.to_xyzw(np.array(quaternions[0]))))
                I_omega_IMU_calib = manif.SO3Tangent(angular_velocities[0])

                nodeData = baf.ik.nodeData()
                nodeData.I_R_IMU = I_R_IMU_calib
                nodeData.I_omega_IMU = I_omega_IMU_calib
                node_struct[item['node_number']] = nodeData

            # Store gravity task data
            elif item_type == "GravityTask":

                # Assumes wxyz format
                quaternions = [utils.normalize_quaternion(quat) for quat in np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['orientation']['data'][start_time_index:])]

                task = motion_data.GravityTask(name=key, orientations=quaternions)
                motiondata.GravityTasks.append(asdict(task))

                # Update node struct for calibration
                I_R_IMU_calib = manif.SO3(quaternion=utils.normalize_quaternion(utils.to_xyzw(np.array(quaternions[0]))))
                I_omega_IMU_calib = manif.SO3Tangent(np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['angVel']['data'][start_time_index]))

                nodeData = baf.ik.nodeData()
                nodeData.I_R_IMU = I_R_IMU_calib
                nodeData.I_omega_IMU = I_omega_IMU_calib
                node_struct[item['node_number']] = nodeData

            # Store position task data
            elif item_type == "FloorContactTask":

                # Take the vertical force from the FT sensor
                forces = np.squeeze(mocap_data_cleaned['shoe' + str(item['node_number'])]['FT']['data'][start_time_index:])[:,2]

                task = motion_data.FloorContactTask(name=key, forces=forces)
                motiondata.FloorContactTasks.append(asdict(task))

        # Update the calibration data once all tasks have been added
        motiondata.CalibrationData = node_struct

        return motiondata
