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

        for key, item in self.mocap_metadata.metadata.items():

            item_type = item['type']

            # Get the index of the timestamp closest to the start time
            zeroed_timestamps = np.squeeze(mocap_data_cleaned['shoe1']['FT']['timestamps'][:] - mocap_data_cleaned['shoe1']['FT']['timestamps'][0])

            start_time_index = np.argmin(np.abs(zeroed_timestamps - self.mocap_metadata.start_time))

            if item_type == "Calibration":
                # Create a node struct for calibration, using the measurements at calibration time
                orientation_nodes = [3, 6, 7, 8, 5, 4, 11, 12, 9, 10]
                floor_contact_nodes = [1, 2]
                node_struct = {}
                for node in orientation_nodes + floor_contact_nodes:
                    # Define time series of rotations for this node
                    I_R_IMU = [manif.SO3(quaternion=utils.normalize_quaternion(utils.to_xyzw(quat))) for quat in np.squeeze(mocap_data_cleaned['node' + str(node)]['orientation']['data'][start_time_index:])]
                    # Define time series of angular velocities for this node
                    I_omega_IMU = [manif.SO3Tangent(omega) for omega in np.squeeze(mocap_data_cleaned['node' + str(node)]['angVel']['data'][start_time_index:])]
                    # Assign these values to the node struct
                    nodeData = baf.ik.nodeData()
                    nodeData.I_R_IMU = I_R_IMU[0]
                    nodeData.I_omega_IMU = I_omega_IMU[0]
                    node_struct[node] = nodeData
                motiondata.CalibrationData = node_struct

            # Retrieve and store timestamps for the entire dataset
            if item_type == "TimeStamp":
                timestamps = zeroed_timestamps[start_time_index:] - zeroed_timestamps[start_time_index]
                motiondata.SampleDurations = timestamps

            # Store orientation task data
            elif item_type == "SO3Task":
                # Assumes wxyz format
                quaternions = [utils.normalize_quaternion(quat) for quat in np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['orientation']['data'][start_time_index:])]
                norms = [np.linalg.norm(quat) for quat in quaternions]
                angular_velocities = np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['angVel']['data'][start_time_index:])

                # Check if norms contain any zeros
                if any(norm == 0 for norm in norms):
                    raise ValueError("One or more quaternions have zero norm and cannot be normalized")

                task = motion_data.SO3Task(name=key, orientations=quaternions, angular_velocities=angular_velocities)
                motiondata.SO3Tasks.append(asdict(task))

            # Store gravity task data
            elif item_type == "GravityTask":

                # Assumes wxyz format
                quaternions = [utils.normalize_quaternion(quat) for quat in np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['orientation']['data'][start_time_index:])]
                norms = [np.linalg.norm(quat) for quat in quaternions]

                # Check if norms contain any zeros
                if any(norm == 0 for norm in norms):
                    raise ValueError("One or more quaternions have zero norm and cannot be normalized")

                task = motion_data.GravityTask(name=key, orientations=quaternions)
                motiondata.GravityTasks.append(asdict(task))

            # Store position task data
            elif item_type == "FloorContactTask":

                # Take the vertical force from the FT sensor
                forces = np.squeeze(mocap_data_cleaned['shoe' + str(item['node_number'])]['FT']['data'][start_time_index:])[:,2]

                task = motion_data.FloorContactTask(name=key, forces=forces)
                motiondata.FloorContactTasks.append(asdict(task))

        return motiondata
