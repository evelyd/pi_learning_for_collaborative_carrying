# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Dict
from dataclasses import dataclass, asdict
from pi_learning_for_collaborative_carrying.data_processing import motion_data
import h5py
from pi_learning_for_collaborative_carrying.data_processing import utils
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

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

    @staticmethod
    def extract_timestamp(mocap_frame: str) -> float:
        """Extract the timestamp from the mocap data associated with a certain frame."""

        # Split the XSens message to identify data related to the links
        mocap_frame_as_list = mocap_frame.split("XsensSuit::vLink::")

        timestamp = float(mocap_frame_as_list[0].split()[1])

        return timestamp

    def extract_links_data(self, mocap_frame: str) -> Dict:
        """Extract links data from the mocap data associated with a certain frame."""

        # Discard from the XSens message data not related to the links
        mocap_frame_as_list = mocap_frame.split("XsensSuit::vLink::")[1:]
        mocap_frame_as_list[-1] = mocap_frame_as_list[-1].split("XsensSuit::vSJoint::")[0]
        mocap_frame_as_list = mocap_frame_as_list[1::2]

        links_data = {}

        for i in range(len(mocap_frame_as_list)):

            # Further cleaning of the XSens message components
            link_info = mocap_frame_as_list[i].strip('" ()').split()
            link_info[0] = link_info[0].strip('"')
            link_name = link_info[0]

            # Skip irrelevant links
            if link_name not in self.mocap_metadata.metadata.keys():
                continue

            if link_name == "Pelvis":
                # Store position and orientation for the base (Pelvis)
                links_data[link_name] = [float(n) for n in link_info[2:9]]
            else:
                # Store orientation only for the other links
                links_data[link_name] = [float(n) for n in link_info[2:6]]

        return links_data

    def clean_mocap_data(self) -> List:
        """Clean the mocap frames collected using XSens from irrelevant information."""

        mocap_data_cleaned = []

        for mocap_frame in self.mocap_data:

            # Skip empty mocap frames that sometimes occur in the dataset
            if len(mocap_frame) <= 1:
                continue

            # Extract timestamp and links data
            timestamp = self.extract_timestamp(mocap_frame)
            links_data = self.extract_links_data(mocap_frame)

            # Store timestamp and links data
            mocap_frame_cleaned = {"timestamp": timestamp}
            for link_name in links_data.keys():
                mocap_frame_cleaned[link_name] = links_data[link_name]

            # Discard the mocap frames containing incomplete information that sometimes occur in the dataset
            if len(mocap_frame_cleaned.keys()) == len(self.mocap_metadata.metadata.keys()):
                mocap_data_cleaned.append(mocap_frame_cleaned)

        return mocap_data_cleaned

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

            # Retrieve and store timestamps for the entire dataset
            if item_type == "TimeStamp":
                timestamps = zeroed_timestamps[start_time_index:] - zeroed_timestamps[start_time_index]
                motiondata.SampleDurations = timestamps

            # Store orientation task data
            elif item_type == "SO3Task":
                # Assumes wxyz format
                quaternions = [utils.normalize_quaternion(quat) for quat in np.squeeze(mocap_data_cleaned['node' + str(item['node_number'])]['orientation']['data'][start_time_index:])]
                norms = [np.linalg.norm(quat) for quat in quaternions]

                # Check if norms contain any zeros
                if any(norm == 0 for norm in norms):
                    raise ValueError("One or more quaternions have zero norm and cannot be normalized")

                task = motion_data.SO3Task(name=key, orientations=quaternions)
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
            elif item_type == "R3Task":

                # Take the vertical force from the FT sensor
                forces = np.squeeze(mocap_data_cleaned['shoe' + str(item['node_number'])]['FT']['data'][start_time_index:])[:,2]

                task = motion_data.R3Task(name=key, forces=forces)
                motiondata.R3Tasks.append(asdict(task))

        return motiondata
