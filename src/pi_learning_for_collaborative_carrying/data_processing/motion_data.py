# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class SampleDurations:
    """Class for storing a sequence of timestamps."""

    timestamps: List[float]


@dataclass
class GravityTask:
    """Class for storing a sequence of orientations of a gravity task."""

    name: str
    orientations: List[float]

@dataclass
class SE3Task:
    """Class for storing a sequence of poses of a pose task."""

    name: str
    positions: List[float]
    orientations: List[float]
    # linear_velocities: List[float]
    # angular_velocities: List[float]

@dataclass
class SO3Task:
    """Class for storing a sequence of orientations of an orientation task."""

    name: str
    orientations: List[float]
    angular_velocities: List[float]

@dataclass
class FloorContactTask:
    """Class for storing a sequence of forces of a floor contact task."""

    name: str
    positions: List[float]
    orientations: List[float]
    forces: List[float]

@dataclass
class MotionData:
    """Class for the intermediate format into which different kinds of MoCap data are converted
    before retargeting. The format includes task data associated with timestamps.
    """
    CalibrationData: Dict = field(default_factory=dict)
    SE3Tasks: List[dict] = field(default_factory=list)
    SO3Tasks: List[dict] = field(default_factory=list)
    FloorContactTasks: List[dict] = field(default_factory=list)
    GravityTasks: List[dict] = field(default_factory=list)
    SampleDurations: List[float] = field(default_factory=list)
    initial_base_pose: List[float] = field(default_factory=lambda: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    @staticmethod
    def build() -> "MotionData":
        """Build an empty MotionData."""

        return MotionData()


@dataclass
class MocapMetadata:
    """Class for the meta information about the collected MoCap data, such as the joints
    and links considered in the data collection as well as the root link of the model.
    """

    start_time: int = 0.0
    end_time: int = 1.0
    metadata: Dict = field(default_factory=dict)

    @staticmethod
    def build(start_time: int = 0, end_time: int = 1.0) -> "MocapMetadata":
        """Build an empty MocapMetadata."""

        return MocapMetadata(start_time=start_time, end_time=end_time)

    def add_timestamp(self) -> None:
        """Indicate that the data samples are associated with timestamps."""

        self.metadata['timestamp'] = {'type': 'TimeStamp'}

    def add_task(self,
                 task_name: str,
                 task_type: str,
                 node_number: int,
                 frame_name: str) -> None:
        """Describe the task."""

        self.metadata[task_name] = {
            'type': task_type,
            'node_number': node_number,
            "frame_name": frame_name
        }
