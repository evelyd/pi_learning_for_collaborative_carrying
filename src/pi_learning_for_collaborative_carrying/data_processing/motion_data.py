# SPDX-FileCopyrightText: Fondazione Istituto Italiano di Tecnologia
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Dict
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

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
class SO3Task:
    """Class for storing a sequence of orientations of an orientation task."""

    name: str
    orientations: List[float]
    angular_velocities: List[float]

@dataclass
class R3Task:
    """Class for storing a sequence of forces of a floor contact task."""

    name: str
    forces: List[float]

@dataclass
class MotionData:
    """Class for the intermediate format into which different kinds of MoCap data are converted
    before retargeting. The format includes task data associated with timestamps.
    """

    SO3Tasks: List[dict] = field(default_factory=list)
    R3Tasks: List[dict] = field(default_factory=list)
    GravityTasks: List[dict] = field(default_factory=list)
    SampleDurations: List[float] = field(default_factory=list)

    @staticmethod
    def build() -> "MotionData":
        """Build an empty MotionData."""

        return MotionData()


@dataclass
class MocapMetadata:
    """Class for the meta information about the collected MoCap data, such as the joints
    and links considered in the data collection as well as the root link of the model.
    """

    start_time: float = 0.0
    root_link: str = ""
    metadata: Dict = field(default_factory=dict)

    @staticmethod
    def build(start_time: float) -> "MocapMetadata":
        """Build an empty MocapMetadata."""

        return MocapMetadata(start_time=start_time)

    def add_timestamp(self) -> None:
        """Indicate that the data samples are associated with timestamps."""

        self.metadata['timestamp'] = {'type': 'TimeStamp'}

    def add_task(self,
                 task_name: str,
                 task_type: str,
                 node_number: int) -> None:
        """Describe the task."""

        self.metadata[task_name] = {
            'type': task_type,
            'node_number': node_number
        }

    def has_entry(self, task_name: str) -> bool:
        """Check if there is a task in the metadata with the given name."""
        return task_name in self.metadata
