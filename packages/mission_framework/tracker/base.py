"""
Tracker interfaces for mission-framework perception.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    from data_types import Detection, TrackedObject
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "sim_interfaces" / "airsim_zenoh_bridge"))
    from data_types import Detection, TrackedObject


@dataclass
class TrackerConfig:
    """Configuration for a swappable multi-object tracker."""
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    match_iou_thresh: float = 0.3
    max_time_lost: int = 20
    min_hits: int = 1


class Tracker(ABC):
    """Abstract tracker interface."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        image_width: int,
        image_height: int,
        timestamp_s: float,
    ) -> List[TrackedObject]:
        """
        Update tracker state from the current frame detections.

        Returns:
            Tracked objects visible in the current frame.
        """

