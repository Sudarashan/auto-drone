"""
Lightweight ByteTrack-style tracker for mock and CPU-only mission testing.

This intentionally keeps dependencies minimal while preserving the important
behavior we need right now: persistent IDs, low-confidence association, and
simple motion continuity across short gaps.
"""

from dataclasses import dataclass
from typing import List, Sequence

try:
    from data_types import Detection, TrackedObject
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "sim_interfaces" / "airsim_zenoh_bridge"))
    from data_types import Detection, TrackedObject

from .base import Tracker, TrackerConfig


def _bbox_xyxy(det: Detection) -> tuple[float, float, float, float]:
    return (
        float(det.bbox_x),
        float(det.bbox_y),
        float(det.bbox_x + det.bbox_w),
        float(det.bbox_y + det.bbox_h),
    )


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0.0 else 0.0


@dataclass
class _TrackState:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    bearing_x: float
    bearing_y: float
    age: int
    hits: int
    time_since_seen: int
    last_timestamp_s: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class SimpleByteTrack(Tracker):
    """
    Minimal ByteTrack-style tracker.

    Matching order:
      1. High-confidence detections with active tracks
      2. Remaining active tracks with low-confidence detections
      3. New tracks for unmatched high-confidence detections
    """

    def __init__(self, config: TrackerConfig | None = None):
        super().__init__(config=config)
        self._next_track_id = 1
        self._tracks: List[_TrackState] = []

    def update(
        self,
        detections: Sequence[Detection],
        image_width: int,
        image_height: int,
        timestamp_s: float,
    ) -> List[TrackedObject]:
        for track in self._tracks:
            track.time_since_seen += 1

        high_conf = [d for d in detections if d.confidence >= self.config.high_thresh]
        low_conf = [
            d for d in detections
            if self.config.low_thresh <= d.confidence < self.config.high_thresh
        ]

        matched_high = self._associate(high_conf, timestamp_s)
        unmatched_track_ids = {track.track_id for track in self._tracks if track.time_since_seen > 0}
        if unmatched_track_ids and low_conf:
            self._associate(low_conf, timestamp_s, allowed_track_ids=unmatched_track_ids)

        matched_detection_ids = matched_high
        for det in high_conf:
            if id(det) not in matched_detection_ids:
                self._start_new_track(det, timestamp_s)

        visible_tracks: List[TrackedObject] = []
        retained_tracks: List[_TrackState] = []
        for track in self._tracks:
            if track.time_since_seen <= self.config.max_time_lost:
                retained_tracks.append(track)
            if track.time_since_seen == 0 and track.hits >= self.config.min_hits:
                visible_tracks.append(self._to_tracked_object(track, image_width, image_height))
            track.age += 1
        self._tracks = retained_tracks
        return visible_tracks

    def _associate(
        self,
        detections: Sequence[Detection],
        timestamp_s: float,
        allowed_track_ids: set[int] | None = None,
    ) -> set[int]:
        matched_detections: set[int] = set()
        candidate_pairs: List[tuple[float, int, Detection]] = []

        for track in self._tracks:
            if allowed_track_ids is not None and track.track_id not in allowed_track_ids:
                continue
            if track.time_since_seen > self.config.max_time_lost:
                continue
            for det in detections:
                if det.class_name.lower() != track.class_name.lower():
                    continue
                iou = _bbox_iou(track.bbox, _bbox_xyxy(det))
                if iou >= self.config.match_iou_thresh:
                    candidate_pairs.append((iou, track.track_id, det))

        used_tracks: set[int] = set()
        for _, track_id, det in sorted(candidate_pairs, key=lambda item: item[0], reverse=True):
            if track_id in used_tracks or id(det) in matched_detections:
                continue
            track = self._get_track(track_id)
            if track is None:
                continue
            self._update_track(track, det, timestamp_s)
            used_tracks.add(track_id)
            matched_detections.add(id(det))

        return matched_detections

    def _start_new_track(self, det: Detection, timestamp_s: float) -> None:
        self._tracks.append(
            _TrackState(
                track_id=self._next_track_id,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=_bbox_xyxy(det),
                bearing_x=det.bearing_x,
                bearing_y=det.bearing_y,
                age=1,
                hits=1,
                time_since_seen=0,
                last_timestamp_s=timestamp_s,
            )
        )
        self._next_track_id += 1

    def _update_track(self, track: _TrackState, det: Detection, timestamp_s: float) -> None:
        prev_cx = (track.bbox[0] + track.bbox[2]) * 0.5
        prev_cy = (track.bbox[1] + track.bbox[3]) * 0.5
        dt = max(timestamp_s - track.last_timestamp_s, 1e-6)
        new_bbox = _bbox_xyxy(det)
        new_cx = (new_bbox[0] + new_bbox[2]) * 0.5
        new_cy = (new_bbox[1] + new_bbox[3]) * 0.5

        track.velocity_x = (new_cx - prev_cx) / dt
        track.velocity_y = (new_cy - prev_cy) / dt
        track.confidence = det.confidence
        track.bbox = new_bbox
        track.bearing_x = det.bearing_x
        track.bearing_y = det.bearing_y
        track.hits += 1
        track.time_since_seen = 0
        track.last_timestamp_s = timestamp_s

    def _get_track(self, track_id: int) -> _TrackState | None:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def _to_tracked_object(
        self,
        track: _TrackState,
        image_width: int,
        image_height: int,
    ) -> TrackedObject:
        x1, y1, x2, y2 = track.bbox
        bbox_x = int(round(x1))
        bbox_y = int(round(y1))
        bbox_w = int(round(x2 - x1))
        bbox_h = int(round(y2 - y1))
        center_x = bbox_x + bbox_w // 2
        center_y = bbox_y + bbox_h // 2
        bearing_x = track.bearing_x
        bearing_y = track.bearing_y
        if image_width > 0 and image_height > 0:
            bearing_x = (center_x - image_width / 2) / (image_width / 2)
            bearing_y = (center_y - image_height / 2) / (image_height / 2)
        return TrackedObject(
            track_id=track.track_id,
            class_id=track.class_id,
            class_name=track.class_name,
            confidence=track.confidence,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            center_x=center_x,
            center_y=center_y,
            area=max(bbox_w * bbox_h, 0),
            bearing_x=bearing_x,
            bearing_y=bearing_y,
            age=track.hits,
            time_since_seen=track.time_since_seen,
            velocity_estimate_x=track.velocity_x,
            velocity_estimate_y=track.velocity_y,
        )
