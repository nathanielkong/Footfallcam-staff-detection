"""ByteTrack adapter; track IDs are associations, not guaranteed identities."""
from types import SimpleNamespace
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


class PersonTracker:
    def __init__(self, fps):
        args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1,
                               new_track_thresh=0.3, track_buffer=max(1, round(fps * 0.6)),
                               match_thresh=0.8, fuse_score=False)
        self.tracker = BYTETracker(args)

    def update(self, detections, frame):
        return self.tracker.update(Boxes(detections, frame.shape[:2]), img=frame)
