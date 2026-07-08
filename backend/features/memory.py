from backend.image_detection.yolo_detector import FrameDetection, Detection
from dataclasses import dataclass

@dataclass
class WorldObject:
    class_name: str
    box: tuple[float, float, float, float]
    last_seen_frame: int
    visible: bool



class World:
    def __init__(self):
        self.memory = {}

    def update(self, frame: FrameDetection):
        # Allow either a FrameDetection object or a plain list of Detection
        for obj in self.memory.values():
            obj.visible = False

        detections = None
        if hasattr(frame, "detections"):
            detections = frame.detections
        elif isinstance(frame, list):
            detections = frame
        else:
            # Unknown frame type: nothing to update
            return

        for det in detections:
            if det.track_id is not None:
                self.memory[det.track_id] = WorldObject(
                    class_name=det.class_name,
                    box=det.box,
                    last_seen_frame=det.frame_index,
                    visible=True
                )
    def get_visible_objects(self):
        return [obj for obj in self.memory.values() if obj.visible]
    def get_object_by_track_id(self, track_id: int):
        return self.memory.get(track_id, None)