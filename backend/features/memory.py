import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.image_detection.yolo_detector import FrameDetection

memory_engine_path = Path(__file__).resolve().parents[2] / "MemoryEngine"
if str(memory_engine_path) not in sys.path:
    sys.path.insert(0, str(memory_engine_path))

from MemoryEngine.memory import MemoryItem
from MemoryEngine.memory_manager import MemoryManager


@dataclass
class WorldObject:
    class_name: str
    box: tuple[float, float, float, float]
    last_seen_frame: int
    visible: bool


class World:
    def __init__(self, capacity: int = 100, db_path: str | None = None):
        self.memory = MemoryManager(capacity=capacity, db_path=db_path or "robot_memory.db")
        self._cache: dict[str, WorldObject] = {}

    def update(self, frame: FrameDetection):
        for track_id in list(self._cache.keys()):
            obj = self._cache[track_id]
            obj.visible = False
            self._persist(track_id, obj)

        detections = None
        if hasattr(frame, "detections"):
            detections = frame.detections
        elif isinstance(frame, list):
            detections = frame
        else:
            return

        for det in detections:
            if det.track_id is None:
                continue

            track_id = str(det.track_id)
            obj = WorldObject(
                class_name=det.class_name,
                box=det.box,
                last_seen_frame=det.frame_index,
                visible=True,
            )
            self._cache[track_id] = obj
            self._persist(track_id, obj)

    def get_visible_objects(self):
        return [obj for obj in self._cache.values() if obj.visible]

    def get_object_by_track_id(self, track_id: int):
        key = str(track_id)
        if key in self._cache:
            return self._cache[key]

        item = self.memory.get(key)
        if item is None:
            return None

        return WorldObject(**json.loads(item.value))

    def _persist(self, key: str, obj: WorldObject) -> None:
        payload = json.dumps(asdict(obj))
        self.memory.add(MemoryItem(key=key, value=payload, timestamp=time.time()))


__all__ = ["MemoryManager", "MemoryItem", "World", "WorldObject"]