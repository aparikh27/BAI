import json
import sys
import threading
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
        # Guards ``_cache`` between the detection thread (writer) and the
        # Executor agent's servo loop (reader).
        self._lock = threading.RLock()

    def update(self, frame: FrameDetection):
        """Replace the visible-object set atomically.

        This runs on the detection thread while the Executor agent polls
        ``get_visible_objects`` from another thread. Clearing visibility
        in-place first — with a synchronous DB write per object — left a window
        in which readers saw *nothing* visible, even though the target was in
        frame the whole time. The Executor's servo loop treats an empty result
        as "target lost" and aborts, so that window surfaced as spurious
        "lost alignment or out of range" failures.

        The new state is therefore built off to the side and swapped in under a
        lock, and persistence happens outside the critical section.
        """
        detections = None
        if hasattr(frame, "detections"):
            detections = frame.detections
        elif isinstance(frame, list):
            detections = frame
        else:
            return

        new_cache: dict[str, WorldObject] = {}
        for det in detections:
            if det.track_id is None:
                continue

            new_cache[str(det.track_id)] = WorldObject(
                class_name=det.class_name,
                box=det.box,
                last_seen_frame=det.frame_index,
                visible=True,
            )

        with self._lock:
            # Objects seen previously but not in this frame stay in memory,
            # flagged as no longer visible.
            for track_id, obj in self._cache.items():
                if track_id not in new_cache:
                    obj.visible = False
                    new_cache[track_id] = obj
            self._cache = new_cache
            snapshot = list(new_cache.items())

        for track_id, obj in snapshot:
            self._persist(track_id, obj)

    def get_visible_objects(self):
        with self._lock:
            return [obj for obj in self._cache.values() if obj.visible]

    def get_visible_items(self) -> dict[str, WorldObject]:
        """Currently visible objects keyed by track id.

        ``self.memory`` is a ``MemoryManager`` (the durable store), not the
        live dictionary, so callers that need track IDs alongside objects must
        go through here rather than iterating it.
        """
        with self._lock:
            return {track_id: obj for track_id, obj in self._cache.items() if obj.visible}

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