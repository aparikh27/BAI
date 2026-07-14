import inspect
import threading
import time

import cv2

from backend.features.camera import CameraService
from backend.image_detection.yolo_detector import YOLODetector, FrameDetection
from backend.features.memory import World, WorldObject


class DetectorService:

    def __init__(self, webots_driver=None):
        """
        Initialize detection service.
        
        :param webots_driver: Optional WebotDriver instance to use robot camera instead of local webcam.
        """
        self.detector = YOLODetector()
        self.camera = CameraService(webots_driver=webots_driver)
        self.world = World()
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None

    def start(self, source, confidence):

        with self.lock:
            if self.running:
                return False

            self.running = True
            self.latest_frame = None
            self.camera.start(source)       

            self.thread = threading.Thread(
                target=self.run_detection,
                args=(source, confidence),
                daemon=True
            )

            self.thread.start()
            return True

    def run_detection(self, source, confidence):
        try:
            frame_index = 0
            last_logged_ids = set()  # Keeps track of what we saw last time to prevent spam

            while self.running:
                import time
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                frame_index += 1
                
                # 1. Process the frame to get detections and annotated frame
                detections, annotated_frame = self.detector.process_frame(frame, confidence, frame_index)
                self._store_latest_frame(annotated_frame)

                # 2. Wrap detections into a FrameDetection and update world memory
                frame_bundle = FrameDetection(frame_index=frame_index, detections=detections)
                self.world.update(frame_bundle)

                # 3. SMART LOGGING: Only print if the items in the room change
                visible_objects = self.world.get_visible_objects()
                
                # Extract current active track IDs from the world memory dictionary keys
                current_ids = {track_id for track_id, obj in self.world.memory.items() if obj.visible}

                if current_ids != last_logged_ids:
                    print("\n--- [WORLD] CURRENT WORLD STATE ---")
                    if not visible_objects:
                        print("[World is empty]")
                    for track_id, obj in self.world.memory.items():
                        if obj.visible:
                            print(f" -> [ID {track_id}] {obj.class_name} | Box: {[round(x, 1) for x in obj.box]}")
                    print("-------------------------------\n")
                    time.sleep(0.1)
                    last_logged_ids = current_ids

        finally:
            self.running = False
            self.camera.stop()
            self.latest_frame = None

    def _store_latest_frame(self, frame):
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if success:
            with self.lock:
                self.latest_frame = encoded.tobytes()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        with self.lock:
            if not self.running:
                return True
            self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        return True
