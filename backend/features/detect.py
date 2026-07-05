import inspect
import threading
import time

import cv2

from backend.features.camera import CameraService
from backend.image_detection.yolo_detector import YOLODetector


class DetectorService:

    def __init__(self):
        self.detector = YOLODetector()
        self.camera = CameraService()
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
            while self.running:
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                frame_index += 1
                detections, annotated_frame = self.detector.process_frame(frame, confidence, frame_index)
                self._store_latest_frame(annotated_frame)

                for detection in detections:
                    if not self.running:
                        break
                    print(detection.verbose())
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
