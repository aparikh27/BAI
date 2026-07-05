import threading
import cv2
from backend.image_detection.yolo_detector import YOLODetector


class DetectorService:

    def __init__(self):
        self.detector = YOLODetector()
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self, source, confidence):

        with self.lock:
            if self.running:
                return False

            self.running = True

            self.thread = threading.Thread(
                target=self.run_detection,
                args=(source, confidence),
                daemon=True
            )

            self.thread.start()
            return True
        
    def run_detection(self, source, confidence):
        try:
            for result in self.detector.detect(
                source=source,
                confidence=confidence,
            ):
                if not self.running:
                    break

                print(result.verbose())
        finally:
            self.running = False
            cv2.destroyAllWindows()


    def stop(self):
        self.running = False

        cv2.destroyAllWindows()
        return True
