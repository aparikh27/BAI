from backend.image_detection.detector import ImageDetector
from ultralytics import YOLO

# documentation: https://docs.ultralytics.com/modes/predict#fixed-shape-vs-minimum-rectangle-rect

class YOLODetector(ImageDetector):
    def __init__(self):
        super().__init__("yolo11n.pt")
        self.model = YOLO("yolo11n.pt")

    def detect(self, source, confidence):

        return self.model.predict(
            source=source,
            stream=True,
            show=True,
            save=True,
            conf=confidence,
        )