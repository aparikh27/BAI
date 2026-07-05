from dataclasses import dataclass

import cv2
from ultralytics import YOLO

from backend.image_detection.detector import ImageDetector


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    frame_index: int

    def verbose(self) -> str:
        return (
            f"frame={self.frame_index} class={self.class_name} "
            f"confidence={self.confidence:.2f} box={self.box}"
        )


class YOLODetector(ImageDetector):
    def __init__(self):
        super().__init__("yolo11n.pt")
        self.model = YOLO("yolo11n.pt")

    def process_frame(self, frame, confidence, frame_index):
        results = self.model(frame, conf=confidence, stream=False, verbose=False)
        detections = []
        annotated_frame = frame

        for result in results:
            annotated_frame = result.plot()
            detections.extend(self._build_detections(frame_index, result))

        return detections, annotated_frame

    def detect(self, source, confidence, frame_callback=None):
        video_source = self._normalize_source(source)
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_index += 1
                detections, annotated_frame = self.process_frame(frame, confidence, frame_index)
                for detection in detections:
                    yield detection

                if annotated_frame is not None and frame_callback is not None:
                    frame_callback(annotated_frame)
        finally:
            cap.release()

    def _normalize_source(self, source):
        if isinstance(source, (int, float)):
            return int(source)
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def _build_detections(self, frame_index, result):
        detections = []
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}

        if not boxes:
            return detections

        for box in boxes:
            coords = self._extract_box_coordinates(box)
            if coords is None:
                continue

            cls_value = self._first_value(getattr(box, "cls", None))
            confidence_value = self._first_value(getattr(box, "conf", None))
            cls_id = int(cls_value) if cls_value is not None else -1
            confidence = float(confidence_value) if confidence_value is not None else 0.0
            class_name = names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    box=coords,
                    frame_index=frame_index,
                )
            )

        return detections

    def _extract_box_coordinates(self, box):
        xyxy = getattr(box, "xyxy", None)
        if xyxy is None:
            return None

        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()

        while isinstance(xyxy, (list, tuple)) and len(xyxy) == 1:
            xyxy = xyxy[0]

        if hasattr(xyxy, "tolist"):
            xyxy = xyxy.tolist()

        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            return None

        try:
            return tuple(float(value) for value in xyxy[:4])
        except (TypeError, ValueError):
            return None

    def _first_value(self, value):
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value
