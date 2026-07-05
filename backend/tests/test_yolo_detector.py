import unittest

from backend.image_detection.yolo_detector import Detection, YOLODetector


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBox:
    def __init__(self):
        self.xyxy = [[FakeTensor([1.0, 2.0, 3.0, 4.0])]]
        self.conf = [0.95]
        self.cls = [0]


class FakeResult:
    def __init__(self):
        self.boxes = [FakeBox()]
        self.names = {0: "person"}


class YOLODetectorTests(unittest.TestCase):
    def test_build_detections_creates_custom_objects(self):
        detector = YOLODetector.__new__(YOLODetector)
        detections = detector._build_detections(2, FakeResult())

        self.assertEqual(len(detections), 1)
        self.assertIsInstance(detections[0], Detection)
        self.assertEqual(detections[0].class_name, "person")
        self.assertEqual(detections[0].frame_index, 2)
        self.assertIn("person", detections[0].verbose())


if __name__ == "__main__":
    unittest.main()
