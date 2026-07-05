import cv2


class CameraService:
    def __init__(self):
        self.cap = None

    def start(self, source=0):
        if self.cap is not None and self.cap.isOpened():
            return self.cap

        self.cap = cv2.VideoCapture(source)
        return self.cap

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None, None

        return self.cap.read()
