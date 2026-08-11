import cv2
import numpy as np


class CameraService:
    def __init__(self, webots_driver=None):
        """
        Initialize camera service.
        
        :param webots_driver: Optional WebotDriver instance. If provided, uses robot camera.
                             Otherwise uses OpenCV VideoCapture (local/default camera).
        """
        self.cap = None
        self.webots_driver = webots_driver
        self.use_webots = webots_driver is not None

    def start(self, source=0):
        """Start the camera. If webots_driver was provided, this initializes the Webots camera."""
        if self.use_webots and self.webots_driver:
            # Already have access to robot camera via webots_driver
            print("[CameraService] Using Webots robot camera (not local webcam)")
            return None  # webots_driver.camera handles everything
        
        if self.cap is not None and self.cap.isOpened():
            return self.cap

        print(f"[CameraService] Using local OpenCV camera source: {source}")
        self.cap = cv2.VideoCapture(source)
        return self.cap

    def stop(self):
        if self.use_webots:
            # Webots camera cleanup is handled by the robot controller
            return
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self):
        """Read a frame from either Webots robot camera or local OpenCV camera."""
        if self.use_webots and self.webots_driver and self.webots_driver.camera:
            try:
                # Step the simulation and grab the frame atomically — the
                # Executor agent drives the same Robot handle from another
                # thread, and the Webots API is not thread-safe.
                raw_image, width, height = self.webots_driver.read_camera_image()
                if raw_image is None or len(raw_image) == 0:
                    return None, None

                if width <= 0 or height <= 0:
                    return None, None

                expected = width * height * 4
                if len(raw_image) < expected:
                    # A short buffer means the frame was torn mid-read; skip it
                    # rather than raising out of the detection loop.
                    return None, None

                # Webots' Camera.getImage() returns BGRA — "a sequence of four
                # bytes representing the blue, green, red and alpha levels of a
                # pixel" — NOT RGBA. Converting as RGBA swaps the red and blue
                # channels, which silently inverts every colour in the frame:
                # tan wood renders cyan and the orange cylinder renders blue,
                # so no colour-based detection can ever match.
                image_array = np.frombuffer(
                    raw_image[:expected], dtype=np.uint8
                ).reshape((height, width, 4))
                frame = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)

                return True, frame
            except (ValueError, RuntimeError) as e:
                print(f"[CameraService] Webots camera error: {e}")
                return None, None
        
        # Fallback to local OpenCV camera
        if self.cap is None or not self.cap.isOpened():
            return None, None

        return self.cap.read()
