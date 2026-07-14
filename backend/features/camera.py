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
                # Step the simulation to advance sensor data and get fresh camera frame
                time_step = self.webots_driver.time_step
                if self.webots_driver.robot.step(time_step) == -1:
                    return None, None
                
                # Get raw image bytes from Webots camera
                raw_image = self.webots_driver.camera.getImage()
                if raw_image is None or len(raw_image) == 0:
                    return None, None
                
                # Convert Webots raw bytes to BGR numpy array for OpenCV/YOLO
                width = self.webots_driver.get_camera_width()
                height = self.webots_driver.camera.getHeight()
                
                # Webots returns RGBA format; convert to BGR for OpenCV compatibility
                image_array = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
                # Drop alpha channel and convert RGBA to BGR
                frame = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                
                return True, frame
            except (ValueError, RuntimeError) as e:
                print(f"[CameraService] Webots camera error: {e}")
                return None, None
        
        # Fallback to local OpenCV camera
        if self.cap is None or not self.cap.isOpened():
            return None, None

        return self.cap.read()
