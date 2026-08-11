import inspect
import threading
import time

import cv2

from backend.features.camera import CameraService
from backend.image_detection.color_detector import ColorObjectDetector
from backend.image_detection.yolo_detector import YOLODetector, FrameDetection
from backend.features.memory import World, WorldObject


class DetectorService:

    def __init__(self, webots_driver=None, world: World | None = None):
        """
        Initialize detection service.

        :param webots_driver: Optional WebotDriver instance to use robot camera instead of local webcam.
        :param world: Shared ``World`` instance (same object the Executor agent reads).
        """
        self.detector = YOLODetector()
        # COCO has no class for the plain coloured Solids used to prop the
        # Webots scene, so colour segmentation runs alongside YOLO and feeds
        # the same World memory.
        self.color_detector = ColorObjectDetector()
        self.camera = CameraService(webots_driver=webots_driver)
        self.world = world if world is not None else World()
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None

    # A detection pass can block on the Webots step lock while the Executor
    # agent drives a motion command, so shutting down needs to outlast a
    # motion step rather than assume the loop exits immediately.
    SHUTDOWN_TIMEOUT = 10.0

    def start(self, source, confidence):

        with self.lock:
            if self.running:
                return False

            previous = self.thread

        # Never run two detection loops at once. Both would step the simulator
        # and call ``world.update()`` from different frames, so tracked objects
        # would flicker in and out of the world model and the Executor's servo
        # loop would keep losing its target.
        if previous is not None and previous.is_alive():
            previous.join(timeout=self.SHUTDOWN_TIMEOUT)
            if previous.is_alive():
                print(
                    "[DetectorService] Previous detection thread is still running; "
                    "refusing to start a second one."
                )
                return False

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

                # 1b. Add colour-segmented props (e.g. the orange cylinder) that
                # the COCO vocabulary cannot express, and draw them on top.
                color_detections = self.color_detector.process_frame(frame, frame_index)
                if color_detections:
                    detections = list(detections) + color_detections
                    annotated_frame = self.color_detector.annotate(
                        annotated_frame, color_detections
                    )

                self._store_latest_frame(annotated_frame)

                # 2. Wrap detections into a FrameDetection and update world memory
                frame_bundle = FrameDetection(frame_index=frame_index, detections=detections)
                self.world.update(frame_bundle)

                # 3. SMART LOGGING: Only print if the items in the room change
                visible_items = self.world.get_visible_items()
                current_ids = set(visible_items)

                if current_ids != last_logged_ids:
                    print("\n--- [WORLD] CURRENT WORLD STATE ---")
                    if not visible_items:
                        print("[World is empty]")
                    for track_id, obj in visible_items.items():
                        print(f" -> [ID {track_id}] {obj.class_name} | Box: {[round(x, 1) for x in obj.box]}")
                    print("-------------------------------\n")
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
            thread = self.thread

        if thread and thread.is_alive():
            # Wait long enough to outlast an in-flight simulator step. Returning
            # while the loop is still alive lets a subsequent start() race a
            # second detection thread against it.
            thread.join(timeout=self.SHUTDOWN_TIMEOUT)
            if thread.is_alive():
                print(
                    "[DetectorService] Detection thread did not stop within "
                    f"{self.SHUTDOWN_TIMEOUT}s."
                )
                return False

        with self.lock:
            self.thread = None

        return True
