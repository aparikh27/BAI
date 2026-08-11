"""
color_detector.py — HSV colour-blob perception for untrained props
──────────────────────────────────────────────────────────────────
``yolo11n.pt`` ships the 80-class COCO vocabulary, which has no concept of the
plain coloured ``Solid`` primitives used to prop a Webots scene.  The orange
cylinder in ``khepera3_gripper.wbt`` (``baseColor 0.976 0.463 0.169``) is
therefore invisible to the YOLO path, and the Executor — which resolves targets
by exact ``class_name`` match — can never satisfy a plan step such as
``{"action": "detect_object", "target": "orange cylinder"}``.

This detector closes that gap with classical CV: threshold the frame in HSV,
keep contours large enough to be the prop, and emit ordinary ``Detection``
records so the rest of the pipeline (World memory, Executor servoing, the
dashboard) needs no special-casing.

Track IDs are allocated from a high, dedicated base so they can never collide
with ByteTrack IDs coming out of the YOLO tracker.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.image_detection.detector import Detection

# ByteTrack allocates small positive integers; start well clear of them.
COLOR_TRACK_ID_BASE = 9000


@dataclass(frozen=True)
class ColorProfile:
    """A named HSV band describing one coloured prop.

    ``hue_low``/``hue_high`` use OpenCV's 0-179 hue scale, not 0-359.
    """

    class_name: str
    class_id: int
    hue_low: int
    hue_high: int
    sat_min: int = 120
    val_min: int = 90
    min_area: int = 60
    # Colour alone cannot separate the cylinder from Webots' wooden boxes and
    # the reddish arena floor — measured on the live feed, the box renders at
    # S≈198/V≈213 against the cylinder's S≈211/V≈249. Geometry does separate
    # them: the prop is a small, tall, narrow blob; boxes and floor are large
    # and wide.
    # Kept loose: when the prop is clipped by a frame edge its apparent aspect
    # collapses well below the true 4:1, and a threshold tight enough to reject
    # the floor would make detection flicker in and out mid-approach.
    min_aspect: float = 0.9       # height / width
    max_aspect: float = 12.0
    max_area_ratio: float = 0.35  # fraction of the frame

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        lower = np.array([self.hue_low, self.sat_min, self.val_min], dtype=np.uint8)
        upper = np.array([self.hue_high, 255, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)


# The Webots prop renders around H≈11, S≈211, V≈249 on OpenCV's scale.  The band
# is deliberately wide: simulator lighting shades the cylinder considerably
# between the lit face and the shadowed side.
ORANGE_CYLINDER = ColorProfile(
    class_name="orange cylinder",
    class_id=900,
    hue_low=5,
    hue_high=22,
    sat_min=150,
    val_min=120,
    min_area=150,
)


class ColorObjectDetector:
    """Detects coloured props by HSV segmentation.

    Kept deliberately separate from ``YOLODetector`` so the two perception
    sources can be merged without either knowing about the other.
    """

    def __init__(self, profiles: list[ColorProfile] | None = None):
        self.profiles = profiles if profiles is not None else [ORANGE_CYLINDER]
        # 5x5 kernel closes the speckle left by the cylinder's specular highlight.
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process_frame(self, frame, frame_index: int) -> list[Detection]:
        """Returns ``Detection`` records for every colour profile match."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: list[Detection] = []

        for profile_index, profile in enumerate(self.profiles):
            mask = profile.mask(hsv)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Largest blob first, so the dominant prop keeps a stable track ID
            # frame to frame without a full tracker.
            blobs = sorted(
                (c for c in contours if cv2.contourArea(c) >= profile.min_area),
                key=cv2.contourArea,
                reverse=True,
            )

            frame_area = float(frame.shape[0] * frame.shape[1])
            rank = 0
            for contour in blobs:
                x, y, w, h = cv2.boundingRect(contour)
                area = float(cv2.contourArea(contour))

                # Reject anything the wrong shape or too large to be the prop.
                aspect = h / float(max(w, 1))
                if not (profile.min_aspect <= aspect <= profile.max_aspect):
                    continue
                if area / frame_area > profile.max_area_ratio:
                    continue

                # Fill ratio doubles as a confidence proxy: a solid cylinder
                # fills most of its bounding box, stray highlights do not.
                confidence = min(1.0, area / float(max(w * h, 1)))

                detections.append(
                    Detection(
                        class_id=profile.class_id,
                        class_name=profile.class_name,
                        confidence=round(confidence, 3),
                        box=(float(x), float(y), float(x + w), float(y + h)),
                        frame_index=frame_index,
                        track_id=COLOR_TRACK_ID_BASE + profile_index * 100 + rank,
                    )
                )
                rank += 1

        return detections

    def annotate(self, frame, detections: list[Detection]):
        """Draws colour detections onto an already-annotated YOLO frame."""
        if frame is None:
            return frame

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            label = f"{det.class_name} {det.confidence:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            # Keep the caption on-screen when the prop sits at the top edge.
            text_top = max(y1 - text_h - baseline, 0)
            cv2.rectangle(
                frame,
                (x1, text_top),
                (x1 + text_w, text_top + text_h + baseline),
                (0, 140, 255),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (x1, text_top + text_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return frame
