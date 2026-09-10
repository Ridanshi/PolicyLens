"""
Detects and tracks people + cartons/pallets across video frames.

Person detection uses stock YOLOv8 (COCO class 'person', works out of the box).
Carton/pallet detection needs a fine-tuned model since COCO has no such class.
Until that weights file exists, falls back to contour-based rectangle detection
so the rest of the pipeline is runnable from day one.

Fine-tuning data: pull a pallet/box dataset from Roboflow Universe (public,
free), export in YOLO format, train with `yolo detect train` (ultralytics CLI),
drop the resulting best.pt at the path passed as `box_model_path`.
"""
from dataclasses import dataclass, field
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedBox:
    track_id: int
    frame_number: int
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    centroid: tuple  # (cx, cy)


@dataclass
class TrackedPerson:
    track_id: int
    frame_number: int
    bbox: tuple
    keypoints: list = field(default_factory=list)


MIN_CONFIRM_HITS = 5   # a fallback track must be seen this many times before it's trusted
TRACK_EXPIRE_FRAMES = 15  # drop a fallback track's identity if unseen this long, stops stale re-matches


class DetectorTracker:
    def __init__(self, box_model_path: str | None = None, person_model_path: str = "yolov8n.pt"):
        self.person_model = YOLO(person_model_path)
        self.box_model = YOLO(box_model_path) if box_model_path else None

    def process_frame(self, frame: np.ndarray, frame_number: int):
        boxes = self._detect_boxes(frame, frame_number)
        people = self._detect_people(frame, frame_number)
        return boxes, people

    def _detect_people(self, frame: np.ndarray, frame_number: int) -> list[TrackedPerson]:
        results = self.person_model.track(frame, persist=True, classes=[0], verbose=False)
        people = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return people
        for box, track_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
            x1, y1, x2, y2 = box
            people.append(TrackedPerson(int(track_id), frame_number, (float(x1), float(y1), float(x2), float(y2))))
        return people

    def _detect_boxes(self, frame: np.ndarray, frame_number: int) -> list[TrackedBox]:
        if self.box_model is not None:
            return self._detect_boxes_yolo(frame, frame_number)
        return self._detect_boxes_contour_fallback(frame, frame_number)

    def _detect_boxes_yolo(self, frame: np.ndarray, frame_number: int) -> list[TrackedBox]:
        results = self.box_model.track(frame, persist=True, verbose=False)
        boxes = []
        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return boxes
        for box, track_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            boxes.append(TrackedBox(int(track_id), frame_number, (float(x1), float(y1), float(x2), float(y2)), (float(cx), float(cy))))
        return boxes

    _next_fallback_id = 1000
    _fallback_tracks: dict = {}  # tid -> {"centroid", "hits", "last_seen"}

    def _detect_boxes_contour_fallback(self, frame: np.ndarray, frame_number: int) -> list[TrackedBox]:
        """MVP substitute until a fine-tuned box detector exists.

        First attempt: raw Canny-edge contours on the whole frame — caught
        real cartons but also shadows/floor seams/texture, showing up as
        spurious dropped/stepped_on/over_stacked events on real footage
        (documented: 4 false drops + 1 false step + 1 false stack on a clip
        that's actually just dragging).

        Second attempt: background subtraction (MOG2) to isolate moving
        foreground instead of raw edges. Made it WORSE on real footage (12
        events with 6 correctly "dragged" → 19 events with only 1 correctly
        "dragged", noise spiking on wrong_orientation/stepped_on) — because
        Godrej's pilot footage is handheld/moving-camera (confirmed: a
        supposedly-static corner region shifted by up to 61/255 intensity
        across 4 seconds), and MOG2 assumes a static camera. Wrong technique
        for this footage — reverted.

        Current approach: back to Canny, plus two camera-motion-agnostic
        additions that don't depend on a static background assumption:
        1. approxPolyDP rectangularity check — real cartons approximate a
           4-6 sided polygon; irregular texture/shadow edges usually don't.
        2. MIN_CONFIRM_HITS confirmation gate — a track must persist across
           several frames before it's trusted, so a one-frame noise blob
           can't fire a behavior event on its own.
        Still a placeholder for a real fine-tuned detector, not a solved
        problem — expect more tuning against the rest of the pilot videos.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = 0.003 * frame_area  # relative, not absolute pixels — stays correct if the caller downscales the frame
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > 0.5 * frame_area:
                continue
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
            if not (4 <= len(approx) <= 6):
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            if aspect < 0.3 or aspect > 3.5:
                continue
            cx, cy = x + w / 2, y + h / 2
            detections.append(((x, y, x + w, y + h), (cx, cy)))

        # expire tracks not seen recently so old centroids can't falsely re-match
        stale = [tid for tid, t in self._fallback_tracks.items() if frame_number - t["last_seen"] > TRACK_EXPIRE_FRAMES]
        for tid in stale:
            del self._fallback_tracks[tid]

        used_ids = set()
        boxes = []
        for bbox, centroid in detections:
            best_id, best_dist = None, 80.0
            for tid, t in self._fallback_tracks.items():
                if tid in used_ids:
                    continue
                dist = np.hypot(centroid[0] - t["centroid"][0], centroid[1] - t["centroid"][1])
                if dist < best_dist:
                    best_id, best_dist = tid, dist
            if best_id is None:
                best_id = DetectorTracker._next_fallback_id
                DetectorTracker._next_fallback_id += 1
                self._fallback_tracks[best_id] = {"centroid": centroid, "hits": 0, "last_seen": frame_number}
            used_ids.add(best_id)
            track = self._fallback_tracks[best_id]
            track["centroid"] = centroid
            track["last_seen"] = frame_number
            track["hits"] += 1

            if track["hits"] >= MIN_CONFIRM_HITS:
                boxes.append(TrackedBox(best_id, frame_number, bbox, centroid))
        return boxes
