"""
Entry point. Run:
    python main.py --video path/to/clip.mp4 --meters-per-pixel 0.004 --bay bay-1

Reads a video, runs detection+tracking+pictogram/OCR+behavior+risk, writes:
  - events.json          (the contract file backend-assistant reads)
  - annotated_output.webm (boxes + risk labels drawn in, for the demo video)
  - clips/<event_id>.webm (short evidence clip per flagged event)

Output format is WebM/VP8, not MP4/H.264 — this OpenCV build has no real H.264
encoder (missing openh264 DLL; every "mp4v" fourcc silently falls back to old
MPEG-4 Part 2), which Chrome's <video> tag cannot decode. Confirmed by testing
in an actual browser: mp4v output loaded (network 200/206) but never rendered
a frame — a false "it works" if you only check the HTTP layer, which is
exactly the mistake that shipped once before this got caught. WebM/VP8 has no
such licensing gap and is verified to actually decode and paint real pixel
content in Chrome (checked via canvas pixel sampling, not just network status).

meters-per-pixel: calibrate once by measuring one known carton's real height
in metres divided by its height in pixels in a representative frame from your
camera. Rough is fine for a hackathon demo — say so on the slide.
"""
import argparse
import json
import os

# Must be set before numpy/torch/easyocr import — uncapped BLAS thread pools
# can fail to allocate under real-world low-free-RAM conditions (confirmed:
# "OpenBLAS error: Memory allocation still failed after 10 retries").
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from collections import deque

import cv2

from detection_tracking import DetectorTracker
from pictogram_ocr import PictogramReader, DEFAULT_POLICY
from behavior_rules import BehaviorAnalyzer
from depth_risk import estimate_impact
from risk_engine import build_event

PRE_EVENT_FRAMES = 30   # ~1s before the event, at 30fps
POST_EVENT_FRAMES = 30  # ~1s after


class ClipRecorder:
    """Buffers recent frames so a short before/after clip can be saved once an
    event fires — this is what backs the dashboard's incident-replay feature."""

    def __init__(self, fps, frame_size):
        self.fps = fps
        self.frame_size = frame_size
        self.pre_buffer = deque(maxlen=PRE_EVENT_FRAMES)
        self.pending = []  # list of {"clip_path": str, "frames": [...], "remaining": int}

    def observe_frame(self, frame):
        self.pre_buffer.append(frame.copy())
        for job in self.pending:
            job["frames"].append(frame.copy())
            job["remaining"] -= 1

    def start_clip(self, clip_path):
        self.pending.append({
            "clip_path": clip_path,
            "frames": list(self.pre_buffer),
            "remaining": POST_EVENT_FRAMES,
        })

    def flush_ready(self):
        still_pending = []
        for job in self.pending:
            if job["remaining"] <= 0:
                self._write(job)
            else:
                still_pending.append(job)
        self.pending = still_pending

    def flush_all(self):
        for job in self.pending:
            self._write(job)
        self.pending = []

    def _write(self, job):
        writer = cv2.VideoWriter(job["clip_path"], cv2.VideoWriter_fourcc(*"VP80"), self.fps, self.frame_size)
        for f in job["frames"]:
            writer.write(f)
        writer.release()


def rescale_bbox(bbox, scale):
    x1, y1, x2, y2 = bbox
    return (x1 / scale, y1 / scale, x2 / scale, y2 / scale)


def rescale_detections(boxes, people, scale):
    """Detection runs on a downscaled frame for speed (CPU cost scales with
    pixel count — Canny/contours especially). Everything downstream (OCR
    crops, drawing, clip recording, behavior thresholds) needs full-res
    coordinates, so rescale immediately after detection, once."""
    if scale == 1.0:
        return boxes, people
    for b in boxes:
        b.bbox = rescale_bbox(b.bbox, scale)
        b.centroid = (b.centroid[0] / scale, b.centroid[1] / scale)
    for p in people:
        p.bbox = rescale_bbox(p.bbox, scale)
    return boxes, people


def crop(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def draw_event(frame, bbox, label, risk_level):
    color = {"low": (0, 200, 0), "medium": (0, 200, 255), "high": (0, 100, 255), "critical": (0, 0, 255)}[risk_level]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--bay", default="bay-1")
    parser.add_argument("--box-model", default=None, help="path to fine-tuned carton/pallet YOLO weights, omit to use contour fallback")
    parser.add_argument("--meters-per-pixel", type=float, default=0.004)
    parser.add_argument("--package-mass-kg", type=float, default=5.0)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--downscale-width", type=int, default=0,
                         help="run detection on frames resized to this width for speed. UNSAFE with the contour-fallback box detector — "
                              "tested, confirmed it breaks track continuity (loses edge detail the Canny/contour heuristic needs, "
                              "causes track-ID resets that wipe out behavior history) and produced 0 events on real footage that "
                              "correctly produced events at full resolution. Safe to enable once a real fine-tuned --box-model is used instead.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clips_dir = os.path.join(args.out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    detector = DetectorTracker(box_model_path=args.box_model)
    pictogram_reader = PictogramReader()
    behavior_analyzer = BehaviorAnalyzer()
    policies = {}  # track_id -> policy dict, read once per package then reused

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        os.path.join(args.out_dir, "annotated_output.webm"),
        cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height),
    )
    clip_recorder = ClipRecorder(fps, (width, height))

    detect_scale = 1.0
    if args.downscale_width and width > args.downscale_width:
        detect_scale = args.downscale_width / width

    events = []
    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        clip_recorder.observe_frame(frame)
        detect_frame = frame
        if detect_scale != 1.0:
            detect_frame = cv2.resize(frame, (int(width * detect_scale), int(height * detect_scale)))
        boxes, people = detector.process_frame(detect_frame, frame_number)
        boxes, people = rescale_detections(boxes, people, detect_scale)

        for box in boxes:
            if box.track_id not in policies:
                package_crop = crop(frame, box.bbox)
                policy = pictogram_reader.read(package_crop).to_dict() if package_crop is not None else dict(DEFAULT_POLICY)
                policies[box.track_id] = policy

        behavior_events = behavior_analyzer.update(boxes, people, frame_height=height)

        box_lookup = {b.track_id: b for b in boxes}
        for bevent in behavior_events:
            box = box_lookup.get(bevent.track_id)
            if box is None:
                continue
            policy = policies.get(bevent.track_id, dict(DEFAULT_POLICY))

            impact = None
            if bevent.pixel_drop:
                impact = estimate_impact(bevent.pixel_drop, args.meters_per_pixel, args.package_mass_kg)

            clip_path = os.path.join(clips_dir, f"track{bevent.track_id}_{bevent.behavior}_{frame_number}.webm")
            event = build_event(
                video_source=args.video,
                bay=args.bay,
                track_id=bevent.track_id,
                behavior=bevent.behavior,
                policy=policy,
                frame_number=frame_number,
                drop_height_m=impact.drop_height_m if impact else None,
                impact_velocity_mps=impact.impact_velocity_mps if impact else None,
                impact_energy_j=impact.impact_energy_j if impact else None,
                clip_path=clip_path,
            )
            events.append(event)
            clip_recorder.start_clip(clip_path)
            draw_event(frame, box.bbox, f"{bevent.behavior} [{event['risk_level']}]", event["risk_level"])

        clip_recorder.flush_ready()
        writer.write(frame)
        frame_number += 1

    clip_recorder.flush_all()
    cap.release()
    writer.release()

    events_path = os.path.join(args.out_dir, "events.json")
    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"Wrote {len(events)} events to {events_path}")
    print(f"Annotated video: {os.path.join(args.out_dir, 'annotated_output.webm')}")


if __name__ == "__main__":
    main()
