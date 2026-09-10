"""
Turns raw tracked-box/person trajectories into a behavior label, by geometry
and motion heuristics — deliberately not a trained action classifier, since no
warehouse-damage-behavior dataset exists publicly (checked). This is the
tradeoff documented in the design: heuristics now, swap for a trained model
later if there's time left after the core pipeline works end to end.

Tune the thresholds below against Godrej's own pilot videos before the demo —
they're starting points, not calibrated numbers.
"""
from collections import defaultdict, deque
from dataclasses import dataclass

HISTORY_LEN = 30  # ~1s at 30fps
DROP_VELOCITY_PX_PER_FRAME = 12.0
DRAG_MAX_VELOCITY_PX_PER_FRAME = 15.0  # was 8 — too strict for kd_packets' noisier real motion (observed mean ~9.3 px/frame, up to 41)
DRAG_MIN_FRAMES = 15
RELATIVE_GROUND_BAND = 0.35  # "near ground" = within the bottom 35% of THIS TRACK's own observed vertical range.
# Originally a scene-wide floor line (first fixed frac-of-frame-height, then a global adaptive frac-of-deepest-
# point-seen) — both failed on kd_packets_dragged_heavy_box_on_other_packets.mp4. Root cause, confirmed by direct
# trajectory instrumentation: that clip's real dragged object moves in the y=500-605 band, which sits just above
# BOTH the fixed (612) and the global-adaptive (635) thresholds — a small miss, but a real one, and it's a second,
# independent scene-specific assumption on top of the velocity threshold above (same underlying problem: 7 clips
# shot handheld at different distances/angles, no single global constant fits all of them). Switching to a
# PER-TRACK relative range sidesteps the "where's the floor in this camera framing" guess entirely: an object
# being dragged near a resting surface, by definition, isn't moving far from wherever it already is.
DRAGGED_COOLDOWN_FRAMES = 90  # ~3s — real footage showed one continuous drag re-firing 40-140x on longer clips without this
THROWN_SPEED_PX_PER_FRAME = 35.0   # higher than a drop — a throw is a forceful, fast motion, not a controlled release
THROWN_MIN_HORIZONTAL_PX_PER_FRAME = 18.0  # distinguishes "thrown" (real horizontal travel) from a near-vertical hard "dropped".
# Both raised after real-footage testing: original values (22/10) mistook handheld-camera shake for
# object motion — the pilot footage is not on a fixed camera (confirmed separately), so any frame-to-frame
# centroid shift is partly camera motion, not just object motion. Raising the bar cuts false "thrown" hits,
# but the correct fix is motion compensation (subtract estimated camera motion via optical flow before
# computing object velocity) — flagged as a known limitation, not fully solved by threshold-raising alone.
THROWN_MIN_STREAK = 2  # require it to persist a frame, not fire on a single-frame camera jerk
ASPECT_FLIP_RATIO = 1.6
ORIENTATION_FLIP_MIN_STREAK = 3   # must persist, not a single-frame blip
STACK_OVERLAP_X_FRAC = 0.5
STEPPED_ON_MIN_OVERLAP_FRAC = 0.3  # real overlap area, not just "feet point falls in box's y-range"
STEPPED_ON_MIN_STREAK = 3


@dataclass
class BehaviorEvent:
    track_id: int
    frame_number: int
    behavior: str  # dropped | dragged | over_stacked | wrong_orientation | thrown | stepped_on | none
    pixel_drop: float = 0.0


class BehaviorAnalyzer:
    def __init__(self):
        self._box_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
        self._first_aspect = {}
        self._already_flagged = defaultdict(set)  # track_id -> set of behaviors already emitted this "episode"
        self._orientation_flip_streak = defaultdict(int)
        self._stepped_on_streak = defaultdict(int)
        self._last_dragged_frame = defaultdict(lambda: -DRAGGED_COOLDOWN_FRAMES)
        self._thrown_streak = defaultdict(int)
        self._track_y_range = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})

    def update(self, boxes, people, ground_y_frac: float = 0.85, frame_height: int = 1080) -> list[BehaviorEvent]:
        events = []

        for box in boxes:
            hist = self._box_history[box.track_id]
            hist.append(box)

            x1, y1, x2, y2 = box.bbox
            w, h = x2 - x1, y2 - y1
            aspect = w / h if h > 0 else 1.0
            self._first_aspect.setdefault(box.track_id, aspect)

            y_range = self._track_y_range[box.track_id]
            y_range["min"] = min(y_range["min"], box.centroid[1])
            y_range["max"] = max(y_range["max"], box.centroid[1])
            ground_y = y_range["max"] - RELATIVE_GROUND_BAND * (y_range["max"] - y_range["min"])

            if len(hist) >= 2:
                prev = hist[-2]
                dy = box.centroid[1] - prev.centroid[1]
                dx = abs(box.centroid[0] - prev.centroid[0])

                speed = (dx ** 2 + dy ** 2) ** 0.5
                thrown_now = speed > THROWN_SPEED_PX_PER_FRAME and dx > THROWN_MIN_HORIZONTAL_PX_PER_FRAME
                if thrown_now:
                    self._thrown_streak[box.track_id] += 1
                else:
                    self._thrown_streak[box.track_id] = 0
                thrown_confirmed = self._thrown_streak[box.track_id] >= THROWN_MIN_STREAK
                if thrown_confirmed and "thrown" not in self._already_flagged[box.track_id]:
                    events.append(BehaviorEvent(box.track_id, box.frame_number, "thrown", pixel_drop=max(dy, 0)))
                    self._already_flagged[box.track_id].add("thrown")
                elif not thrown_now and dy > DROP_VELOCITY_PX_PER_FRAME and "dropped" not in self._already_flagged[box.track_id]:
                    events.append(BehaviorEvent(box.track_id, box.frame_number, "dropped", pixel_drop=dy))
                    self._already_flagged[box.track_id].add("dropped")

                near_ground = box.centroid[1] > ground_y
                dragging_now = near_ground and 0 < dx <= DRAG_MAX_VELOCITY_PX_PER_FRAME and dy < 5
                if dragging_now:
                    sustained = sum(
                        1 for i in range(max(1, len(hist) - DRAG_MIN_FRAMES), len(hist))
                        if hist[i].centroid[1] > ground_y
                    )
                    cooldown_ok = box.frame_number - self._last_dragged_frame[box.track_id] >= DRAGGED_COOLDOWN_FRAMES
                    if sustained >= DRAG_MIN_FRAMES and cooldown_ok:
                        events.append(BehaviorEvent(box.track_id, box.frame_number, "dragged"))
                        self._last_dragged_frame[box.track_id] = box.frame_number

            base_aspect = self._first_aspect[box.track_id]
            flipped_now = base_aspect > 0 and (aspect / base_aspect > ASPECT_FLIP_RATIO or base_aspect / max(aspect, 1e-6) > ASPECT_FLIP_RATIO)
            if flipped_now:
                self._orientation_flip_streak[box.track_id] += 1
            else:
                self._orientation_flip_streak[box.track_id] = 0
            if self._orientation_flip_streak[box.track_id] >= ORIENTATION_FLIP_MIN_STREAK:
                if "wrong_orientation" not in self._already_flagged[box.track_id]:
                    events.append(BehaviorEvent(box.track_id, box.frame_number, "wrong_orientation"))
                    self._already_flagged[box.track_id].add("wrong_orientation")

            for other in boxes:
                if other.track_id == box.track_id:
                    continue
                ox1, oy1, ox2, oy2 = other.bbox
                x_overlap = max(0, min(x2, ox2) - max(x1, ox1))
                if x_overlap / max(w, 1) > STACK_OVERLAP_X_FRAC and oy2 < y1:
                    stack_key = f"stack_{box.track_id}"
                    if stack_key not in self._already_flagged[box.track_id]:
                        events.append(BehaviorEvent(box.track_id, box.frame_number, "over_stacked"))
                        self._already_flagged[box.track_id].add(stack_key)

            box_area = max(w * h, 1)
            stepped_on_now = False
            for person in people:
                px1, py1, px2, py2 = person.bbox
                feet_y = py2
                feet_over_box = x1 < (px1 + px2) / 2 < x2 and y1 < feet_y < y2
                if not feet_over_box:
                    continue
                ix1, iy1 = max(x1, px1), max(y1, py1)
                ix2, iy2 = min(x2, px2), min(y2, py2)
                overlap_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if overlap_area / box_area >= STEPPED_ON_MIN_OVERLAP_FRAC:
                    stepped_on_now = True
                    break
            if stepped_on_now:
                self._stepped_on_streak[box.track_id] += 1
            else:
                self._stepped_on_streak[box.track_id] = 0
            if self._stepped_on_streak[box.track_id] >= STEPPED_ON_MIN_STREAK:
                if "stepped_on" not in self._already_flagged[box.track_id]:
                    events.append(BehaviorEvent(box.track_id, box.frame_number, "stepped_on"))
                    self._already_flagged[box.track_id].add("stepped_on")

        return events
