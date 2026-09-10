"""
Combines all 7 real Godrej pilot-video outputs into one events.json for the
backend to serve — gives the dashboard realistic multi-clip, multi-behavior
data instead of one thin sample. Assigns a distinct bay label per source clip
so bay-filtering/summary features have something real to show.

Run after run_all_pilot_videos.py has produced output_real/<clip>/events.json
for all clips.
"""
import json
import os

OUTPUT_REAL_DIR = "output_real"
BAY_LABELS = {
    "rolling_and_dragging_on_wet_floor": "bay-1",
    "rolling_and_dropping_carton": "bay-1",
    "dock_level_dragging_cupboard": "bay-2",
    "kd_packets_dragged_heavy_box_on_other_packets": "bay-2",
    "stepping_on_cartons_vertical_product_horizontal_heavy_on_top": "bay-3",
    "throwing_mattresses": "bay-3",
    "throwing_seating_cartons_using_strap_to_hold": "bay-4",
}

merged = []
for clip_dir, bay in BAY_LABELS.items():
    events_path = os.path.join(OUTPUT_REAL_DIR, clip_dir, "events.json")
    if not os.path.exists(events_path):
        print(f"skip (not found): {events_path}")
        continue
    with open(events_path) as f:
        events = json.load(f)
    for e in events:
        e["bay"] = bay
        # clip_path was written relative to that clip's own --out-dir when it ran; make it findable
        # from the backend's working directory regardless of which clip produced it.
        if e.get("evidence", {}).get("clip_path"):
            e["evidence"]["clip_path"] = os.path.join(OUTPUT_REAL_DIR, clip_dir, "clips",
                                                        os.path.basename(e["evidence"]["clip_path"]))
    merged.extend(events)
    print(f"{clip_dir}: {len(events)} events -> {bay}")

merged.sort(key=lambda e: e["timestamp"])

out_path = os.path.join(OUTPUT_REAL_DIR, "merged_events.json")
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"\nWrote {len(merged)} total events to {out_path}")
