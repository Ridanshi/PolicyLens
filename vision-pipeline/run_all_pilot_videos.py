"""Runs main.py sequentially on every real Godrej pilot video. One-off batch
helper, not part of the pipeline."""
import subprocess
import os
import time

VIDEOS = [
    "rolling_and_dropping_carton.mp4",
    "dock_level_dragging_cupboard.mp4",
    "kd_packets_dragged_heavy_box_on_other_packets.mp4",
    "stepping_on_cartons_vertical_product_horizontal_heavy_on_top.mp4",
    "throwing_mattresses.mp4",
    "throwing_seating_cartons_using_strap_to_hold.mp4",
]

pilot_dir = os.path.join("..", "sample_data", "godrej_pilot")
python_exe = os.path.join("venv", "Scripts", "python.exe")

for video in VIDEOS:
    name = os.path.splitext(video)[0]
    out_dir = os.path.join("output_real", name)
    print(f"=== Running {video} ===", flush=True)
    result = subprocess.run([
        python_exe, "main.py",
        "--video", os.path.join(pilot_dir, video),
        "--bay", "bay-1",
        "--meters-per-pixel", "0.004",
        "--out-dir", out_dir,
    ], capture_output=True, text=True)
    print(result.stdout[-500:], flush=True)
    if result.returncode != 0:
        print(f"FAILED: {video}", flush=True)
        print(result.stderr[-2000:], flush=True)
    print(f"=== Done {video} ===\n", flush=True)
    time.sleep(5)

print("ALL DONE", flush=True)
