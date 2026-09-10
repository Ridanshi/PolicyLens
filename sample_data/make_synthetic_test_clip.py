"""
No real warehouse footage available yet (Godrej's pilot videos need manual
download from the Drive link — auth-gated, can't be fetched automatically).
This generates a synthetic clip so the pipeline's PLUMBING can be verified
end-to-end today: a box that sits still, then free-falls, then a second box
gets stacked on a third. Tests mechanics only, not real-world accuracy —
real accuracy validation still needs the actual pilot footage.
"""
import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 480, 30, 6
frames = FPS * SECONDS

writer = cv2.VideoWriter("synthetic_drop.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

box_w, box_h = 80, 60
box_x = 280

for i in range(frames):
    frame = np.full((H, W, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (0, int(H * 0.85)), (W, H), (60, 60, 60), -1)  # floor line

    if i < 60:
        box_y = 100  # sitting still
    elif i < 75:
        progress = (i - 60) / 15
        box_y = int(100 + progress * (int(H * 0.85) - box_h - 100))  # fast fall
    else:
        box_y = int(H * 0.85) - box_h  # landed, stays

    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 140, 255), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)

    writer.write(frame)

writer.release()
print("Wrote synthetic_drop.mp4")
