"""
python smoke_test_fake_data.py
Makes 600 fake "road" pictures in fake_recording
"""
import csv
from pathlib import Path

import cv2
import numpy as np

out = Path("fake_recording")
(out / "IMG").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def fake_frame(offset):
    """160x320 picture; road centre shifted by offset (-1..1) at the top of the road."""
    img = np.full((160, 320, 3), (40, 90, 40), np.uint8)             # grass
    centre_top = 160 + int(offset * 100)
    road = np.array([[centre_top - 25, 60], [centre_top + 25, 60], [250, 135], [70, 135]])
    cv2.fillPoly(img, [road], (120, 120, 120))                       # road
    noise = rng.integers(-8, 8, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


with open(out / "driving_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for i in range(600):
        steering = float(np.clip(rng.normal(0, 0.4), -1, 1))
        paths = []
        for cam in ("center", "left", "right"):
            p = out / "IMG" / f"{cam}_{i:04d}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(fake_frame(steering), cv2.COLOR_RGB2BGR))
            paths.append(str(p))
        writer.writerow(paths + [round(steering, 4), 1, 0, 15])
print("Wrote fake_recording/ with 600 moments.")
