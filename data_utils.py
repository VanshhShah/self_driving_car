"""Reading driving_log.csv written by the Udacity simulator."""
from pathlib import Path

import cv2
import pandas as pd

COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def load_log(data_dir: str) -> pd.DataFrame:
    """Return a table with image paths fixed to point inside <data_dir>/IMG."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "driving_log.csv", header=None, names=COLUMNS)
    df["steering"] = pd.to_numeric(df["steering"], errors="coerce")
    df = df.dropna(subset=["steering"]).reset_index(drop=True)   # drops a header row if present

    # The CSV stores the full path from the machine that recorded it (often Windows
    # "C:\...\IMG\center_xxx.jpg"). Keep only the file name and look in our IMG folder.
    for col in ("center", "left", "right"):
        df[col] = df[col].apply(
            lambda p: str(data_dir / "IMG" / Path(str(p).strip().replace("\\", "/")).name)
            if isinstance(p, str) and p.strip() else None      # Donkey recordings: no side cameras
        )
    return df


def read_rgb(path: str):
    """Read an image file as RGB (OpenCV reads BGR by default)."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
