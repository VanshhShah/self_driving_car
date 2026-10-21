"""
Shared code used by BOTH training and driving.

Why it lives in one file: the document's #1 gotcha is "different pictures in
training and driving". If train.py and drive.py both call preprocess() from
here, they cannot drift apart.
"""
import cv2
import numpy as np

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Crop as a FRACTION of the picture height, so it works for any camera size:
#   Donkey simulator picture: 120 rows x 160 columns -> top 0.375 = 45 rows of sky.
#   (Udacity simulator, 160x320, used 60 top / 25 bottom rows = 0.375 / 0.156.)
CROP_TOP_FRAC = 0.375      # sky + horizon
CROP_BOTTOM_FRAC = 0.0     # Donkey's camera shows no car hood


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 picture (any size) -> network-ready uint8 picture (66x200x3, YUV)."""
    h = image_rgb.shape[0]
    top, bottom = int(h * CROP_TOP_FRAC), int(h * CROP_BOTTOM_FRAC)
    image = image_rgb[top:h - bottom, :, :]                              # crop
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)  # shrink
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)                       # Nvidia used YUV
    return image
