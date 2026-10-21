"""
python step5_drive.py model.keras --sim-path "D:\\DonkeySimWin\\donkey_sim.exe"
"""
import argparse
import time

import numpy as np
from tensorflow import keras

from common import preprocess
from donkey_common import make_env, reset, step

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--sim-path", default=None)
parser.add_argument("--track", default="donkey-generated-track-v0")
parser.add_argument("--throttle", type=float, default=0.3)
args = parser.parse_args()

model = keras.models.load_model(args.model)
env = make_env(args.sim_path, args.track)
obs = reset(env)

takeovers, start = 0, time.time()
try:
    while True:
        x = preprocess(obs)[None].astype("float32")             # SAME preprocessing as training
        steering = float(np.clip(model(x, training=False).numpy()[0, 0], -1, 1))
        throttle = args.throttle * (0.6 if abs(steering) > 0.5 else 1.0)   # ease off in sharp turns
        obs, done = step(env, steering, throttle)
        if done:
            takeovers += 1
            print(f"Left the track (#{takeovers}) after {time.time() - start:.0f}s - resetting")
            obs = reset(env)
except KeyboardInterrupt:
    pass
finally:
    env.close()
    print(f"Stopped. Crashes/resets: {takeovers}")
