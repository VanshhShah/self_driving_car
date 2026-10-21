"""
python step1_connection_test.py --sim-path "D:\\DonkeySimWin\\donkey_sim.exe"
"""
import argparse

import cv2

from donkey_common import make_env, reset, step

parser = argparse.ArgumentParser()
parser.add_argument("--sim-path", default=None)
parser.add_argument("--track", default="donkey-generated-track-v0")
args = parser.parse_args()

env = make_env(args.sim_path, args.track)
obs = reset(env)
print("Connected! picture shape:", obs.shape)
for i in range(60):
    obs, done = step(env, 0.0, 0.3)
    if done:
        print("car left the track at step", i)
        break
cv2.imwrite("hello.jpg", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
env.close()
print("Saved hello.jpg - open it. If you see a road, everything works.")
