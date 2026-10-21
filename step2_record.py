"""
python step2_record.py --out my_recording --sim-path "D:\\DonkeySimWin\\donkey_sim.exe"

A small window shows what the car's camera sees. CLICK ON THAT WINDOW so it gets your keys.
    Left/Right arrows (or A/D) : steer          Up/W : go      Down/S : brake
    R : start/stop recording    Esc : quit and save
Recording writes <out>/IMG/*.jpg and <out>/driving_log.csv (same format as before).
Drive smoothly. Also record recoveries: let the car drift to the edge, THEN start
recording and steer back to the middle.
"""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import cv2
import numpy as np
import pygame

from donkey_common import make_env, reset, step

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="my_recording")
parser.add_argument("--sim-path", default=None, help="full path to donkey_sim.exe (omit if already running)")
parser.add_argument("--track", default="donkey-generated-track-v0")
parser.add_argument("--throttle", type=float, default=0.3, help="throttle while Up is held")
parser.add_argument("--fps", type=int, default=20)
args = parser.parse_args()

out = Path(args.out)
(out / "IMG").mkdir(parents=True, exist_ok=True)
log_path = out / "driving_log.csv"
existing = len(list((out / "IMG").glob("*.jpg")))        # lets you add more laps to the same folder
log_file = open(log_path, "a", newline="")
writer = csv.writer(log_file)

env = make_env(args.sim_path, args.track)
obs = reset(env)

pygame.init()
SCALE = 4
screen = pygame.display.set_mode((obs.shape[1] * SCALE, obs.shape[0] * SCALE))
pygame.display.set_caption("Donkey recorder - click here, arrows to drive, R = record, Esc = quit")
font = pygame.font.SysFont(None, 28)
clock = pygame.time.Clock()

steering, recording, saved = 0.0, False, 0
running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    recording = not recording

        keys = pygame.key.get_pressed()
        target = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            target -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            target += 1.0
        if target:                                          # hold a key to steer harder
            steering = float(np.clip(steering + 0.1 * target, -1.0, 1.0))
        else:                                               # let go -> wheel returns to centre
            steering -= float(np.sign(steering)) * min(abs(steering), 0.15)
        throttle = args.throttle if (keys[pygame.K_UP] or keys[pygame.K_w]) else 0.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle = -0.2

        if recording and throttle > 0:                      # record the picture BEFORE acting on it
            name = f"center_{existing + saved:06d}.jpg"
            cv2.imwrite(str(out / "IMG" / name), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            writer.writerow([name, "", "", round(float(steering), 4), throttle, 0, 0])
            saved += 1

        obs, done = step(env, steering, throttle)
        if done:                                            # crashed / left the track
            obs = reset(env)
            steering = 0.0

        frame = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(pygame.transform.scale(frame, screen.get_size()), (0, 0))
        label = f"{'REC' if recording else 'not recording'}  steer {steering:+.2f}  saved {saved}"
        screen.blit(font.render(label, True, (255, 60, 60) if recording else (255, 255, 255)), (8, 8))
        pygame.display.flip()
        clock.tick(args.fps)
finally:
    log_file.close()
    env.close()
    pygame.quit()
    print(f"Saved {saved} pictures to {out}. Next: python step1_check_data.py --data {out}")
