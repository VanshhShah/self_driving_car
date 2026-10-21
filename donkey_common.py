"""Connecting to the Donkey Car simulator through the gym-donkeycar library."""
import os

import gym_donkeycar  # noqa: F401  (importing registers the environments)

try:
    import gymnasium as gym
except ImportError:          # older gym-donkeycar uses classic gym
    import gym

DEFAULT_TRACK = "donkey-generated-track-v0"


def make_env(sim_path=None, track=DEFAULT_TRACK, port=9091):
    """sim_path: full path to donkey_sim.exe, or None if you started the simulator yourself."""
    os.environ["DONKEY_SIM_PORT"] = str(port)
    if sim_path:
        os.environ["DONKEY_SIM_PATH"] = sim_path
    else:
        os.environ.pop("DONKEY_SIM_PATH", None)   # => "you must start the sim manually"
    return gym.make(track)


def reset(env):
    """Works with both old (obs) and new (obs, info) gym APIs."""
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step(env, steering, throttle):
    """Returns (obs, done). Works with both 4-tuple and 5-tuple gym APIs."""
    import numpy as np
    out = env.step(np.array([steering, throttle], dtype=np.float32))
    if len(out) == 5:
        obs, _reward, terminated, truncated, _info = out
        return obs, bool(terminated or truncated)
    obs, _reward, done, _info = out
    return obs, bool(done)
