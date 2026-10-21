"""
python step3_check_data.py --data path/to/your/recording
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import preprocess
from data_utils import load_log, read_rgb

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="folder containing driving_log.csv and IMG/")
args = parser.parse_args()

df = load_log(args.data)
print(f"{len(df)} recorded moments ({len(df) * 3} pictures with left/right cameras)")
print(df[["steering", "throttle", "speed"]].describe().round(3))
print(f"Share of moments with steering almost 0: {(df.steering.abs() < 0.05).mean():.0%}")

plt.figure(figsize=(7, 3.5))
plt.hist(df.steering, bins=51)
plt.title("Steering values in your recording")
plt.xlabel("steering (-1 = full left, +1 = full right)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("steering_hist.png", dpi=120)

sample = read_rgb(df.center.iloc[len(df) // 2])
seen = preprocess(sample)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].imshow(sample); axes[0].set_title("raw 160x320")
axes[1].imshow(seen);   axes[1].set_title("network input 66x200 (YUV, colours look odd)")
for a in axes:
    a.axis("off")
plt.tight_layout()
plt.savefig("preview.png", dpi=120)
print("Saved steering_hist.png and preview.png - open them and check they look sensible.")
