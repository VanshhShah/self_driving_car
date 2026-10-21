"""
python step4_train.py --data path/to/your/recording --epochs 10
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from common import INPUT_SHAPE, preprocess
from data_utils import load_log, read_rgb
from model import build_model

SIDE_CAMERA_CORRECTION = 0.2   # left camera: steer a bit right; right camera: a bit left


class DrivingData(keras.utils.Sequence):
    """Feeds batches of (picture, steering). Augments only when training=True."""

    def __init__(self, df, batch_size=64, training=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.training = training
        self.order = np.arange(len(self.df))
        self.rng = np.random.default_rng(0)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.order)

    def _one(self, row):
        steering = float(row.steering)
        if not self.training:                       # validation: honest, centre camera only
            return preprocess(read_rgb(row.center)), steering
        cameras = [c for c in ("center", "left", "right") if isinstance(getattr(row, c), str)]
        camera = self.rng.choice(cameras)
        image = read_rgb(getattr(row, camera))
        if camera == "left":
            steering += SIDE_CAMERA_CORRECTION
        elif camera == "right":
            steering -= SIDE_CAMERA_CORRECTION
        if self.rng.random() < 0.5:                 # mirror image -> mirrored steering
            image = image[:, ::-1, :]
            steering = -steering
        return preprocess(image), steering

    def __getitem__(self, idx):
        ids = self.order[idx * self.batch_size:(idx + 1) * self.batch_size]
        pairs = [self._one(r) for r in self.df.iloc[ids].itertuples()]
        images = np.stack([p[0] for p in pairs]).astype("float32")
        steering = np.array([p[1] for p in pairs], dtype="float32")
        return images, steering


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--keep-straight", type=float, default=0.3,
                        help="fraction of near-zero-steering moments to keep (they dominate laps)")
    parser.add_argument("--out", default="model.keras")
    args = parser.parse_args()

    df = load_log(args.data)

    # Split FIRST so validation moments are never seen (even flipped) during training.
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = int(len(shuffled) * args.val_fraction)
    val_df, train_df = shuffled.iloc[:n_val], shuffled.iloc[n_val:]

    # Most of a lap is straight: keep only some of those so the net doesn't just say "0".
    straight = train_df.steering.abs() < 0.05
    keep = ~straight | (np.random.default_rng(1).random(len(train_df)) < args.keep_straight)
    train_df = train_df[keep]
    print(f"train moments: {len(train_df)}   validation moments: {len(val_df)}")

    model = build_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True, verbose=1),
    ]
    history = model.fit(
        DrivingData(train_df, args.batch, training=True),
        validation_data=DrivingData(val_df, args.batch, training=False),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plt.figure(figsize=(6, 3.5))
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.xlabel("epoch"); plt.ylabel("mean squared error"); plt.legend(); plt.tight_layout()
    plt.savefig("loss.png", dpi=120)
    print(f"Best model saved to {args.out}. Loss curves in loss.png")
    print("Training loss falling while validation loss rises = overfitting.")


if __name__ == "__main__":
    main()
