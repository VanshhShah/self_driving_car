"""Nvidia's "End to End Learning for Self-Driving Cars" network, in Keras."""
from tensorflow import keras
from tensorflow.keras import layers

from common import INPUT_SHAPE


def build_model(dropout: float = 0.5) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        layers.Rescaling(1.0 / 127.5, offset=-1.0),            # pixels 0..255 -> -1..1

        # 5 convolutional layers: learn edges, then shapes, then road features
        layers.Conv2D(24, 5, strides=2, activation="elu"),
        layers.Conv2D(36, 5, strides=2, activation="elu"),
        layers.Conv2D(48, 5, strides=2, activation="elu"),
        layers.Conv2D(64, 3, activation="elu"),
        layers.Conv2D(64, 3, activation="elu"),

        layers.Dropout(dropout),                                # fights overfitting
        layers.Flatten(),

        # 3 dense layers + 1 output (the steering number)
        layers.Dense(100, activation="elu"),
        layers.Dense(50, activation="elu"),
        layers.Dense(10, activation="elu"),
        layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    return model


if __name__ == "__main__":
    build_model().summary()   # python model.py  -> see the layers and ~250k weights
