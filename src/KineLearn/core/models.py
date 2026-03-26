from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input,
        Dense,
        LSTM,
        Bidirectional,
        TimeDistributed,
    )
    from tensorflow.keras.models import Model
except Exception:
    tf = None
    Input = Dense = LSTM = Bidirectional = TimeDistributed = Model = None

# Path: src/KineLearn/core/models.py


def build_keypoint_bilstm(window_size: int, derived_dim: int) -> "tf.keras.Model":
    """
    Keypoints-only sequence model.
    Output is per-timestep sigmoid for ONE behavior: shape (B, T, 1)
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build models.")

    inp = Input(shape=(int(window_size), int(derived_dim)), name="keypoints")
    x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm_128")(inp)
    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_64")(x)
    x = TimeDistributed(Dense(32, activation="relu"), name="td_dense_32")(x)
    x = TimeDistributed(Dense(64, activation="relu"), name="td_dense_64")(x)
    out = TimeDistributed(Dense(1, activation="sigmoid"), name="y")(x)
    return Model(inp, out, name="kinelearn_keypoints_bilstm")
