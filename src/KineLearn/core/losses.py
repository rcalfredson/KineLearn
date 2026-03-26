from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
except Exception:
    tf = None
    K = None

# Path: src/KineLearn/core/losses.py


def focal_loss(alpha: float, gamma: float = 2.0):
    """
    Binary focal loss for a single sigmoid output per timestep.

    alpha: weight for positive class (y=1)
    gamma: focusing parameter
    """
    if tf is None or K is None:
        raise ImportError("TensorFlow is required for focal_loss.")

    alpha_t = tf.constant(float(alpha), dtype=tf.float32)
    gamma_t = tf.constant(float(gamma), dtype=tf.float32)

    def loss(y_true, y_pred):
        eps = K.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        w_t = y_true * alpha_t + (1.0 - y_true) * (1.0 - alpha_t)
        mod = tf.pow(1.0 - p_t, gamma_t)

        return tf.reduce_mean(-w_t * mod * tf.math.log(p_t))

    return loss
