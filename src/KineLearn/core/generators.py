from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf

    SequenceBase = tf.keras.utils.Sequence
except Exception:
    tf = None
    SequenceBase = object

# Path: src/KineLearn/core/generators.py


class KeypointWindowGenerator(SequenceBase):
    """
    Memmap-backed generator yielding:
      X: (B, T, D) float32
      y: (B, T, 1) float32   for a single behavior_idx

    Notes:
    - Uses window indices (0..N-1) as the sampling unit.
    - Works with np.memmap or ndarray.
    """

    def __init__(
        self,
        mmX: np.ndarray,
        mmY: np.ndarray,
        *,
        behavior_idx: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        noise_std: float = 0.0,
    ):
        if tf is None:
            raise ImportError(
                "TensorFlow is required for KeypointWindowGenerator "
                "(tf.keras.utils.Sequence)."
            )

        self.mmX = mmX
        self.mmY = mmY
        self.behavior_idx = int(behavior_idx)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.noise_std = float(noise_std)
        self.rng = (
            np.random.default_rng(self.seed) if (self.shuffle or self.noise_std > 0.0) else None
        )

        self.n = int(mmX.shape[0])
        if int(mmY.shape[0]) != self.n:
            raise ValueError(
                f"mmX/mmY count mismatch: {mmX.shape[0]} vs {mmY.shape[0]}"
            )
        if not (0 <= self.behavior_idx < int(mmY.shape[-1])):
            raise ValueError(f"behavior_idx out of range: {self.behavior_idx}")

        self.indices = np.arange(self.n, dtype=np.int64)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, i: int):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.indices[sl]

        Xb = np.asarray(self.mmX[idx], dtype=np.float32)  # (B,T,D)
        yb = np.asarray(self.mmY[idx, :, self.behavior_idx], dtype=np.float32)  # (B,T)
        yb = yb[..., None]  # (B,T,1)

        if self.noise_std > 0.0:
            Xb = Xb + self.rng.normal(0.0, self.noise_std, size=Xb.shape).astype(
                np.float32
            )

        return Xb, yb

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
