# Saved Models Directory

This directory stores trained model checkpoints and serialized weights for the Super-Resolution Convolutional Neural Network (SRCNN).

## Stored Artifacts
- `best_srcnn.keras`: The best-performing model checkpoint saved during training based on minimum validation MSE loss / maximum validation PSNR.
- Architecture: 3-layer SRCNN (9x9 conv with 64 filters -> ReLU -> 1x1 conv with 32 filters -> ReLU -> 5x5 conv with 3 filters -> Linear RGB output).
- Input: Bicubic upsampled RGB image normalized to $[0.0, 1.0]$ with shape `(batch, 224, 224, 3)`.
- Output: Super-resolved reconstructed RGB image normalized to $[0.0, 1.0]$ with shape `(batch, 224, 224, 3)`.

## Loading Model
```python
import tensorflow as tf

model = tf.keras.models.load_model("models/best_srcnn.keras", compile=False)
reconstructed = model.predict(lr_bicubic_batch)
```
