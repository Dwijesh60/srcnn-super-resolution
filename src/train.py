"""
Training pipeline and experiment runner for SRCNN.
"""

import os
import tensorflow as tf
from src import config
from src.metrics import psnr_metric, ssim_metric

def train_srcnn(
    model,
    train_dataset,
    val_dataset,
    epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    checkpoint_path=None,
    verbose=1
):
    """
    Compiles and trains SRCNN model with Adam optimizer and standard callbacks.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[psnr_metric, ssim_metric]
    )
    
    callbacks = []
    
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0
        )
        callbacks.append(ckpt_cb)
        
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=0
    )
    callbacks.append(early_stop_cb)
    
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=0
    )
    callbacks.append(reduce_lr_cb)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history
