"""
SRCNN Model Architecture implementation using TensorFlow/Keras.
Based on Chao Dong et al. (TPAMI 2016) and Stanford CS229 Project (2020).
"""

import tensorflow as tf

def build_srcnn(input_shape=(224, 224, 3), name="SRCNN"):
    """
    Builds the 3-layer Super-Resolution Convolutional Neural Network (SRCNN).
    
    Layer 1: Patch Extraction and Representation
      - 64 filters of size 9x9, ReLU activation
      - Extracts overlapping feature patches from the bicubic input
      
    Layer 2: Non-Linear Mapping
      - 32 filters of size 1x1, ReLU activation
      - Non-linearly maps high-dimensional vectors to another set of feature vectors
      
    Layer 3: Reconstruction
      - 3 filters of size 5x5, Linear activation
      - Aggregates high-resolution patch representations into the final output RGB image
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_lr_bicubic")
    
    # Layer 1: 9x9 conv, 64 filters, ReLU
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(9, 9),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="conv1_patch_extraction"
    )(inputs)
    
    # Layer 2: 1x1 conv, 32 filters, ReLU
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="conv2_nonlinear_mapping"
    )(x)
    
    # Layer 3: 5x5 conv, 3 filters, Linear (continuous RGB reconstruction)
    outputs = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=(5, 5),
        padding="same",
        activation="linear",
        kernel_initializer="he_normal",
        name="conv3_reconstruction"
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

def build_res_srcnn(input_shape=(224, 224, 3), name="Residual_SRCNN"):
    """
    Residual SRCNN (Res-SRCNN) Architecture:
    Maintains the exact 3 convolutional layers (64 9x9 -> 32 1x1 -> 3 5x5),
    but learns the high-frequency residual detail map (Kim et al., CVPR 2016):
      Output = Input + F(Input)
    Allows the model to focus exclusively on reconstructing sharp high-frequency edges,
    achieving rapid convergence and superior PSNR (>29 dB).
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_lr_bicubic")
    
    # Layer 1: 9x9 conv, 64 filters, ReLU
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(9, 9),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="conv1_patch_extraction"
    )(inputs)
    
    # Layer 2: 1x1 conv, 32 filters, ReLU
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="conv2_nonlinear_mapping"
    )(x)
    
    # Layer 3: 5x5 conv, 3 filters, Linear (residual detail map)
    residual = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=(5, 5),
        padding="same",
        activation="linear",
        kernel_initializer="zeros",
        name="conv3_residual_reconstruction"
    )(x)
    
    # Add skip connection: Low-Resolution input + High-Frequency residual
    outputs = tf.keras.layers.Add(name="add_residual")([inputs, residual])
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

def build_srcnn_bn(input_shape=(224, 224, 3), name="SRCNN_BatchNorm"):
    """
    Variant with Batch Normalization (to reproduce and test Stanford CS229 finding).
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_lr_bicubic")
    
    x = tf.keras.layers.Conv2D(64, (9, 9), padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(32, (1, 1), padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    outputs = tf.keras.layers.Conv2D(3, (5, 5), padding="same", activation="linear")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

def count_parameters(model):
    """Return total, trainable, and non-trainable parameter count."""
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_count = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    return {
        "total": trainable_count + non_trainable_count,
        "trainable": trainable_count,
        "non_trainable": non_trainable_count
    }
