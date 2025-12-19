"""
Model definitions for HAM10000 skin lesion classification.

Contains:
- build_resnet50_base: baseline ResNet50 classifier
- build_resnet50_modified: ResNet50 + Squeeze-and-Excitation (SE) channel attention
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

# Default number of classes for HAM10000
NUM_CLASSES_DEFAULT = 7


def build_resnet50_base(num_classes: int = NUM_CLASSES_DEFAULT) -> keras.Model:
    """
    Build the baseline ResNet50 model for HAM10000 classification.

    Args:
        num_classes: Number of output classes (default 7 for HAM10000).

    Returns:
        A compiled Keras Model.
    """
    # Base ResNet50 (ImageNet weights)
    base_model_res = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    # First training phase: freeze backbone
    base_model_res.trainable = False

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(base_model_res.output)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(
        inputs=base_model_res.input,
        outputs=outputs,
        name="ResNet50_HAM10000_Base",
    )

    return model


def se_block(x: tf.Tensor, reduction_ratio: int = 16, name: str | None = None) -> tf.Tensor:
    """
    Squeeze-and-Excitation (SE) block for channel attention.

    Args:
        x: 4D input tensor (batch, H, W, C).
        reduction_ratio: Reduction factor for the bottleneck.
        name: Optional base name for the layers.

    Returns:
        Tensor after applying SE scaling.
    """
    channels = x.shape[-1]
    if channels is None:
        raise ValueError("Input tensor to SE block must have known channel dimension")

    prefix = "" if name is None else name + "_"

    # Squeeze: global average pooling over spatial dimensions
    se = layers.GlobalAveragePooling2D(name=prefix + "squeeze")(x)

    # Excitation: bottleneck MLP
    se = layers.Dense(
        units=channels // reduction_ratio,
        activation="relu",
        name=prefix + "excitation_dense1",
    )(se)
    se = layers.Dense(
        units=channels,
        activation="sigmoid",
        name=prefix + "excitation_dense2",
    )(se)

    # Reshape to (batch, 1, 1, C) for channel-wise multiplication
    se = layers.Reshape((1, 1, channels), name=prefix + "reshape")(se)

    # Scale: channel-wise multiplication
    x = layers.Multiply(name=prefix + "scale")([x, se])

    return x


def build_resnet50_modified(num_classes: int = NUM_CLASSES_DEFAULT) -> keras.Model:
    """
    Build a modified ResNet50 with a Squeeze-and-Excitation (SE) channel-attention block.

    Args:
        num_classes: Number of output classes (default 7 for HAM10000).

    Returns:
        A compiled Keras Model.
    """
    # Base ResNet50 (ImageNet weights)
    base_model_ch = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    # First training phase: freeze backbone
    base_model_ch.trainable = False

    # Take the convolutional output
    x = base_model_ch.output

    # Add channel-attention SE block
    x = se_block(x, reduction_ratio=16, name="resnet50_se")

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(
        inputs=base_model_ch.input,
        outputs=outputs,
        name="ResNet50_HAM10000_SE",
    )

    return model
