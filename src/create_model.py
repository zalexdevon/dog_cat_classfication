import tensorflow as tf
from src.create_object import create_object


def create_model(param):
    # Inputs
    inputs = tf.keras.Input(
        shape=(param["image_size"], param["image_size"], param["channels"])
    )

    # Layer
    resize_layer = tf.keras.layers.Resizing(param["image_size"], param["image_size"])
    data_augmentation_layer = create_object(param, "data_augmentation")
    rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255)
    conv2D_layer = create_object(param, "conv")
    flatten_layer = create_object(param, "flatten")
    dense_layer = create_object(param, "dense")
    output_layer = tf.keras.layers.Dense(units=2, activation="softmax")

    # Xây dựng model
    x = resize_layer(inputs)
    x = data_augmentation_layer(x)
    x = rescaling_layer(x)
    x = conv2D_layer(x)
    x = flatten_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = tf.keras.models.Model(inputs, x)
    return model
