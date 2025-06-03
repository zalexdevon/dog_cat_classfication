import os
from Mylib import (
    myfuncs,
    tf_myfuncs,
    tf_model_training_funcs,
    tf_model_training_classes,
    tf_myclasses,
)
import tensorflow as tf
from src.utils import classes
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import gc


def load_data(data_transformation_path, class_names_path):
    train_ds = tf.data.Dataset.load(f"{data_transformation_path}/train_ds")
    val_ds = tf.data.Dataset.load(f"{data_transformation_path}/val_ds")
    image_size = myfuncs.load_python_object(
        f"{data_transformation_path}/image_size.pkl"
    )
    channels = myfuncs.load_python_object(f"{data_transformation_path}/channels.pkl")
    num_classes = len(myfuncs.load_python_object(class_names_path))

    return train_ds, val_ds, image_size, channels, num_classes


def create_model(param):
    # Layer input và output
    input_layer = tf.keras.Input(
        shape=(param["image_size"], param["image_size"], param["channels"])
    )
    output_layer = tf_model_training_funcs.get_output_layer(param["num_classes"])

    # Layer ở giữa
    data_augmentation_layer = tf_model_training_classes.LayerCreator(
        param, "layer0"
    ).next()
    rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255)
    conv2D_layer = tf_model_training_classes.LayerCreator(param, "layer1").next()
    flatten_layer = tf_model_training_classes.LayerCreator(param, "layer2").next()
    dense_layer = tf_model_training_classes.LayerCreator(param, "layer3").next()

    # Xây dựng model
    x = data_augmentation_layer(input_layer)
    x = rescaling_layer(x)
    x = conv2D_layer(x)
    x = flatten_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = tf.keras.models.Model(input_layer, x)
    return model
