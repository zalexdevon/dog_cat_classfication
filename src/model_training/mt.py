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
from pathlib import Path


def create_train_val_ds(param):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        Path(f"{param['train_val_path']}/train"),
        shuffle=True,
        image_size=(param["image_size"], param["image_size"]),
        batch_size=param["batch_size"],
        class_names=param["class_names"],
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        Path(f"{param['train_val_path']}/val"),
        shuffle=True,
        image_size=(param["image_size"], param["image_size"]),
        batch_size=param["batch_size"],
        class_names=param["class_names"],
    )

    # Cache và prefetch dataset
    train_ds = tf_myfuncs.cache_prefetch_tfdataset(train_ds)
    val_ds = tf_myfuncs.cache_prefetch_tfdataset(val_ds)

    return train_ds, val_ds


def create_model(param):
    # Inputs
    inputs = tf.keras.Input(
        shape=(param["image_size"], param["image_size"], param["channels"])
    )

    # Layer
    resize_layer = tf.keras.layers.Resizing(param["image_size"], param["image_size"])
    data_augmentation_layer = tf_model_training_classes.LayerCreator(
        param, "layer0"
    ).next()
    rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255)
    conv2D_layer = tf_model_training_classes.LayerCreator(param, "layer1").next()
    flatten_layer = tf_model_training_classes.LayerCreator(param, "layer2").next()
    dense_layer = tf_model_training_classes.LayerCreator(param, "layer3").next()
    output_layer = tf_model_training_funcs.get_output_layer_for_classification(
        param["num_classes"]
    )

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
