import tensorflow as tf
import os
from pathlib import Path
import os
from Mylib import tf_myfuncs, tf_metrics, tf_create_object
import tensorflow as tf
from pathlib import Path
import pandas as pd
from src import const


def create_train_test_ds(param, test_data_path):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        Path(f"{param['train_val_path']}/train"),
        shuffle=True,
        image_size=(param["image_size"], param["image_size"]),
        batch_size=param["batch_size"],
        class_names=param["class_names"],
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_path,
        shuffle=True,
        image_size=(param["image_size"], param["image_size"]),
        batch_size=param["batch_size"],
        class_names=param["class_names"],
    )

    # Cache và prefetch dataset
    train_ds = tf_myfuncs.cache_prefetch_tfdataset(train_ds)
    test_ds = tf_myfuncs.cache_prefetch_tfdataset(test_ds)

    return train_ds, test_ds


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
    data_augmentation_layer = tf_create_object.ObjectCreatorFromDict(
        param, "layer0"
    ).next()
    rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255)
    conv2D_layer = tf_create_object.ObjectCreatorFromDict(param, "layer1").next()
    flatten_layer = tf_create_object.ObjectCreatorFromDict(param, "layer2").next()
    dense_layer = tf_create_object.ObjectCreatorFromDict(param, "layer3").next()
    output_layer = get_output_layer_for_classification(param["num_classes"])

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


def get_run_folders(model_training_path):
    run_folders = pd.Series(os.listdir(model_training_path))
    run_folders = run_folders[run_folders.str.startswith("run")]
    return run_folders


def get_metrics(scoring):
    if scoring in const.BUILT_IN_METRICS:
        return [scoring]

    if scoring == "bleu":
        return [tf_metrics.BleuScoreCustomMetric()]

    raise ValueError(f"Chưa định nghĩa cho {scoring}")


def get_reverse_param_in_sorted(scoring):
    if scoring in const.SCORINGS_PREFER_MAXIMUM:
        return True

    if scoring in const.SCORINGS_PREFER_MININUM:
        return False

    raise ValueError(f"Chưa định nghĩa cho {scoring}")


def get_output_layer_for_classification(num_classes=None):
    if len(num_classes) == 2:
        return tf.keras.layers.Dense(1, activation="sigmoid")

    # Phân loại nhiều class
    return tf.keras.layers.Dense(num_classes, activation="softmax")
