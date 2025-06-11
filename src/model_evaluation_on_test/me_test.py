import tensorflow as tf
from Mylib import myfuncs, tf_myclasses, tf_myfuncs, tf_model_evaluator
import os
from pathlib import Path


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
