import tensorflow as tf
from Mylib import tf_myfuncs, myfuncs
import os


def load_data(data_ingestion_path):
    class_names = myfuncs.load_python_object(f"{data_ingestion_path}/class_names.pkl")

    return class_names


def create_train_val_ds(data_ingestion_path, class_names, image_size, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_ingestion_path}/train",
        shuffle=True,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        class_names=class_names,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f"{data_ingestion_path}/val",
        shuffle=True,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        class_names=class_names,
    )

    # Cache và prefetch dataset
    train_ds = tf_myfuncs.cache_prefetch_tfdataset_2(train_ds)
    val_ds = tf_myfuncs.cache_prefetch_tfdataset_2(val_ds)

    return train_ds, val_ds


def save_data(
    data_transformation_path, train_ds, val_ds, image_size, batch_size, channels
):
    train_ds.save(f"{data_transformation_path}/train_ds")
    val_ds.save(f"{data_transformation_path}/val_ds")
    myfuncs.save_python_object(f"{data_transformation_path}/image_size.pkl", image_size)
    myfuncs.save_python_object(f"{data_transformation_path}/batch_size.pkl", batch_size)
    myfuncs.save_python_object(f"{data_transformation_path}/channels.pkl", channels)
