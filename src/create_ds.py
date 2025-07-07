from tensorflow.keras.preprocessing import image_dataset_from_directory
from Mylib import tf_myfuncs


def create_train_val_ds(param, train_val_path, class_names):
    train_ds = create_ds(param, train_val_path / "train", class_names)
    val_ds = create_ds(param, train_val_path / "val", class_names)

    return train_ds, val_ds


def create_train_test_ds(param, train_data_path, test_data_path, class_names):
    train_ds = create_ds(param, train_data_path, class_names)
    test_ds = create_ds(param, test_data_path, class_names)

    return train_ds, test_ds


def create_ds(param, ds_path, class_names):
    ds = image_dataset_from_directory(
        ds_path,
        shuffle=True,
        image_size=(param["image_size"], param["image_size"]),
        batch_size=param["batch_size"],
        class_names=class_names,
    )

    ds = tf_myfuncs.cache_prefetch_tfdataset(ds)

    return ds
