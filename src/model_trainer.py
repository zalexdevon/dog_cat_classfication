import os
from Mylib import (
    myfuncs,
)
import tensorflow as tf
import numpy as np
from pathlib import Path
import traceback
from src import create_ds
from src import create_model
from src import create_object
from tensorflow.keras.callbacks import EarlyStopping
import time
import pandas as pd


def train(
    param_dict,
    model_training_path,
    num_models,
    train_val_path,
    loss,
    patience,
    min_delta,
    epochs,
    class_names,
):
    list_param = get_list_param(param_dict, model_training_path, num_models)

    model_training_run_path = Path(
        f"{model_training_path}/{get_folder_name(model_training_path)}"
    )
    myfuncs.create_directories([model_training_run_path])

    myfuncs.save_python_object(
        Path(f"{model_training_run_path}/list_param.pkl"), list_param
    )

    best_val_scoring_path = Path(f"{model_training_run_path}/best_val_scoring.pkl")
    myfuncs.save_python_object(best_val_scoring_path, -np.inf)

    best_result_path = Path(f"{model_training_run_path}/best_result.pkl")
    myfuncs.save_python_object(best_result_path, -np.inf)

    for i, param in enumerate(list_param):
        print(f"Train model {i} / {num_models}")
        print(f"Param: {param}")
        train_model(
            param,
            best_result_path,
            best_val_scoring_path,
            loss,
            patience,
            min_delta,
            epochs,
            train_val_path,
            class_names,
        )

    best_model_result = myfuncs.load_python_object(best_result_path)
    print("Model tốt nhất")
    print(f"Param: {best_model_result[0]}")
    print(
        f"Val scoring: {best_model_result[1]}, Train scoring: {best_model_result[2]}, Best epoch: {best_model_result[3]}, training time: {best_model_result[4]}"
    )


def train_model(
    param,
    best_model_result_path,
    best_val_scoring_path,
    loss,
    patience,
    min_delta,
    epochs,
    train_val_path,
    class_names,
):
    try:
        train_ds, val_ds = create_ds.create_train_val_ds(
            param, train_val_path, class_names
        )
        start_do_something_before_epoch1 = time.time()

        callbacks = create_callbacks(
            patience, min_delta, start_do_something_before_epoch1
        )

        optimizer = create_object.create_object(param, "optimizer")

        model = create_model.create_model(param)
        print("model summary")
        model.summary()
        print()

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )

        history = model.fit(
            x=train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=callbacks,
        ).history

        val_scoring, train_scoring, best_epoch, training_time = callbacks[1].best_result
        print("Kết quả train model")
        print(
            f"Val scoring: {val_scoring}, Train scoring: {train_scoring}, Best epoch: {best_epoch}, training time: {training_time} (mins)"
        )
        print("\n")

        best_val_scoring = myfuncs.load_python_object(best_val_scoring_path)
        if best_val_scoring < val_scoring:
            myfuncs.save_python_object(best_val_scoring_path, val_scoring)
            myfuncs.save_python_object(
                best_model_result_path,
                (param, val_scoring, train_scoring, best_epoch, training_time, history),
            )

    except Exception as e:
        print(f"Lỗi: {e}")
        traceback.print_exc()


def create_callbacks(patience, min_delta, start_do_something_before_epoch1):
    earlystopping = EarlyStopping(
        monitor=f"val_accuracy",
        patience=patience,
        min_delta=min_delta,
        mode="max",
    )
    model_checkpoint = BestEpochResultSearcher(start_do_something_before_epoch1)

    return [earlystopping, model_checkpoint]


def get_list_param(param_dict, model_training_path, num_models):
    full_list_param = myfuncs.get_full_list_dict(param_dict)

    run_folders = pd.Series(os.listdir(model_training_path))

    if len(run_folders) > 0:
        for run_folder in run_folders:
            list_param = myfuncs.load_python_object(
                Path(f"{model_training_path}/{run_folder}/list_param.pkl")
            )
            full_list_param = myfuncs.subtract_2list_set(full_list_param, list_param)

    return myfuncs.randomize_list(full_list_param, num_models)


def get_folder_name(model_training_path):
    run_folders = os.listdir(model_training_path)

    if len(run_folders) == 0:
        return "run0"

    number_in_run_folders = run_folders.str.extract(r"(\d+)").astype("int")[0]
    folder_name = f"run{number_in_run_folders.max() +1}"
    return folder_name


class BestEpochResultSearcher(tf.keras.callbacks.Callback):
    def __init__(self, start_do_something_before_epoch1):
        super().__init__()
        self.start_do_something_before_epoch1 = start_do_something_before_epoch1

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

        amount_time_for_do_something_before_epoch1 = (
            self.start_time - self.start_do_something_before_epoch1
        )
        print(
            f"Thời gian chờ trước khi vào epoch 1 = {amount_time_for_do_something_before_epoch1} (s)"
        )
        self.train_scorings = []
        self.val_scorings = []

    def on_epoch_end(self, epoch, logs=None):

        self.train_scorings.append(logs.get("accuracy"))
        self.val_scorings.append(logs.get("val_accuracy"))

    def on_train_end(self, logs=None):
        training_time = (time.time() - self.start_time) / 60
        index_best_model = np.argmax(self.val_scorings)
        best_model_val_scoring = np.abs(self.val_scorings[index_best_model])
        best_model_train_scoring = np.abs(self.train_scorings[index_best_model])
        best_epoch = index_best_model + 1

        self.best_result = (
            best_model_val_scoring,
            best_model_train_scoring,
            best_epoch,
            training_time,
        )
