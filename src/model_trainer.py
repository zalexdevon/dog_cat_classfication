import os
from Mylib import (
    myfuncs,
)
import tensorflow as tf
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue
import traceback
from src.create_ds import create_train_val_ds
from src.create_model import create_model
from src.create_object import create_object
from tensorflow.keras.callbacks import EarlyStopping


# Tìm theo accuracy tốt nhất
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
    # Get danh sách các tham số để tìm kiếm model tốt nhất
    list_param = get_list_param(param_dict, model_training_path, num_models)

    # Tạo thư mục lưu kết quả mô hình tốt nhất
    model_training_run_path = Path(
        f"{model_training_path}/{get_folder_name(model_training_path)}"
    )
    myfuncs.create_directories([model_training_run_path])

    # Save list_param
    myfuncs.save_python_object(
        Path(f"{model_training_run_path}/list_param.pkl"), list_param
    )

    # Save val scoring lớn nhất
    best_val_scoring_path = Path(f"{model_training_run_path}/best_val_scoring.pkl")
    myfuncs.save_python_object(best_val_scoring_path, -np.inf)

    # Save kết quả tốt nhất sau khi train model
    best_result_path = Path(f"{model_training_run_path}/best_result.pkl")
    myfuncs.save_python_object(best_result_path, -np.inf)

    # Duyệt qua từng tham số trong list_param
    for i, param in enumerate(list_param):
        print(f"Train model {i} / {num_models}")
        print(f"Param: {param}")
        p = Process(
            target=train_model,
            args=(
                param,
                best_result_path,
                best_val_scoring_path,
                loss,
                patience,
                min_delta,
                epochs,
                train_val_path,
                class_names,
            ),
        )
        p.start()
        p.join()

    # In ra kết quả của model tốt nhất
    best_model_result = myfuncs.load_python_object(best_result_path)
    print("Model tốt nhất")
    print(f"Param: {best_model_result[0]}")
    print(
        f"Val scoring: {best_model_result[1]}, Train scoring: {best_model_result[2]}, Best epoch: {best_model_result[3]}"
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
        # tạo train_ds, và val_ds
        train_ds, val_ds = create_train_val_ds(param, train_val_path, class_names)

        # Tạo callbacks
        callbacks = create_callbacks(patience, min_delta)

        # tạo optimizer
        optimizer = create_object(param, "optimizer")

        # tạo model
        model = create_model(param)

        # compile model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )

        # train model với callbacks
        model.fit(
            train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=callbacks,
        )

        # in kết quả
        val_scoring, train_scoring, best_epoch = callbacks[1].best_result
        print(
            f"Val scoring: {val_scoring}, Train scoring: {train_scoring}, Best epoch: {best_epoch}"
        )

        # Cập nhật best val scoring lớn nhất
        best_val_scoring = myfuncs.load_python_object(best_val_scoring_path)

        if best_val_scoring < val_scoring:
            myfuncs.save_python_object(best_val_scoring_path, val_scoring)

            # Lưu kết quả
            myfuncs.save_python_object(
                best_model_result_path,
                (param, val_scoring, train_scoring, best_epoch),
            )

    except Exception as e:
        # Nếu có exception thì bỏ qua vòng lặp đi
        print(f"Lỗi: {e}")
        traceback.print_exc()


def create_callbacks(patience, min_delta):
    earlystopping = EarlyStopping(
        monitor=f"val_accuracy",
        patience=patience,
        min_delta=min_delta,
        mode="max",
    )
    model_checkpoint = BestEpochResultSearcher()

    return [earlystopping, model_checkpoint]


def get_list_param(param_dict, model_training_path, num_models):
    # Get full_list_param
    full_list_param = myfuncs.get_full_list_dict(param_dict)

    # Get folder của run
    run_folders = os.listdir(model_training_path)

    if len(run_folders) > 0:
        # Get list param còn lại
        for run_folder in run_folders:
            list_param = myfuncs.load_python_object(
                Path(f"{model_training_path}/{run_folder}/list_param.pkl")
            )
            full_list_param = myfuncs.subtract_2list_set(full_list_param, list_param)

    # Random list
    return myfuncs.randomize_list(full_list_param, num_models)


def get_folder_name(model_training_path):
    # Get các folder lưu model tốt nhất
    run_folders = os.listdir(model_training_path)

    if len(run_folders) == 0:  # Lần đầu tiên chạy thì là run0
        return "run0"

    number_in_run_folders = run_folders.str.extract(r"(\d+)").astype("int")[
        0
    ]  # Các con số trong run0, run1, ... (0, 1, )
    folder_name = f"run{number_in_run_folders.max() +1}"  # Tên folder sẽ là số lớn nhất để prevent trùng
    return folder_name


class BestEpochResultSearcher(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        self.train_scorings = []
        self.val_scorings = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_scorings.append(logs.get("accuracy"))
        self.val_scorings.append(logs.get("val_accuracy"))

    def on_train_end(self, logs=None):
        # Tìm model ứng với val scoring tốt nhất
        index_best_model = np.argmax(self.val_scorings)
        best_model_val_scoring = np.abs(self.val_scorings[index_best_model])
        best_model_train_scoring = np.abs(self.train_scorings[index_best_model])
        best_epoch = index_best_model + 1

        self.best_result = (
            best_model_val_scoring,
            best_model_train_scoring,
            best_epoch,
        )
