import os
from Mylib import (
    myfuncs,
    tf_create_object,
    tf_model_evaluator_classes,
)
import tensorflow as tf
import numpy as np
from pathlib import Path
from src import funcs, const
from multiprocessing import Process, Queue
import traceback
import random
import shutil


class BestEpochResultSearcher(tf.keras.callbacks.Callback):
    def __init__(self, sign_for_val_scoring_find_best_model, scoring):
        super().__init__()
        self.sign_for_val_scoring_find_best_model = sign_for_val_scoring_find_best_model
        self.scoring = scoring

    def on_train_begin(self, logs=None):
        self.train_scorings = []
        self.val_scorings = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_scorings.append(
            logs.get(self.scoring) * self.sign_for_val_scoring_find_best_model
        )
        self.val_scorings.append(
            logs.get(f"val_{self.scoring}") * self.sign_for_val_scoring_find_best_model
        )

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


class BestEpochResultSearcherWithModelSaving(tf.keras.callbacks.Callback):
    def __init__(
        self,
        sign_for_val_scoring_find_best_model,
        scoring,
    ):
        super().__init__()
        self.sign_for_val_scoring_find_best_model = sign_for_val_scoring_find_best_model
        self.scoring = scoring

    def on_train_begin(self, logs=None):
        self.train_scorings = []
        self.val_scorings = []
        self.model_weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_scorings.append(
            logs.get(self.scoring) * self.sign_for_val_scoring_find_best_model
        )
        self.val_scorings.append(
            logs.get(f"val_{self.scoring}") * self.sign_for_val_scoring_find_best_model
        )
        self.model_weights.append(self.model.get_weights())

    def on_train_end(self, logs=None):
        # Tìm model ứng với val scoring tốt nhất
        index_best_model = np.argmax(self.val_scorings)
        best_model_val_scoring = np.abs(self.val_scorings[index_best_model])
        best_model_train_scoring = np.abs(self.train_scorings[index_best_model])

        best_weights = self.model_weights[index_best_model]
        self.model.set_weights(best_weights)

        best_epoch = index_best_model + 1

        self.best_result = (
            best_model_val_scoring,
            best_model_train_scoring,
            best_epoch,
            self.model,
        )


class ModelTrainer:
    def __init__(
        self,
        model_training_path,
        num_models,
        scoring,
    ):
        self.model_training_path = model_training_path
        self.num_models = num_models
        self.scoring = scoring

    def train(self):
        # Tạo thư mục lưu kết quả mô hình tốt nhất
        list_param = funcs.get_list_param(self.model_training_path, self.num_models)
        model_training_run_path = Path(
            f"{self.model_training_path}/{funcs.get_folder_name(self.model_training_path)}"
        )
        myfuncs.create_directories([model_training_run_path])
        myfuncs.save_python_object(
            Path(f"{model_training_run_path}/list_param.pkl"), list_param
        )

        # Get các tham số cần thiết khác
        best_val_scoring_path = Path(f"{model_training_run_path}/best_val_scoring.pkl")
        myfuncs.save_python_object(best_val_scoring_path, -np.inf)
        sign_for_val_scoring_find_best_model = (
            funcs.get_sign_for_val_scoring_to_find_best_model(self.scoring)
        )

        best_model_result_path = Path(f"{model_training_run_path}/best_result.pkl")
        myfuncs.save_python_object(best_model_result_path, -np.inf)

        for i, param in enumerate(list_param):
            print(f"Train model {i} / {self.num_models}")
            print(f"Param: {param}")
            p = Process(
                target=self.train_model,
                args=(
                    param,
                    sign_for_val_scoring_find_best_model,
                    best_model_result_path,
                    best_val_scoring_path,
                ),
            )
            p.start()
            p.join()

        # In ra kết quả của model tốt nhất
        best_model_result = myfuncs.load_python_object(best_model_result_path)
        print("Model tốt nhất")
        print(f"Param: {best_model_result[0]}")
        print(
            f"Val scoring: {best_model_result[1]}, Train scoring: {best_model_result[2]}, Best epoch: {best_model_result[3]}"
        )

    def train_model(
        self,
        param,
        sign_for_val_scoring_find_best_model,
        best_model_result_path,
        best_val_scoring_path,
    ):
        try:
            # tạo train_ds, và val_ds
            train_ds, val_ds = funcs.create_train_val_ds(param)

            # Tạo callbacks
            callbacks = self.create_callbacks(
                param, sign_for_val_scoring_find_best_model
            )

            # tạo optimizer
            optimizer = tf_create_object.ObjectCreatorFromDict(
                param, "optimizer"
            ).next()

            # tạo model
            model = funcs.create_model(param)

            # compile model
            model.compile(
                optimizer=optimizer,
                loss=param["loss"],
                metrics=funcs.get_metrics(self.scoring),
            )

            # trian model với callbacks
            model.fit(
                train_ds,
                epochs=param["epochs"],
                verbose=1,
                validation_data=val_ds,
                callbacks=callbacks,
            )

            # in kết quả
            val_scoring, train_scoring, best_epoch = callbacks[1].best_result
            print(
                f"Val scoring: {val_scoring}, Train scoring: {train_scoring}, Best epoch: {best_epoch}"
            )

            # Cập nhật best model và lưu lại
            val_scoring_find_best_model = (
                val_scoring * sign_for_val_scoring_find_best_model
            )

            best_val_scoring = myfuncs.load_python_object(best_val_scoring_path)

            if best_val_scoring < val_scoring_find_best_model:
                myfuncs.save_python_object(
                    best_val_scoring_path, val_scoring_find_best_model
                )

                # Lưu kết quả
                myfuncs.save_python_object(
                    best_model_result_path,
                    (param, val_scoring, train_scoring, best_epoch),
                )

            # Giải phóng bộ nhớ model
        except Exception as e:
            # Nếu có exception thì bỏ qua vòng lặp đi
            print(f"Lỗi: {e}")
            traceback.print_exc()

    def create_callbacks(self, param, sign_for_val_scoring_find_best_model):
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=f"val_{self.scoring}",
            patience=param["patience"],
            min_delta=param["min_delta"],
            mode=funcs.get_mode_for_EarlyStopping(self.scoring),
        )
        model_checkpoint = BestEpochResultSearcher(
            sign_for_val_scoring_find_best_model, self.scoring
        )

        return [earlystopping, model_checkpoint]


class ModelTrainerWithModelSaving:
    def __init__(
        self,
        model_training_path,
        num_models,
        scoring,
    ):
        self.model_training_path = model_training_path
        self.num_models = num_models
        self.scoring = scoring

    def train(self):
        # Tạo thư mục lưu kết quả mô hình tốt nhất
        list_param = funcs.get_list_param(self.model_training_path, self.num_models)
        model_training_run_path = Path(
            f"{self.model_training_path}/{funcs.get_folder_name(self.model_training_path)}"
        )
        myfuncs.create_directories([model_training_run_path])
        myfuncs.save_python_object(
            Path(f"{model_training_run_path}/list_param.pkl"), list_param
        )

        # Get các tham số cần thiết khác
        best_val_scoring_path = Path(f"{model_training_run_path}/best_val_scoring.pkl")
        myfuncs.save_python_object(best_val_scoring_path, -np.inf)
        sign_for_val_scoring_find_best_model = (
            funcs.get_sign_for_val_scoring_to_find_best_model(self.scoring)
        )

        best_model_result_path = Path(f"{model_training_run_path}/best_result.pkl")
        best_model_path = Path(f"{model_training_run_path}/best_model.keras")
        myfuncs.save_python_object(best_model_result_path, -np.inf)

        for i, param in enumerate(list_param):
            print(f"Train model {i} / {self.num_models}")
            print(f"Param: {param}")
            p = Process(
                target=self.train_model,
                args=(
                    param,
                    sign_for_val_scoring_find_best_model,
                    best_model_result_path,
                    best_model_path,
                    best_val_scoring_path,
                ),
            )
            p.start()
            p.join()

        # In ra kết quả của model tốt nhất
        best_model_result = myfuncs.load_python_object(best_model_result_path)
        print("Model tốt nhất")
        print(f"Param: {best_model_result[0]}")
        print(
            f"Val scoring: {best_model_result[1]}, Train scoring: {best_model_result[2]}, Best epoch: {best_model_result[3]}"
        )

    def train_model(
        self,
        param,
        sign_for_val_scoring_find_best_model,
        best_model_result_path,
        best_model_path,
        best_val_scoring_path,
    ):
        try:
            # tạo train_ds, và val_ds
            train_ds, val_ds = funcs.create_train_val_ds(param)

            # Tạo callbacks
            callbacks = self.create_callbacks(
                param,
                sign_for_val_scoring_find_best_model,
            )

            # tạo optimizer
            optimizer = tf_create_object.ObjectCreatorFromDict(
                param, "optimizer"
            ).next()

            # tạo model
            model = funcs.create_model(param)

            # compile model
            model.compile(
                optimizer=optimizer,
                loss=param["loss"],
                metrics=funcs.get_metrics(self.scoring),
            )

            # trian model với callbacks
            model.fit(
                train_ds,
                epochs=param["epochs"],
                verbose=1,
                validation_data=val_ds,
                callbacks=callbacks,
            )

            # in kết quả
            val_scoring, train_scoring, best_epoch, model = callbacks[1].best_result
            print(
                f"Val scoring: {val_scoring}, Train scoring: {train_scoring}, Best epoch: {best_epoch}"
            )

            # Cập nhật model tốt nhất
            val_scoring_to_find_best_model = (
                val_scoring * sign_for_val_scoring_find_best_model
            )
            best_val_scoring = myfuncs.load_python_object(best_val_scoring_path)

            if val_scoring_to_find_best_model > best_val_scoring:
                myfuncs.save_python_object(
                    best_val_scoring_path, val_scoring_to_find_best_model
                )
                myfuncs.save_python_object(
                    best_model_result_path,
                    (param, val_scoring, train_scoring, best_epoch),
                )
                model.save(best_model_path)

        except Exception as e:
            # Nếu có exception thì bỏ qua vòng lặp đi
            print(f"Lỗi: {e}")
            traceback.print_exc()

    def create_callbacks(self, param, sign_for_val_scoring_find_best_model):
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=f"val_{self.scoring}",
            patience=param["patience"],
            min_delta=param["min_delta"],
            mode=funcs.get_mode_for_EarlyStopping(self.scoring),
        )
        model_checkpoint = BestEpochResultSearcherWithModelSaving(
            sign_for_val_scoring_find_best_model, self.scoring
        )

        return [earlystopping, model_checkpoint]


class ModelRetrainer:
    def __init__(self, best_param, train_ds, best_epoch, scoring, loss):
        self.best_param = best_param
        self.train_ds = train_ds
        self.best_epoch = best_epoch
        self.scoring = scoring
        self.loss = loss

    def next(self):
        # tạo model
        model = funcs.create_model(self.best_param)

        # compile
        optimizer = tf_create_object.ObjectCreatorFromDict(self.best_param, "optimizer")
        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=funcs.get_metrics(self.scoring),
        )

        # Train
        model.fit(
            self.train_ds,
            epochs=self.best_epoch,
            verbose=1,
        )

        return model


class BestResultSearcher:
    def __init__(self, model_training_path, scoring):
        self.model_training_path = model_training_path
        self.scoring = scoring

    def next(self):
        run_folders = funcs.get_run_folders(self.model_training_path)
        list_result = []

        for run_folder in run_folders:
            best_result_path = self.model_training_path / run_folder / "best_result.pkl"
            list_result.append(myfuncs.load_python_object(best_result_path))

        best_result = self.get_best_result(list_result)
        return best_result

    def get_best_result(self, list_result):
        list_result = sorted(
            list_result,
            key=lambda item: item[1],
            reverse=funcs.get_reverse_param_in_sorted(self.scoring),
        )  # Sort theo val scoring
        return list_result[0]


class BestResultAndModelSearcher:
    def __init__(self, model_training_path, scoring):
        self.model_training_path = model_training_path
        self.scoring = scoring

    def next(self):
        run_folders = funcs.get_run_folders(self.model_training_path)
        list_result = []

        for run_folder in run_folders:
            best_result = myfuncs.load_python_object(
                self.model_training_path / run_folder / "best_result.pkl"
            )
            val_scoring = best_result[1]
            list_result.append((val_scoring, run_folder))

        list_result = sorted(
            list_result,
            key=lambda item: item[0],
            reverse=funcs.get_reverse_param_in_sorted(self.scoring),
        )
        best_run_folder_path = self.model_training_path / list_result[0][1]
        best_result = myfuncs.load_python_object(
            best_run_folder_path / "best_result.pkl"
        )
        best_model = myfuncs.load_python_object(best_run_folder_path / "best_model.pkl")

        return best_result, best_model


class ModelEvalutor:
    def __init__(self, model, class_names, val_ds, model_evaluation_on_val_path):
        self.model = model
        self.class_names = class_names
        self.val_ds = val_ds
        self.model_evaluation_on_val_path = model_evaluation_on_val_path

    def next(self):
        # Đánh giá
        model_result_text = "===============Kết quả đánh giá model==================\n"

        # Đánh giá model trên tập val
        result_text, val_confusion_matrix = (
            tf_model_evaluator_classes.ClassifierEvaluator(
                model=self.model,
                class_names=self.class_names,
                train_ds=self.val_ds,
            ).evaluate()
        )
        model_result_text += result_text  # Thêm đoạn đánh giá vào

        # Lưu lại confusion matrix cho tập val
        val_confusion_matrix_path = Path(
            f"{self.model_evaluation_on_val_path}/confusion_matrix.png"
        )
        val_confusion_matrix.savefig(
            val_confusion_matrix_path, dpi=None, bbox_inches="tight", format=None
        )

        # Lưu vào file results.txt
        with open(
            Path(f"{self.model_evaluation_on_val_path}/result.txt"), mode="w"
        ) as file:
            file.write(model_result_text)


class ParamDictFixer:
    def __init__(self, param_dict):
        self.param_dict = param_dict

    def next(self):
        keys = list(self.param_dict.keys())
        values = list(self.param_dict.values())

        values = [item if isinstance(item, list) else [item] for item in values]

        values = [self.fix_value(item) for item in values]

        return dict(zip(keys, values))

    def fix_value(self, value):
        value = [tuple(item) if isinstance(item, list) else item for item in value]
        return value
