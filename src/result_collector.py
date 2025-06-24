import os
import pandas as pd
from Mylib.myfuncs import load_python_object


# Collect theo accuracy lớn nhất
def collect(full_model_training_path):
    full_results = []

    # Get các thư mục model training
    list_model_training_name = os.listdir(full_model_training_path)
    list_model_training_path = [
        full_model_training_path / item for item in list_model_training_name
    ]

    for model_training_path, model_training_name in zip(
        list_model_training_path, list_model_training_name
    ):
        full_results += collect_from_one_model_training_path(
            model_training_path, model_training_name
        )

    # Sort theo val scoring (phần tử đầu trong tuple), giảm dần theo accuracy
    full_results = sorted(full_results, key=lambda item: item[0], reverse=True)

    # Get table để hiển thị rõ ràng
    table = show_on_table(full_results)

    return full_results, table


# Show theo table
def show_on_table(full_results):
    (
        list_val_scoring,
        list_model_training_name,
        list_run_folder_name,
        list_train_scoring,
        list_best_epoch,
    ) = zip(*full_results)
    return pd.DataFrame(
        data={
            "val": list_val_scoring,
            "mt": list_model_training_name,
            "run": list_run_folder_name,
            "train": list_train_scoring,
            "epoch": list_best_epoch,
        }
    )


def collect_from_one_model_training_path(model_training_path, model_training_name):
    results = []
    run_folder_names = os.listdir(model_training_path)
    run_folder_paths = [model_training_path / item for item in run_folder_names]

    for run_folder_path, run_folder_name in zip(run_folder_paths, run_folder_names):
        best_result_path = run_folder_path / "best_result.pkl"
        best_result = load_python_object(best_result_path)
        param, val_scoring, train_scoring, best_epoch = best_result
        results.append(
            (
                val_scoring,
                model_training_name,
                run_folder_name,
                train_scoring,
                best_epoch,
            )
        )

    return results
