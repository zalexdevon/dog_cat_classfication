import tensorflow as tf
from Mylib import myfuncs
import os
import pandas as pd
import plotly.express as px

SCORINGS_PREFER_MININUM = ["loss", "mse", "mae"]
SCORINGS_PREFER_MAXIMUM = ["accuracy", "roc_auc"]


def gather_result_from_model_training():
    components = []
    model_training_path = "artifacts/model_training"

    list_models_folder_path = [
        f"{model_training_path}/{item}" for item in os.listdir(model_training_path)
    ]
    for models_folder_path in list_models_folder_path:
        list_model_path = [
            f"{models_folder_path}/{item}" for item in os.listdir(models_folder_path)
        ]
        for model_path in list_model_path:
            result = myfuncs.load_python_object(f"{model_path}/result.pkl")
            components.append(result)

    return components


def get_reverse_param_in_sorted(scoring):
    return scoring in SCORINGS_PREFER_MAXIMUM


def get_list_best_models(scoring):
    list_result = gather_result_from_model_training()
    reverse_param_in_sort = get_reverse_param_in_sorted(scoring)
    list_result = sorted(list_result, key=lambda x: x[2], reverse=reverse_param_in_sort)
    model_indices, train_scorings, val_scorings, training_times, list_epochs = zip(
        *list_result
    )
    return pd.DataFrame(
        data={
            "model_index": model_indices,
            "train_scoring": train_scorings,
            "val_scoring": val_scorings,
            "training_time (s)": training_times,  # Thời gian theo second
            "epochs": list_epochs,  # Thời gian theo second
        }
    )
