import pandas as pd
from src.layers import *
from src.layer_lists import *
from tensorflow.keras.layers import (
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
)
from tensorflow.keras.optimizers import (
    RMSprop,
    SGD,
    Adam,
)


def create_object(param, layer_text):
    # Get param ứng với layer_text
    keys = pd.Series(param.keys())
    values = pd.Series(param.values())

    filter_mask = keys.str.startswith(layer_text)
    keys = keys[filter_mask]
    values = values[filter_mask]

    keys = keys.apply(get_param_name)
    layer_param = dict(zip(keys, values))

    # Tạo class
    class_name = layer_param.pop("name")
    ClassName = globals()[class_name]

    # Tạo object
    layer = ClassName(**layer_param)
    return layer


def get_param_name(key):
    parts = key.split("__", 1)
    return parts[1]
