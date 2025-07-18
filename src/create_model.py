from tensorflow.keras import layers, models
from src import create_object


def create_model(param):
    inputs = layers.Input(
        shape=(param["image_size"], param["image_size"], param["channels"])
    )

    resize_layer = layers.Resizing(param["image_size"], param["image_size"])
    data_augmentation_layer = create_object.create_object(param, "data_augmentation")
    rescaling_layer = layers.Rescaling(1.0 / 255)
    conv2D_layer = create_object.create_object(param, "conv")
    flatten_layer = create_object.create_object(param, "flatten")
    dense_layer = create_object.create_object(param, "dense")
    output_layer = layers.Dense(units=2, activation="softmax")

    x = resize_layer(inputs)
    x = data_augmentation_layer(x)
    x = rescaling_layer(x)
    x = conv2D_layer(x)
    x = flatten_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = models.Model(inputs, x)
    return model


def create_model_demoTfFunction(param):
    inputs = layers.Input(
        shape=(param["image_size"], param["image_size"], param["channels"])
    )

    resize_layer = layers.Resizing(param["image_size"], param["image_size"])
    flatten_layer = create_object.create_object(param, "flatten")
    dense_layer = create_object.create_object(param, "dense")
    output_layer = layers.Dense(units=2, activation="softmax")

    x = resize_layer(inputs)
    x = flatten_layer(x)
    x = dense_layer(x)
    x = output_layer(x)

    model = models.Model(inputs, x)
    return model
