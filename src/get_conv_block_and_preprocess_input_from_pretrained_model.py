from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import vgg19


def get_conv_block_and_preprocess_input_from_pretrained_model(model_name):
    conv_block = None
    preprocess_input = None
    if model_name == "VGG19":
        conv_block = VGG19(
            include_top=False,
        )
        conv_block = conv_block.layers[1:]
        preprocess_input = vgg19.preprocess_input
    else:
        raise ValueError(
            f"Chưa định nghĩa cho  {model_name} trong hàm get_conv_block_and_preprocess_input_from_pretrained_model"
        )

    return conv_block, preprocess_input
