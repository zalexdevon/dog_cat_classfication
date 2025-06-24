from src.layers import *
from tensorflow.keras import layers


class DenseLayerList(layers.Layer):
    def __init__(self, dropout_rate, list_units, do_have_last_layer, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.list_units = list_units
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "list_units": self.list_units,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        self.DenseLayers = [
            DenseLayer(units=units, dropout_rate=self.dropout_rate)
            for units in self.list_units
        ]
        self.lastDenseLayer = (
            DenseLayer(units=self.list_units[-1], dropout_rate=0)
            if self.do_have_last_layer
            else PassThroughLayer()
        )

        super().build(input_shape)

    def call(self, x):
        # Xử lí x
        for layer in self.DenseLayers:
            x = layer(x)

        x = self.lastDenseLayer(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class Conv2DBlockList(layers.Layer):
    def __init__(
        self, list_filters, num_conv_block, layer_name, do_have_last_layer, **kwargs
    ):
        super().__init__(**kwargs)
        self.list_filters = list_filters
        self.num_conv_block = num_conv_block
        self.layer_name = layer_name
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "list_filters": self.list_filters,
                "num_conv_block": self.num_conv_block,
                "layer_name": self.layer_name,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        ConvName = globals()[self.layer_name]

        self.Conv2DBlocks = [
            ConvName(
                filters=filters,
                num_conv_block=self.num_conv_block,
            )
            for filters in self.list_filters
        ]

        self.lastConv2DBlock = (
            Conv2DBlockNoMaxPooling(filters=self.list_filters[-1])
            if self.do_have_last_layer
            else PassThroughLayer()
        )

        super().build(input_shape)

    def call(self, x):
        # Xử lí x
        for layer in self.Conv2DBlocks:
            x = layer(x)

        x = self.lastConv2DBlock(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class Conv2DBlockListMixNumConvBlock(layers.Layer):
    def __init__(
        self, list_filters_num_conv_block, layer_name, do_have_last_layer, **kwargs
    ):
        super().__init__(**kwargs)
        self.list_filters_num_conv_block = list_filters_num_conv_block
        self.layer_name = layer_name
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "list_filters_num_conv_block": self.list_filters_num_conv_block,
                "layer_name": self.layer_name,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        ConvName = globals()[self.layer_name]

        self.Conv2DBlocks = [
            ConvName(
                filters=item[0],
                num_conv_block=item[1],
            )
            for item in self.list_filters_num_conv_block
        ]

        self.lastConv2DBlock = (
            Conv2DBlockNoMaxPooling(filters=self.list_filters_num_conv_block[-1][0])
            if self.do_have_last_layer
            else PassThroughLayer()
        )

        super().build(input_shape)

    def call(self, x):
        # Xử lí x
        for layer in self.Conv2DBlocks:
            x = layer(x)

        x = self.lastConv2DBlock(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class Conv2DBlockListMixNumConvBlockLayerName(layers.Layer):
    def __init__(
        self, list_filters_num_conv_block_layer_name, do_have_last_layer, **kwargs
    ):
        super().__init__(**kwargs)
        self.list_filters_num_conv_block_layer_name = (
            list_filters_num_conv_block_layer_name
        )
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "list_filters_num_conv_block_layer_name": self.list_filters_num_conv_block_layer_name,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        self.Conv2DBlocks = [
            globals()[item[2]](
                filters=item[0],
                num_conv_block=item[1],
            )
            for item in self.list_filters_num_conv_block_layer_name
        ]

        self.lastConv2DBlock = (
            Conv2DBlockNoMaxPooling(
                filters=self.list_filters_num_conv_block_layer_name[-1][0]
            )
            if self.do_have_last_layer
            else PassThroughLayer()
        )

        super().build(input_shape)

    def call(self, x):
        # Xử lí x
        for layer in self.Conv2DBlocks:
            x = layer(x)

        x = self.lastConv2DBlock(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)
